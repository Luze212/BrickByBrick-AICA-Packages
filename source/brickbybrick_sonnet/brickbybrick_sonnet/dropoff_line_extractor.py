"""
DropoffLineExtractor
─────────────────────────────────────────────────────────────────────────────
Event-getriebener Block. Feuert ausschließlich in Phase 1 (Exploration) und
nur wenn yolo_done_trigger eine steigende Flanke zeigt.

Zweck: Erkennt Ablagelinien im Bild ("Bernhards Algorithmus") und gibt
       fertige 3D-TCP-Ablageposen als flaches Array aus.

Bypass in Phase 2: Sobald trigger_ppl == True, kehrt der Callback sofort
zurück – kein CPU-Verbrauch, kein Schreiben auf den Output.

Output-Format line_ex_list:
  Stride 7 pro Ablagepose: [X, Y, Z, Qx, Qy, Qz, Qw, X2, Y2, Z2, ...]
  Quaternion-Reihenfolge: Qx, Qy, Qz, Qw (w am Ende – nicht AICA-native [w,x,y,z])!
  Alle Werte in Metern / Quaternion-Einheiten.
  AKTUEL!
"""

import matplotlib
matplotlib.use('Agg')  # Kein GUI-Fenster in AICA

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from modulo_components.lifecycle_component import LifecycleComponent
from modulo_core.encoded_state import EncodedState
import state_representation as sr
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Bool, Float64MultiArray
import traceback

# ============================================================
# MODUL-LEVEL VARIABLEN – werden vom Component vor jedem
# Pipeline-Aufruf über 'global' aktualisiert.
# ============================================================
APPLY_UNDISTORTION = False
SHOW_INTERMEDIATE_IMAGES = False
PIPELINE_ACTIVE = True
FALLBACK_Z_HEIGHT_MM = 700.0

# ── Steinhöhe: hier anpassen wenn sich der Stein ändert ──────────────────────
STONE_HEIGHT_MM = 10.0  # Höhe eines Legosteins in mm (Übergangswert)
MIN_LEN_PX = 30.0
L_MM = 35.0
W_MM = 10.0
CLEARANCE_MM = 1.0
DS_MM = 1.0
STEP_MM = L_MM * 0.95
SHIFT_DELTA_MM = 3.0
MAX_SHIFT_TRIES = 25

# Platzhalter für Funktionen die auf bgr0 aus globals() zugreifen
bgr0 = None
bgr = None
paper_mask = None
corners_global = None
_paper_commit_image_key = None

# ============================================================
# KAMERA-INTRINSIK UND ENTZERRUNGSPARAMETER (Konstanten)
# ============================================================
K = np.array([
    [960.2567, 0.0, 639.5],
    [0.0, 960.2567, 359.5],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

D = np.array([0.23155, -0.65347, 0.0, 0.0, 0.71843], dtype=np.float32)

fx_new = 1004.33728
fy_new = 1003.76068
cx_new = 639.05038
cy_new = 359.01649

P_new = np.array([
    [fx_new, 0.0, cx_new],
    [0.0, fy_new, cy_new],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

fx = float(P_new[0, 0])
fy = float(P_new[1, 1])
cx = float(P_new[0, 2])
cy = float(P_new[1, 2])

# ============================================================
# TCP→KAMERA TRANSFORM (Konstante, optional)
# ============================================================
TCP_TO_CAMERA_T_DEFAULT = np.array([
    [0.0, 1.0, 0.0, -0.0205],
    [-1.0, 0.0, 0.0, 0.0175],
    [0.0, 0.0, 1.0, -0.072475],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

# ============================================================
# STANDARD-PARAMETER
# ============================================================
DEFAULT_PAPER_PARAMS = dict(
    downscale=0.41,
    wL=0.7,
    wC=1.1,
    wT=10.0,
    bL=1.0,
    bC=0.4,
    bT=0.34,
    p_fg=65,
    p_sfg=76,
    p_bg=48,
    use_grabcut=True,
    gc_iters=4,
    ksize=9,
    close_iters=1,
    open_iters=6,
    band_frac=0.043,
    min_pts=170,
    out_w=2400,
)

DEFAULT_LINE_PARAMS = dict(
    erode_paper=12,
    border_margin=20,
    blackhat_ksize=61,
    close_ksize=5,
    close_iter=2,
    min_area=350,
    max_mean_width=7.0,
    bridge_gaps=True,
    bridge_dist=24,
    bridge_band=6,
    junction_dilate=5,
    port_cluster=3,
    pair_ang=61,
    merge_dist=18,
    merge_ang=39,
    min_len=10,
    simpl_eps=1.5,
    prune_spurs=False,
    prune_len=6,
)

paper_params_global = dict(DEFAULT_PAPER_PARAMS)
line_params_global = dict(DEFAULT_LINE_PARAMS)
_last_paper_preview = {"ok": False, "corners": None, "result": None}
_last_line_preview = {"ok": False, "params": None, "result": None}

# ============================================================
# ALGORITHMUS-FUNKTIONEN (unverändert vom Kollegen)
# ============================================================

def _compute_image_key(arr, image_path, input_bgr_used):
    """Kompakter Fingerprint zur Erkennung von Bildwechseln ohne teures Voll-Hashing."""
    a = np.asarray(arr)
    if a.size == 0:
        return ('empty', str(image_path), bool(input_bgr_used))
    flat = a.reshape(-1)
    step = max(1, flat.size // 12000)
    sample = flat[::step]
    checksum = int(np.sum(sample, dtype=np.uint64) % np.uint64(2**32))
    return (
        'input_bgr' if input_bgr_used else str(image_path),
        tuple(int(v) for v in a.shape),
        str(a.dtype),
        int(sample.size),
        checksum,
    )


def _yaw_rad_from_q0_qz(q0, qz=None):
    q0_clamped = float(np.clip(float(q0), -1.0, 1.0))
    yaw = 2.0 * float(np.arccos(q0_clamped))
    if qz is not None:
        qz_f = float(qz)
        if abs(qz_f) > 1e-12:
            yaw = float(np.sign(qz_f) * yaw)
    return yaw


def _q0_qz_from_yaw_rad(yaw):
    half = 0.5 * float(yaw)
    return float(np.cos(half)), float(np.sin(half))


def mask_from_corners(shape_hw, corners_xy):
    h, w = shape_hw[:2]
    m = np.zeros((h, w), np.uint8)
    pts = np.round(corners_xy).astype(np.int32)
    cv2.fillConvexPoly(m, pts, 255)
    return m


def paper_likelihood_map(bgr_small, wL, wC, wT, bL, bC, bT):
    lab = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2) / 128.0

    gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad = cv2.GaussianBlur(grad, (0, 0), 1.2)
    grad = grad / (grad.max() + 1e-6)

    S = (wL * (L - bL)) - (wC * (chroma - bC)) - (wT * (grad - bT))
    P = 1.0 / (1.0 + np.exp(-S))
    return P.astype(np.float32)


def build_mask_from_params(
    bgr,
    downscale,
    wL,
    wC,
    wT,
    bL,
    bC,
    bT,
    p_fg,
    p_sfg,
    p_bg,
    use_grabcut=True,
    gc_iters=4,
    ksize=11,
    close_iters=0,
    open_iters=1,
):
    ds = float(downscale)
    small = cv2.resize(bgr, (0, 0), fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
    P = paper_likelihood_map(small, wL, wC, wT, bL, bC, bT)

    th_fg = np.percentile(P, p_fg)
    th_sfg = np.percentile(P, p_sfg)
    th_bg = np.percentile(P, p_bg)

    th_prob = min(th_fg, th_sfg)
    th_sure = max(th_fg, th_sfg)

    prob_fg = P >= th_prob
    sure_fg = P >= th_sure
    sure_bg = P <= th_bg

    gc = np.full(P.shape, cv2.GC_PR_BGD, np.uint8)
    gc[prob_fg] = cv2.GC_PR_FGD
    gc[sure_fg] = cv2.GC_FGD
    gc[sure_bg] = cv2.GC_BGD

    if use_grabcut:
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, gc, None, bgdModel, fgdModel, int(gc_iters), cv2.GC_INIT_WITH_MASK)

    mask_small = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    k = int(ksize)
    if k % 2 == 0:
        k += 1
    k = max(3, k)

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if close_iters > 0:
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, ker, iterations=int(close_iters))
    if open_iters > 0:
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, ker, iterations=int(open_iters))

    mask_full = cv2.resize(mask_small, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return P, small, mask_small, mask_full


def largest_component(mask):
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = 1 + np.argmax(areas)
    return (lbl == i).astype(np.uint8) * 255


def get_largest_contour(binmask):
    if binmask is None:
        return None
    cnts, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def fit_line_trim(pts, trim_k=2.5):
    pts = pts.astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    p = np.array([x0, y0], np.float32)
    v = np.array([vx, vy], np.float32)

    d = pts - p
    dist = np.abs(v[0] * d[:, 1] - v[1] * d[:, 0])
    med = np.median(dist)
    keep = dist < max(2.0, trim_k * med)
    pts2 = pts[keep] if keep.sum() > 20 else pts

    vx, vy, x0, y0 = cv2.fitLine(pts2, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return np.array([x0, y0], np.float32), np.array([vx, vy], np.float32)


def intersect_lines(l1, l2):
    p1, v1 = l1
    p2, v2 = l2
    A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]], np.float32)
    b = (p2 - p1).astype(np.float32)
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None
    t, _ = np.linalg.solve(A, b)
    return p1 + t * v1


def order_points(pts4):
    pts = np.asarray(pts4, np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], np.float32)


def contour_to_quad_by_lines(cnt, band_frac=0.03, min_pts=200):
    pts = cnt.reshape(-1, 2).astype(np.float32)

    rect = cv2.minAreaRect(cnt)
    (cx_, cy_), _, ang = rect
    center = np.array([cx_, cy_], np.float32)

    theta = math.radians(ang)
    R = np.array(
        [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]],
        np.float32,
    )

    pr = (pts - center) @ R.T
    xmin, ymin = pr.min(axis=0)
    xmax, ymax = pr.max(axis=0)

    band_x = max(2.0, float(band_frac) * (xmax - xmin))
    band_y = max(2.0, float(band_frac) * (ymax - ymin))

    left_pts = pts[pr[:, 0] < (xmin + band_x)]
    right_pts = pts[pr[:, 0] > (xmax - band_x)]
    top_pts = pts[pr[:, 1] < (ymin + band_y)]
    bottom_pts = pts[pr[:, 1] > (ymax - band_y)]

    if min(len(left_pts), len(right_pts), len(top_pts), len(bottom_pts)) < int(min_pts):
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        if len(approx) == 4:
            return order_points(approx.reshape(-1, 2))
        return order_points(cv2.boxPoints(rect).astype(np.float32))

    L = fit_line_trim(left_pts)
    Rr = fit_line_trim(right_pts)
    T = fit_line_trim(top_pts)
    B = fit_line_trim(bottom_pts)

    tl = intersect_lines(L, T)
    tr = intersect_lines(Rr, T)
    br = intersect_lines(Rr, B)
    bl = intersect_lines(L, B)

    corners = [tl, tr, br, bl]
    if any(c is None for c in corners):
        return order_points(cv2.boxPoints(rect).astype(np.float32))

    return order_points(np.stack(corners, axis=0))


def warp_to_a4(bgr, corners, out_w=1200):
    corners = order_points(corners)
    out_w = int(out_w)
    out_h = int(round(out_w * 1.41421356))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    warp = cv2.warpPerspective(bgr, M, (out_w, out_h))
    return warp, M


def resolve_paper_runtime_inputs():
    """Liefert bgr/paper_mask fuer Pipelines. Fallback: Auto-Paper ohne Commit."""
    runtime_key = globals().get('_runtime_image_key', None)
    commit_key = globals().get('_paper_commit_image_key', None)

    committed_available = ("bgr" in globals()) and ("paper_mask" in globals())
    committed_valid_for_current_image = committed_available and (runtime_key is not None) and (commit_key == runtime_key)

    if committed_valid_for_current_image:
        corners_use = globals().get("corners_global", None)
        if corners_use is not None:
            corners_use = np.asarray(corners_use, dtype=np.float32).reshape(-1, 2)
        return bgr, paper_mask, corners_use, "committed"

    params = dict(DEFAULT_PAPER_PARAMS)
    params.update(dict(globals().get("paper_params_global", {})))

    bgr_use = np.asarray(bgr0).copy()
    paper_mask_use = np.ones(bgr_use.shape[:2], np.uint8) * 255
    corners_use = None
    mode = "auto-full"

    try:
        _, _, _, mask_full = build_mask_from_params(
            bgr_use,
            params["downscale"],
            params["wL"],
            params["wC"],
            params["wT"],
            params["bL"],
            params["bC"],
            params["bT"],
            params["p_fg"],
            params["p_sfg"],
            params["p_bg"],
            use_grabcut=params["use_grabcut"],
            gc_iters=params["gc_iters"],
            ksize=params["ksize"],
            close_iters=params["close_iters"],
            open_iters=params["open_iters"],
        )

        comp = largest_component(mask_full)
        if comp is not None and cv2.countNonZero(comp) > 0:
            cnt = get_largest_contour(comp)
            if cnt is not None:
                corners_tmp = contour_to_quad_by_lines(
                    cnt,
                    band_frac=params["band_frac"],
                    min_pts=params["min_pts"],
                )
                if corners_tmp is not None:
                    corners_arr = np.asarray(corners_tmp, dtype=np.float32).reshape(-1, 2)
                    if len(corners_arr) >= 4:
                        corners_use = corners_arr[:4].copy()
                        paper_mask_use = mask_from_corners(bgr_use.shape, corners_use)
                        mode = "auto-quad"
                    else:
                        paper_mask_use = comp
                        mode = "auto-mask"
                else:
                    paper_mask_use = comp
                    mode = "auto-mask"
            else:
                paper_mask_use = comp
                mode = "auto-mask"
    except Exception as e:
        globals()["_last_auto_paper_error"] = str(e)

    return bgr_use, paper_mask_use, corners_use, mode


def _paper_preview_from_params(params, commit=False, show_debug=True):
    global paper_params_global, bgr, corners_global, paper_mask, _paper_commit_image_key

    P, _, mask_small, mask_full = build_mask_from_params(
        bgr0,
        params["downscale"],
        params["wL"],
        params["wC"],
        params["wT"],
        params["bL"],
        params["bC"],
        params["bT"],
        params["p_fg"],
        params["p_sfg"],
        params["p_bg"],
        use_grabcut=params["use_grabcut"],
        gc_iters=params["gc_iters"],
        ksize=params["ksize"],
        close_iters=params["close_iters"],
        open_iters=params["open_iters"],
    )

    comp = largest_component(mask_full)
    cnt = get_largest_contour(comp)

    overlay = bgr0.copy()
    corners = None
    warp = None
    ok = False

    if cnt is not None and cv2.contourArea(cnt) >= 1000:
        corners = contour_to_quad_by_lines(
            cnt,
            band_frac=params["band_frac"],
            min_pts=params["min_pts"],
        )
        if corners is not None:
            ok = True
            cv2.polylines(
                overlay,
                [np.round(corners).astype(np.int32)],
                True,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
            warp, _ = warp_to_a4(bgr0, corners, out_w=params["out_w"])

    corners_np = corners.copy().astype(np.float32) if corners is not None else None
    _last_paper_preview["ok"] = ok
    _last_paper_preview["corners"] = corners_np
    _last_paper_preview["result"] = dict(
        likelihood=P,
        mask_small=mask_small,
        mask_full=mask_full,
        largest_component=comp,
        overlay=overlay,
        warped_a4=warp,
    )

    if show_debug:
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 3, 1)
        plt.title("Mask full")
        plt.imshow(mask_full, cmap="gray")
        plt.axis("off")
        plt.subplot(2, 3, 2)
        plt.title("Largest component")
        plt.imshow(comp if comp is not None else np.zeros_like(mask_full), cmap="gray")
        plt.axis("off")
        plt.subplot(2, 3, 3)
        plt.title("Quad overlay")
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.subplot(2, 3, 4)
        plt.title("Mask small")
        plt.imshow(mask_small, cmap="gray")
        plt.axis("off")
        plt.subplot(2, 3, 5)
        plt.title("Likelihood P")
        plt.imshow(P, cmap="magma")
        plt.axis("off")
        plt.subplot(2, 3, 6)
        plt.title("Warped A4")
        if warp is None:
            plt.imshow(np.zeros((10, 10, 3), dtype=np.uint8))
        else:
            plt.imshow(cv2.cvtColor(warp, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    if commit:
        if not ok:
            print("\nCommit abgebrochen: Preview ist nicht valide.")
            return _last_paper_preview
        paper_params_global = dict(params)
        bgr = bgr0.copy()
        corners_global = corners_np.copy().astype(np.float32)
        paper_mask = mask_from_corners(bgr.shape, corners_global)
        _paper_commit_image_key = globals().get("_runtime_image_key", None)
        print("\nCommitted: paper_params_global, bgr, corners_global, paper_mask")

    return _last_paper_preview


def remove_small_components(binmask_u8, min_area=250):
    num, lbl, stats, _ = cv2.connectedComponentsWithStats((binmask_u8 > 0).astype(np.uint8), 8)
    out = np.zeros_like(binmask_u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lbl == i] = 255
    return out


def skeletonize(binmask_u8):
    binmask_u8 = (binmask_u8 > 0).astype(np.uint8) * 255

    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binmask_u8, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    skel = np.zeros_like(binmask_u8)
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binmask_u8.copy()
    while True:
        eroded = cv2.erode(img, elem)
        opened = cv2.dilate(eroded, elem)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def marker_mask_from_paper(
    bgr,
    paper_mask_u8,
    erode_paper=0,
    border_margin=0,
    blackhat_ksize=61,
    close_ksize=9,
    close_iter=2,
    min_area=250,
):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if paper_mask_u8 is None:
        raise ValueError("paper_mask_u8 is None")

    pm = (paper_mask_u8 > 0).astype(np.uint8) * 255
    if pm.shape[:2] != gray.shape[:2]:
        pm = cv2.resize(pm, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    if erode_paper > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_paper + 1, 2 * erode_paper + 1))
        pm = cv2.erode(pm, k, 1)

    if border_margin > 0:
        dist = cv2.distanceTransform((pm > 0).astype(np.uint8), cv2.DIST_L2, 3)
        inner = (dist >= float(border_margin)).astype(np.uint8) * 255
    else:
        inner = pm.copy()

    roi = gray.copy()
    roi[inner == 0] = 255

    kbh = cv2.getStructuringElement(cv2.MORPH_RECT, (blackhat_ksize, blackhat_ksize))
    bh = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, kbh)

    _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kcl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kcl, iterations=close_iter)
    th = cv2.bitwise_and(th, inner)

    th = remove_small_components(th, min_area=min_area)
    return th, bh, inner


def filter_components_by_width(marker_mask_u8, max_mean_width=7.0):
    m = (marker_mask_u8 > 0).astype(np.uint8) * 255
    num, lbl, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)

    out = np.zeros_like(m)
    for i in range(1, num):
        comp = (lbl == i).astype(np.uint8) * 255
        area = float(stats[i, cv2.CC_STAT_AREA])
        sk = skeletonize(comp)
        length = float(cv2.countNonZero(sk))
        mean_w = area / (length + 1e-6)
        if mean_w <= max_mean_width:
            out[comp > 0] = 255

    return out


def pruned_neighbors(skel_bool, y, x):
    h, w = skel_bool.shape
    nbs = []

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and skel_bool[ny, nx]:
            nbs.append((ny, nx))

    for dy, dx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and skel_bool[ny, nx]:
            if (0 <= y + dy < h and skel_bool[y + dy, x]) and (0 <= x + dx < w and skel_bool[y, x + dx]):
                continue
            nbs.append((ny, nx))

    return nbs


def degree_map_pruned(skel_bool):
    deg = np.zeros(skel_bool.shape, np.uint8)
    ys, xs = np.where(skel_bool)
    for y, x in zip(ys, xs):
        deg[int(y), int(x)] = len(pruned_neighbors(skel_bool, int(y), int(x)))
    return deg


def prune_spurs(skel_u8, max_len=20, iterations=3):
    skel = (skel_u8 > 0).astype(np.uint8)
    for _ in range(iterations):
        skel_bool = skel.astype(bool)
        deg = degree_map_pruned(skel_bool)
        endpoints = list(zip(*np.where(skel_bool & (deg == 1))))
        if not endpoints:
            break

        removed_any = False
        for ep in endpoints:
            path = [ep]
            prev = None
            curr = ep
            while True:
                nbs = pruned_neighbors(skel_bool, curr[0], curr[1])
                if prev is not None:
                    nbs = [p for p in nbs if p != prev]
                if len(nbs) != 1:
                    break
                nxt = nbs[0]
                path.append(nxt)
                prev, curr = curr, nxt

                if deg[curr[0], curr[1]] != 2:
                    break
                if len(path) > max_len:
                    break

            if len(path) <= max_len:
                for y, x in path:
                    skel[y, x] = 0
                removed_any = True

        if not removed_any:
            break

    return (skel * 255).astype(np.uint8)


def bridge_gaps_endpoints(skel_u8, marker_mask_u8, max_dist=15, band_dilate=5):
    skel = (skel_u8 > 0).astype(np.uint8)
    skel_bool = skel.astype(bool)

    deg = degree_map_pruned(skel_bool)
    eps = list(zip(*np.where(skel_bool & (deg == 1))))
    if len(eps) < 2:
        return (skel * 255).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_dilate + 1, 2 * band_dilate + 1))
    band = cv2.dilate((marker_mask_u8 > 0).astype(np.uint8) * 255, k, 1).astype(bool)

    used = set()
    for i in range(len(eps)):
        if i in used:
            continue

        y1, x1 = eps[i]
        best_j = None
        best_d2 = (max_dist + 1) ** 2

        for j in range(i + 1, len(eps)):
            if j in used:
                continue
            y2, x2 = eps[j]
            d2 = (y2 - y1) ** 2 + (x2 - x1) ** 2
            if d2 >= best_d2:
                continue
            tmp = np.zeros_like(skel, np.uint8)
            cv2.line(tmp, (x1, y1), (x2, y2), 1, 1, cv2.LINE_8)
            if np.all(band[tmp.astype(bool)]):
                best_d2 = d2
                best_j = j

        if best_j is not None:
            y2, x2 = eps[best_j]
            cv2.line(skel, (x1, y1), (x2, y2), 1, 1, cv2.LINE_8)
            used.add(i)
            used.add(best_j)

    return (skel * 255).astype(np.uint8)


def cluster_junctions_from_deg(skel_bool, deg, dilate_ksize=15):
    junc = skel_bool & (deg >= 3)
    if not junc.any():
        return {}, []

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
    junc_d = cv2.dilate(junc.astype(np.uint8) * 255, k, 1)

    num, lbl, _, cent = cv2.connectedComponentsWithStats((junc_d > 0).astype(np.uint8), 8)

    mapping = {}
    reps = []
    for i in range(1, num):
        ys, xs = np.where(lbl == i)

        cand = np.stack(np.where(junc & (lbl == i)), axis=1)
        if cand.size == 0:
            cand = np.stack([ys, xs], axis=1)

        cy_, cx_ = cent[i]
        d2 = (cand[:, 0] - cy_) ** 2 + (cand[:, 1] - cx_) ** 2
        ry, rx = map(int, cand[int(np.argmin(d2))])
        reps.append((ry, rx))

        for y, x in zip(ys, xs):
            mapping[(int(y), int(x))] = (ry, rx)

    return mapping, reps


def extract_skeleton_graph_ports(skel_u8, junction_dilate_ksize=15, port_cluster_dilate=3):
    out = {"nodes": [], "edges": [], "loops": [], "polylines": [], "deg": None, "ports": []}
    skel_bool = skel_u8 > 0
    if not skel_bool.any():
        return out

    deg = degree_map_pruned(skel_bool)
    out["deg"] = deg

    jmap, jreps = cluster_junctions_from_deg(skel_bool, deg, dilate_ksize=junction_dilate_ksize)

    blob_mask = np.zeros_like(skel_bool, bool)
    for y, x in jmap.keys():
        blob_mask[y, x] = True

    skel_core = skel_bool & (~blob_mask)

    deg_core = degree_map_pruned(skel_core)
    endpoints = list(zip(*np.where(skel_core & (deg_core == 1))))
    isolated = list(zip(*np.where(skel_core & (deg_core == 0))))

    node_px = list(dict.fromkeys([(int(y), int(x)) for y, x in (jreps + endpoints + isolated)]))
    node_id = {p: i for i, p in enumerate(node_px)}
    out["nodes"] = [(p[1], p[0]) for p in node_px]

    h, w = skel_core.shape
    ports_raw = []
    ys, xs = np.where(skel_core)
    for y, x in zip(ys, xs):
        y = int(y)
        x = int(x)
        if deg_core[y, x] == 0:
            continue

        attached = None
        for ny, nx in pruned_neighbors(skel_bool, y, x):
            if 0 <= ny < h and 0 <= nx < w and blob_mask[ny, nx]:
                rep = jmap.get((ny, nx), None)
                if rep is not None:
                    attached = rep
                    break

        if attached is not None:
            ports_raw.append(((y, x), attached))

    ports_by_rep = defaultdict(list)
    for p, rep in ports_raw:
        ports_by_rep[rep].append(p)

    ports = []
    for rep, plist in ports_by_rep.items():
        if len(plist) == 1:
            ports.append((plist[0], rep))
            continue

        tmp = np.zeros((h, w), np.uint8)
        for py, px in plist:
            tmp[py, px] = 255

        if port_cluster_dilate > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * port_cluster_dilate + 1, 2 * port_cluster_dilate + 1)
            )
            tmp = cv2.dilate(tmp, k, 1)

        num, lbl = cv2.connectedComponents((tmp > 0).astype(np.uint8), 8)

        ry, rx = rep
        for ci in range(1, num):
            members = [(py, px) for (py, px) in plist if lbl[py, px] == ci]
            if not members:
                for py, px in plist:
                    if lbl[max(0, py - 1): min(h, py + 2), max(0, px - 1): min(w, px + 2)].max() == ci:
                        members.append((py, px))
                if not members:
                    continue

            d2 = [(py - ry) ** 2 + (px - rx) ** 2 for py, px in members]
            ports.append((members[int(np.argmax(d2))], rep))

    dedup = {}
    for p, rep in ports:
        dedup[p] = rep
    ports = [(p, rep) for p, rep in dedup.items()]

    out["ports"] = [{"port": (p[1], p[0]), "node": (rep[1], rep[0])} for p, rep in ports]

    core_stop = set(endpoints) | set(isolated)
    port_map = {p: rep for p, rep in ports}

    visited_dir = set()

    def step_key(a, b):
        return a[0], a[1], b[0], b[1]

    def trace_from(start_px, next_px):
        poly = [(start_px[1], start_px[0]), (next_px[1], next_px[0])]
        prev, curr = start_px, next_px

        for _ in range(200000):
            if curr in core_stop and curr != start_px:
                return poly, curr
            if curr in port_map and curr != start_px:
                return poly, curr

            nbs = [p for p in pruned_neighbors(skel_core, curr[0], curr[1]) if p != prev]
            nbs = list(dict.fromkeys(nbs))
            if len(nbs) != 1:
                return poly, curr

            nxt = nbs[0]
            if step_key(curr, nxt) in visited_dir:
                return poly, curr

            prev, curr = curr, nxt
            poly.append((curr[1], curr[0]))

        return poly, curr

    def node_for(px):
        if px in port_map:
            return node_id[port_map[px]]
        if px in node_id:
            return node_id[px]
        node_id[px] = len(out["nodes"])
        out["nodes"].append((px[1], px[0]))
        return node_id[px]

    edges = []
    start_points = set(core_stop) | set(port_map.keys())

    for s in list(start_points):
        for nb in pruned_neighbors(skel_core, s[0], s[1]):
            if step_key(s, nb) in visited_dir:
                continue

            visited_dir.add(step_key(s, nb))
            poly, end_px = trace_from(s, nb)

            for i in range(1, len(poly)):
                a = (poly[i - 1][1], poly[i - 1][0])
                b = (poly[i][1], poly[i][0])
                visited_dir.add(step_key(a, b))

            pl = np.array(poly, np.int32)
            if len(pl) >= 2:
                edges.append({"u": node_for(s), "v": node_for(end_px), "polyline": pl})

    out["edges"] = edges
    out["polylines"] = [e["polyline"] for e in edges]
    return out


def _unit(v):
    n = float(np.linalg.norm(v)) + 1e-9
    return (v / n).astype(np.float32)


def _angle_deg(u, v):
    c = float(np.clip(np.dot(_unit(u), _unit(v)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _edge_dir_at_node(edge_poly, node_xy, k=8):
    pl = edge_poly.astype(np.float32)
    node = np.array(node_xy, np.float32)

    d0 = np.linalg.norm(pl[0] - node)
    d1 = np.linalg.norm(pl[-1] - node)

    if d0 <= d1:
        j = min(len(pl) - 1, k)
        v = pl[j] - pl[0]
        start_at_node = True
    else:
        j = max(0, len(pl) - 1 - k)
        v = pl[j] - pl[-1]
        start_at_node = False

    return _unit(v), start_at_node


def build_strokes_cover_all(graph, angle_threshold_deg=70, dir_k=10):
    nodes = graph["nodes"]
    edges = graph["edges"]
    if not edges:
        return [], {}

    adj = defaultdict(list)
    for ei, e in enumerate(edges):
        adj[e["u"]].append(ei)
        adj[e["v"]].append(ei)

    visited = np.zeros(len(edges), dtype=bool)
    strokes = []

    def oriented_poly(ei, from_node):
        pl = edges[ei]["polyline"]
        u, v = edges[ei]["u"], edges[ei]["v"]
        if u == v:
            return pl
        _, start_at = _edge_dir_at_node(pl, nodes[from_node], k=dir_k)
        return pl if start_at else pl[::-1].copy()

    node_degree = {n: len(adj[n]) for n in adj}

    for ei0 in range(len(edges)):
        if visited[ei0]:
            continue

        e0 = edges[ei0]
        u0, v0 = e0["u"], e0["v"]

        cand = [u0, v0]
        cand.sort(key=lambda n: (node_degree.get(n, 0) == 2, -node_degree.get(n, 0)))
        curr_node = cand[0]

        stroke_pts = oriented_poly(ei0, curr_node).copy()
        visited[ei0] = True

        prev_edge = ei0
        curr_node = v0 if curr_node == u0 else u0

        while True:
            incoming = _unit(stroke_pts[-1].astype(np.float32) - stroke_pts[-2].astype(np.float32)) if len(stroke_pts) >= 2 else np.array([1.0, 0.0], np.float32)

            best = None
            for ei in adj.get(curr_node, []):
                if visited[ei] or ei == prev_edge:
                    continue

                dir_out, _ = _edge_dir_at_node(edges[ei]["polyline"], nodes[curr_node], k=dir_k)
                ang = _angle_deg(incoming, dir_out)
                if ang <= angle_threshold_deg and (best is None or ang < best[0]):
                    best = (ang, ei)

            if best is None:
                break

            _, ei_next = best
            eN = edges[ei_next]
            uN, vN = eN["u"], eN["v"]
            visited[ei_next] = True

            plN = oriented_poly(ei_next, curr_node)
            stroke_pts = np.vstack([stroke_pts, plN[1:]])

            prev_edge = ei_next
            curr_node = vN if curr_node == uN else uN

        strokes.append(stroke_pts.astype(np.int32))

    dbg = {"uncovered_edges": np.where(~visited)[0].tolist(), "visited_ratio": float(np.mean(visited))}
    return strokes, dbg


def strokes_to_mask(shape_hw, strokes, thickness=1):
    h, w = shape_hw[:2]
    m = np.zeros((h, w), np.uint8)
    for pl in strokes:
        if pl is None or len(pl) < 2:
            continue
        cv2.polylines(m, [pl.reshape(-1, 1, 2)], False, 255, thickness, cv2.LINE_8)
    return m


def ensure_skeleton_covered_by_strokes(
    strokes,
    skeleton_u8,
    min_cc_area=25,
    junc_dilate=3,
    port_cluster=1,
    angle_thr=85,
):
    sk = (skeleton_u8 > 0).astype(np.uint8) * 255
    if cv2.countNonZero(sk) == 0:
        return strokes

    sm = strokes_to_mask(sk.shape, strokes, thickness=1)
    miss = cv2.bitwise_and(sk, cv2.bitwise_not(sm))
    if cv2.countNonZero(miss) == 0:
        return strokes

    num, lbl, stats, _ = cv2.connectedComponentsWithStats((miss > 0).astype(np.uint8), 8)
    extra = []

    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_cc_area:
            continue

        comp = (lbl == i).astype(np.uint8) * 255
        g = extract_skeleton_graph_ports(
            comp,
            junction_dilate_ksize=junc_dilate,
            port_cluster_dilate=port_cluster,
        )
        st, _ = build_strokes_cover_all(g, angle_threshold_deg=angle_thr)
        extra.extend(st)

    return strokes + extra if extra else strokes


def polyline_length(pl):
    if pl is None or len(pl) < 2:
        return 0.0
    d = np.diff(pl.astype(np.float32), axis=0)
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))


def end_tangent(pl, at_end=True, k=8):
    if pl is None or len(pl) < 2:
        return np.array([1.0, 0.0], np.float32)

    pl = pl.astype(np.float32)
    if at_end:
        i0 = max(0, len(pl) - 1 - k)
        v = pl[-1] - pl[i0]
    else:
        i1 = min(len(pl) - 1, k)
        v = pl[i1] - pl[0]

    n = np.linalg.norm(v) + 1e-9
    return (v / n).astype(np.float32)


def angle_deg(u, v):
    u = u / (np.linalg.norm(u) + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def dedup_strokes(strokes, eps_end=2.0):
    kept, seen = [], set()
    for pl in strokes:
        if pl is None or len(pl) < 2:
            continue

        a = tuple(np.round(pl[0].astype(np.float32) / eps_end).astype(int))
        b = tuple(np.round(pl[-1].astype(np.float32) / eps_end).astype(int))
        key = (a, b, int(round(polyline_length(pl)))) if a <= b else (b, a, int(round(polyline_length(pl))))
        if key in seen:
            continue

        seen.add(key)
        kept.append(pl)

    return kept


def merge_two(pl1, pl2, connect_mode):
    if connect_mode == 0:
        return np.vstack([pl1, pl2[1:]])
    if connect_mode == 1:
        return np.vstack([pl1, pl2[::-1][1:]])
    if connect_mode == 2:
        return np.vstack([pl1[::-1], pl2[1:]])
    if connect_mode == 3:
        return np.vstack([pl1[::-1], pl2[::-1][1:]])
    raise ValueError("bad connect_mode")


def postprocess_strokes(
    strokes,
    min_len_px=15,
    simplify_eps=1.5,
    merge_dist_px=14,
    merge_angle_deg=30,
    max_merge_iters=80,
):
    simp = []
    for pl in strokes:
        if pl is None or len(pl) < 2:
            continue
        appr = cv2.approxPolyDP(pl.reshape(-1, 1, 2).astype(np.int32), simplify_eps, False).reshape(-1, 2)
        if len(appr) >= 2:
            simp.append(appr)

    simp = dedup_strokes(simp, eps_end=2.0)

    for _ in range(max_merge_iters):
        if len(simp) < 2:
            break

        best = None
        for i in range(len(simp)):
            for j in range(i + 1, len(simp)):
                A, B = simp[i], simp[j]
                a0, a1 = A[0].astype(np.float32), A[-1].astype(np.float32)
                b0, b1 = B[0].astype(np.float32), B[-1].astype(np.float32)

                d = [
                    np.linalg.norm(a1 - b0),
                    np.linalg.norm(a1 - b1),
                    np.linalg.norm(a0 - b0),
                    np.linalg.norm(a0 - b1),
                ]
                mode = int(np.argmin(d))
                dist = float(d[mode])
                if dist > merge_dist_px:
                    continue

                if mode == 0:
                    tA, tB = end_tangent(A, at_end=True), end_tangent(B, at_end=False)
                elif mode == 1:
                    tA, tB = end_tangent(A, at_end=True), -end_tangent(B, at_end=True)
                elif mode == 2:
                    tA, tB = -end_tangent(A, at_end=False), end_tangent(B, at_end=False)
                else:
                    tA, tB = -end_tangent(A, at_end=False), -end_tangent(B, at_end=True)

                ang = angle_deg(tA, tB)
                if ang > merge_angle_deg:
                    continue

                score = dist + 0.25 * ang
                if best is None or score < best[0]:
                    best = (score, i, j, mode)

        if best is None:
            break

        _, i, j, mode = best
        merged = merge_two(simp[i], simp[j], mode)
        new_list = [merged if k == i else pl for k, pl in enumerate(simp) if k != j]
        simp = dedup_strokes(new_list, eps_end=2.0)

    return [pl for pl in simp if polyline_length(pl) >= float(min_len_px)]


def draw_polylines_overlay(bgr, polylines, color=(0, 0, 255), thickness=2):
    out = bgr.copy()
    for pl in polylines:
        if pl is None or len(pl) < 2:
            continue
        cv2.polylines(out, [pl.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)
    return out


def run_line_pipeline(bgr, paper_mask, params, show_debug=True):
    marker_mask, _, _ = marker_mask_from_paper(
        bgr,
        paper_mask,
        erode_paper=params["erode_paper"],
        border_margin=params["border_margin"],
        blackhat_ksize=params["blackhat_ksize"],
        close_ksize=params["close_ksize"],
        close_iter=params["close_iter"],
        min_area=params["min_area"],
    )

    marker_mask = filter_components_by_width(marker_mask, max_mean_width=float(params["max_mean_width"]))
    skeleton = skeletonize(marker_mask)

    if params.get("bridge_gaps", True) and int(params["bridge_dist"]) > 0:
        skeleton = bridge_gaps_endpoints(
            skeleton,
            marker_mask,
            max_dist=int(params["bridge_dist"]),
            band_dilate=int(params["bridge_band"]),
        )

    if params.get("prune_spurs", False):
        skeleton = prune_spurs(skeleton, max_len=int(params["prune_len"]), iterations=2)

    graph = extract_skeleton_graph_ports(
        skeleton,
        junction_dilate_ksize=int(params["junction_dilate"]),
        port_cluster_dilate=int(params["port_cluster"]),
    )

    strokes, dbg_cover = build_strokes_cover_all(graph, angle_threshold_deg=float(params["pair_ang"]))

    strokes = postprocess_strokes(
        strokes,
        min_len_px=0,
        simplify_eps=float(params["simpl_eps"]),
        merge_dist_px=float(params["merge_dist"]),
        merge_angle_deg=float(params["merge_ang"]),
    )

    strokes = ensure_skeleton_covered_by_strokes(
        strokes,
        skeleton,
        min_cc_area=25,
        junc_dilate=3,
        port_cluster=1,
        angle_thr=85,
    )

    strokes = postprocess_strokes(
        strokes,
        min_len_px=float(params["min_len"]),
        simplify_eps=float(params["simpl_eps"]),
        merge_dist_px=float(params["merge_dist"]),
        merge_angle_deg=float(params["merge_ang"]),
    )

    overlay = draw_polylines_overlay(bgr, strokes, (0, 0, 255), 2)

    if show_debug:
        plt.figure(figsize=(16, 5))
        plt.subplot(1, 3, 1)
        plt.title("marker_mask")
        plt.imshow(marker_mask, cmap="gray")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.title("skeleton")
        plt.imshow(skeleton, cmap="gray")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.title(f"strokes ({len(strokes)})")
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return dict(
        marker_mask=marker_mask,
        skeleton=skeleton,
        graph=graph,
        strokes=strokes,
        overlay=overlay,
        debug=dbg_cover,
    )


def _line_preview_from_params(params, commit=False, show_debug=True):
    global line_params_global

    bgr_use, paper_mask_use, _, paper_mode = resolve_paper_runtime_inputs()
    result = run_line_pipeline(bgr_use, paper_mask_use, params, show_debug=bool(show_debug))

    _last_line_preview["ok"] = True
    _last_line_preview["params"] = dict(params)
    _last_line_preview["result"] = result

    if commit:
        line_params_global = dict(params)
        print("Committed: line_params_global")

    return result, paper_mode


def undistort_points_px(points_xy, K=K, D=D, P=P_new):
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if not APPLY_UNDISTORTION:
        return pts
    und = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, D, P=P)
    return und[:, 0, :]


def undistort_strokes_px(strokes):
    return [undistort_points_px(pl) for pl in strokes if pl is not None and len(pl) > 0]


def undistort_image(bgr, K=K, D=D, P=P_new):
    if not APPLY_UNDISTORTION:
        return np.asarray(bgr).copy()
    return cv2.undistort(bgr, K, D, None, P)


def undpx_to_cam_mm_robot_frame(pts_und_px, z_mm):
    """Undistorted Pixel -> Kamera-mm mit +x oben, +y links, Ursprung Bildmitte."""
    pts = np.asarray(pts_und_px, dtype=float)
    u = pts[:, 0]
    v = pts[:, 1]
    x_up_mm = (cy - v) * z_mm / fy
    y_left_mm = (cx - u) * z_mm / fx
    return np.stack([x_up_mm, y_left_mm], axis=1)


def cam_mm_robot_frame_to_undpx(pts_mm, z_mm):
    """Kamera-mm (+x oben, +y links) -> undistorted Pixel."""
    pts = np.asarray(pts_mm, dtype=float)
    x_up_mm = pts[:, 0]
    y_left_mm = pts[:, 1]
    u = cx - y_left_mm * fx / z_mm
    v = cy - x_up_mm * fy / z_mm
    return np.stack([u, v], axis=1)


def resample_polyline_mm(pts, ds=1.0):
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts

    seg = pts[1:] - pts[:-1]
    seglen = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = s[-1]
    if total < 1e-9:
        return pts[:1].copy()

    n = int(np.floor(total / ds)) + 1
    s_new = np.linspace(0.0, total, n)

    out = np.zeros((n, 2), float)
    j = 0
    for i, si in enumerate(s_new):
        while j < len(seglen) - 1 and s[j + 1] < si:
            j += 1
        if seglen[j] < 1e-12:
            out[i] = pts[j]
        else:
            t = (si - s[j]) / seglen[j]
            out[i] = pts[j] + t * seg[j]
    return out


def rect_corners(center, theta, L, W):
    cx0, cy0 = center
    c, s_ = np.cos(theta), np.sin(theta)
    ux = np.array([c, s_])
    uy = np.array([-s_, c])
    hl = 0.5 * L
    hw = 0.5 * W
    return np.array([
        [cx0, cy0] + ux * hl + uy * hw,
        [cx0, cy0] + ux * hl - uy * hw,
        [cx0, cy0] - ux * hl - uy * hw,
        [cx0, cy0] - ux * hl + uy * hw,
    ], dtype=float)


def _proj_interval(poly, axis):
    dots = poly @ axis
    return dots.min(), dots.max()


def rects_intersect(polyA, polyB):
    for poly in (polyA, polyB):
        for i in range(4):
            p0 = poly[i]
            p1 = poly[(i + 1) % 4]
            edge = p1 - p0
            axis = np.array([-edge[1], edge[0]])
            n = np.linalg.norm(axis)
            if n < 1e-12:
                continue
            axis /= n
            a0, a1 = _proj_interval(polyA, axis)
            b0, b1 = _proj_interval(polyB, axis)
            if (a1 < b0) or (b1 < a0):
                return False
    return True


def tangent_theta(points, i):
    i0 = max(i - 1, 0)
    i1 = min(i + 1, len(points) - 1)
    d = points[i1] - points[i0]
    return np.arctan2(d[1], d[0])


def polyline_length_px(pts):
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return 0.0
    d = pts[1:] - pts[:-1]
    return float(np.sum(np.linalg.norm(d, axis=1)))


def place_blocks_global(points_mm_rs, polys_global, L, W, ds, step_mm,
                        clearance=0.0, max_shift_tries=20, shift_delta_mm=3.0):
    step_idx = max(1, int(round(step_mm / ds)))
    shift_idx = max(1, int(round(shift_delta_mm / ds)))

    placed_local = []
    i = 0
    while i < len(points_mm_rs):
        center = points_mm_rs[i]
        theta = tangent_theta(points_mm_rs, i)
        poly = rect_corners(center, theta, L + clearance, W + clearance)

        ok = True
        for prev in polys_global:
            if rects_intersect(poly, prev):
                ok = False
                break

        if ok:
            placed_local.append({'center': center, 'theta': theta, 'corners': poly})
            polys_global.append(poly)
            i += step_idx
        else:
            moved = False
            for k in range(1, max_shift_tries + 1):
                j = i + k * shift_idx
                if j >= len(points_mm_rs):
                    break
                c2 = points_mm_rs[j]
                t2 = tangent_theta(points_mm_rs, j)
                p2 = rect_corners(c2, t2, L + clearance, W + clearance)

                ok2 = True
                for prev in polys_global:
                    if rects_intersect(p2, prev):
                        ok2 = False
                        break

                if ok2:
                    placed_local.append({'center': c2, 'theta': t2, 'corners': p2})
                    polys_global.append(p2)
                    i = j + step_idx
                    moved = True
                    break

            if not moved:
                i += 1

    return placed_local, polys_global


def run_block_pipeline_world(
    bgr_img,
    result,
    active,
    camera_pos_world_m,
    camera_rot_world_q0=1.0,
    camera_rot_world_qz=None,
    table_z_world_mm=170.0,
    block_z_world_mm=None,
):
    camera_pos_world_m = np.asarray(camera_pos_world_m, dtype=float).reshape(3,)
    camera_pos_world_mm = camera_pos_world_m * 1000.0
    if block_z_world_mm is None:
        block_z_world_mm = table_z_world_mm

    z_height_mm = float(camera_pos_world_mm[2] - table_z_world_mm)
    if z_height_mm <= 5.0:
        print(f"Warnung: unplausible z_height_mm={z_height_mm:.2f}; fallback={FALLBACK_Z_HEIGHT_MM:.2f} wird genutzt.")
        z_height_mm = float(FALLBACK_Z_HEIGHT_MM)

    q0 = float(np.clip(camera_rot_world_q0, -1.0, 1.0))
    alpha = 2.0 * float(np.arccos(q0))
    if camera_rot_world_qz is not None:
        qz = float(camera_rot_world_qz)
        if abs(qz) > 1e-12:
            alpha = float(np.sign(qz) * alpha)
    c_a = np.cos(alpha)
    s_a = np.sin(alpha)
    R_cam_to_world = np.array([[c_a, -s_a], [s_a, c_a]], dtype=float)

    strokes_px = result['strokes']
    strokes_undpx = [undistort_points_px(pl) for pl in strokes_px if pl is not None and len(pl) > 0]
    strokes_sorted = sorted(strokes_undpx, key=polyline_length_px, reverse=True)

    vis = undistort_image(bgr_img).copy()

    if not bool(active):
        return dict(
            active=False,
            z_height_mm=z_height_mm,
            blocks_px=[],
            blocks_world=[],
            vis_bgr=vis,
            strokes_sorted=strokes_sorted,
        )

    placed_all = []
    polys_global = []

    for stroke_undpx in strokes_sorted:
        if polyline_length_px(stroke_undpx) < MIN_LEN_PX:
            continue

        stroke_mm = undpx_to_cam_mm_robot_frame(stroke_undpx, z_height_mm)
        stroke_mm_rs = resample_polyline_mm(stroke_mm, ds=DS_MM)
        if len(stroke_mm_rs) < 2:
            continue

        placed_local, polys_global = place_blocks_global(
            stroke_mm_rs,
            polys_global,
            L=L_MM,
            W=W_MM,
            ds=DS_MM,
            step_mm=STEP_MM,
            clearance=CLEARANCE_MM,
            max_shift_tries=MAX_SHIFT_TRIES,
            shift_delta_mm=SHIFT_DELTA_MM,
        )
        placed_all.extend(placed_local)

    blocks_px = []
    blocks_world = []

    for i, b in enumerate(placed_all):
        center_cam_xy = np.asarray(b['center'], dtype=float).reshape(1, 2)
        corners_cam_xy = np.asarray(b['corners'], dtype=float).reshape(4, 2)

        center_px = cam_mm_robot_frame_to_undpx(center_cam_xy, z_height_mm)[0]
        corners_px = cam_mm_robot_frame_to_undpx(corners_cam_xy, z_height_mm)

        center_world_xy = (center_cam_xy @ R_cam_to_world.T)[0] + camera_pos_world_mm[:2]
        corners_world_xy = (corners_cam_xy @ R_cam_to_world.T) + camera_pos_world_mm[:2]

        yaw_world = float(np.arctan2(np.sin(b['theta'] + alpha), np.cos(b['theta'] + alpha)))

        corners_world_mm = np.column_stack([
            corners_world_xy[:, 0],
            corners_world_xy[:, 1],
            np.full(4, float(block_z_world_mm), dtype=float),
        ])

        blocks_px.append(dict(
            id=i,
            center_px_und=[float(center_px[0]), float(center_px[1])],
            yaw_rad=float(b['theta']),
            yaw_deg=float(np.degrees(b['theta'])),
            corners_px_und=corners_px.tolist(),
        ))

        blocks_world.append(dict(
            id=i,
            center_world_mm=[
                float(center_world_xy[0]),
                float(center_world_xy[1]),
                float(block_z_world_mm),
            ],
            yaw_world_rad=yaw_world,
            yaw_world_deg=float(np.degrees(yaw_world)),
            corners_world_mm=corners_world_mm.tolist(),
        ))

    for pl in strokes_sorted:
        pl_i = np.round(np.asarray(pl, float)).astype(np.int32).reshape(-1, 1, 2)
        if len(pl_i) >= 2:
            cv2.polylines(vis, [pl_i], False, (0, 0, 255), 2, cv2.LINE_AA)

    for b_px, b_world in zip(blocks_px, blocks_world):
        c = np.round(np.asarray(b_px['corners_px_und'], float)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [c], True, (255, 0, 0), 2, cv2.LINE_AA)

        center_px = np.round(np.asarray(b_px['center_px_und'], float)).astype(np.int32)
        cx_, cy_ = int(center_px[0]), int(center_px[1])
        cv2.circle(vis, (cx_, cy_), 3, (0, 255, 255), -1, cv2.LINE_AA)

        xw, yw, _ = b_world['center_world_mm']
        label = f"id={b_px['id']} ({xw:.1f},{yw:.1f})"
        text_pos = (cx_ + 6, cy_ - 6)
        cv2.putText(vis, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    return dict(
        active=True,
        z_height_mm=z_height_mm,
        blocks_px=blocks_px,
        blocks_world=blocks_world,
        vis_bgr=vis,
        strokes_sorted=strokes_sorted,
    )


def angle_to_quat_wxyz(angle_rad, axis='z'):
    half = 0.5 * float(angle_rad)
    c = float(np.cos(half))
    s = float(np.sin(half))
    if axis == 'x':
        return c, s, 0.0, 0.0
    if axis == 'y':
        return c, 0.0, s, 0.0
    if axis == 'z':
        return c, 0.0, 0.0, s
    raise ValueError(f'Unbekannte Achse: {axis}')


def fmt_num(v, digits=10):
    if abs(float(v)) < 1e-12:
        return '0'
    return f"{float(v):.{digits}f}"


# ============================================================
# AICA LIFECYCLE COMPONENT
# ============================================================

class DropoffLineExtractor(LifecycleComponent):

    def __init__(self, node_name: str, *args, **kwargs):
        super().__init__(node_name, *args, **kwargs)

        # ── Parameter ─────────────────────────────────────────────────────────
        self._table_z_world_mm = sr.Parameter("table_z_world_mm", 170.0, sr.ParameterType.DOUBLE)
        self.add_parameter("_table_z_world_mm", "Tischoberfläche in Weltkoordinaten (mm)")

        # block_z = table_z + STONE_HEIGHT_MM (Tisch + Steinhöhe)
        self._block_z_world_mm = sr.Parameter("block_z_world_mm", 170.0 + STONE_HEIGHT_MM, sr.ParameterType.DOUBLE)
        self.add_parameter("_block_z_world_mm", "TCP-Z beim Ablegen in Weltkoordinaten (mm) – typisch: table_z + Steinhöhe")

        self._l_mm = sr.Parameter("l_mm", 35.0, sr.ParameterType.DOUBLE)
        self.add_parameter("_l_mm", "Länge eines Bausteins (mm)")

        self._w_mm = sr.Parameter("w_mm", 10.0, sr.ParameterType.DOUBLE)
        self.add_parameter("_w_mm", "Breite eines Bausteins (mm)")

        self._step_mm = sr.Parameter("step_mm", 33.25, sr.ParameterType.DOUBLE)
        self.add_parameter("_step_mm", "Schrittweite zwischen Baustein-Mittelpunkten entlang der Linie (mm)")

        self._min_len_px = sr.Parameter("min_len_px", 30.0, sr.ParameterType.DOUBLE)
        self.add_parameter("_min_len_px", "Minimale Strich-Länge in Pixeln – kürzere Striche werden ignoriert")

        # ── Inputs ────────────────────────────────────────────────────────────
        self._image_in = RosImage()
        self.add_input("image_in", "_image_in", RosImage)

        self._cam_ist_pose = sr.CartesianPose("cam_ist_pose", "world")
        self.add_input("cam_ist_pose", "_cam_ist_pose", EncodedState)

        # yolo_done_trigger: steigende Flanke startet Linienerkennung
        self._yolo_done_trigger = False
        self.add_input(
            "yolo_done_trigger", "_yolo_done_trigger", Bool,
            user_callback=self._on_yolo_trigger,
        )

        self._trigger_ppl = False
        self.add_input("trigger_ppl", "_trigger_ppl", Bool)

        # ── Outputs ───────────────────────────────────────────────────────────
        self._line_ex_list = []
        self.add_output("line_ex_list", "_line_ex_list", Float64MultiArray)

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle-Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def on_validate_parameter_callback(self, parameter: sr.Parameter) -> bool:
        return True

    def on_configure_callback(self) -> bool:
        self._line_ex_list = []
        return True

    def on_activate_callback(self) -> bool:
        self.get_logger().info("DropoffLineExtractor: Aktiviert – wartet auf yolo_done_trigger.")
        return True

    def on_deactivate_callback(self) -> bool:
        return True

    def on_step_callback(self):
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Event-Callback: yolo_done_trigger (steigende Flanke)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_yolo_trigger(self):
        # ── Bypass Phase 2 ────────────────────────────────────────────────────
        if self._trigger_ppl:
            return

        # ── Nur steigende Flanke auswerten ────────────────────────────────────
        if not self._yolo_done_trigger:
            return

        # ── Kamerapose prüfen ─────────────────────────────────────────────────
        if self._cam_ist_pose.is_empty():
            self.get_logger().warn(
                "DropoffLineExtractor: Kamerapose fehlt – überspringe Inferenz."
            )
            return

        self.get_logger().info(
            "DropoffLineExtractor: Trigger empfangen – starte Linienerkennung."
        )

        # ── Bilddaten extrahieren ─────────────────────────────────────────────
        msg = self._image_in
        if not msg.data:
            self.get_logger().warn(
                "DropoffLineExtractor: Leeres Bild empfangen – überspringe Erkennung."
            )
            return

        bgr_use = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, -1)
        )

        # bgr0 global setzen, damit Hilfsfunktionen (resolve_paper_runtime_inputs
        # etc.) auf das aktuelle Bild zugreifen können.
        global bgr0
        bgr0 = bgr_use

        # ── Modul-Variablen aus Parametern aktualisieren ──────────────────────
        global L_MM, W_MM, STEP_MM, MIN_LEN_PX
        L_MM = self._l_mm.get_value()
        W_MM = self._w_mm.get_value()
        STEP_MM = self._step_mm.get_value()
        MIN_LEN_PX = self._min_len_px.get_value()

        # ── Kamerapose extrahieren ────────────────────────────────────────────
        pos = self._cam_ist_pose.get_position()
        camera_pos_m = np.array([float(pos[0]), float(pos[1]), float(pos[2])])
        q = self._cam_ist_pose.get_orientation()
        # state_representation: Quaternion als [w, x, y, z]
        camera_q0 = float(q[0])   # w-Komponente
        camera_qz = float(q[3])   # z-Komponente

        table_z = self._table_z_world_mm.get_value()
        block_z = self._block_z_world_mm.get_value()

        # ── Papiererkennung (Auto-Modus, kein Commit nötig) ───────────────────
        try:
            params = dict(DEFAULT_PAPER_PARAMS)
            _, _, _, mask_full = build_mask_from_params(
                bgr_use,
                params["downscale"], params["wL"], params["wC"], params["wT"],
                params["bL"], params["bC"], params["bT"],
                params["p_fg"], params["p_sfg"], params["p_bg"],
                use_grabcut=params["use_grabcut"], gc_iters=params["gc_iters"],
                ksize=params["ksize"], close_iters=params["close_iters"],
                open_iters=params["open_iters"],
            )
            comp = largest_component(mask_full)
            if comp is not None and cv2.countNonZero(comp) > 0:
                cnt = get_largest_contour(comp)
                if cnt is not None:
                    corners = contour_to_quad_by_lines(
                        cnt,
                        band_frac=params["band_frac"],
                        min_pts=params["min_pts"],
                    )
                    if corners is not None and len(np.asarray(corners)) >= 4:
                        paper_mask_use = mask_from_corners(
                            bgr_use.shape,
                            np.asarray(corners, dtype=np.float32)[:4],
                        )
                    else:
                        paper_mask_use = comp
                else:
                    paper_mask_use = comp
            else:
                paper_mask_use = np.ones(bgr_use.shape[:2], np.uint8) * 255
        except Exception as e:
            self.get_logger().warn(
                f"DropoffLineExtractor: Papiererkennung fehlgeschlagen ({e}), nutze Vollbild."
            )
            paper_mask_use = np.ones(bgr_use.shape[:2], np.uint8) * 255

        # ── Linien-Pipeline ───────────────────────────────────────────────────
        try:
            result = run_line_pipeline(
                bgr_use, paper_mask_use, dict(DEFAULT_LINE_PARAMS), show_debug=False
            )
        except Exception as e:
            self.get_logger().error(
                f"DropoffLineExtractor: Linien-Pipeline fehlgeschlagen: {e}\n"
                f"{traceback.format_exc()}"
            )
            return

        # ── Block-Platzierung und Weltkoordinaten ─────────────────────────────
        try:
            run_out = run_block_pipeline_world(
                bgr_img=bgr_use,
                result=result,
                active=True,
                camera_pos_world_m=camera_pos_m,
                camera_rot_world_q0=camera_q0,
                camera_rot_world_qz=camera_qz,
                table_z_world_mm=table_z,
                block_z_world_mm=block_z,
            )
        except Exception as e:
            self.get_logger().error(
                f"DropoffLineExtractor: Block-Pipeline fehlgeschlagen: {e}\n"
                f"{traceback.format_exc()}"
            )
            return

        # ── Output: Stride 7 [X, Y, Z, Qx, Qy, Qz, Qw] in Metern ───────────
        flat = []
        for b in run_out['blocks_world']:
            x_m, y_m, z_m = [v / 1000.0 for v in b['center_world_mm']]
            yaw = b['yaw_world_rad']
            qx = 0.0
            qy = 0.0
            qz = float(np.sin(yaw / 2.0))
            qw = float(np.cos(yaw / 2.0))
            flat.extend([x_m, y_m, z_m, qx, qy, qz, qw])

        self._line_ex_list = flat

        self.get_logger().info(
            f"DropoffLineExtractor: {len(flat) // 7} Ablageposen erkannt "
            f"und auf line_ex_list geschrieben."
        )
