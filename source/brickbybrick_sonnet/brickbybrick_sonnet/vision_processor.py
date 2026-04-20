"""
VisionProcessor (VP)
─────────────────────────────────────────────────────────────────────────────
Monolithischer Bildverarbeitungs-Block. Ersetzt als experimentelle
Alternative die Kette:
    PoseTriggeredCamera → YoloObjectDetector → DropoffLineExtractor
                                              → MasterListManager

Ziel: Race-freie, streng sequenzielle Ausführung der gesamten Pipeline
innerhalb EINES on_step_callback. Keine separaten Topics zwischen den
vier Stufen → keine Topic-Reihenfolge-Probleme.

Interner Ablauf (pro Bildaufnahme):
  IDLE
   └─ take_img=True + trajectory_success=True
  SETTLING (0,3 s Timer, parameter)
   └─ Timer abgelaufen
  SNAPSHOT: image + ist_pose einfrieren, cam_ist_pose intern berechnen
  YOLO: Inferenz → Stride-8 corners
  DLE: Linienerkennung (nur Phase 1, Latch bei trigger_ppl)
  MLM: Pinhole-Rückprojektion, Twist-Minimierung, Filter, Listen-Update
  DONE: vision_processor_done = True, img_taken = True
   └─ trajectory_success fällt
  RESET: img_taken = False, Trigger-Reset, zurück zu IDLE

Outputs:
  master_overview, master_dropoff, filtered_yolo
  vision_processor_done (1-Takt-Puls, analog mlm_done_trigger)
  img_taken            (Handshake an ExplorationNavigator, analog PTC)
"""

import math
import os
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import state_representation as sr
from clproto import MessageType
from modulo_core.encoded_state import EncodedState
from modulo_components.lifecycle_component import LifecycleComponent
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Bool, Float64MultiArray

from brickbybrick_sonnet.geometry_utils import (
    quaternion_from_euler,
    yaw_from_quaternion,
    minimize_twist,
    gauss_shoelace_area,
    pinhole_ray,
    ray_table_intersect,
)

# DLE-Algorithmus-Funktionen inkl. Modul-Globals (bgr0, L_MM, ...) wiederverwenden.
from brickbybrick_sonnet import dropoff_line_extractor as _dle_algo

try:
    from ament_index_python.packages import get_package_share_directory as _get_share
    _SHARE = _get_share("brickbybrick_sonnet")
except Exception:
    _SHARE = "."

_DEFAULT_MODEL_PATH = os.path.join(_SHARE, "data", "model", "best.pt")

# ── Kamera-Intrinsik (RealSense D435i, 1280×720 color) ─────────────────────
# Muss synchron zu MasterListManager gehalten werden.
_CAM_FX = 910.7815551757812
_CAM_FY = 910.9876708984375
_CAM_CX = 644.165283203125
_CAM_CY = 367.8723449707031

# ── Kamera-Montage-Offset relativ zum TCP (TCP-Frame) ──────────────────────
# Muss synchron zu pose_triggered_camera.py gehalten werden.
# Quelle: Magnus RKIM, 14.04.2026.
_CAM_OFFSET_XYZ = [-0.0205, 0.01750, -0.072475]
# scipy-Quat-Format [qx, qy, qz, qw] – 180° um Z.
_CAM_OFFSET_QUAT = [0.0, 0.0, -0.7071, 0.7071]


class VisionProcessor(LifecycleComponent):

    def __init__(self, node_name: str, *args, **kwargs):
        super().__init__(node_name, *args, **kwargs)

        # ── Parameter ─────────────────────────────────────────────────────────
        self._settling_delay_s = sr.Parameter(
            "settling_delay_s", 0.3, sr.ParameterType.DOUBLE,
        )
        self.add_parameter(
            "_settling_delay_s",
            "Wartezeit nach Zielerreichung bis Snapshot [s] – gibt Roboter Zeit zum Ausschwingen",
        )

        self._model_path = sr.Parameter(
            "model_path", _DEFAULT_MODEL_PATH, sr.ParameterType.STRING,
        )
        self.add_parameter(
            "_model_path",
            "Pfad zur trainierten YOLOv11-Seg Gewichtsdatei (.pt)",
        )

        self._z_table_m = sr.Parameter("z_table_m", 0.170, sr.ParameterType.DOUBLE)
        self.add_parameter(
            "_z_table_m",
            "Tischhöhe im Weltframe [m] – Pinhole-Projektion & DLE-Tisch-Z",
        )

        self._block_z_world_mm = sr.Parameter(
            "block_z_world_mm", 181.0, sr.ParameterType.DOUBLE,
        )
        self.add_parameter(
            "_block_z_world_mm",
            "TCP-Z beim Ablegen [mm] – typisch: z_table + Steinhöhe",
        )

        self._l_mm = sr.Parameter("l_mm", 75.0, sr.ParameterType.DOUBLE)
        self.add_parameter("_l_mm", "Länge eines Bausteins [mm]")

        self._w_mm = sr.Parameter("w_mm", 25.0, sr.ParameterType.DOUBLE)
        self.add_parameter("_w_mm", "Breite eines Bausteins [mm]")

        self._step_mm = sr.Parameter("step_mm", 77.5, sr.ParameterType.DOUBLE)
        self.add_parameter(
            "_step_mm", "Schrittweite zwischen Baustein-Mittelpunkten [mm]",
        )

        self._min_len_px = sr.Parameter("min_len_px", 30.0, sr.ParameterType.DOUBLE)
        self.add_parameter(
            "_min_len_px", "Minimale Linienlänge [px] – kürzere werden verworfen",
        )

        self._emit_debug = sr.Parameter("emit_debug", False, sr.ParameterType.BOOL)
        self.add_parameter(
            "_emit_debug",
            "Wenn True: debug_image mit YOLO-Overlay + Pose-/Timing-Text publizieren",
        )

        # ── Inputs ────────────────────────────────────────────────────────────
        self._take_img = False
        self.add_input("take_img", "_take_img", Bool)

        self._trajectory_success = False
        self.add_input("trajectory_success", "_trajectory_success", Bool)

        self._ist_pose_in = sr.CartesianPose("ist_pose_in", "world")
        self.add_input("ist_pose_in", "_ist_pose_in", EncodedState)

        self._image_stream = RosImage()
        self.add_input("image_stream", "_image_stream", RosImage)

        self._trigger_ppl = False
        self.add_input("trigger_ppl", "_trigger_ppl", Bool)

        # ── Outputs ───────────────────────────────────────────────────────────
        self._master_overview = []
        self.add_output("master_overview", "_master_overview", Float64MultiArray)

        self._master_dropoff = []
        self.add_output("master_dropoff", "_master_dropoff", Float64MultiArray)

        self._filtered_yolo = []
        self.add_output("filtered_yolo", "_filtered_yolo", Float64MultiArray)

        self._vision_processor_done = False
        self.add_output(
            "vision_processor_done", "_vision_processor_done", Bool,
        )

        self._img_taken = False
        self.add_output("img_taken", "_img_taken", Bool)

        self._debug_image = RosImage()
        self.add_output("debug_image", "_debug_image", RosImage)

        # ── Interne Zustandsvariablen ─────────────────────────────────────────
        self._state: str = "IDLE"          # IDLE | SETTLING | DONE
        self._timer_start = None
        self._model = None                 # YOLO-Handle, lazy in on_configure

        # DLE-Latch: sobald trigger_ppl einmal True war, bleibt DLE aus.
        self._ppl_started_latch: bool = False

        # Verzögerter Reset für vision_processor_done (analog mlm_done_trigger).
        self._reset_done_next_step: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle-Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def on_validate_parameter_callback(self, parameter: sr.Parameter) -> bool:
        return True

    def on_configure_callback(self) -> bool:
        path = self._model_path.get_value()
        try:
            from ultralytics import YOLO
            self._model = YOLO(path)
            self.get_logger().info(
                f"VisionProcessor: YOLO-Modell geladen von '{path}'."
            )
        except FileNotFoundError:
            self.get_logger().error(
                f"VisionProcessor: Modelldatei nicht gefunden: '{path}'"
            )
            return False
        except Exception as exc:
            self.get_logger().error(
                f"VisionProcessor: Fehler beim Laden des Modells: {exc}"
            )
            return False

        self._master_overview = []
        self._master_dropoff = []
        self._filtered_yolo = []
        self._vision_processor_done = False
        self._img_taken = False
        self._state = "IDLE"
        self._timer_start = None
        self._ppl_started_latch = False
        self._reset_done_next_step = False
        self.get_logger().info("VisionProcessor: Konfiguriert – bereit.")
        return True

    def on_activate_callback(self) -> bool:
        self.get_logger().info(
            "VisionProcessor: Aktiviert – wartet auf take_img + trajectory_success."
        )
        return True

    def on_deactivate_callback(self) -> bool:
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Haupt-Taktschleife
    # ─────────────────────────────────────────────────────────────────────────

    def on_step_callback(self):
        # Latch setzen sobald Phase 2 beginnt (unabhängig vom Zustand).
        if self._trigger_ppl:
            self._ppl_started_latch = True

        # Verzögerter Reset von vision_processor_done (1 Takt True → False).
        if self._reset_done_next_step:
            self._vision_processor_done = False
            self._reset_done_next_step = False
        elif self._vision_processor_done:
            self._reset_done_next_step = True

        # img_taken-Reset: sobald trajectory_success abfällt, Handshake lösen.
        if not self._trajectory_success and self._img_taken:
            self._img_taken = False

        # ── Zustandsmaschine ─────────────────────────────────────────────────
        if self._state == "IDLE":
            # Trigger-Bedingung: take_img UND trajectory_success UND noch kein
            # offener Handshake (img_taken muss zuerst zurückgesetzt sein).
            if self._take_img and self._trajectory_success and not self._img_taken:
                self._state = "SETTLING"
                self._timer_start = self.get_clock().now()
                self.get_logger().info(
                    f"VisionProcessor: Trigger – starte "
                    f"{self._settling_delay_s.get_value():.2f}-s-Settling."
                )

        elif self._state == "SETTLING":
            # Safety: Wenn trajectory_success während Settling abfällt, abbrechen.
            if not self._trajectory_success:
                self._state = "IDLE"
                self._timer_start = None
                self.get_logger().info(
                    "VisionProcessor: Settling abgebrochen – "
                    "trajectory_success gefallen."
                )
                return

            elapsed = (
                self.get_clock().now() - self._timer_start
            ).nanoseconds / 1e9
            if elapsed < self._settling_delay_s.get_value():
                return

            # ── Timer fertig: komplette Pipeline in EINEM Step ───────────────
            self._run_pipeline()

            # Handshake-Signale setzen (nach Pipeline-Ende).
            self._img_taken = True
            self._vision_processor_done = True
            self._state = "DONE"

        elif self._state == "DONE":
            # Bleibe in DONE bis trajectory_success abfällt (→ img_taken=False).
            # Danach zurück in IDLE für den nächsten Zyklus.
            if not self._trajectory_success:
                self._state = "IDLE"

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline: Snapshot → YOLO → DLE → MLM
    # ─────────────────────────────────────────────────────────────────────────

    def _run_pipeline(self):
        # 1) Snapshot einfrieren (Bild + Posen).
        msg = self._image_stream
        if not msg.data:
            self.get_logger().warn(
                "VisionProcessor: Leeres Bild – überspringe Pipeline."
            )
            self._filtered_yolo = []
            return

        image_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, -1)
        ).copy()

        ist_pose = (
            sr.CartesianPose(self._ist_pose_in)
            if not self._ist_pose_in.is_empty() else None
        )

        # Kamerapose intern aus TCP-Pose + Montage-Offset berechnen.
        cam_pos, cam_quat_scipy = self._compute_cam_pose(ist_pose)

        # 2) YOLO-Inferenz.
        t0 = time.perf_counter()
        corners = self._run_yolo(image_array, msg.height, msg.width)
        t_yolo_ms = (time.perf_counter() - t0) * 1000.0
        self.get_logger().info(
            f"VisionProcessor: YOLO – {len(corners) // 8} Klotz/Klötze erkannt "
            f"[{t_yolo_ms:.0f} ms]."
        )

        # 3) DLE (nur Phase 1, mit Latch).
        t_dle_ms = 0.0
        dle_skipped = self._trigger_ppl or self._ppl_started_latch
        if not dle_skipped:
            t0 = time.perf_counter()
            new_dropoff = self._run_dle(image_array, cam_pos, cam_quat_scipy)
            t_dle_ms = (time.perf_counter() - t0) * 1000.0
            if len(new_dropoff) > len(self._master_dropoff):
                self._master_dropoff = new_dropoff
                self.get_logger().info(
                    f"VisionProcessor: master_dropoff aktualisiert – "
                    f"{len(new_dropoff) // 7} Ablagepose(n) [{t_dle_ms:.0f} ms]."
                )

        # 4) MLM-Logik (Geometrie + Filter).
        t0 = time.perf_counter()
        self._run_mlm(corners, ist_pose, cam_pos, cam_quat_scipy)
        t_mlm_ms = (time.perf_counter() - t0) * 1000.0

        self.get_logger().info(
            f"VisionProcessor: Pipeline-Timing – YOLO {t_yolo_ms:.0f} ms, "
            f"DLE {t_dle_ms:.0f} ms, MLM {t_mlm_ms:.0f} ms, "
            f"Σ {t_yolo_ms + t_dle_ms + t_mlm_ms:.0f} ms."
        )

        # 5) Debug-Image optional rendern.
        if self._emit_debug.get_value():
            self._publish_debug_image(
                image_array, msg, corners, cam_pos,
                t_yolo_ms, t_dle_ms, t_mlm_ms, dle_skipped,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Teilschritt: Kamerapose berechnen
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_cam_pose(self, ist_pose):
        if ist_pose is None:
            return None, None
        tcp_pos = np.array(ist_pose.get_position(), dtype=float)
        tcp_ori = ist_pose.get_orientation()  # AICA: [qw, qx, qy, qz]
        tcp_quat_scipy = [
            float(tcp_ori[1]), float(tcp_ori[2]),
            float(tcp_ori[3]), float(tcp_ori[0]),
        ]
        R_tcp = Rotation.from_quat(tcp_quat_scipy)
        cam_pos = (tcp_pos + R_tcp.apply(_CAM_OFFSET_XYZ)).tolist()
        q = (R_tcp * Rotation.from_quat(_CAM_OFFSET_QUAT)).as_quat()
        # scipy → AICA [qw, qx, qy, qz]
        cam_quat_aica = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        return cam_pos, cam_quat_aica

    # ─────────────────────────────────────────────────────────────────────────
    # Teilschritt: YOLO-Inferenz (ehem. yolo_object_detector._run_yolo_inference)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_yolo(self, image_array, height, width):
        # RGB → BGR (Simulator/Kamera liefert rgb8, Ultralytics erwartet BGR).
        image_bgr = image_array[:, :, ::-1]
        results = self._model(image_bgr, verbose=False, device='cpu')

        corners_flat = []
        if results and len(results) > 0 and results[0].masks is not None:
            for mask_xy in results[0].masks.xy:
                if len(mask_xy) < 5:
                    continue
                pts = mask_xy.astype(np.float32).reshape(-1, 1, 2)
                rect = cv2.minAreaRect(pts)
                box_corners = cv2.boxPoints(rect)

                # Rand-Filter (5 px).
                on_border = False
                for (u, v) in box_corners:
                    if u < 5 or u > width - 5 or v < 5 or v > height - 5:
                        on_border = True
                        break
                if on_border:
                    continue

                for (u, v) in box_corners:
                    corners_flat.extend([float(u), float(v)])

        return corners_flat

    # ─────────────────────────────────────────────────────────────────────────
    # Teilschritt: MLM-Logik (ehem. master_list_manager._on_yolo_trigger)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_mlm(self, corners, ist_pose, cam_pos, cam_quat):
        if len(corners) == 0:
            self._filtered_yolo = []
            return

        # Phase 1: TCP-Pose zur master_overview hinzufügen.
        if (not self._trigger_ppl and not self._ppl_started_latch
                and ist_pose is not None):
            pos = ist_pose.get_position()
            ori = ist_pose.get_orientation()
            self._master_overview.extend([
                float(pos[0]), float(pos[1]), float(pos[2]),
                float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3]),
            ])
            self.get_logger().info(
                f"VisionProcessor: master_overview erweitert "
                f"({len(self._master_overview) // 7} Posen gesamt)."
            )

        # Aktuellen Roboter-Yaw für Twist-Minimierung bestimmen.
        current_robot_yaw = 0.0
        if ist_pose is not None:
            ori = ist_pose.get_orientation()
            current_robot_yaw = yaw_from_quaternion(
                float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3]),
            )

        z_table = self._z_table_m.get_value()
        pending_bricks = []

        for i in range(0, len(corners), 8):
            if i + 8 > len(corners):
                break
            u1, v1 = float(corners[i]),     float(corners[i + 1])
            u2, v2 = float(corners[i + 2]), float(corners[i + 3])
            u3, v3 = float(corners[i + 4]), float(corners[i + 5])
            u4, v4 = float(corners[i + 6]), float(corners[i + 7])

            u_center = (u1 + u3) / 2.0
            v_center = (v1 + v3) / 2.0
            area = gauss_shoelace_area([(u1, v1), (u2, v2), (u3, v3), (u4, v4)])
            theta = math.atan2(v2 - v1, u2 - u1)

            ray = pinhole_ray(u_center, v_center, _CAM_FX, _CAM_FY, _CAM_CX, _CAM_CY)
            if cam_pos is not None:
                X_klotz, Y_klotz = ray_table_intersect(ray, cam_pos, cam_quat, z_table)
            else:
                X_klotz, Y_klotz = 0.0, 0.0

            opt_yaw = minimize_twist(theta, current_robot_yaw)
            quat = quaternion_from_euler(math.pi, 0.0, opt_yaw)  # [qx,qy,qz,qw]

            pending_bricks.append([
                X_klotz, Y_klotz, area, u_center, v_center,
                quat[0], quat[1], quat[2], quat[3],
            ])

        # 1-cm-Ablage-Filter.
        new_filtered = []
        for brick in pending_bricks:
            X_b, Y_b = brick[0], brick[1]
            too_close = False
            for j in range(0, len(self._master_dropoff), 7):
                if j + 7 > len(self._master_dropoff):
                    break
                X_d = float(self._master_dropoff[j])
                Y_d = float(self._master_dropoff[j + 1])
                if math.sqrt((X_b - X_d) ** 2 + (Y_b - Y_d) ** 2) < 0.01:
                    too_close = True
                    break
            if not too_close:
                new_filtered.extend(brick)

        self._filtered_yolo = new_filtered
        self.get_logger().info(
            f"VisionProcessor: MLM – {len(pending_bricks)} Rohdaten, "
            f"{len(new_filtered) // 9} nach Ablage-Filter in filtered_yolo."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Teilschritt: DLE (Bernhards Algorithmus) – delegiert ans bestehende Modul
    # ─────────────────────────────────────────────────────────────────────────

    def _run_dle(self, image_array, cam_pos, cam_quat_aica):
        if cam_pos is None:
            self.get_logger().warn(
                "VisionProcessor: DLE – Kamerapose fehlt, überspringe."
            )
            return []

        # Modul-Globals des DLE vorbereiten.
        _dle_algo.bgr0 = image_array
        _dle_algo.L_MM = self._l_mm.get_value()
        _dle_algo.W_MM = self._w_mm.get_value()
        _dle_algo.STEP_MM = self._step_mm.get_value()
        _dle_algo.MIN_LEN_PX = self._min_len_px.get_value()

        table_z_mm = self._z_table_m.get_value() * 1000.0
        block_z_mm = self._block_z_world_mm.get_value()

        camera_pos_m = np.array(cam_pos, dtype=float)
        # cam_quat_aica = [qw, qx, qy, qz] (DLE erwartet q0/qz aus w/z)
        camera_q0 = float(cam_quat_aica[0])
        camera_qz = float(cam_quat_aica[3])

        # Papiererkennung (Auto-Modus).
        try:
            params = dict(_dle_algo.DEFAULT_PAPER_PARAMS)
            _, _, _, mask_full = _dle_algo.build_mask_from_params(
                image_array,
                params["downscale"], params["wL"], params["wC"], params["wT"],
                params["bL"], params["bC"], params["bT"],
                params["p_fg"], params["p_sfg"], params["p_bg"],
                use_grabcut=params["use_grabcut"], gc_iters=params["gc_iters"],
                ksize=params["ksize"], close_iters=params["close_iters"],
                open_iters=params["open_iters"],
            )
            comp = _dle_algo.largest_component(mask_full)
            if comp is not None and cv2.countNonZero(comp) > 0:
                cnt = _dle_algo.get_largest_contour(comp)
                if cnt is not None:
                    corners = _dle_algo.contour_to_quad_by_lines(
                        cnt,
                        band_frac=params["band_frac"],
                        min_pts=params["min_pts"],
                    )
                    if corners is not None and len(np.asarray(corners)) >= 4:
                        paper_mask_use = _dle_algo.mask_from_corners(
                            image_array.shape,
                            np.asarray(corners, dtype=np.float32)[:4],
                        )
                    else:
                        paper_mask_use = comp
                else:
                    paper_mask_use = comp
            else:
                paper_mask_use = np.ones(image_array.shape[:2], np.uint8) * 255
        except Exception as e:
            self.get_logger().warn(
                f"VisionProcessor: DLE-Papiererkennung fehlgeschlagen ({e})."
            )
            paper_mask_use = np.ones(image_array.shape[:2], np.uint8) * 255

        # Linien-Pipeline.
        try:
            result = _dle_algo.run_line_pipeline(
                image_array, paper_mask_use,
                dict(_dle_algo.DEFAULT_LINE_PARAMS), show_debug=False,
            )
        except Exception as e:
            self.get_logger().error(
                f"VisionProcessor: DLE-Linien-Pipeline fehlgeschlagen: {e}"
            )
            return []

        # Block-Platzierung (Welt).
        try:
            run_out = _dle_algo.run_block_pipeline_world(
                bgr_img=image_array,
                result=result,
                active=True,
                camera_pos_world_m=camera_pos_m,
                camera_rot_world_q0=camera_q0,
                camera_rot_world_qz=camera_qz,
                table_z_world_mm=table_z_mm,
                block_z_world_mm=block_z_mm,
            )
        except Exception as e:
            self.get_logger().error(
                f"VisionProcessor: DLE-Block-Pipeline fehlgeschlagen: {e}"
            )
            return []

        # Stride 7 [X, Y, Z, Qx, Qy, Qz, Qw] in Metern.
        flat = []
        for b in run_out['blocks_world']:
            x_m, y_m, z_m = [v / 1000.0 for v in b['center_world_mm']]
            yaw = b['yaw_world_rad']
            qx = float(np.sin(yaw / 2.0))
            qy = float(np.cos(yaw / 2.0))
            qz = 0.0
            qw = 0.0
            flat.extend([x_m, y_m, z_m, qx, qy, qz, qw])

        self.get_logger().info(
            f"VisionProcessor: DLE – {len(flat) // 7} Ablageposen erkannt."
        )
        return flat
