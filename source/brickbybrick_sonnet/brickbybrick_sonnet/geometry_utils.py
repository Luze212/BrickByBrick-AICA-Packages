"""
geometry_utils.py
─────────────────────────────────────────────────────────────────────────────
Gemeinsame mathematische Hilfsfunktionen für alle BrickByBrick-Komponenten.

Implementiert:
  - quaternion_from_euler      : Euler (RPY) → Quaternion [qx, qy, qz, qw]
  - yaw_from_quaternion        : Quaternion → Yaw-Winkel (Z-Rotation, Radiant)
  - minimize_twist             : Optimaler Greif-Yaw bei 180°-symmetrischem Klotz
  - gauss_shoelace_area        : Polygonfläche via Gaußsche Trapezformel
  - pinhole_ray                : Pixel → normalisierter 3D-Sichtstrahl (Kameraframe)
  - ray_table_intersect        : Strahl + Kamerapose → Weltkoordinaten (X, Y)

Platzhalter (TO-DO, physikalische Parameter noch offen):
  - depth_to_world_z           : Tiefenbild-Pixel → Weltkoordinaten Z_pick

WICHTIG: Diese Datei ist KEINE AICA-Komponente und darf NICHT in setup.cfg
         oder einer component_descriptions/*.json registriert werden.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation


# ─────────────────────────────────────────────────────────────────────────────
# Implementierte Funktionen
# ─────────────────────────────────────────────────────────────────────────────

def quaternion_from_euler(roll_rad: float, pitch_rad: float, yaw_rad: float) -> list:
    """
    Wandelt Euler-Winkel (extrinsisch, XYZ-Reihenfolge) in ein Quaternion um.

    Konvention für den Sauger-Greifer:
        roll  = π  (180°) → Greifer zeigt nach unten (Z-Achse negativ)
        pitch = 0.0
        yaw   = opt_yaw   → Ausrichtung des Klotzes

    Args:
        roll_rad:  Rotation um X-Achse in Radiant
        pitch_rad: Rotation um Y-Achse in Radiant
        yaw_rad:   Rotation um Z-Achse in Radiant

    Returns:
        [qx, qy, qz, qw] als Python-Liste
    """
    rot = Rotation.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad])
    return rot.as_quat().tolist()  # scipy liefert [qx, qy, qz, qw]


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """
    Extrahiert den Yaw-Winkel (Z-Rotation) aus einem Quaternion.

    Wird im MasterListManager verwendet, um den aktuellen TCP-Yaw des Roboters
    für die Twist-Minimierung zu bestimmen.

    Args:
        qx, qy, qz, qw: Quaternion-Komponenten

    Returns:
        Yaw-Winkel in Radiant
    """
    rot = Rotation.from_quat([qx, qy, qz, qw])
    euler = rot.as_euler('xyz')   # [roll, pitch, yaw]
    return float(euler[2])


def minimize_twist(theta_rad: float, robot_yaw_rad: float) -> float:
    """
    Wählt den optimalen Greif-Yaw unter Berücksichtigung der 180°-Symmetrie
    eines rechteckigen Klotzes.

    Ein Klotz kann bei Winkel θ ODER bei θ + 180° gegriffen werden –
    mechanisch identisch, aber unterschiedlicher Arm-Verdrehungsaufwand.
    Diese Funktion wählt den Kandidaten mit der geringsten Winkeldifferenz
    zum aktuellen Roboter-Yaw, um Gelenk-Überschläge zu vermeiden.

    Args:
        theta_rad:     Roher Kantenwinkel des Klotzes aus arctan2 (Radiant)
        robot_yaw_rad: Aktueller Yaw-Winkel des TCP (Radiant)

    Returns:
        Optimaler Yaw-Winkel in Radiant
    """
    candidate1 = theta_rad
    candidate2 = theta_rad + math.pi

    def _angular_diff(a: float, b: float) -> float:
        """Normalisierte absolute Winkeldifferenz, beschränkt auf [0, π]."""
        diff = (a - b + math.pi) % (2.0 * math.pi) - math.pi
        return abs(diff)

    diff1 = _angular_diff(candidate1, robot_yaw_rad)
    diff2 = _angular_diff(candidate2, robot_yaw_rad)
    return candidate1 if diff1 <= diff2 else candidate2


def gauss_shoelace_area(corners: list) -> float:
    """
    Berechnet die Fläche eines Polygons nach der Gaußschen Trapezformel
    (Shoelace-Algorithmus).

    Wird im MasterListManager verwendet, um die Klotzgröße als Proxy für
    die Priorität beim Greifen zu bestimmen (größte Fläche = größter Klotz).

    Args:
        corners: Liste von (u, v) Pixelkoordinaten-Tupeln,
                 z.B. [(u1, v1), (u2, v2), (u3, v3), (u4, v4)]

    Returns:
        Fläche in Pixeln²  (immer positiv)
    """
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    return abs(area) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# TO-DO Platzhalter – physikalische Parameter noch nicht vollständig geklärt
# ─────────────────────────────────────────────────────────────────────────────

def pinhole_ray(u: float, v: float,
                fx: float, fy: float, cx: float, cy: float) -> list:
    """
    Wandelt eine Pixelkoordinate in einen normalisierten 3D-Sichtstrahl
    im Kamera-Koordinatensystem um (Standard-Pinhole-Projektion).

    Kamera-Frame-Konvention (ROS / RealSense): X rechts, Y nach unten, Z in
    Blickrichtung (optische Achse).

    Args:
        u, v:       Pixelkoordinate im Bild (Spalte, Zeile)
        fx, fy:     Brennweiten in Pixel  ← auflösungsabhängig, siehe MLM-Parameter
        cx, cy:     Hauptpunkt in Pixel   ← auflösungsabhängig, siehe MLM-Parameter

    Returns:
        Normalisierter Richtungsvektor [dx, dy, dz] im Kamera-Frame
    """
    dx = (u - cx) / fx
    dy = (v - cy) / fy
    dz = 1.0
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    return [dx / norm, dy / norm, dz / norm]


def ray_table_intersect(ray_cam: list,
                        cam_pos: list, cam_quat: list,
                        z_table: float) -> tuple:
    """
    Berechnet den Schnittpunkt eines Kamera-Sichtstrahls mit der Tischebene
    (Z = z_table im Weltframe) und gibt die Weltkoordinaten (X, Y) zurück.

    Algorithmus:
        1. Strahl in Weltframe rotieren: ray_world = R(cam_quat) @ ray_cam
        2. Parametergleichung: P = cam_pos + t * ray_world
        3. t = (z_table - cam_pos.z) / ray_world.z
        4. X = cam_pos.x + t * ray_world.x
           Y = cam_pos.y + t * ray_world.y

    Args:
        ray_cam:   Normalisierter Sichtstrahl im Kamera-Frame [dx, dy, dz]
        cam_pos:   Kameraposition im Weltframe [x, y, z]
        cam_quat:  Kameraquaternion im Weltframe [qx, qy, qz, qw]
        z_table:   Tischhöhe im Weltframe [m]  ← gemessen 170 mm, bei Umbau anpassen

    Returns:
        (X_welt, Y_welt) – Schnittpunkt mit der Tischebene.
        Falls der Strahl parallel zur Tischebene verläuft (ray_world.z ≈ 0),
        wird (cam_pos.x, cam_pos.y) als Fallback zurückgegeben.
    """
    rot = Rotation.from_quat(cam_quat)          # [qx, qy, qz, qw]
    ray_world = rot.apply(ray_cam).tolist()

    ox, oy, oz = cam_pos
    dx, dy, dz = ray_world

    if abs(dz) < 1e-9:
        # Strahl parallel zur Tischebene – kein sinnvoller Schnittpunkt
        return (ox, oy)

    t = (z_table - oz) / dz
    return (ox + t * dx, oy + t * dy)


def depth_to_world_z(u: float, v: float, depth_m: float,
                     fx: float, fy: float, cx: float, cy: float,
                     cam_pos: list, cam_quat: list) -> float:
    """
    Wandelt einen Tiefenbild-Pixel in die Höhe Z_pick im Weltkoordinatensystem.

    Wird im PickPlaceController (WAIT_IMG_1 / WAIT_IMG_2) verwendet, um die
    exakte Pick-Höhe eines Klotzes zu bestimmen.

    Da die Kamera nicht senkrecht montiert ist, wird der vollständige 3D-Punkt
    im Kamera-Frame rekonstruiert und anschließend in den Weltframe rotiert.

    Tiefenformat: RealSense D435i liefert aligned-depth als uint16 in mm.
    Der Aufrufer übergibt den bereits in Meter umgerechneten Medianwert.

    Args:
        u, v:       Pixelkoordinate des Klotz-Zentrums (aus YOLO)
        depth_m:    Tiefenwert in Metern (Median aus 5×5-Patch, uint16 mm / 1000)
        fx, fy:     Linsenparameter [px] – D435i 640×480: fx=322, fy=322
        cx, cy:     Linsenparameter [px] – D435i 640×480: cx=320, cy=240
        cam_pos:    Kamera-Weltpose – Position [x, y, z] im Weltframe
        cam_quat:   Kamera-Weltpose – Orientierung [qx, qy, qz, qw] im Weltframe

    Returns:
        Z_pick – Höhe des Klotzes im Weltkoordinatensystem (Meter)
    """
    X_cam = (u - cx) * depth_m / fx
    Y_cam = (v - cy) * depth_m / fy
    Z_cam = depth_m
    rot = Rotation.from_quat(cam_quat)
    point_world = rot.apply([X_cam, Y_cam, Z_cam]) + np.array(cam_pos)
    return float(point_world[2])


