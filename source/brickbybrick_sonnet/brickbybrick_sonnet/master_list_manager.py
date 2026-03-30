"""
MasterListManager (DataHandler)
─────────────────────────────────────────────────────────────────────────────
Empfängt YOLO-Eckpunkte und Ablagelinien-Posen, führt geometrische
Berechnungen durch und erzeugt drei persistente Ausgabe-Arrays:

  master_overview  – Roboter-TCP-Posen zum Zeitpunkt jeder Bildaufnahme
                     (nur in Phase 1 befüllt, Stride 7: [X,Y,Z,Qx,Qy,Qz,Qw,...])

  master_dropoff   – 3D-Ablageposen vom DropoffLineExtractor
                     (Stride 7: [X,Y,Z,Qx,Qy,Qz,Qw,...])

  filtered_yolo    – Gefilterte Klotz-Posen, wird bei JEDER neuen YOLO-Inferenz
                     (Phase 1 und Phase 2!) neu berechnet und überschrieben.
                     Stride 9: [X, Y, Area, u_center, v_center, Qx, Qy, Qz, Qw,...]

Wichtig: master_overview wird nur in Phase 1 (trigger_ppl == False) befüllt.
         filtered_yolo wird immer neu berechnet – das ist der Closed-Loop-Mechanismus.
"""

import math

import state_representation as sr
from modulo_core.encoded_state import EncodedState
from modulo_components.lifecycle_component import LifecycleComponent
from std_msgs.msg import Bool, Float64MultiArray

from brickbybrick_sonnet.geometry_utils import (
    quaternion_from_euler,
    yaw_from_quaternion,
    minimize_twist,
    gauss_shoelace_area,
)


class MasterListManager(LifecycleComponent):

    def __init__(self, node_name: str, *args, **kwargs):
        super().__init__(node_name, *args, **kwargs)

        # ── Inputs ────────────────────────────────────────────────────────────
        # yolo_done_trigger: Event-Trigger – user_callback prüft steigende Flanke
        self._yolo_done_trigger = False
        self.add_input(
            "yolo_done_trigger", "_yolo_done_trigger", Bool,
            user_callback=self._on_yolo_trigger,
        )

        # YOLO-Eckpunkte werden synchron mit yolo_done_trigger geliefert
        self._yolo_corners_list_in = []
        self.add_input("yolo_corners_list_in", "_yolo_corners_list_in", Float64MultiArray)

        # line_ex_list_in: Event-Trigger – user_callback prüft Längenänderung
        self._line_ex_list_in = []
        self.add_input(
            "line_ex_list_in", "_line_ex_list_in", Float64MultiArray,
            user_callback=self._on_line_data,
        )

        self._ist_pose_in = sr.CartesianPose("ist_pose_in", "world")
        self.add_input("ist_pose_in", "_ist_pose_in", EncodedState)

        self._cam_ist_pose_in = sr.CartesianPose("cam_ist_pose_in", "world")
        self.add_input("cam_ist_pose_in", "_cam_ist_pose_in", EncodedState)

        self._trigger_ppl = False
        self.add_input("trigger_ppl", "_trigger_ppl", Bool)

        # ── Outputs ───────────────────────────────────────────────────────────
        self._master_overview = []
        self.add_output("master_overview", "_master_overview", Float64MultiArray)

        self._master_dropoff = []
        self.add_output("master_dropoff", "_master_dropoff", Float64MultiArray)

        self._filtered_yolo = []
        self.add_output("filtered_yolo", "_filtered_yolo", Float64MultiArray)

        # Stafetten-Trigger: signalisiert dem PPL, dass filtered_yolo frisch ist.
        # Wird am Ende von _on_yolo_trigger auf True gesetzt und im
        # on_step_callback sofort wieder auf False (saubere 1-Takt-Flanke).
        self._mlm_done_trigger = False
        self.add_output("mlm_done_trigger", "_mlm_done_trigger", Bool)

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle-Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def on_validate_parameter_callback(self, parameter: sr.Parameter) -> bool:
        return True

    def on_configure_callback(self) -> bool:
        self._master_overview = []
        self._master_dropoff = []
        self._filtered_yolo = []
        self._mlm_done_trigger = False
        self.get_logger().info("MasterListManager: Konfiguriert – Arrays geleert.")
        return True

    def on_activate_callback(self) -> bool:
        self.get_logger().info(
            "MasterListManager: Aktiviert – wartet auf YOLO- und Linien-Events."
        )
        return True

    def on_deactivate_callback(self) -> bool:
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Taktschleife – vollständig event-getrieben, kein periodischer Code
    # ─────────────────────────────────────────────────────────────────────────

    def on_step_callback(self):
        # Saubere steigende Flanke: Trigger im darauffolgenden Takt zurücksetzen
        if self._mlm_done_trigger:
            self._mlm_done_trigger = False

    # ─────────────────────────────────────────────────────────────────────────
    # Event 1: YOLO-Erkennung abgeschlossen (steigende Flanke von yolo_done_trigger)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_yolo_trigger(self):
        # Nur bei steigender Flanke (True) verarbeiten
        if not self._yolo_done_trigger:
            return

        corners = self._yolo_corners_list_in
        if len(corners) == 0:
            # Kein Klotz erkannt – filtered_yolo leeren (physikalisch entfernt)
            self._filtered_yolo = []
            self._mlm_done_trigger = True
            self.get_logger().info(
                "MasterListManager: YOLO lieferte 0 Klötze – filtered_yolo geleert."
            )
            return

        # ── Snapshot der Posen für diesen Erkennungszyklus ───────────────────
        # Deep-Copy via C++ Klon-Konstruktor – verhindert Race Conditions,
        # falls das Framework die Input-Variable zwischen get_position() und
        # get_orientation() aktualisiert.
        ist_pose = sr.CartesianPose(self._ist_pose_in) if not self._ist_pose_in.is_empty() else None
        cam_ist_pose = sr.CartesianPose(self._cam_ist_pose_in) if not self._cam_ist_pose_in.is_empty() else None

        # ── Phase 1: TCP-Pose zur Übersichtspose-Liste hinzufügen ────────────
        # trigger_ppl == True bedeutet Phase 2 → kein Anhängen mehr
        if not self._trigger_ppl and ist_pose is not None:
            pos = ist_pose.get_position()
            ori = ist_pose.get_orientation()
            self._master_overview.extend([
                float(pos[0]), float(pos[1]), float(pos[2]),
                float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3]),
            ])
            self.get_logger().info(
                f"MasterListManager: master_overview erweitert "
                f"({len(self._master_overview) // 7} Posen gesamt)."
            )

        # ── Aktuellen Roboter-Yaw für Twist-Minimierung bestimmen ────────────
        current_robot_yaw = 0.0
        if ist_pose is not None:
            ori = ist_pose.get_orientation()
            current_robot_yaw = yaw_from_quaternion(
                float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])
            )

        # ── Geometrie-Berechnung für jeden Klotz (Stride 8) ──────────────────
        pending_bricks = []  # je Eintrag: [X, Y, Area, u_center, v_center, Qx, Qy, Qz, Qw]

        for i in range(0, len(corners), 8):
            if i + 8 > len(corners):
                break

            u1, v1 = float(corners[i]),     float(corners[i + 1])
            u2, v2 = float(corners[i + 2]), float(corners[i + 3])
            u3, v3 = float(corners[i + 4]), float(corners[i + 5])
            u4, v4 = float(corners[i + 6]), float(corners[i + 7])

            # Diagonalschnittpunkt als Klotz-Center
            u_center = (u1 + u3) / 2.0
            v_center = (v1 + v3) / 2.0

            # Polygonfläche (Gauß-Shoelace) als Proxy für Klotzgröße
            area = gauss_shoelace_area([(u1, v1), (u2, v2), (u3, v3), (u4, v4)])

            # Kantenwinkel der ersten Kante (Ausrichtung im 2D-Bild)
            theta = math.atan2(v2 - v1, u2 - u1)

            # ══════════════════════════════════════════════════════════════════
            # TO-DO: 3D-Projektion (Pinhole-Kamera + Strahl-Schnitt mit Tisch)
            #
            # Benötigt: Verifizierte Kamerakalibrierung (fx, fy, cx, cy) und
            #           Tischhöhe (z_table) – aus KUKA URDF / ROS camera_info.
            #
            # Voraussichtlicher Code:
            #   from brickbybrick_sonnet.geometry_utils import (
            #       pinhole_ray, ray_table_intersect
            #   )
            #   ray = pinhole_ray(u_center, v_center, fx, fy, cx, cy)
            #   X_klotz, Y_klotz = ray_table_intersect(ray, cam_ist_pose, z_table)
            # ══════════════════════════════════════════════════════════════════
            X_klotz = 0.0
            Y_klotz = 0.0

            # Twist-Minimierung: wähle optimalen Greif-Yaw (θ oder θ+180°)
            opt_yaw = minimize_twist(theta, current_robot_yaw)

            # Quaternion: Sauger zeigt strikt nach unten (Roll=π), Yaw=opt_yaw
            quat = quaternion_from_euler(math.pi, 0.0, opt_yaw)  # [qx, qy, qz, qw]

            pending_bricks.append([
                X_klotz, Y_klotz, area, u_center, v_center,
                quat[0], quat[1], quat[2], quat[3],
            ])

        # ── Filter: Klötze auf Ablagepositionen verwerfen ─────────────────────
        # Distanz < 0.01 m → Klotz liegt bereits auf einer Ablagelinie
        new_filtered = []
        for brick in pending_bricks:
            X_b, Y_b = brick[0], brick[1]
            too_close = False
            for j in range(0, len(self._master_dropoff), 7):
                if j + 7 > len(self._master_dropoff):
                    break
                X_d = float(self._master_dropoff[j])
                Y_d = float(self._master_dropoff[j + 1])
                dist = math.sqrt((X_b - X_d) ** 2 + (Y_b - Y_d) ** 2)
                if dist < 0.01:
                    too_close = True
                    break
            if not too_close:
                new_filtered.extend(brick)

        self._filtered_yolo = new_filtered
        self._mlm_done_trigger = True
        self.get_logger().info(
            f"MasterListManager: YOLO-Callback abgeschlossen – "
            f"{len(pending_bricks)} Klotz/Klötze erkannt, "
            f"{len(new_filtered) // 9} nach Ablage-Filter in filtered_yolo."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Event 2: Neue Ablage-Posen vom DropoffLineExtractor
    # ─────────────────────────────────────────────────────────────────────────

    def _on_line_data(self):
        # Überschreibe master_dropoff nur wenn die neue Liste länger ist
        if len(self._line_ex_list_in) > len(self._master_dropoff):
            self._master_dropoff = list(self._line_ex_list_in)
            self.get_logger().info(
                f"MasterListManager: master_dropoff aktualisiert – "
                f"{len(self._master_dropoff) // 7} Ablageposition(en)."
            )
