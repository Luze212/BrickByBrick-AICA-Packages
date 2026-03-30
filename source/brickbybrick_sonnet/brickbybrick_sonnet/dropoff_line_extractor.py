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
"""

import state_representation as sr
from modulo_core.encoded_state import EncodedState
from modulo_components.lifecycle_component import LifecycleComponent
from std_msgs.msg import Bool, Float64MultiArray


class DropoffLineExtractor(LifecycleComponent):

    def __init__(self, node_name: str, *args, **kwargs):
        super().__init__(node_name, *args, **kwargs)

        # ── Inputs ────────────────────────────────────────────────────────────
        self._image_in = sr.Image()
        self.add_input("image_in", "_image_in", EncodedState)

        self._cam_ist_pose = sr.CartesianPose("cam_ist_pose", "world")
        self.add_input("cam_ist_pose", "_cam_ist_pose", EncodedState)

        # yolo_done_trigger: Event-Trigger – user_callback prüft steigende Flanke
        self._yolo_done_trigger = False
        self.add_input(
            "yolo_done_trigger", "_yolo_done_trigger", Bool,
            user_callback=self._on_yolo_trigger,
        )

        self._trigger_ppl = False
        self.add_input("trigger_ppl", "_trigger_ppl", Bool)

        # ── Outputs ───────────────────────────────────────────────────────────
        # Persistenter Output: wird nur bei neuer Linien-Erkennung überschrieben.
        # MasterListManager vergleicht Länge und übernimmt das größere Ergebnis.
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
        return True

    def on_deactivate_callback(self) -> bool:
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Taktschleife – rein event-getrieben, kein periodischer Code nötig
    # ─────────────────────────────────────────────────────────────────────────

    def on_step_callback(self):
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Event-Callback: yolo_done_trigger (steigende Flanke)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_yolo_trigger(self):
        # ── Bypass Phase 2: In Phase 2 läuft dieser Block nicht mehr ─────────
        # trigger_ppl wird dauerhaft True, sobald Exploration abgeschlossen ist.
        if self._trigger_ppl:
            return

        # ── Steigende Flanke: nur bei True auswerten ──────────────────────────
        # Der Callback feuert bei jeder eingehenden Nachricht auf dem Signal.
        # Da yolo_done_trigger für genau einen Takt True ist, filtert diese
        # Prüfung alle "False"-Nachrichten heraus.
        if not self._yolo_done_trigger:
            return

        self.get_logger().info(
            "DropoffLineExtractor: Trigger empfangen – starte Linienerkennung."
        )

        # ── Bilddaten laden ───────────────────────────────────────────────────
        image_array = self._image_in.get_data()
        if image_array is None or image_array.size == 0:
            self.get_logger().warn(
                "DropoffLineExtractor: Leeres Bild empfangen – überspringe Erkennung."
            )
            return

        # ══════════════════════════════════════════════════════════════════════
        # TO-DO: Bernhards Linienerkennungs-Algorithmus (Farbabgleich)
        #
        # Input:
        #   image_array   – numpy array des eingefrorenen Kamerabildes (H×W×C)
        #   cam_ist_pose  – sr.CartesianPose der Kamera im Weltframe
        #                   (self._cam_ist_pose, bereits im Klon-Konstruktor verfügbar)
        #
        # Aufgabe:
        #   Finde alle sichtbaren Ablagelinien im Bild via Farberkennung
        #   (Bernhards Algorithmus, z.B. HSV-Maskierung o.ä.).
        #   Berechne für jede Ablagestelle eine 3D-TCP-Pose im Weltframe.
        #
        # Output (in line_poses_flat schreiben):
        #   Flaches Array, Stride 7 pro Ablagepose:
        #   [X, Y, Z, Qx, Qy, Qz, Qw, X2, Y2, Z2, Qx2, Qy2, Qz2, Qw2, ...]
        #
        #   Beispiel für zwei Ablageposen:
        #   line_poses_flat = [
        #       0.50, 0.10, 0.02, 0.0, 0.0, 0.0, 1.0,   # Pose 1
        #       0.50, 0.20, 0.02, 0.0, 0.0, 0.0, 1.0,   # Pose 2
        #   ]
        #
        # Hinweis:
        #   Die Rotation in Qx,Qy,Qz,Qw soll bereits die TCP-Ausrichtung für
        #   das Ablegen codieren (Sauger nach unten, korrekte Linienverdrehung).
        #   Siehe geometry_utils.quaternion_from_euler() für Quaternion-Erstellung.
        # ══════════════════════════════════════════════════════════════════════

        line_poses_flat = []   # Platzhalter – bis Bernhards Algorithmus steht

        self._line_ex_list = line_poses_flat

        self.get_logger().info(
            f"DropoffLineExtractor: {len(line_poses_flat) // 7} Ablageposen erkannt "
            f"und auf line_ex_list geschrieben."
        )
