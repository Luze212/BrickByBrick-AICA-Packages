"""
ExplorationNavigator
─────────────────────────────────────────────────────────────────────────────
Phase 1 (EXPLORATION):
    Fährt vordefinierte Posen aus ExplCords.yaml nacheinander ab.
    Bei Zielerreichung (< 2 mm) wird ein Kamera-Handshake durchgeführt:
    Trigger senden → auf img_taken warten → nächste Pose.

Phase 2 (GATEWAY):
    Leitet alle Kommandos des PickPlaceControllers 1:1 an den Roboter weiter
    und berechnet weiterhin den Distanz-basierten trajectory_success.
"""

import math
import os
import yaml
import state_representation as sr

try:
    from ament_index_python.packages import get_package_share_directory as _get_share
    _SHARE = _get_share("brickbybrick_fix")
except Exception:
    _SHARE = "."

_DEFAULT_EXPL_PATH = os.path.join(_SHARE, "data", "exploration", "ExplCords.yaml")
from clproto import MessageType
from modulo_core.encoded_state import EncodedState
from modulo_components.lifecycle_component import LifecycleComponent
from std_msgs.msg import Bool


class ExplorationNavigator(LifecycleComponent):

    def __init__(self, node_name: str, *args, **kwargs):
        super().__init__(node_name, *args, **kwargs)

        # ── Parameter ─────────────────────────────────────────────────────────
        self._expl_coords_path = sr.Parameter(
            "expl_coords_path",
            _DEFAULT_EXPL_PATH,
            sr.ParameterType.STRING,
        )
        self.add_parameter(
            "_expl_coords_path",
            "Pfad zur YAML-Datei mit den Explorationsposen (ExplCords.yaml)",
        )

        # ── Inputs ────────────────────────────────────────────────────────────
        self._ist_pose = sr.CartesianPose("ist_pose", "world")
        self.add_input("ist_pose", "_ist_pose", EncodedState)

        self._target_pose_in = sr.CartesianPose("target_pose_in", "world")
        self.add_input("target_pose_in", "_target_pose_in", EncodedState)

        self._take_img_in = False
        self.add_input("take_img_in", "_take_img_in", Bool)

        self._img_taken = False
        self.add_input("img_taken", "_img_taken", Bool)

        # ── Outputs ───────────────────────────────────────────────────────────
        self._target_pose_out = sr.CartesianPose("target_pose_out", "world")
        self.add_output(
            "target_pose_out", "_target_pose_out",
            EncodedState, MessageType.CARTESIAN_POSE_MESSAGE,
        )

        self._take_img_out = False
        self.add_output("take_img_out", "_take_img_out", Bool)

        self._trajectory_success = False
        self.add_output("trajectory_success", "_trajectory_success", Bool)

        self._trigger_ppl = False
        self.add_output("trigger_ppl", "_trigger_ppl", Bool)

        # ── Interne Zustandsvariablen (über Taktzyklen hinweg persistent) ─────
        self._state: str = "EXPLORATION"
        self._exploration_pose_list: list = []
        self._waiting_for_camera_reset: bool = False
        self._phase_done_timer = None   # gesetzt wenn Liste leer wird

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle-Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def on_validate_parameter_callback(self, parameter: sr.Parameter) -> bool:
        return True

    def on_configure_callback(self) -> bool:
        path = self._expl_coords_path.get_value()
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            self.get_logger().error(
                f"ExplorationNavigator: ExplCords.yaml nicht gefunden: {path}"
            )
            return False
        except Exception as exc:
            self.get_logger().error(
                f"ExplorationNavigator: Fehler beim Lesen von ExplCords.yaml: {exc}"
            )
            return False

        self._exploration_pose_list = []

        if data is None:
            self.get_logger().warn(
                "ExplorationNavigator: ExplCords.yaml ist leer – "
                "startet direkt im GATEWAY-Modus."
            )
            return True

        for pose_name, entry in data.items():
            try:
                pos = entry["position"]
                ori = entry["orientation"]
                pose = sr.CartesianPose(pose_name, "world")
                pose.set_position([pos["x"], pos["y"], pos["z"]])
                pose.set_orientation([ori["qw"], ori["qx"], ori["qy"], ori["qz"]])
                self._exploration_pose_list.append(pose)
            except KeyError as exc:
                self.get_logger().error(
                    f"ExplorationNavigator: Ungültiges Format in Pose '{pose_name}': "
                    f"fehlendes Feld {exc}"
                )
                return False

        # Interne Flags zurücksetzen
        self._state = "EXPLORATION"
        self._waiting_for_camera_reset = False
        self._trajectory_success = False
        self._trigger_ppl = False
        self._phase_done_timer = None

        # Fix L-3: target_pose_out auf leere/neutrale Pose zurücksetzen,
        # um "Geister-Bewegungen" beim Neustart zu verhindern.
        self._target_pose_out = sr.CartesianPose("target_pose_out", "world")

        self.get_logger().info(
            f"ExplorationNavigator konfiguriert: "
            f"{len(self._exploration_pose_list)} Pose(n) aus '{path}' geladen."
        )
        return True

    def on_activate_callback(self) -> bool:
        self.get_logger().info(
            f"ExplorationNavigator aktiviert. Startzustand: {self._state}, "
            f"ausstehende Posen: {len(self._exploration_pose_list)}"
        )
        return True

    def on_deactivate_callback(self) -> bool:
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Haupt-Taktschleife
    # ─────────────────────────────────────────────────────────────────────────

    def on_step_callback(self):
        if self._state == "EXPLORATION":
            self._run_exploration_step()
        elif self._state == "GATEWAY":
            self._run_gateway_step()

    # ─────────────────────────────────────────────────────────────────────────
    # Private Zustandslogik
    # ─────────────────────────────────────────────────────────────────────────

    def _run_exploration_step(self):
        # ── Liste erschöpft: 0,5-s-Timer für YOLO/MLM-Pipeline ──────────────
        if len(self._exploration_pose_list) == 0:
            if self._phase_done_timer is None:
                self._phase_done_timer = self.get_clock().now()
                self.get_logger().info(
                    "ExplorationNavigator: Explorationsliste leer – "
                    "starte 0,5-s-Warten auf letzte YOLO/MLM-Verarbeitung."
                )
                return

            elapsed = (
                self.get_clock().now() - self._phase_done_timer
            ).nanoseconds / 1e9
            if elapsed < 0.5:
                return

            # Timer abgelaufen – GATEWAY aktivieren
            self._trigger_ppl = True
            self._state = "GATEWAY"
            self.get_logger().info(
                "ExplorationNavigator: ZUSTANDSWECHSEL EXPLORATION → GATEWAY. "
                "trigger_ppl = True."
            )
            return

        # ── Aktuelle Zielpose konstant an Roboter senden ─────────────────────
        current_target = self._exploration_pose_list[0]
        self._target_pose_out.set_position(current_target.get_position())
        self._target_pose_out.set_orientation(current_target.get_orientation())

        # Nur auswerten, wenn ist_pose bereits gültige Daten trägt
        if self._ist_pose.is_empty():
            return

        dist = _euclidean_distance(self._ist_pose, current_target)

        # ── Zielerreichung: Kamera-Trigger senden ────────────────────────────
        if dist < 0.002 and not self._waiting_for_camera_reset:
            self._trajectory_success = True
            self._take_img_out = True

        # ── Handshake: Kamera bestätigt Snapshot ─────────────────────────────
        # Guard: not _waiting_for_camera_reset verhindert Doppel-Pop falls
        # img_taken über mehrere Zyklen True bleibt.
        if self._img_taken and not self._waiting_for_camera_reset:
            self._take_img_out = False
            self._trajectory_success = False
            self._exploration_pose_list.pop(0)
            self._waiting_for_camera_reset = True
            self.get_logger().info(
                f"ExplorationNavigator: Kamera-Handshake abgeschlossen. "
                f"Noch {len(self._exploration_pose_list)} Pose(n) ausstehend."
            )

        # ── Safety-Lock lösen: img_taken ist zurückgefallen ──────────────────
        if self._waiting_for_camera_reset and not self._img_taken:
            self._waiting_for_camera_reset = False
            self.get_logger().info(
                "ExplorationNavigator: Kamera-Reset bestätigt – "
                "nächste Pose wird angefahren."
            )

    def _run_gateway_step(self):
        # ── Pass-Through: PPL-Kommandos 1:1 an Roboter weiterleiten ──────────
        if not self._target_pose_in.is_empty():
            self._target_pose_out.set_position(self._target_pose_in.get_position())
            self._target_pose_out.set_orientation(
                self._target_pose_in.get_orientation()
            )

        self._take_img_out = self._take_img_in
        # trigger_ppl bleibt dauerhaft True – kein Reset

        # ── Distanzbasierter trajectory_success auch im GATEWAY-Modus ────────
        if not self._ist_pose.is_empty() and not self._target_pose_in.is_empty():
            dist = _euclidean_distance(self._ist_pose, self._target_pose_in)
            self._trajectory_success = dist < 0.002
        else:
            self._trajectory_success = False


# ─────────────────────────────────────────────────────────────────────────────
# Modul-Hilfsfunktion (kein Import von geometry_utils nötig, da reine Geometrie)
# ─────────────────────────────────────────────────────────────────────────────

def _euclidean_distance(pose_a: sr.CartesianPose, pose_b: sr.CartesianPose) -> float:
    """Euklidische Distanz zwischen den Positionen zweier CartesianPose-Objekte."""
    pa = pose_a.get_position()
    pb = pose_b.get_position()
    return math.sqrt(
        (pa[0] - pb[0]) ** 2 +
        (pa[1] - pb[1]) ** 2 +
        (pa[2] - pb[2]) ** 2
    )
