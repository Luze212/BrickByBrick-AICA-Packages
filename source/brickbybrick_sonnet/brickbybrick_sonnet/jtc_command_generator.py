import numpy as np
from modulo_components.lifecycle_component import LifecycleComponent
from modulo_core.encoded_state import EncodedState
import state_representation as sr


class JtcCommandGenerator(LifecycleComponent):
    def __init__(self, node_name: str, *args, **kwargs):
        super().__init__(node_name, *args, **kwargs)

        # --- PARAMETER ---
        self._v_max = sr.Parameter("v_max", 0.1, sr.ParameterType.DOUBLE)
        self.add_parameter("_v_max", "Gewünschte Geschwindigkeit in m/s")

        self._target_tf_name = sr.Parameter("target_tf_name", "mein_ziel_frame", sr.ParameterType.STRING)
        self.add_parameter("_target_tf_name", "Name des Frames im TF-Tree (muss mit Signal-to-TF Block übereinstimmen)")

        # --- INPUTS ---
        self._ist_pose = sr.CartesianPose("ist_pose", "world")
        self.add_input("ist_pose", "_ist_pose", EncodedState)

        self._target_pose = sr.CartesianPose("target_pose", "world")
        self.add_input("target_pose", "_target_pose", EncodedState)

        # --- OUTPUTS ---
        self._jtc_command = ""
        self.add_output("jtc_command", "_jtc_command", str)

        # --- INTERNER STATUS ---
        self._last_target_pos = None

    def on_validate_parameter_callback(self, parameter: sr.Parameter) -> bool:
        return True

    def on_configure_callback(self) -> bool:
        self._last_target_pos = None
        self._jtc_command = ""
        self.get_logger().info("JtcCommandGenerator: Konfiguriert.")
        return True

    def on_activate_callback(self) -> bool:
        self.get_logger().info("JtcCommandGenerator: Aktiviert – warte auf Zielpose.")
        return True

    def on_step_callback(self):
        # 1. Schutzabfrage: Warten bis echte Roboterdaten da sind
        if self._ist_pose.is_empty() or self._target_pose.is_empty():
            return

        current_target_pos = self._target_pose.get_position()

        # 2. Flankenauswertung (Wurde eine NEUE Pose geschickt?)
        if self._last_target_pos is not None:
            dist_to_last_target = np.linalg.norm(current_target_pos - self._last_target_pos)
            if dist_to_last_target < 0.001:
                return

        # 3. Mathematik: Euklidische Distanz vom aktuellen TCP zum neuen Ziel
        robot_pos = self._ist_pose.get_position()
        distance = np.linalg.norm(current_target_pos - robot_pos)

        # 4. Dauer berechnen (Zeit = Strecke / Geschwindigkeit)
        v_max = self._v_max.get_value()
        if v_max <= 0.0:
            v_max = 0.1

        duration = distance / v_max

        # Sicherheits-Limit: Niemals weniger als 0.5 Sekunden
        duration = max(duration, 0.5)

        # 5. String-Befehl zusammenbauen
        frame_name = self._target_tf_name.get_value()
        self._jtc_command = f"{{frames: [{frame_name}], durations: [{duration:.2f}]}}"

        self.get_logger().info(
            f"JtcCommandGenerator: Neues Ziel erkannt! "
            f"Distanz: {distance:.3f}m, Duration: {duration:.2f}s. "
            f"Befehl: {self._jtc_command}"
        )

        # 6. Status aktualisieren (verhindert Doppel-Sendung)
        self._last_target_pos = current_target_pos.copy()
