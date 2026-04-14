import numpy as np
from modulo_components.lifecycle_component import LifecycleComponent
from modulo_core.encoded_state import EncodedState
import state_representation as sr
from modulo_interfaces.srv import StringTrigger


class JtcCommandGenerator(LifecycleComponent):
    def __init__(self, node_name: str, *args, **kwargs):
        super().__init__(node_name, *args, **kwargs)

        # --- PARAMETER ---
        self._v_max = sr.Parameter("v_max", 0.1, sr.ParameterType.DOUBLE)
        self.add_parameter("_v_max", "Gewünschte Geschwindigkeit in m/s")

        self._target_tf_name = sr.Parameter("target_tf_name", "mein_ziel_frame", sr.ParameterType.STRING)
        self.add_parameter("_target_tf_name", "Name des Frames im TF-Tree (muss mit Signal-to-TF Block übereinstimmen)")

        self._service_name = sr.Parameter("service_name", "jtc_trigger_service", sr.ParameterType.STRING)
        self.add_parameter("_service_name", "ROS 2 Service-Name des JTC set_trajectory Service")

        # --- INPUTS ---
        self._ist_pose = sr.CartesianPose("ist_pose", "world")
        self.add_input("ist_pose", "_ist_pose", EncodedState)

        self._target_pose = sr.CartesianPose("target_pose", "world")
        self.add_input("target_pose", "_target_pose", EncodedState)

        # --- INTERNER STATUS ---
        self._jtc_client = None
        self._last_target_pos = None

    def on_validate_parameter_callback(self, parameter: sr.Parameter) -> bool:
        return True

    def on_configure_callback(self) -> bool:
        # Service Client wird erst hier erstellt, damit der service_name-Parameter
        # bereits durch die UI überschrieben werden konnte.
        srv_name = self._service_name.get_value()
        self._jtc_client = self.create_client(StringTrigger, srv_name)
        self._last_target_pos = None
        self.get_logger().info(f"JtcCommandGenerator: Konfiguriert. Service: '{srv_name}'")
        return True

    def on_activate_callback(self) -> bool:
        self.get_logger().info("JtcCommandGenerator: Aktiviert – warte auf Zielpose.")
        return True

    def on_deactivate_callback(self) -> bool:
        self._last_target_pos = None
        return True

    def on_step_callback(self):
        # 1. Schutzabfrage: Warten bis echte Roboterdaten da sind
        if self._ist_pose.is_empty() or self._target_pose.is_empty():
            return

        current_target_pos = self._target_pose.get_position()

        # 2. Flankenauswertung: Nur feuern, wenn sich das Ziel um > 1mm ändert
        if self._last_target_pos is not None:
            dist_to_last_target = np.linalg.norm(current_target_pos - self._last_target_pos)
            if dist_to_last_target < 0.001:
                return

        # 3. Euklidische Distanz vom aktuellen TCP zum neuen Ziel
        robot_pos = self._ist_pose.get_position()
        distance = np.linalg.norm(current_target_pos - robot_pos)

        # 4. Dauer berechnen (Zeit = Strecke / Geschwindigkeit), mind. 0.5s
        v_max = self._v_max.get_value()
        if v_max <= 0.0:
            v_max = 0.1
        duration = max(distance / v_max, 0.5)

        # 5. Payload-String zusammenbauen
        frame_name = self._target_tf_name.get_value()
        command_string = f"{{frames: [{frame_name}], durations: [{duration:.2f}]}}"

        # 6. Asynchron an den JTC Service senden
        self._send_jtc_service_request(command_string)

        # 7. Ziel verriegeln, um Doppelsendung zu verhindern
        self._last_target_pos = current_target_pos.copy()

    def _send_jtc_service_request(self, command_string: str):
        """Sendet den Payload-String nicht-blockierend an den JTC set_trajectory Service."""
        if self._jtc_client is None:
            self.get_logger().error("JTC Service Client nicht initialisiert (on_configure aufgerufen?).")
            return

        # Nicht-blockierende Verfügbarkeitsprüfung
        if not self._jtc_client.service_is_ready():
            self.get_logger().warn("JTC Service nicht erreichbar – Befehl wird verworfen.")
            return

        req = StringTrigger.Request()
        req.payload = command_string

        future = self._jtc_client.call_async(req)
        future.add_done_callback(self._service_response_callback)
        self.get_logger().info(f"JtcCommandGenerator: Service Request gesendet: {command_string}")

    def _service_response_callback(self, future):
        """Wird aufgerufen, sobald der JTC den Befehl verarbeitet hat."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"JTC: Trajektorie akzeptiert. '{response.message}'")
            else:
                self.get_logger().warn(f"JTC: Trajektorie abgelehnt! '{response.message}'")
        except Exception as e:
            self.get_logger().error(f"JTC Service Call fehlgeschlagen: {e}")
