"""
PoseTriggeredCamera
─────────────────────────────────────────────────────────────────────────────
Schießt einen verzögerten Snapshot und friert Roboter- und Kameradaten
100 % synchron ein.

Ablauf:
  1. Trigger: take_img == True UND trajectory_success == True
  2. Warte 0,3 s (Vibrations-Settling-Delay)
  3. Deep-Copy von image_stream, ist_pose_in, cam_ist_pose_in auf Outputs
  4. img_taken = True (Handshake-Signal an ExplorationNavigator / PPL)
  5. Reset: sobald trajectory_success wieder False wird → img_taken = False
"""

import state_representation as sr
from clproto import MessageType
from modulo_core.encoded_state import EncodedState
from modulo_components.lifecycle_component import LifecycleComponent
from std_msgs.msg import Bool


class PoseTriggeredCamera(LifecycleComponent):

    def __init__(self, node_name: str, *args, **kwargs):
        super().__init__(node_name, *args, **kwargs)

        # ── Inputs ────────────────────────────────────────────────────────────
        self._take_img = False
        self.add_input("take_img", "_take_img", Bool)

        self._trajectory_success = False
        self.add_input("trajectory_success", "_trajectory_success", Bool)

        self._ist_pose_in = sr.CartesianPose("ist_pose_in", "world")
        self.add_input("ist_pose_in", "_ist_pose_in", EncodedState)

        self._cam_ist_pose_in = sr.CartesianPose("cam_ist_pose_in", "world")
        self.add_input("cam_ist_pose_in", "_cam_ist_pose_in", EncodedState)

        self._image_stream = sr.Image()
        self.add_input("image_stream", "_image_stream", EncodedState)

        # ── Outputs ───────────────────────────────────────────────────────────
        self._img_taken = False
        self.add_output("img_taken", "_img_taken", Bool)

        self._ist_pose_out = sr.CartesianPose("ist_pose_out", "world")
        self.add_output(
            "ist_pose_out", "_ist_pose_out",
            EncodedState, MessageType.CARTESIAN_POSE_MESSAGE,
        )

        self._cam_ist_pose_out = sr.CartesianPose("cam_ist_pose_out", "world")
        self.add_output(
            "cam_ist_pose_out", "_cam_ist_pose_out",
            EncodedState, MessageType.CARTESIAN_POSE_MESSAGE,
        )

        self._image_out = sr.Image()
        self.add_output("image_out", "_image_out", EncodedState)

        # ── Interne Zustandsvariablen (über Taktzyklen hinweg persistent) ─────
        self._is_delaying: bool = False
        self._timer_start = None    # ROS-Zeitstempel; gesetzt wenn Delay startet

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle-Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def on_validate_parameter_callback(self, parameter: sr.Parameter) -> bool:
        return True

    def on_configure_callback(self) -> bool:
        self._is_delaying = False
        self._timer_start = None
        self._img_taken = False
        return True

    def on_activate_callback(self) -> bool:
        return True

    def on_deactivate_callback(self) -> bool:
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Haupt-Taktschleife
    # ─────────────────────────────────────────────────────────────────────────

    def on_step_callback(self):
        # ── Abbruch-Sicherheit: Roboter wurde weggeschoben während Delay ──────
        if self._is_delaying and not self._trajectory_success:
            self._is_delaying = False
            self._timer_start = None
            self.get_logger().info(
                "PoseTriggeredCamera: Delay abgebrochen – "
                "trajectory_success fiel während Wartezeit ab."
            )

        # ── Trigger-Bedingung ─────────────────────────────────────────────────
        if self._take_img and self._trajectory_success and not self._img_taken:
            if not self._is_delaying:
                # Delay-Timer starten
                self._is_delaying = True
                self._timer_start = self.get_clock().now()
                self.get_logger().info(
                    "PoseTriggeredCamera: Trigger empfangen – starte 0,3-s-Settling-Delay."
                )

            # Laufenden Delay auswerten
            if self._is_delaying:
                elapsed = (
                    self.get_clock().now() - self._timer_start
                ).nanoseconds / 1e9

                if elapsed >= 0.3:
                    # ── Snapshot-Moment: Klon-Konstruktor für C++-gebundene Objekte ─
                    self._image_out = sr.Image(self._image_stream)
                    self._ist_pose_out = sr.CartesianPose(self._ist_pose_in)
                    self._cam_ist_pose_out = sr.CartesianPose(self._cam_ist_pose_in)
                    self._img_taken = True
                    self._is_delaying = False
                    self._timer_start = None
                    self.get_logger().info(
                        "PoseTriggeredCamera: Snapshot eingefroren – img_taken = True."
                    )

        # ── Handshake-Reset: Roboter ist abgefahren ───────────────────────────
        # trajectory_success fällt auf False → Handshake-Zyklus abgeschlossen
        if not self._trajectory_success and self._img_taken:
            self._img_taken = False
            self.get_logger().info(
                "PoseTriggeredCamera: Handshake-Reset – img_taken = False."
            )
