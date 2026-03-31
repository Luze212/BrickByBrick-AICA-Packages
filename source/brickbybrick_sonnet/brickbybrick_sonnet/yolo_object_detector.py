"""
YoloObjectDetector
─────────────────────────────────────────────────────────────────────────────
Event-getriebener Block. Wartet passiv auf neue Bilder von PoseTriggeredCamera,
jagt sie durch das YOLOv11-OBB Modell und gibt die 2D-Eckpunkte aller
erkannten Klötze als flaches Array weiter.

Ablauf (user_callback auf image_in):
  1. Bild + synchrone Posen ankommen
  2. YOLO OBB Inferenz
  3. Rand-Filter: Masken < 5 px vom Bildrand aussortieren
  4. Eckpunkte in flaches Array packen [u1,v1, u2,v2, u3,v3, u4,v4, ...]
  5. Posen 1:1 weiterleiten
  6. yolo_done_trigger = True setzen (on_step_callback setzt ihn nächsten Takt zurück)

Array-Format Ausgang yolo_corners_list:
  Stride 8 pro Klotz: [u1, v1, u2, v2, u3, v3, u4, v4, u1_B, v1_B, ...]
"""

import os
import numpy as np

try:
    from ament_index_python.packages import get_package_share_directory as _get_share
    _SHARE = _get_share("brickbybrick_sonnet")
except Exception:
    _SHARE = "."

_DEFAULT_MODEL_PATH = os.path.join(_SHARE, "data", "model", "best.pt")
import state_representation as sr
from clproto import MessageType
from modulo_core.encoded_state import EncodedState
from modulo_components.lifecycle_component import LifecycleComponent
from std_msgs.msg import Bool, Float64MultiArray


class YoloObjectDetector(LifecycleComponent):

    def __init__(self, node_name: str, *args, **kwargs):
        super().__init__(node_name, *args, **kwargs)

        # ── Parameter ─────────────────────────────────────────────────────────
        self._model_path = sr.Parameter(
            "model_path",
            _DEFAULT_MODEL_PATH,
            sr.ParameterType.STRING,
        )
        self.add_parameter(
            "_model_path",
            "Pfad zur trainierten YOLOv11-OBB Gewichtsdatei (.pt)",
        )

        # ── Inputs ────────────────────────────────────────────────────────────
        # image_in: Event-getrieben – user_callback wird bei jedem neuen Bild gefeuert
        self._image_in = sr.Image()
        self.add_input(
            "image_in", "_image_in", EncodedState,
            user_callback=self._on_new_image,
        )

        self._ist_pose_in = sr.CartesianPose("ist_pose_in", "world")
        self.add_input("ist_pose_in", "_ist_pose_in", EncodedState)

        self._cam_ist_pose_in = sr.CartesianPose("cam_ist_pose_in", "world")
        self.add_input("cam_ist_pose_in", "_cam_ist_pose_in", EncodedState)

        # ── Outputs ───────────────────────────────────────────────────────────
        self._yolo_corners_list = []
        self.add_output("yolo_corners_list", "_yolo_corners_list", Float64MultiArray)

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

        self._yolo_done_trigger = False
        self.add_output("yolo_done_trigger", "_yolo_done_trigger", Bool)

        # ── Internes Modell-Handle (persistent, über Taktzyklen hinweg) ───────
        self._model = None      # wird in on_configure_callback geladen

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle-Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def on_validate_parameter_callback(self, parameter: sr.Parameter) -> bool:
        return True

    def on_configure_callback(self) -> bool:
        """
        Lädt das YOLOv11-OBB Modell einmalig in den RAM/GPU.
        Dies ist ein zeitintensiver Vorgang – darf NICHT im on_step_callback passieren.
        """
        path = self._model_path.get_value()
        try:
            from ultralytics import YOLO
            self._model = YOLO(path)
            self.get_logger().info(
                f"YoloObjectDetector: Modell erfolgreich geladen von '{path}'."
            )
        except FileNotFoundError:
            self.get_logger().error(
                f"YoloObjectDetector: Modelldatei nicht gefunden: '{path}'"
            )
            return False
        except Exception as exc:
            self.get_logger().error(
                f"YoloObjectDetector: Fehler beim Laden des Modells: {exc}"
            )
            return False

        self._yolo_done_trigger = False
        return True

    def on_activate_callback(self) -> bool:
        self.get_logger().info(
            "YoloObjectDetector: Aktiviert – wartet auf erstes Bild-Event."
        )
        return True

    def on_deactivate_callback(self) -> bool:
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Taktschleife – nur für yolo_done_trigger Flanken-Reset
    # ─────────────────────────────────────────────────────────────────────────

    def on_step_callback(self):
        # Saubere steigende Flanke: Trigger im darauffolgenden Takt zurücksetzen
        if self._yolo_done_trigger:
            self._yolo_done_trigger = False

    # ─────────────────────────────────────────────────────────────────────────
    # Event-Callback: neues (eingefrorenes) Bild von PoseTriggeredCamera
    # ─────────────────────────────────────────────────────────────────────────

    def _on_new_image(self):
        if self._model is None:
            self.get_logger().warn(
                "YoloObjectDetector: Bild empfangen, aber Modell noch nicht geladen."
            )
            return

        # ── Bilddaten extrahieren ─────────────────────────────────────────────
        image_array = self._image_in.get_data()
        if image_array is None or image_array.size == 0:
            self.get_logger().warn(
                "YoloObjectDetector: Leeres Bild empfangen – überspringe Inferenz."
            )
            return

        height, width = image_array.shape[:2]

        results = self._model(image_array, verbose=False)

        corners_flat = []

        if results and len(results) > 0 and results[0].obb is not None:
            obb = results[0].obb

            if len(obb) > 0:
                all_corners = obb.xyxyxyxy.cpu().numpy()   # shape: (N, 4, 2)

                for box_corners in all_corners:
                    # ── Rand-Filter: Klotz aussortieren wenn Ecke < 5 px vom Rand
                    on_border = False
                    for (u, v) in box_corners:
                        if u < 5 or u > width - 5 or v < 5 or v > height - 5:
                            on_border = True
                            break
                    if on_border:
                        continue

                    # ── Eckpunkte in flaches Array packen (Stride 8) ──────────
                    for (u, v) in box_corners:
                        corners_flat.extend([float(u), float(v)])

        self._yolo_corners_list = corners_flat

        # ── Posen synchron 1:1 weiterleiten ──────────────────────────────────
        if not self._ist_pose_in.is_empty():
            self._ist_pose_out = sr.CartesianPose(self._ist_pose_in)

        if not self._cam_ist_pose_in.is_empty():
            self._cam_ist_pose_out = sr.CartesianPose(self._cam_ist_pose_in)

        # ── Trigger setzen (on_step_callback setzt ihn nächsten Takt zurück) ──
        self._yolo_done_trigger = True
        self.get_logger().info(
            f"YoloObjectDetector: Inferenz abgeschlossen – "
            f"{len(corners_flat) // 8} Klotz/Klötze nach Rand-Filter erkannt."
        )
