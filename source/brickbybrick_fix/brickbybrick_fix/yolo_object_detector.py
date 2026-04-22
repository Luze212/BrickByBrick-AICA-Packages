"""
YoloObjectDetector
─────────────────────────────────────────────────────────────────────────────
Event-getriebener Block. Wartet passiv auf neue Bilder von PoseTriggeredCamera,
jagt sie durch das YOLOv11-Seg Modell und gibt die 2D-Eckpunkte aller
erkannten Klötze als flaches Array weiter.

Modell-Typ: yolo11m-seg (Instanz-Segmentierung, kein OBB).
  Aus jeder Segmentierungsmaske wird via cv2.minAreaRect eine rotierte
  Bounding Box berechnet. Das Ausgabeformat bleibt Stride-8 (4 Ecken × 2
  Koordinaten), identisch zum OBB-Format für den Rest der Pipeline.

Threading-Architektur (wichtig!):
  user_callback (_on_new_image) läuft im ROS-Subscriber-Thread.
  PyTorch ist NICHT thread-safe über Thread-Grenzen hinweg – der Dispatch-Stack
  wird thread-lokal initialisiert. Deshalb darf die YOLO-Inferenz NICHT im
  Subscriber-Callback aufgerufen werden.

Ablauf:
  1. _on_new_image (Subscriber-Thread): Bild + Posen kopieren, Flag setzen
  2. on_step_callback (Haupt-Thread):   YOLO-Inferenz ausführen, Outputs schreiben

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
import cv2
import state_representation as sr
from clproto import MessageType
from modulo_core.encoded_state import EncodedState
from modulo_components.lifecycle_component import LifecycleComponent
from sensor_msgs.msg import Image as RosImage
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
        self._image_in = RosImage()
        self.add_input(
            "image_in", "_image_in", RosImage,
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

        # Trigger wird einen Step VERZÖGERT gesetzt, damit yolo_corners_list
        # garantiert vor dem Trigger beim MLM ankommt (ROS hat keine Topic-
        # übergreifende Empfangsreihenfolge → sonst Race im MLM-user_callback).
        self._trigger_pending: bool = False
        self._reset_trigger_next_step: bool = False

        # ── Internes Modell-Handle (persistent, über Taktzyklen hinweg) ───────
        self._model = None      # wird in on_configure_callback geladen

        # ── Bild-Puffer: vom Subscriber-Thread befüllt, von on_step_callback gelesen ──
        # Kein Mutex nötig: AICA serialisiert Callbacks und Step innerhalb des Nodes.
        self._pending_image_array = None   # np.ndarray oder None
        self._pending_image_width: int = 0
        self._pending_image_height: int = 0
        self._pending_ist_pose = None      # sr.CartesianPose-Klon oder None
        self._pending_cam_pose = None      # sr.CartesianPose-Klon oder None
        self._new_image_pending: bool = False
        self._last_image_stamp: tuple = (-1, -1)   # (sec, nanosec) des zuletzt verarbeiteten Snapshots

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
        self._trigger_pending = False
        self._reset_trigger_next_step = False
        self._new_image_pending = False
        self._pending_image_array = None
        self._pending_ist_pose = None
        self._pending_cam_pose = None
        self._last_image_stamp = (-1, -1)
        return True

    def on_activate_callback(self) -> bool:
        self.get_logger().info(
            "YoloObjectDetector: Aktiviert – wartet auf erstes Bild-Event."
        )
        return True

    def on_deactivate_callback(self) -> bool:
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Taktschleife – YOLO-Inferenz im Haupt-Thread (PyTorch-Thread-Sicherheit)
    # ─────────────────────────────────────────────────────────────────────────

    def on_step_callback(self):
        # ── YOLO-Inferenz: nur wenn ein neues Bild vorliegt ──────────────────
        # _run_yolo_inference setzt _trigger_pending = True, NICHT
        # _yolo_done_trigger direkt. Der Trigger wird einen Step später
        # gesetzt, damit corners garantiert vorher beim MLM ankommen.
        if self._new_image_pending:
            self._new_image_pending = False
            self._run_yolo_inference()

        # ── Verzögerter Trigger-Reset ─────────────────────────────────────────
        # Erst im übernächsten Takt zurücksetzen, damit AICA beim Publish noch True sieht.
        if self._reset_trigger_next_step:
            self._yolo_done_trigger = False
            self._reset_trigger_next_step = False
        elif self._yolo_done_trigger:
            self._reset_trigger_next_step = True
        elif self._trigger_pending:
            # Vorheriger Step hat corners publiziert → jetzt Trigger feuern.
            self._yolo_done_trigger = True
            self._trigger_pending = False

    def _run_yolo_inference(self):
        """
        YOLO-Inferenz im Haupt-Component-Thread (aufgerufen von on_step_callback).
        PyTorchs Dispatch-Stack ist hier korrekt initialisiert.
        """
        image_array = self._pending_image_array
        height = self._pending_image_height
        width = self._pending_image_width

        # ── BGR-Konvertierung: ROS/Simulator publiziert rgb8, YOLO erwartet BGR ──
        # Ultralytics wendet intern BGR→RGB an. Ohne diese Konvertierung sieht
        # das Modell vertauschte R↔B Kanäle und erkennt nichts.
        image_bgr = image_array[:, :, ::-1]

        results = self._model(image_bgr, verbose=False, device='cpu')

        corners_flat = []

        # ── Segmentierungsmasken auswerten ───────────────────────────────────
        # Modell ist yolo11m-seg (Segmentierung, kein OBB).
        # Aus jeder Maske wird via cv2.minAreaRect eine rotierte Bounding Box
        # abgeleitet – das Ausgabeformat (Stride 8: 4 Ecken × 2 Koordinaten)
        # bleibt identisch mit dem OBB-Format für den Rest der Pipeline.
        if results and len(results) > 0 and results[0].masks is not None:
            masks_xy = results[0].masks.xy   # Liste von np.arrays (N_pts, 2) in Bildkoordinaten
            for mask_xy in masks_xy:
                if len(mask_xy) < 5:
                    continue

                pts = mask_xy.astype(np.float32).reshape(-1, 1, 2)
                rect = cv2.minAreaRect(pts)           # ((cx,cy), (w,h), angle)
                box_corners = cv2.boxPoints(rect)     # (4, 2) – rotierte Ecken

                # ── Rand-Filter: Klotz aussortieren wenn Ecke < 5 px vom Rand
                on_border = False
                for (u, v) in box_corners:
                    if u < 5 or u > width - 5 or v < 5 or v > height - 5:
                        on_border = True
                        break
                if on_border:
                    continue

                # ── Eckpunkte in flaches Array packen (Stride 8) ──────────────
                for (u, v) in box_corners:
                    corners_flat.extend([float(u), float(v)])

        self._yolo_corners_list = corners_flat

        # ── Posen aus dem synchron gespeicherten Snapshot weiterleiten ───────
        if self._pending_ist_pose is not None:
            self._ist_pose_out = self._pending_ist_pose
        if self._pending_cam_pose is not None:
            self._cam_ist_pose_out = self._pending_cam_pose

        # ── Trigger erst NÄCHSTEN Step setzen – siehe on_step_callback ───────
        # Grund: corners müssen erst publiziert sein, bevor der Trigger den
        # MLM-user_callback feuert (sonst Race: trigger vor corners).
        self._trigger_pending = True
        self.get_logger().info(
            f"YoloObjectDetector: Inferenz abgeschlossen – "
            f"{len(corners_flat) // 8} Klotz/Klötze nach Rand-Filter erkannt."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Event-Callback: neues (eingefrorenes) Bild von PoseTriggeredCamera
    # Läuft im ROS-Subscriber-Thread → NUR Datenkopie, KEIN PyTorch!
    # ─────────────────────────────────────────────────────────────────────────

    def _on_new_image(self):
        if self._model is None:
            self.get_logger().warn(
                "YoloObjectDetector: Bild empfangen, aber Modell noch nicht geladen."
            )
            return

        msg = self._image_in
        if not msg.data:
            self.get_logger().warn(
                "YoloObjectDetector: Leeres Bild empfangen – überspringe Inferenz."
            )
            return

        # ── Duplikat-Filter: PoseTriggeredCamera publiziert denselben Snapshot
        # jeden Takt erneut. Der ROS-Timestamp identifiziert eindeutig ob ein
        # neuer Snapshot vorliegt – nur dann YOLO auslösen.
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == self._last_image_stamp:
            return
        self._last_image_stamp = stamp

        # ── Bild als eigene Kopie sichern (Buffer kann nach Callback ungültig werden) ──
        self._pending_image_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, -1)
        ).copy()
        self._pending_image_height = msg.height
        self._pending_image_width = msg.width

        # ── Posen synchron zum Bild einfrieren (PoseTriggeredCamera-Snapshot) ──
        self._pending_ist_pose = (
            sr.CartesianPose(self._ist_pose_in)
            if not self._ist_pose_in.is_empty() else None
        )
        self._pending_cam_pose = (
            sr.CartesianPose(self._cam_ist_pose_in)
            if not self._cam_ist_pose_in.is_empty() else None
        )

        # ── Flag für on_step_callback setzen ─────────────────────────────────
        self._new_image_pending = True
