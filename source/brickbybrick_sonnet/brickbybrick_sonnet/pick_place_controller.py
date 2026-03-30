"""
PickPlaceController (PPL)
─────────────────────────────────────────────────────────────────────────────
State-Machine für den vollautomatischen Pick-&-Place-Ablauf (Phase 2).

Standby solange trigger_ppl == False (Exploration läuft noch).
Beim ersten trigger_ppl == True: master_overview und master_dropoff werden
EINMALIG in lokale, veränderbare Listen kopiert.

filtered_yolo wird NIEMALS lokal kopiert – er ist ein Live-Input, der vom
MasterListManager bei jeder neuen YOLO-Erkennung überschrieben wird.
Das ist der physikalische Closed-Loop-Mechanismus:
  Klotz gegriffen → nächstes Bild zeigt einen Klotz weniger → filtered_yolo kürzer.

Synchronisation (Stafetten-Trigger):
  WAIT_IMG_1/WAIT_IMG_2 reagieren auf die steigende Flanke von mlm_done_trigger
  (NICHT img_taken oder yolo_done_trigger!). Der MLM setzt mlm_done_trigger
  erst NACH der filtered_yolo-Aktualisierung, sodass der PPL garantiert
  frische Daten liest.

Zustandsmaschine:
  INIT → CHECK_LISTS → MOVE_OVERVIEW → WAIT_IMG_1
       → MOVE_PICK_HOVER → WAIT_IMG_2 → EXECUTE_PICK
       → PREPARE_PLACE → EXECUTE_PLACE → CHECK_LISTS (Loop)
       → FINISHED (wenn Listen erschöpft – Terminal-Zustand)

Sub-Zustände (self._sub_state):
  EXECUTE_PICK:    APPROACH_PICK → VACUUM_DELAY → APPROACH_RETRACT
  EXECUTE_PLACE:   APPROACH_DROP → RELEASE_DELAY → APPROACH_RETRACT
"""

import state_representation as sr
from clproto import MessageType
from modulo_core.encoded_state import EncodedState
from modulo_components.lifecycle_component import LifecycleComponent
from std_msgs.msg import Bool, Float64MultiArray


# ── Hardware-Konstanten (TO-DO: nach KUKA-Inbetriebnahme verifizieren) ────────
_HOVER_HEIGHT_M = 0.10     # Hover-Abstand über Klotz/Ablage in Metern
_Z_SAUGER_M     = 0.05     # Länge des Saugnapf-Aufsatzes vom TCP in Metern
_Z_PICK_DEFAULT = 0.02     # Fallback Pick-Höhe bis Tiefenkamera kalibriert ist

# Hardcodierte Home-Pose (TO-DO: Koordinaten nach KUKA-Inbetriebnahme eintragen)
_HOME_POSITION    = [0.0, 0.0, 0.50]      # [X, Y, Z] in Meter
_HOME_ORIENTATION = [0.0, 0.0, 0.0, 1.0]  # [Qx, Qy, Qz, Qw] – Identitätsquaternion


class PickPlaceController(LifecycleComponent):

    def __init__(self, node_name: str, *args, **kwargs):
        super().__init__(node_name, *args, **kwargs)

        # ── Inputs ────────────────────────────────────────────────────────────
        self._trigger_ppl = False
        self.add_input("trigger_ppl", "_trigger_ppl", Bool)

        self._trajectory_success = False
        self.add_input("trajectory_success", "_trajectory_success", Bool)

        # Stafetten-Trigger vom MasterListManager: feuert erst NACHDEM
        # filtered_yolo mit den frischen Daten überschrieben wurde.
        self._mlm_done_trigger = False
        self.add_input("mlm_done_trigger", "_mlm_done_trigger", Bool)

        # Live-Input – wird vom MasterListManager laufend überschrieben
        self._filtered_yolo = []
        self.add_input("filtered_yolo", "_filtered_yolo", Float64MultiArray)

        self._master_dropoff = []
        self.add_input("master_dropoff", "_master_dropoff", Float64MultiArray)

        self._master_overview = []
        self.add_input("master_overview", "_master_overview", Float64MultiArray)

        self._depth_image = sr.Image()
        self.add_input("depth_image", "_depth_image", EncodedState)

        # ── Outputs ───────────────────────────────────────────────────────────
        self._target_pose_out = sr.CartesianPose("target_pose_out", "world")
        self.add_output(
            "target_pose_out", "_target_pose_out",
            EncodedState, MessageType.CARTESIAN_POSE_MESSAGE,
        )

        self._take_img_out = False
        self.add_output("take_img_out", "_take_img_out", Bool)

        self._vacuum_on = False
        self.add_output("vacuum_on", "_vacuum_on", Bool)

        # ── State-Machine-Zustand ─────────────────────────────────────────────
        self._state: str = "INIT"
        self._sub_state: str = ""      # Unter-Zustand für EXECUTE_PICK / _PLACE

        # ── Lokale Listen (einmalig aus Inputs kopiert beim PPL-Start) ────────
        self._master_overview_local: list = []
        self._master_dropoff_local: list  = []
        self._init_done: bool = False  # Sicherheits-Flag: Kopieren nur 1× erlaubt

        # ── Flanken-Erkennung (over-cycle edge detection) ─────────────────────
        self._prev_trajectory_success: bool = False
        self._prev_mlm_done_trigger: bool = False

        # ── Zustandsvariablen (persistent über Taktzyklen) ───────────────────
        self._z_pick: float = _Z_PICK_DEFAULT
        # _current_brick: ausschließlich 9-Elemente-Format
        # [X, Y, Area, u_center, v_center, Qx, Qy, Qz, Qw]
        self._current_brick: list = []
        # _retract_pose_local: separater Speicher für Retract-Koordinaten (Fix M-1)
        # [X, Y, Z_retract, Qx, Qy, Qz, Qw]
        self._retract_pose_local: list = []
        self._timer_start = None       # Zeitstempel für nicht-blockierende Delays

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle-Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def on_validate_parameter_callback(self, parameter: sr.Parameter) -> bool:
        return True

    def on_configure_callback(self) -> bool:
        # Fix M-4: ALLE persistenten Variablen vollständig zurücksetzen
        self._state = "INIT"
        self._sub_state = ""
        self._init_done = False
        self._vacuum_on = False
        self._take_img_out = False
        self._timer_start = None
        self._prev_trajectory_success = False
        self._prev_mlm_done_trigger = False
        self._master_overview_local = []
        self._master_dropoff_local = []
        self._z_pick = _Z_PICK_DEFAULT
        self._current_brick = []
        self._retract_pose_local = []
        self.get_logger().info("PickPlaceController: Konfiguriert – Zustand INIT.")
        return True

    def on_activate_callback(self) -> bool:
        self.get_logger().info(
            "PickPlaceController: Aktiviert – Standby bis trigger_ppl == True."
        )
        return True

    def on_deactivate_callback(self) -> bool:
        self._vacuum_on = False  # Sicherheit: Vakuum beim Deaktivieren abschalten
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Haupt-Taktschleife
    # ─────────────────────────────────────────────────────────────────────────

    def on_step_callback(self):
        # ── Flanken-Erkennung (einmal pro Takt berechnen) ─────────────────────
        traj_rising = self._trajectory_success and not self._prev_trajectory_success
        mlm_rising = self._mlm_done_trigger and not self._prev_mlm_done_trigger
        self._prev_trajectory_success = self._trajectory_success
        self._prev_mlm_done_trigger = self._mlm_done_trigger

        # ── Zustandsmaschine ──────────────────────────────────────────────────
        if self._state == "INIT":
            self._run_init()

        elif self._state == "CHECK_LISTS":
            self._run_check_lists()

        elif self._state == "MOVE_OVERVIEW":
            self._run_move_overview(traj_rising)

        elif self._state == "WAIT_IMG_1":
            self._run_wait_img_1(mlm_rising)

        elif self._state == "MOVE_PICK_HOVER":
            self._run_move_pick_hover(traj_rising)

        elif self._state == "WAIT_IMG_2":
            self._run_wait_img_2(mlm_rising)

        elif self._state == "EXECUTE_PICK":
            self._run_execute_pick(traj_rising)

        elif self._state == "PREPARE_PLACE":
            self._run_prepare_place(traj_rising)

        elif self._state == "EXECUTE_PLACE":
            self._run_execute_place(traj_rising)

        elif self._state == "FINISHED":
            pass  # Terminal-Zustand – keine weitere Aktion

    # ─────────────────────────────────────────────────────────────────────────
    # Hilfsmethode: Zustandswechsel mit Logging + Flanken-Synchronisation
    # ─────────────────────────────────────────────────────────────────────────

    def _transition(self, new_state: str, new_sub: str = ""):
        """
        Wechselt den Zustand und synchronisiert die Flanken-Erkennung.
        Die Synchronisation verhindert, dass ein bereits anstehendes True-Signal
        sofort im neuen Zustand feuert.
        """
        self.get_logger().info(
            f"PickPlaceController: ZUSTANDSWECHSEL {self._state} → {new_state}"
            + (f" [{new_sub}]" if new_sub else "")
        )
        self._state = new_state
        self._sub_state = new_sub
        # Flanken auf aktuellen Wert synchronisieren
        self._prev_trajectory_success = self._trajectory_success
        self._prev_mlm_done_trigger = self._mlm_done_trigger

    # ─────────────────────────────────────────────────────────────────────────
    # Zustand: INIT
    # ─────────────────────────────────────────────────────────────────────────

    def _run_init(self):
        self._vacuum_on = False

        if not self._trigger_ppl:
            return  # Standby während Phase 1

        # trigger_ppl = True: Listen EINMALIG kopieren und starten
        if not self._init_done:
            self._master_overview_local = list(self._master_overview)
            self._master_dropoff_local  = list(self._master_dropoff)
            self._init_done = True
            self.get_logger().info(
                f"PickPlaceController: Initialisierung – "
                f"{len(self._master_overview_local) // 7} Übersichtsposen, "
                f"{len(self._master_dropoff_local) // 7} Ablagepositionen kopiert."
            )
        self._transition("CHECK_LISTS")

    # ─────────────────────────────────────────────────────────────────────────
    # Zustand: CHECK_LISTS
    # ─────────────────────────────────────────────────────────────────────────

    def _run_check_lists(self):
        overview_ok = len(self._master_overview_local) >= 7
        dropoff_ok  = len(self._master_dropoff_local)  >= 7

        if overview_ok and dropoff_ok:
            self._transition("MOVE_OVERVIEW")
        else:
            # Fix M-3: Wechsel in Terminal-Zustand statt Log-Spam bei 50 Hz
            # ══════════════════════════════════════════════════════════════════
            # TO-DO: Home-Pose – Koordinaten nach KUKA-Inbetriebnahme eintragen
            # Aktuelle Werte in _HOME_POSITION / _HOME_ORIENTATION sind Platzhalter!
            # ══════════════════════════════════════════════════════════════════
            self._target_pose_out.set_position(_HOME_POSITION)
            self._target_pose_out.set_orientation(_HOME_ORIENTATION)
            reason = "Keine Übersichtsposen" if not overview_ok else "Keine Ablagepositionen"
            self.get_logger().warn(
                f"PickPlaceController: CHECK_LISTS – {reason} verfügbar. "
                f"HOME-POSE gesendet. Wechsle zu FINISHED."
            )
            self._transition("FINISHED")

    # ─────────────────────────────────────────────────────────────────────────
    # Zustand: MOVE_OVERVIEW
    # ─────────────────────────────────────────────────────────────────────────

    def _run_move_overview(self, traj_rising: bool):
        # Sende konstant die erste Übersichtspose
        pos = self._master_overview_local[0:3]
        ori = self._master_overview_local[3:7]
        self._target_pose_out.set_position(pos)
        self._target_pose_out.set_orientation(ori)

        if traj_rising:
            self._take_img_out = True
            self._transition("WAIT_IMG_1")

    # ─────────────────────────────────────────────────────────────────────────
    # Zustand: WAIT_IMG_1 (Grob-Auswertung + Tiefen-Berechnung)
    #
    # Fix H-1: Reagiert auf yolo_done_trigger statt img_taken.
    # Damit ist garantiert, dass filtered_yolo bereits frisch ist.
    # ─────────────────────────────────────────────────────────────────────────

    def _run_wait_img_1(self, mlm_rising: bool):
        if not mlm_rising:
            return

        # yolo_done_trigger gefeuert → filtered_yolo ist frisch
        self._take_img_out = False

        # ── filtered_yolo auswerten (Live-Input vom MasterListManager) ────────
        if len(self._filtered_yolo) == 0:
            # Keine Klötze an dieser Position → nächste Übersichtspose
            self._master_overview_local = self._master_overview_local[7:]
            self.get_logger().info(
                "PickPlaceController: WAIT_IMG_1 – keine Klötze. "
                f"Übersichtspose gepopped ({len(self._master_overview_local) // 7} verbleiben)."
            )
            self._transition("CHECK_LISTS")
            return

        # ── Klotz mit größter Fläche wählen (Stride 9: X,Y,Area,u_c,v_c,Qx,Qy,Qz,Qw)
        best_brick = None
        best_area  = -1.0
        filt = self._filtered_yolo
        for i in range(0, len(filt), 9):
            if i + 9 > len(filt):
                break
            area = float(filt[i + 2])
            if area > best_area:
                best_area  = area
                best_brick = [float(v) for v in filt[i:i + 9]]

        if best_brick is None:
            self._transition("CHECK_LISTS")
            return

        # Fix M-1: _current_brick ausschließlich im 9-Elemente-Format speichern
        self._current_brick = best_brick
        u_c  = best_brick[3]
        v_c  = best_brick[4]
        X_b  = best_brick[0]
        Y_b  = best_brick[1]
        quat = best_brick[5:9]  # [Qx, Qy, Qz, Qw]

        # ── Tiefenbild auswerten → Z_pick ────────────────────────────────────
        # ══════════════════════════════════════════════════════════════════════
        # TO-DO: Z_pick aus Tiefenbild berechnen
        #
        # Algorithmus:
        #   depth_array = self._depth_image.get_data()   # numpy array (H×W)
        #   u, v = int(round(u_c)), int(round(v_c))
        #   patch = depth_array[max(0,v-2):v+3, max(0,u-2):u+3]
        #   depth_m = float(np.median(patch))
        #   self._z_pick = depth_to_world_z(u_c, v_c, depth_m, fx, fy, cx, cy, cam_pose)
        #
        # Benötigt: Kamerakalibrierung (fx, fy, cx, cy) und Kamerapose im
        #           Weltframe. Bis dahin gilt der Fallback-Wert _Z_PICK_DEFAULT.
        # ══════════════════════════════════════════════════════════════════════
        self._z_pick = _Z_PICK_DEFAULT
        self.get_logger().warn(
            "PickPlaceController: WAIT_IMG_1 – Z_pick-Berechnung (TO-DO). "
            f"Verwende Fallback: {self._z_pick} m."
        )

        # ── Kamera-TCP-Offset: X/Y korrigieren ───────────────────────────────
        # ══════════════════════════════════════════════════════════════════════
        # TO-DO: Kamera-TCP-Versatz aufaddieren, damit der SAUGER (nicht
        #        die Kamera) über dem Klotz zentriert ist.
        #
        # Voraussichtlicher Code:
        #   from brickbybrick_sonnet.geometry_utils import camera_tcp_offset
        #   X_hover, Y_hover = camera_tcp_offset(X_b, Y_b, cam_pose, tcp_pose)
        #
        # Benötigt: Verifizierter Kamera-Montageoffset aus KUKA-URDF.
        # ══════════════════════════════════════════════════════════════════════
        X_hover = X_b
        Y_hover = Y_b

        # Hover-Pose setzen und weiterfahren
        hover_z = self._z_pick + _HOVER_HEIGHT_M
        self._target_pose_out.set_position([X_hover, Y_hover, hover_z])
        self._target_pose_out.set_orientation(quat)

        self.get_logger().info(
            f"PickPlaceController: WAIT_IMG_1 – Klotz gewählt "
            f"(Area={best_area:.1f}px², X={X_b:.3f}, Y={Y_b:.3f}). → MOVE_PICK_HOVER"
        )
        self._transition("MOVE_PICK_HOVER")

    # ─────────────────────────────────────────────────────────────────────────
    # Zustand: MOVE_PICK_HOVER
    # ─────────────────────────────────────────────────────────────────────────

    def _run_move_pick_hover(self, traj_rising: bool):
        # Hover-Pose konstant senden (target_pose_out wurde in WAIT_IMG_1 gesetzt)
        if traj_rising:
            self._take_img_out = True
            self._transition("WAIT_IMG_2")

    # ─────────────────────────────────────────────────────────────────────────
    # Zustand: WAIT_IMG_2 (Fein-Auswertung vom zentrierten Hover-Bild)
    #
    # Fix H-1: Reagiert auf yolo_done_trigger statt img_taken.
    # Fix M-5: Z_pick wird aus dem frischen Tiefenbild neu berechnet.
    # ─────────────────────────────────────────────────────────────────────────

    def _run_wait_img_2(self, mlm_rising: bool):
        if not mlm_rising:
            return

        # yolo_done_trigger gefeuert → filtered_yolo ist frisch
        self._take_img_out = False

        # ── Frisches filtered_yolo vom nahen Hover-Bild auswerten ────────────
        best_brick = None
        best_area  = -1.0
        filt = self._filtered_yolo
        for i in range(0, len(filt), 9):
            if i + 9 > len(filt):
                break
            area = float(filt[i + 2])
            if area > best_area:
                best_area  = area
                best_brick = [float(v) for v in filt[i:i + 9]]

        if best_brick is None:
            self.get_logger().warn(
                "PickPlaceController: WAIT_IMG_2 – kein Klotz im Hover-Bild. "
                "Zurück zu CHECK_LISTS."
            )
            self._transition("CHECK_LISTS")
            return

        X_fein = best_brick[0]
        Y_fein = best_brick[1]
        u_c    = best_brick[3]
        v_c    = best_brick[4]
        quat   = best_brick[5:9]  # [Qx, Qy, Qz, Qw]

        # ── Fix M-5: Z_pick aus frischem Tiefenbild aktualisieren ─────────────
        # ══════════════════════════════════════════════════════════════════════
        # TO-DO: Z_pick aus dem nahen Hover-Tiefenbild neu berechnen
        #
        # Algorithmus (analog zu WAIT_IMG_1):
        #   depth_array = self._depth_image.get_data()
        #   u, v = int(round(u_c)), int(round(v_c))
        #   patch = depth_array[max(0,v-2):v+3, max(0,u-2):u+3]
        #   depth_m = float(np.median(patch))
        #   self._z_pick = depth_to_world_z(u_c, v_c, depth_m, fx, fy, cx, cy, cam_pose)
        #
        # Hinweis: Das Hover-Bild ist deutlich näher als das Overview-Bild und
        #          liefert eine präzisere Tiefenmessung. Bis die Kalibrierung
        #          steht, verwenden wir den Fallback (ggf. überschrieben aus WAIT_IMG_1).
        # ══════════════════════════════════════════════════════════════════════
        self.get_logger().warn(
            "PickPlaceController: WAIT_IMG_2 – Z_pick-Erneuerung (TO-DO). "
            f"Aktueller Wert: {self._z_pick} m."
        )

        # ── Pick-Pose und Retract-Pose generieren ─────────────────────────────
        z_pick    = self._z_pick
        z_drop    = z_pick - _Z_SAUGER_M
        z_retract = z_pick + _HOVER_HEIGHT_M

        # Pick-Pose sofort als target setzen
        self._target_pose_out.set_position([X_fein, Y_fein, z_drop])
        self._target_pose_out.set_orientation(quat)

        # Fix M-1: _current_brick bleibt im 9er-Format.
        # Retract-Koordinaten in separater Variable speichern.
        self._current_brick = best_brick
        self._retract_pose_local = [X_fein, Y_fein, z_retract] + list(quat)

        self.get_logger().info(
            f"PickPlaceController: WAIT_IMG_2 – Fein-Pose "
            f"(X={X_fein:.3f}, Y={Y_fein:.3f}, Z_drop={z_drop:.3f}). → EXECUTE_PICK"
        )
        self._transition("EXECUTE_PICK", "APPROACH_PICK")

    # ─────────────────────────────────────────────────────────────────────────
    # Zustand: EXECUTE_PICK (mit Sub-Zuständen)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_execute_pick(self, traj_rising: bool):
        if self._sub_state == "APPROACH_PICK":
            # Pick-Pose wird bereits gesendet (gesetzt in WAIT_IMG_2)
            if traj_rising:
                self._vacuum_on = True
                self._timer_start = self.get_clock().now()
                self._sub_state = "VACUUM_DELAY"
                self.get_logger().info(
                    "PickPlaceController: EXECUTE_PICK – Ankunft an Pick-Pose. "
                    "Vakuum EIN, starte 0,3s Delay."
                )

        elif self._sub_state == "VACUUM_DELAY":
            elapsed = (self.get_clock().now() - self._timer_start).nanoseconds / 1e9
            if elapsed >= 0.3:
                # Fix M-1: Retract-Pose aus separater Variable lesen
                rp = self._retract_pose_local
                self._target_pose_out.set_position([rp[0], rp[1], rp[2]])
                self._target_pose_out.set_orientation(rp[3:7])
                self._sub_state = "APPROACH_RETRACT"
                self.get_logger().info(
                    "PickPlaceController: EXECUTE_PICK – Delay abgelaufen. "
                    "Sende Retract-Pose."
                )

        elif self._sub_state == "APPROACH_RETRACT":
            if traj_rising:
                self._transition("PREPARE_PLACE")

    # ─────────────────────────────────────────────────────────────────────────
    # Zustand: PREPARE_PLACE
    # ─────────────────────────────────────────────────────────────────────────

    def _run_prepare_place(self, traj_rising: bool):
        # Erste Ablageposition aus der lokalen Liste lesen (ohne zu poppen)
        pos  = self._master_dropoff_local[0:3]
        ori  = self._master_dropoff_local[3:7]
        X_d, Y_d, Z_d = float(pos[0]), float(pos[1]), float(pos[2])
        hover_z = Z_d + _HOVER_HEIGHT_M

        self._target_pose_out.set_position([X_d, Y_d, hover_z])
        self._target_pose_out.set_orientation(ori)

        if traj_rising:
            self._transition("EXECUTE_PLACE", "APPROACH_DROP")

    # ─────────────────────────────────────────────────────────────────────────
    # Zustand: EXECUTE_PLACE (mit Sub-Zuständen)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_execute_place(self, traj_rising: bool):
        pos = self._master_dropoff_local[0:3]
        ori = self._master_dropoff_local[3:7]
        X_d, Y_d, Z_d = float(pos[0]), float(pos[1]), float(pos[2])

        if self._sub_state == "APPROACH_DROP":
            # Drop-Pose (direkte Ablage-Höhe)
            self._target_pose_out.set_position([X_d, Y_d, Z_d])
            self._target_pose_out.set_orientation(ori)
            if traj_rising:
                self._vacuum_on = False
                self._timer_start = self.get_clock().now()
                self._sub_state = "RELEASE_DELAY"
                self.get_logger().info(
                    "PickPlaceController: EXECUTE_PLACE – Ablage-Pose erreicht. "
                    "Vakuum AUS, starte 0,5s Release-Delay."
                )

        elif self._sub_state == "RELEASE_DELAY":
            elapsed = (self.get_clock().now() - self._timer_start).nanoseconds / 1e9
            if elapsed >= 0.5:
                # Retract nach Ablage
                hover_z = Z_d + _HOVER_HEIGHT_M
                self._target_pose_out.set_position([X_d, Y_d, hover_z])
                self._target_pose_out.set_orientation(ori)
                self._sub_state = "APPROACH_RETRACT"
                self.get_logger().info(
                    "PickPlaceController: EXECUTE_PLACE – Release-Delay abgelaufen. "
                    "Sende Retract-Pose."
                )

        elif self._sub_state == "APPROACH_RETRACT":
            if traj_rising:
                # Ablageposition verbraucht – aus lokaler Liste entfernen
                self._master_dropoff_local = self._master_dropoff_local[7:]
                self.get_logger().info(
                    f"PickPlaceController: EXECUTE_PLACE – Klotz abgelegt. "
                    f"Dropoff-Liste: {len(self._master_dropoff_local) // 7} verbleiben."
                )
                self._transition("CHECK_LISTS")
