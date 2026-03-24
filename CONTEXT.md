# CONTEXT.md

## Projektübersicht

Dieses Repository basiert auf dem **AICA Package Template** und dient der Erstellung von Custom Components für die AICA-Robotersteuerungssoftware. Ziel ist die dynamische Steuerung eines **KUKA-Roboters** zur Erkennung und Ablage von Bausteinen auf einer Linie — ein klassischer **Pick & Place** Ablauf mit Bildverarbeitungs-Pipeline.

Ein **YOLOv11-Modell** zur Bausteinerkennung sowie Code zur Linienerkennung per Farbabgleich sind separat vorhanden und werden zu einem späteren Zeitpunkt durch den Nutzer manuell eingebracht.

---

## Systemarchitektur: Zwei Phasen

### Phase 1: Exploration
### Phase 2: Pick & Place Loop (PPL)

Die gesamte Logik ist auf **8 AICA Components (Blöcke)** verteilt:

| Block | Phase | Funktion |
|---|---|---|
| `PosFeedCam` | 1 + 2 (Gateway) | Pose-Steuerung & Gateway-Umschaltung |
| `CamSnapshot` | 1 + 2 | Bildaufnahme mit Pose-Stempel |
| `YoloExe` | 1 + 2 | YOLOv11 Inferenz |
| `LineCoordExtraction` | 1 (deaktiviert in Phase 2) | Linienerkennung & Ablagekoordinaten |
| `DataHandler` | 1 + 2 | Datenaggregation, Masterlisten-Verwaltung |
| `PickPlaceLoop` | 2 | Pick & Place Ablaufsteuerung |
| `TiefenkameraStream` | 2 | Tiefenwert-Auswertung für z_pick |

---

## Phase 1: Exploration — Detaillierte Logik

### Ziel
Den Arbeitsbereich abfahren, Bilder aufnehmen, Bausteinpositionen (YoloList) und Ablagekoordinaten auf der Linie (LineExList) erfassen. Ergebnis: `KlotzOverviewList` & `DropoffCoordsList` für Phase 2.

### Block-Beschreibungen

#### `PosFeedCam`
- **Init**: Liest `ExplCoords.yaml` → befüllt `ExplorationPoseList`; Output `takeIMG = True` (dauerhaft während Exploration)
- **Loop-Logik**:
  - `len(ExplorationPoseList) > 0`: Sendet obersten Eintrag (Frame) als `nextPose` an JTC (Joint Trajectory Controller)
  - `len(ExplorationPoseList) == 0`: **Gateway-Modus** — leitet `takeIMG` und `nextPose` von PPL-Inputs stumpf weiter; Output `Trigger_PickPlaceLoop = True` (aktiviert PPL, deaktiviert `LineCoordExtraction`)
- **Nach IMGtaken=True**: Aktuellen obersten Eintrag aus `ExplorationPoseList` poppen → nächste Pose senden
- Interne Variable verhindert Doppel-Aufnahmen / Race Conditions

#### `CamSnapshot`
- **Trigger**: Beide Inputs `takeIMG = True` AND `trajectory_success = True` → Bild speichern nach 0.3s Delay
- **Output** `IMGtaken = True` → zurück an `PosFeedCam` (im Gateway-Modus: setzt `takeIMG` auf False; während Exploration: ignoriert)
- **Greift im Aufnahmemoment**: `istPose` & `CamIstPose` von Inputs
- **Output** `CamIstPose` → an `YoloExe` & `LineCoordExtraction` (für World-Koordinaten-Berechnung)

#### `YoloExe`
- **Inputs**: Bild von `CamSnapshot`, `CamIstPose`, `istPose`
- **Verarbeitung**: YOLOv11 Inferenz
- **Outputs**: `YoloList` (Detektionen mit World-Koordinaten), `debugIMG`, `istPose` → an `DataHandler`
- **Trigger**: Nach Abschluss → Trigger an `LineCoordExtraction`

#### `LineCoordExtraction`
- **Inputs**: `CamIstPose`, Bild, `PPL_Trigger`
- **Wenn `PPL_Trigger = True`**: Block deaktiviert sich selbst (Phase 2 läuft)
- **Verarbeitung**: Linienerkennung per Farbabgleich → berechnet `dropoffCoords` aus Linienpositionen
- **Outputs**: `LineExList` (enthält `dropoffCoords`), `debugIMG` → an `DataHandler`

#### `DataHandler`
- **Inputs**: Von `YoloExe`: `istPose`, `YoloList`, `debugIMG`; Von `LineCoordExtraction`: `LineExList`, `debugIMG`
- **Verarbeitung**:
  - Speichert Debug-Bilder
  - `KlotzOverviewList`: Wenn `len(YoloList) >= 1` → `istPose` appenden
  - `DropoffCoordsList`: Wenn `len(LineExList) > 0` AND `len(LineExList) > len(DropoffCoordsList)` → `DropoffCoordsList` komplett durch `LineExList`-Einträge ersetzen
    > **Begründung**: Die längste `LineExList` repräsentiert den vollständigsten Belegungsplan — dieser soll umgesetzt werden
- **Outputs**: `KlotzOverviewList` & `DropoffCoordsList` → an `PickPlaceLoop`

---

## Phase 2: Pick & Place Loop — Detaillierte Logik

### Ausgangssituation
`PosFeedCam` läuft als Gateway. `DataHandler` hat `KlotzOverviewList` & `DropoffCoordsList` bereit. `LineCoordExtraction` ist deaktiviert.

### Loop-Bedingung
`len(DropoffCoordsList) > 0` AND `len(KlotzOverviewList) > 0`

#### `PickPlaceLoop` — Init
- Empfängt `KlotzOverviewList` & `DropoffCoordsList` von `DataHandler`
- Lädt `zHover` (Hover-Höhe)
- Erstellt interne Listen `pickList` und `placeList`

---

### Pick-Sequenz (Schritt für Schritt)

**Schritt 1 — Klotz-Übersichtspose anfahren**
- Sendet ersten `KlotzOverviewList`-Eintrag als `nextPose` via Gateway an JTC
- `takeIMG = True` → Pipeline läuft automatisch bei Ankunft
- `DataHandler` liefert neue `YoloList` zurück
- Wenn `len(YoloList) == 0`: Oberen Eintrag aus `KlotzOverviewList` poppen → zu Schritt 1 (nächste Übersichtspose)

**Schritt 2 — Grobe Pick-Pose berechnen**
- Warten auf `trajectory_success = True` (TCP hat `KlotzOverviewPos` erreicht)
- `YoloList` nach Maskengröße sortieren → Zentrum der größten Maske = Zielblock
- TCP-Pose berechnen: Kamera senkrecht über Klotzzentrum → `x_grob`, `y_grob`
- Tiefenkamera: 5×5 Patch auf Klotzzentrum auswerten → `z_pick`
- `pickList` leeren
- Neuen Eintrag in `pickList`: `(x_grob, y_grob, z_pick + zHover - KameraTCPAbstand)`

**Schritt 3 — Erste Annäherung & Feinkorrektur**
- `pickList[0]` als `nextPose` publishen; `takeIMG = True`
- Kamera senkrecht über Block → `CamSnapshot` → `DataHandler` → neue `YoloList`
- Größte Maske = Zielblock → `x_fein`, `y_fein`; `z_fein = z_pick - zSauger`
- Neuen Eintrag in `pickList` appenden: `(x_fein, y_fein, z_fein)`
- Wenn `YoloList` leer: Abbruch → zurück zu Schritt 1

**Schritt 4 — Post-Pick-Pose berechnen**
- `pickList` erweitern um: `(x_fein, y_fein, z_pick + zHover)` → `pickList[2]`

**Schritt 5 — Pick-Anfahrt**
- `pickList[1]` publishen; `takeIMG = False`
- Warten auf `trajectory_success = True` (mit Delay, um vorherigen Success zu ignorieren)
- `vacuum_on = True` setzen; 0.3s Delay (Vakuum aufbauen)

**Schritt 6 — Retract**
- `pickList[2]` publishen → Klotz 15cm anheben
- Pick abgeschlossen

---

### Place-Sequenz (Schritt für Schritt)

**Schritt 1 — Place-Posen vorbereiten**
- `placeList` leeren
- Aus oberstem `DropoffCoordsList`-Eintrag: `x_drop`, `y_drop`, `z_drop`
- `placeList[0]`: `(x_drop, y_drop, z_drop + zHover)` — Hover über Ablageposition
- `placeList[1]`: `(x_drop, y_drop, z_drop - zSauger)` — Ablageposition (Klotz wird 1–2mm fallen gelassen)
- `placeList[2]`: `(x_drop, y_drop, z_drop + zHover)` — Retract nach Ablage

**Schritt 2 — Zu placeHover fahren**
- Wechsel von `pickList` auf `placeList`; `placeList[0]` publishen

**Schritt 3 — Ablegen**
- Bei `trajectory_success`: `placeList[1]` publishen (Place)
- Bei `trajectory_success`: `vacuum_on = False`; 0.5s Delay
- `placeList[2]` publishen (Retract)
- Bei `trajectory_success`: Oberen `DropoffCoordsList`-Eintrag poppen

**Schritt 4 — Loop-Abfrage**
- `len(DropoffCoordsList) > 0` AND `len(KlotzOverviewList) > 0` → zurück zur Pick-Sequenz
- Sonst: Abschlusspose `frame3` (Homeposition, hardcodiert im Block) publishen

---

## Datenkanäle (Kommunikation zwischen Blöcken)

Die Kommunikationsstruktur zwischen den Blöcken ist zunächst nicht fest vorgegeben. Claude wählt geeignete Kanal-Typen im Sinne der AICA/ROS 2 Konventionen (z.B. Topics, Services). Folgende Datenflüsse sind definiert:

```
PosFeedCam  ──nextPose──►  JTC (Joint Trajectory Controller, extern)
PosFeedCam  ──takeIMG───►  CamSnapshot
Robot-State-Broadcaster ──istPose──► PosFeedCam, CamSnapshot, YoloExe
JTC ──trajectory_success──► PosFeedCam, CamSnapshot, PickPlaceLoop

CamSnapshot ──IMGtaken───►  PosFeedCam
CamSnapshot ──CamIstPose──► YoloExe, LineCoordExtraction
CamSnapshot ──Bild───────►  YoloExe, LineCoordExtraction

YoloExe ──YoloList, debugIMG, istPose──► DataHandler
YoloExe ──Trigger──────────────────────► LineCoordExtraction
YoloExe ──YoloList─────────────────────► PickPlaceLoop (Phase 2)

LineCoordExtraction ──LineExList, debugIMG──► DataHandler

DataHandler ──KlotzOverviewList, DropoffCoordsList──► PickPlaceLoop

PickPlaceLoop ──nextPose, takeIMG──► PosFeedCam (Gateway)
PickPlaceLoop ──vacuum_on──────────► Vakuumgreifer (extern)
PickPlaceLoop ──PPL_Trigger────────► LineCoordExtraction (deaktiviert)

TiefenkameraStream ──Tiefenbild──► PickPlaceLoop
```

---

## Technische Randbedingungen

- **YOLOv11-Modell**: Separat verfügbar, wird manuell eingebracht — Code muss entsprechende Import-Pfade und Schnittstellen vorsehen
- **Linienerkennung**: Farbabgleich-basiert, ebenfalls separat — Integration analog zu YOLOv11
- **`zSauger`**: Einheit mm; definiert wie tief die Saugglocke in den Klotz fährt
- **`zHover`**: Hover-Höhe, wird bei PPL-Init geladen
- **`KameraTCPAbstand`**: Fester Offset zwischen Kamera-Sensor und TCP
- **`frame3`**: Hardcodierte Homeposition im `PickPlaceLoop`-Block
- **`ExplCoords.yaml`**: Datei mit Explorationsposen, vom Nutzer bereitzustellen
