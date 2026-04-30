# BrickByBrick – AICA Custom Component Package

Steuerungspaket für einen KUKA-Roboter, der Bausteine per Pick-and-Place auf einer Linie ablegt. Gebaut auf dem AICA Modulo Framework (ROS 2 Lifecycle Nodes, visuell verbunden in AICA Studio).

## Was das System macht

1. **Exploration:** Der Roboter fährt vordefinierte Kamerapositionen ab und nimmt an jeder Station ein Bild auf.
2. **Erkennung:** YOLOv11 erkennt Bausteine im Bild, eine Linienerkennung (DLE) bestimmt die Ablagepositionen.
3. **Pick & Place:** Der Roboter greift Bausteine und legt sie nacheinander auf der erkannten Linie ab.

## Komponenten

| Block | Aufgabe |
|---|---|
| `ExplorationNavigator` | Fährt Explorationsposen ab, wechselt danach in Gateway-Modus |
| `PoseTriggeredCamera` | Friert Bild + TCP-Pose synchron ein |
| `YoloObjectDetector` | YOLOv11-Seg Inferenz → rotierte Bounding Boxes | YOD
| `DropoffLineExtractor` | Erkennt Ablagelinie im Bild, berechnet 3D-Ablageposen | DLE
| `MasterListManager` | Pinhole-Rückprojektion, Filterung, Listenverwaltung | MLM
| `JtcCommandGenerator` | Berechnet Fahrtdauer und sendet TF-Frame-Kommando an JTC |
| `PickPlaceController` | Steuert den Pick-and-Place-Ablauf in Phase 2 | PTC
| `VisionProcessor` | Fasst PTC + YOD + DLE + MLM in einem Block zusammen |

## Build

```bash
docker build -f aica-package.toml .

# nur Tests für Docker-Build
docker build -f aica-package.toml --target test .
```

## Modell & Konfiguration

- YOLO-Modell: `data/model/best.pt` (im installierten Share-Verzeichnis)
- Explorationsposen: `data/exploration/ExplCords.yaml`
- Beide Pfade sind per AICA-Parameter zur Laufzeit überschreibbar.

---

## Dokumentation

| Datei | Inhalt |
|---|---|
| `ARCHITECTURE.md` | Vollständige AICA-Framework-Referenz: Komponentenstruktur, Signal-Typen, Parameter-System, Service Clients, JTC-Integration, Build-System. Erste Anlaufstelle für alle Framework-Fragen. |
| `CONTEXT.md` | Projektspezifische Datenstrukturen: Stride-Formate der Signalarrays (Stride 7, 8, 9), Erklärung des Flattening-Prinzips, Hinweis auf `geometry_utils.py`. |
| `CLAUDE.md` | Anweisungen für KI-Assistenten Claude beim Arbeiten im Projekt: kritische Regeln zum AICA-Framework, Dateistruktur, verbotene Muster. |
| `README_AICA-Package.md` | Originale Vorlage des AICA Package Templates mit allgemeinen Hinweisen zu DevContainer, Wizard und Abhängigkeiten. |
