# CONTEXT.md

## Projektübersicht

Dieses Repository basiert auf dem **AICA Package Template** und dient der Erstellung von Custom Components für die AICA-Robotersteuerungssoftware. Ziel ist die dynamische Steuerung eines **KUKA-Roboters** zur Erkennung und Ablage von Bausteinen auf einer Linie — ein klassischer **Pick & Place** Ablauf mit Bildverarbeitungs-Pipeline.

Das zu verwendende YOLOv11 Model liegt im Ordner source/sonnet_small/model
Die Datein ExplCords.yaml für die Exploration liegt im Ordner source/sonnet_small/exploration

Die Pfade zu ExplCords.yaml und dem YOLOv11-Modell sollen in den jeweiligen Python-Komponenten zwingend als AICA-Parameter (sr.Parameter("_model_path", "source/sonnet_small/...", sr.ParameterType.STRING)) angelegt werden. Sie dürfen als Default-Werte im Code stehen, müssen aber über die UI rekonfigurierbar bleiben.


## Projektspezifische Datenstrukturen (Flattening & Strides)

Da AICA komplexe Listen (wie Listen von Posen oder Objekten) nur als flache double_arrays übertragen kann, nutzen wir im gesamten Projekt feste "Strides" (Schrittweiten), um Daten in flache Python-Listen zu packen und wieder zu entpacken:

    2D YOLO Eckpunkte (Stride 8): [u1, v1, u2, v2, u3, v3, u4, v4, ...]

    3D Basis-Posen (Stride 7): [X, Y, Z, Qx, Qy, Qz, Qw, ...] (Verwendet für master_dropoff, line_ex_list und master_overview)

    3D Klotz-Daten inkl. Pixel (Stride 9): [X, Y, Area, u_center, v_center, Qx, Qy, Qz, Qw, ...] (Spezialformat für filtered_yolo)

    Shared Utils: Reine mathematische Logik wird in geometry_utils.py ausgelagert. Diese Datei ist keine AICA-Komponente und wird nicht registriert.