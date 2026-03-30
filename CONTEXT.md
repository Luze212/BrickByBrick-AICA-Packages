# CONTEXT.md

## Projektübersicht

Dieses Repository basiert auf dem **AICA Package Template** und dient der Erstellung von Custom Components für die AICA-Robotersteuerungssoftware. Ziel ist die dynamische Steuerung eines **KUKA-Roboters** zur Erkennung und Ablage von Bausteinen auf einer Linie — ein klassischer **Pick & Place** Ablauf mit Bildverarbeitungs-Pipeline.

Das zu verwendende YOLOv11 Model liegt im Ordner source/sonnet_small/model
Die Datein ExplCords.yaml für die Exploration liegt im Ordner source/sonnet_small/exploration

Die Pfade zu ExplCords.yaml und dem YOLOv11-Modell sollen in den jeweiligen Python-Komponenten zwingend als AICA-Parameter (sr.Parameter("_model_path", "source/sonnet_small/...", sr.ParameterType.STRING)) angelegt werden. Sie dürfen als Default-Werte im Code stehen, müssen aber über die UI rekonfigurierbar bleiben.