# System-Briefing für KI-Assistenten: AICA Modulo Framework

**Anweisung an die KI:** Dieses Dokument beschreibt die Architektur und Programmierrichtlinien für das AICA-Framework. Das System basiert unter der Haube auf ROS 2 Lifecycle Nodes, abstrahiert dieses aber vollständig durch das `modulo_components`-Framework. 

**WICHTIG:** Vergiss klassische ROS 2 Konzepte in Python (wie `rclpy.create_publisher`, `rclpy.create_subscription`, `TimerCallbacks` oder manuelle Message-Serialisierung zu JSON-Strings). Nutze ausschließlich die hier definierten AICA-Paradigmen!

---

## 1. Kernphilosophie: Komponenten & Ports
Ein AICA-Programm besteht aus Blöcken (Komponenten), die über grafische Kabel (Ports) in einer Web-UI verbunden werden.

* **Basisklasse:** Alle Python-Skripte erben von `from modulo_components.lifecycle_component import LifecycleComponent`.
* **Kein Pub/Sub:** Inputs und Outputs werden im Konstruktor (`__init__`) als Variablen registriert und an die AICA-Engine gebunden.
* **Das Publizieren:** Um Daten zu senden, wird **kein** `.publish()` aufgerufen. Die Python-Variable wird im Code einfach überschrieben. Die AICA-Engine liest die Variablen am Ende jedes Taktzyklus automatisch aus und verschickt sie über das ROS-Netzwerk.

---

## 2. Code-Struktur & Signale (I/O)
* **Inputs anlegen:** `self.add_input("_variablen_name", "ui_port_name", DataType)`
* **Outputs anlegen:** `self.add_output("_variablen_name", "ui_port_name", DataType)`
* **Datentypen:** AICA nutzt die eigene Bibliothek `state_representation` (importiert als `sr`).
    * *Posen:* `sr.CartesianPose("name", "reference_frame")`
    * *Bilder:* `sr.Image()`
    * *Listen/Arrays:* Einfache Python `list` (wird in AICA intern als `double_array` behandelt).

---

## 3. Ausführungsmodelle (Callbacks)
AICA bietet zwei Wege, wie Code ausgeführt wird. **Blockierendes `time.sleep()` ist streng verboten**, da es die Node einfriert. Zeitmessungen erfolgen über `(self.get_clock().now() - start_time).nanoseconds / 1e9`.

* **A) Zyklisch (Step-basiert):**
  Die Funktion `def on_step_callback(self):` wird automatisch mit der in der UI konfigurierten Frequenz (z.B. 50 Hz) aufgerufen. Hier laufen State Machines und kontinuierliche Berechnungen.
* **B) Event-basiert (Data-driven):**
  Inputs können mit Callbacks verknüpft werden. Der Code läuft nur, wenn neue Daten ankommen. Das spart massiv CPU-Leistung.
  *Syntax:* `self.add_input("_img", "image_in", sr.Image, user_callback=self._on_new_image)`

---

## 4. UI-Integration: Die JSON-Beschreibungen
Jede Python-Komponente benötigt zwingend eine `.json`-Datei im Ordner `component_descriptions`. Diese definiert, wie der Block in der Web-Oberfläche aussieht.

* **WICHTIG:** Das JSON-Schema akzeptiert **nur AICA-spezifische `signal_type` Strings!** Verwende niemals ROS-Typen wie `sensor_msgs/Image` oder `PoseStamped`.
* **Erlaubte Signal-Typen (Mapping):**
    * Python `bool` $\rightarrow$ `"signal_type": "bool"`
    * Python `list` $\rightarrow$ `"signal_type": "double_array"`
    * `sr.CartesianPose` $\rightarrow$ `"signal_type": "cartesian_pose"`
* **Komplexe Objekte (Bilder):**
  Bilder haben keinen Standard-Typ in der UI. Sie müssen als `"other"` deklariert werden, zwingend gefolgt vom C++-Typ:
  ```json
  "signal_type": "other",
  "custom_signal_type": "state_representation::Image"


## AICA SDK — Technische Referenz

### Paketstruktur (Component Package)

Minimale Verzeichnisstruktur für ein Paket `custom_component_package`:

```
custom_component_package/
├── component_descriptions/
│   ├── custom_component_package_cpp_component.json
│   └── custom_component_package_py_component.json
├── custom_component_package/
│   └── py_component.py          ← Python-Komponenten (Unterverzeichnis = Paketname)
├── include/custom_component_package/
│   └── CppComponent.hpp         ← C++ Header
├── src/
│   └── CppComponent.cpp         ← C++ Implementierung
├── CMakeLists.txt
├── package.xml
└── setup.cfg                    ← Nur wenn Python-Komponenten enthalten
```

**package.xml** (minimal):
```xml
<depend>modulo_components</depend>
<buildtool_depend>ament_cmake_auto</buildtool_depend>
<buildtool_depend>ament_cmake_python</buildtool_depend>
<export><build_type>ament_cmake</build_type></export>
```

**CMakeLists.txt** (minimal):
```cmake
find_package(ament_cmake_auto REQUIRED)
find_package(ament_cmake_python REQUIRED)
ament_auto_find_build_dependencies()
ament_python_install_package(${PROJECT_NAME} SCRIPTS_DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY ./component_descriptions DESTINATION .)
ament_auto_package()
```

**setup.cfg** (Python-Registrierung):
```ini
[options.entry_points]
python_components =
    custom_component_package::PyComponent = custom_component_package.py_component:PyComponent
```
> Wichtig: Klassenname muss als `paketname::KlassenName` registriert werden (doppelte `::`)

---

### Komponente implementieren (Python)

#### Vererbung

```python
from modulo_components.component import Component
# oder: from modulo_components.lifecycle_component import LifecycleComponent

class MyComponent(Component):
    def __init__(self, node_name: str, *args, **kwargs):
        super().__init__(node_name, *args, **kwargs)
        # Parameter, Signale, Callbacks hier deklarieren
```

`LifecycleComponent` zusätzliche Override-Methoden: `on_configure_callback()`, `on_activate_callback()`, `on_deactivate_callback()`, `on_cleanup_callback()`, `on_shutdown_callback()`, `on_error_callback()`

#### Parameter

```python
import state_representation as sr

# Als Klassenattribut
self._param_a = sr.Parameter("A", sr.ParameterType.INT)
self.add_parameter("_param_a", "Beschreibung")

# Inline
self.add_parameter(sr.Parameter("B", 1.0, sr.ParameterType.DOUBLE), "Beschreibung")

# Wert lesen
self._param_a.get_value()
self.get_parameter("B").get_value()
```

Leerer Parameter: `sr.Parameter("X", sr.ParameterType.INT)` → `is_empty() == True`, `get_value()` wirft `EmptyStateError`

**Validierung** (wird bei jeder Parameteränderung aufgerufen):
```python
def on_validate_parameter_callback(self, parameter: sr.Parameter) -> bool:
    if parameter.get_name() == "A":
        if parameter.is_empty():
            self.get_logger().warn("Parameter A darf nicht leer sein")
            return False  # Änderung ablehnen
    return True  # Änderung akzeptieren (Mutation des Werts vor return möglich)
```

#### Signale (Ein- und Ausgänge)

Signalnamen: `lower_snake_case`, einzigartig, darf nicht mit Zahl/Unterstrich beginnen. Standard-Topic: `~/signal_name`

**Unterstützte Nachrichtentypen:**
```python
from std_msgs.msg import Bool, Int32, Float64, Float64MultiArray, String
from modulo_core.encoded_state import EncodedState  # für state_representation Typen
```

**Eingang (Input):**
```python
from state_representation import JointPositions

self._input_positions = JointPositions()
self.add_input("positions", "_input_positions", EncodedState)

# Mit Callback:
self.add_input("number", "_input_number", Int32, user_callback=self._my_callback)
# Attribut wird VOR dem Callback aktualisiert
```

**Ausgang (Output):**
```python
from clproto import MessageType
from state_representation import CartesianPose

self._output_pose = CartesianPose()
self.add_output("pose", "_output_pose", EncodedState, MessageType.CARTESIAN_POSE_MESSAGE)

self._output_number = 3.14
self.add_output("number", "_output_number", Float64)
```
> Leere Zustände werden nicht publiziert. LifecycleComponent publiziert nur im Zustand `ACTIVE`.

#### Periodisches Verhalten

```python
def on_step_callback(self):
    # Wird mit der konfigurierten `rate` (Hz) aufgerufen
    # Wird VOR dem Publizieren der Ausgänge ausgewertet
    self._output_value = compute_something()
```

---

### Komponentenbeschreibung (JSON)

Jede Komponente braucht eine JSON-Datei in `component_descriptions/`. Dateiname-Konvention: `paketname_komponentenname.json`

**Minimale Struktur:**
```json
{
  "name": "My Component",
  "description": {
    "brief": "Einzeilige Beschreibung",
    "details": "Ausführliche Beschreibung"
  },
  "registration": "my_package::MyComponent",
  "inherits": "modulo_components::Component"
}
```

**Signale:**
```json
{
  "inputs": [{
    "display_name": "Gelenkzustand",
    "description": "Aktueller Gelenkzustand",
    "signal_name": "state",
    "signal_type": "joint_state"
  }],
  "outputs": [{
    "display_name": "Gelenkbefehl",
    "description": "Gewünschter Befehl",
    "signal_name": "command",
    "signal_type": "joint_state"
  }]
}
```

**Parameter:**
```json
{
  "parameters": [{
    "display_name": "Verstärkung",
    "description": "Skalierungsfaktor",
    "parameter_name": "gain",
    "parameter_type": "double",
    "default_value": "1.0"
  }]
}
```
`default_value: null` → Pflichtparameter (muss gesetzt werden). `default_value: ""` → gültiger leerer Zustand.
Optionale Felder: `dynamic` (laufzeit-rekonfigurierbar), `internal` (versteckt in UI)

**Prädikate:**
```json
{
  "predicates": [{
    "display_name": "Ist aktiv",
    "description": "True wenn Komponente aktiv verarbeitet",
    "predicate_name": "is_active"
  }]
}
```

**Dienste (Services):**
```json
{
  "services": [
    {
      "display_name": "Zurücksetzen",
      "description": "Interne Zustände zurücksetzen",
      "service_name": "reset"
    },
    {
      "display_name": "Frame aufzeichnen",
      "description": "Zeichnet einen TF-Frame auf",
      "service_name": "record_frame",
      "payload_format": "YAML-Dict mit 'frame' und optionalem 'reference_frame'"
    }
  ]
}
```
Ohne `payload_format` → leerer Trigger-Service. Mit `payload_format` → String-Payload-Service.

**Virtuelle Komponente** (abstrakte Basisklasse, nicht direkt instanziierbar):
```json
{ "virtual": true }
```

### Externe Abhängigkeiten (aica-package.toml)

```toml
# System-Bibliotheken
[build.packages.component.dependencies.apt]
libyaml-cpp-dev = "*"

# Python-Pakete via requirements.txt
[build.packages.component.dependencies.pip]
file = "requirements.txt"

# Python-Pakete direkt
[build.packages.component.dependencies.pip.packages]
numpy = "1.0.0"
```

---

## Entwicklungsrichtlinien für dieses Projekt

- **Nur notwendige Dateien anpassen** im Sinne der AICA-Vorlage (keine Änderungen an `.init_wizard/` nach Wizard-Ausführung)
- **Requirements pflegen**: Alle neuen Python-Abhängigkeiten in `requirements.txt` des jeweiligen Pakets eintragen; C++-Deps in `aica-package.toml`
- **Imports**: Alle neuen Imports müssen in den jeweiligen Paket-Anforderungen gepflegt werden
- **YOLOv11 & Linienerkennung**: Als optionale/externe Module behandeln — Code so strukturieren, dass diese nachträglich eingebracht werden können
- **Nicht selbstständig weiterarbeiten** ohne Nutzeranweisung — jeder Schritt wird explizit vom Nutzer vorgegeben
