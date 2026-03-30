# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

Custom AICA component package for controlling a **KUKA robot** in a pick-and-place workflow: YOLOv11 detects bricks, the robot picks them and places them on a line. This is built on top of the AICA Modulo framework, which abstracts ROS 2 Lifecycle Nodes into a visual block-based programming model.

- YOLOv11 model: `source/sonnet_small/model/best.pt`
- Exploration config: `source/sonnet_small/exploration/ExplCords.yaml`
- Both paths **must** be exposed as reconfigurable `sr.Parameter` in any component that uses them

## Build & Test Commands

```bash
# Build the package
docker build -f aica-package.toml .

# Run tests only
docker build -f aica-package.toml --target test .
```

Tests use pytest with ROS context fixtures (see `test/python_tests/conftest.py`).

## AICA Modulo Framework — Critical Rules

**Forget standard ROS 2 Python patterns** (`rclpy.create_publisher`, `create_subscription`, `TimerCallback`, manual JSON serialization). Use only AICA paradigms.

### Component Inheritance

```python
from modulo_components.lifecycle_component import LifecycleComponent
from modulo_components.component import Component
```

`LifecycleComponent` adds: `on_configure_callback()`, `on_activate_callback()`, `on_deactivate_callback()`, `on_cleanup_callback()`, `on_shutdown_callback()`, `on_error_callback()`

### Inputs / Outputs

Register in `__init__`, never use `.publish()`. The AICA engine reads output variables automatically at the end of each cycle.

```python
import state_representation as sr
from modulo_core.encoded_state import EncodedState
from std_msgs.msg import Bool, Int32, Float64, Float64MultiArray

# Input
self._input_pose = sr.CartesianPose("name", "frame")
self.add_input("pose_in", "_input_pose", EncodedState)

# Input with event callback (data-driven, saves CPU)
self.add_input("image_in", "_img", sr.Image, user_callback=self._on_new_image)

# Output — just assign the variable, engine publishes it
self._output_pose = sr.CartesianPose("name", "frame")
self.add_output("pose_out", "_output_pose", EncodedState, MessageType.CARTESIAN_POSE_MESSAGE)
```

Empty states are not published. `LifecycleComponent` only publishes in `ACTIVE` state.

### Parameters

```python
# Declare
self._model_path = sr.Parameter("model_path", "source/sonnet_small/model/best.pt", sr.ParameterType.STRING)
self.add_parameter("_model_path", "Path to YOLOv11 model")

# Read
self._model_path.get_value()

# Validate on every change
def on_validate_parameter_callback(self, parameter: sr.Parameter) -> bool:
    return True  # return False to reject
```

### Execution

- `on_step_callback(self)` — called at configured rate (e.g. 50 Hz), runs before outputs are published
- **Never use `time.sleep()`** — freezes the node. Use: `(self.get_clock().now() - start_time).nanoseconds / 1e9`

## Component Description JSON

Every Python component requires a JSON file in `component_descriptions/`. Filename: `packagename_componentname.json`.

**Use AICA signal types only — never ROS types like `sensor_msgs/Image`:**

| Python type | `signal_type` |
|---|---|
| `bool` | `"bool"` |
| `list` / double array | `"double_array"` |
| `sr.CartesianPose` | `"cartesian_pose"` |
| `sr.JointState` | `"joint_state"` |
| `sr.Image` | `"other"` + `"custom_signal_type": "state_representation::Image"` |

## File Structure Rules (Ament Build System — Strict)

- `CMakeLists.txt`, `package.xml`, `setup.cfg`, `requirements.txt` **must** be at the package root (`source/brickbybrick_sonnet/`), never in subdirectories
- Python components live in `source/brickbybrick_sonnet/brickbybrick_sonnet/` (subfolder named after the package)

## CMakeLists.txt Anti-Patterns

**Forbidden — these macros do not exist and will break the build:**
```cmake
include(InstallAicaDescriptions)   # ← DOES NOT EXIST
install_aica_descriptions(...)     # ← DOES NOT EXIST
```

**Correct way to install component descriptions:**
```cmake
install(DIRECTORY ./component_descriptions DESTINATION .)
```

## Adding Dependencies

- Python packages: add to `source/brickbybrick_sonnet/requirements.txt` and reference in `aica-package.toml` under `[build.packages.brickbybrick-sonnet.dependencies.pip]`
- C++/system libraries: add to `aica-package.toml` under `[build.packages.brickbybrick-sonnet.dependencies.apt]`

## Development Guidelines

- Wait for explicit user instruction before implementing each step
- Treat YOLOv11 and line detection as optional/external modules — structure code so they can be integrated later
- All new Python imports must be tracked in `requirements.txt`
