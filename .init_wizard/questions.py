try:
    import questionary  # type: ignore
    from questionary import Choice  # type: ignore
except ImportError as exc:
    raise ImportError("Please install the 'questionary' package to use this script.") from exc

import re

BLOCKLIST = ["controller", "component", "template_component_package", "template_controller_package", "src", "test"]


def is_snake_case(value: str) -> bool:
    pattern = r"^[a-z]+(?:_[a-z0-9]+)*$"
    if re.match(pattern, value):
        return True
    return False


def is_camel_case(value: str) -> bool:
    pattern = r"^[a-zA-Z]+(?:[A-Z][a-z0-9]*)*$"
    if re.match(pattern, value):
        return True
    return False


def camel_to_snake(name: str) -> str:
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s1)
    return s2.lower()


def is_valid_package_name(value: str) -> bool:
    return value not in BLOCKLIST


RERUN_Q = questionary.confirm(
    "The wizard has already been run before, do you want re-run it? (This will remove all previous configurations!)",
    default=True,
)

CONFIRMATION_Q = questionary.confirm("Are you happy with this configuration?", default=True)

COLLECTION_NAME_Q = questionary.text(
    "Choose a ROS collection name (snake_case):",
    default="",
    validate=lambda text: text.strip() != "" and is_snake_case(text) and is_valid_package_name(text),
)

# Available templates
AVAILABLE_TEMPLATES_Q = questionary.checkbox(
    "Which packages would you like to include in your development environment?",
    choices=[
        {"name": "Components", "checked": True},
        {"name": "Controllers", "checked": True},
    ],
    validate=lambda choices: len(choices) > 0,
)

# Controllers
CONTROLLER_PACKAGE_Q = questionary.text(
    "Enter the new controller package name (snake_case):",
    default="",
    validate=lambda text: text.strip() != "" and is_snake_case(text) and is_valid_package_name(text),
)

CONTROLLER_NAME_Q = questionary.text(
    "Enter the desired controller name (CamelCase):",
    default="",
    validate=lambda text: text.strip() != "" and is_camel_case(text),
)

HARDWARE_IF_Q = questionary.select(
    "Select the type of hardware interface you want to use:",
    choices=["Position", "Velocity", "Effort"],
)

# Components
COMPONENT_PACKAGE_Q = questionary.text(
    "Enter the new component package name (snake_case):",
    default="",
    validate=lambda text: text.strip() != "" and is_snake_case(text) and is_valid_package_name(text),
)

AVAILABLE_TEMPLATES = ["py_lifecycle_component", "py_component", "CPPLifecycleComponent", "CPPComponent"]
TEMPLATES_TO_INCLUDE_Q = questionary.checkbox(
    "Select the types of components you want to include:",
    choices=[
        Choice(title="Python Lifecycle", checked=True, value=AVAILABLE_TEMPLATES[0]),
        Choice(title="Python", checked=True, value=AVAILABLE_TEMPLATES[1]),
        Choice(title="C++ Lifecycle", checked=True, value=AVAILABLE_TEMPLATES[2]),
        Choice(title="C++", checked=True, value=AVAILABLE_TEMPLATES[3]),
    ],
    validate=lambda choices: len(choices) > 0,
)


# VSCode Defaults
def VSCODE_DEFAULTS_Q(configuration: dict):
    return questionary.select(
        "Which package would you like to use for VSCode and devcontainer settings?",
        choices=[
            {"name": configuration["component"]["package_name"]},
            {"name": configuration["controller"]["package_name"]},
        ],
    )
