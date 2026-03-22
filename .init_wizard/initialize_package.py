#!/usr/bin/env python3
import os
import re
import shutil
from glob import glob
from pathlib import Path
from fnmatch import translate
from typing import Optional, List, Dict, LiteralString

from questions import (
    AVAILABLE_TEMPLATES,
    AVAILABLE_TEMPLATES_Q,
    CONTROLLER_PACKAGE_Q,
    CONTROLLER_NAME_Q,
    HARDWARE_IF_Q,
    COMPONENT_PACKAGE_Q,
    TEMPLATES_TO_INCLUDE_Q,
    VSCODE_DEFAULTS_Q,
    COLLECTION_NAME_Q,
    CONFIRMATION_Q,
    RERUN_Q,
    camel_to_snake,
)

from jinja2 import Environment, FileSystemLoader


TEMPLATE_SOURCES = os.getenv("TEMPLATE_SOURCES", ".")
TEMPLATE_TARGET_DIR = os.getenv("TEMPLATE_TARGET_DIR", "/sources")


def run_wizard():
    """
    :brief: Wizard to initialize a package with templates for controllers and components.
    :details: This wizard will guide you through the process of setting up your package by asking a series of questions.
              It will create the necessary files and directories based on your responses, and then self-delete to keep
              the workspace clean.
    """

    if os.path.exists(f"{TEMPLATE_TARGET_DIR}/aica-package.toml"):
        if RERUN_Q.ask():
            try:
                rm_files(
                    [
                        f"{TEMPLATE_TARGET_DIR}/source",
                        f"{TEMPLATE_TARGET_DIR}/.devcontainer",
                        f"{TEMPLATE_TARGET_DIR}/.vscode",
                        f"{TEMPLATE_TARGET_DIR}/.github",
                        f"{TEMPLATE_TARGET_DIR}/aica-package.toml",
                    ]
                )
            except OSError as e:
                print(f"Error removing files: {e}")
        else:
            print("Package initialization aborted.")
            exit(0)

    env = Environment(loader=FileSystemLoader(Path(__file__).parent.joinpath("templates")))

    print(
        "This package initialization wizard will help you set up your development environment,"
        " please carefully follow the prompts."
    )

    component_context = {}
    controller_context = {}
    configuration = {
        "controller": controller_context,
        "component": component_context,
        "collection_name": "",
    }

    # template selection
    configuration["templates_chosen"] = AVAILABLE_TEMPLATES_Q.ask()

    if "Controllers" in configuration["templates_chosen"]:
        controller_context["package_name"] = CONTROLLER_PACKAGE_Q.ask()
        controller_context["controller_name"] = CONTROLLER_NAME_Q.ask()
        controller_context["hardware_interface"] = HARDWARE_IF_Q.ask().lower()
        print("-- Controller configuration complete\n")

    if "Components" in configuration["templates_chosen"]:
        component_context["package_name"] = COMPONENT_PACKAGE_Q.ask()
        component_context["component_templates_included"] = TEMPLATES_TO_INCLUDE_Q.ask()
        print("-- Component configuration complete\n")

    if len(controller_context) and len(component_context):
        if controller_context["package_name"] == component_context["package_name"]:
            print("Controller and component package names must be different.")
            return
        configuration["vs_code_package"] = VSCODE_DEFAULTS_Q(configuration).ask()
        configuration["collection_name"] = COLLECTION_NAME_Q.ask()
        print("-- VSCode and collection configuration complete\n")
    elif len(controller_context):
        configuration["vs_code_package"] = controller_context["package_name"]
    else:
        configuration["vs_code_package"] = component_context["package_name"]

    print_configuration(configuration)
    exclude = []

    if not CONFIRMATION_Q.ask():
        print("Package initialization aborted.")
        return

    if configuration["controller"]:
        controller_context["controller_name_snake"] = camel_to_snake(controller_context["controller_name"])
        populate_templates(
            env,
            controller_context,
            f"{TEMPLATE_SOURCES}/templates",
            f"{TEMPLATE_TARGET_DIR}/source/{controller_context['package_name']}",
            "controller",
        )
        rename_files_and_directories(
            controller_context, f"{TEMPLATE_TARGET_DIR}/source/{controller_context['package_name']}"
        )

    if configuration["component"]:
        exclude += list(set(AVAILABLE_TEMPLATES) - set(component_context["component_templates_included"]))
        exclude += [camel_to_snake(t) for t in exclude]
        exclude = [t + "*" for t in exclude]
        if not any(t.startswith("py_") for t in component_context["component_templates_included"]):
            exclude += ["*.py*"]
            exclude += ["requirements.txt.j2"]
            exclude += ["setup.cfg.j2"]
        else:
            component_context["has_py_files"] = True

        if not any(t.startswith("CPP") for t in component_context["component_templates_included"]):
            exclude += ["*.cpp*"]
            exclude += ["*.hpp*"]
        else:
            component_context["has_cpp_files"] = True

        populate_templates(
            env,
            component_context,
            f"{TEMPLATE_SOURCES}/templates",
            f"{TEMPLATE_TARGET_DIR}/source/{component_context['package_name']}",
            "component",
            exclude,
        )
        rename_files_and_directories(
            component_context, f"{TEMPLATE_TARGET_DIR}/source/{component_context['package_name']}"
        )
    populate_common_files(env, configuration, f"{TEMPLATE_SOURCES}/templates", f"{TEMPLATE_TARGET_DIR}")

    fix_permissions(f"{TEMPLATE_TARGET_DIR}/source")
    fix_permissions(f"{TEMPLATE_TARGET_DIR}/.devcontainer")
    fix_permissions(f"{TEMPLATE_TARGET_DIR}/.vscode")
    fix_permissions(f"{TEMPLATE_TARGET_DIR}/.github")
    fix_permissions(f"{TEMPLATE_TARGET_DIR}/aica-package.toml")
    try:
        rm_files([f"{TEMPLATE_SOURCES}/__pycache__"])
    except OSError as e:
        print(f"Error removing files: {e}")
    print("-- Package initialization complete\n")


def populate_templates(
    env: Environment,
    cfg: Dict,
    templates_dir: LiteralString,
    target_dir: LiteralString,
    sub_dir: LiteralString = "",
    exclude: Optional[List[LiteralString]] = None,
):
    """
    :brief: Populate the target directory with rendered templates from the templates directory.
    :param env: The Jinja2 environment used for rendering templates.
    :param cfg: The configuration dictionary used to render templates.
    :param templates_dir: The directory containing the template files.
    :param target_dir: The directory where the rendered templates will be saved.
    :param sub_dir: The subdirectory within the templates directory to search for templates.
    :param exclude: A list of file patterns to exclude from processing.
    """
    pattern = os.path.join(templates_dir, sub_dir, "**", "*.j2")
    absolute_files = glob(pattern, recursive=True)
    if exclude:
        exclude_patterns = [translate(p) for p in exclude]
        absolute_files = [f for f in absolute_files if not any(re.search(p, f) for p in exclude_patterns)]
    relative_files = [os.path.relpath(f, templates_dir) for f in absolute_files]

    for rel_path in relative_files:
        template = env.get_template(rel_path)
        out_path = str(Path(rel_path)).replace(f"{sub_dir}/", "")
        try:
            write_to_file(f"{target_dir}/{out_path.replace('.j2', '')}", template.render(cfg))
        except OSError as e:
            print(f"Error writing file: {e}")


def populate_common_files(
    env: Environment, configuration: Dict, templates_dir: LiteralString, target_dir: LiteralString
):
    """
    :brief: Populate common files in the target directory from the templates directory.
    :param env: The Jinja2 environment used for rendering templates.
    :param configuration: The configuration dictionary used to render templates.
    :param templates_dir: The directory containing the template files.
    :param target_dir: The directory where the rendered templates will be saved.
    """
    config_files = glob(os.path.join(templates_dir, ".*", "*.json.j2"))
    toml = env.get_template("aica-package.toml.j2")

    for cfg in config_files:
        filepath = os.path.relpath(cfg, templates_dir)
        template = env.get_template(filepath)
        write_to_file(f"{target_dir}/{filepath.replace('.j2', '')}", template.render(configuration))

    if len(configuration["collection_name"]) > 0:
        cfg = {"image_name": configuration["collection_name"]}
    elif len(configuration["controller"]) > 0:
        cfg = {"image_name": configuration["controller"]["package_name"]}
    else:
        cfg = {"image_name": configuration["component"]["package_name"]}

    og_delimiters = change_env_delimiters(
        env,
        {
            "variable_start_string": "<<",
            "variable_end_string": ">>",
            "block_start_string": "<%",
            "block_end_string": "%>",
        },
    )
    workflows = env.get_template(".github/workflows/build-test.yml.j2")
    write_to_file(f"{target_dir}/.github/workflows/build-test.yml", workflows.render(cfg))
    change_env_delimiters(env, og_delimiters)

    write_to_file(f"{target_dir}/aica-package.toml", toml.render(configuration))


def rename_files_and_directories(context: Dict, sources_dir: LiteralString):
    """
    :brief: Rename files and directories in the sources directory based on a Jinja2-compatible context dictionary.
    :param context: The context dictionary used for renaming.
    :param sources_dir: The directory containing the source files and directories.
    """
    pattern = os.path.join(sources_dir, "**", "*")
    absolute_files = glob(pattern, recursive=True)
    absolute_files.sort(key=lambda p: len(Path(p).parts), reverse=True)

    for abs_path in absolute_files:
        leaf = Path(abs_path).name
        for key in sorted(context.keys(), key=len, reverse=True):  # reverse ensures longer strings are replaced first
            if key in leaf:
                leaf = leaf.replace(key, context[key])
        path = "/".join(list(Path(abs_path).parts[1:-1]))
        renamed = Path(path, leaf)
        try:
            os.rename(abs_path, renamed)
        except OSError as e:
            print(f"Error renaming {abs_path} to {renamed}: {e}")


def write_to_file(filepath: LiteralString, content: LiteralString):
    """
    :brief: Write content to a file, creating any necessary directories in the way.
    :param filepath: The path to the file to write.
    :param content: The content to write to the file.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        print(f"Error writing file: {e}")


def fix_permissions(path: LiteralString):
    """
    :brief: Fix the permissions of a file or directory (recursively; if this is run in a container, this step may be
            necessary).
    :param path: The path to the file or directory.
    """
    uid = int(os.getenv("UID"))
    gid = int(os.getenv("GID"))
    try:
        if os.path.isfile(path):
            os.chown(path, uid, gid)
        else:
            for root, dirs, files in os.walk(path):
                os.chown(root, uid, gid)
                for d in dirs:
                    os.chown(os.path.join(root, d), uid, gid)
                for f in files:
                    os.chown(os.path.join(root, f), uid, gid)
    except OSError as e:
        print(f"Error fixing permissions: {e}")


def rm_files(files: List):
    """
    :brief: Remove files and directories.
    :param files: The list of files and directories to remove.
    """
    for file in files:
        if os.path.isdir(file):
            shutil.rmtree(file, ignore_errors=False)
        else:
            os.remove(file)


def change_env_delimiters(env: Environment, new_delimiter: Dict):
    """
    :brief: Change the delimiters used in the Jinja2 environment.
    :param env: The Jinja2 environment to modify.
    :param new_delimiter: A dictionary containing the new delimiter values.
    :return: A dictionary containing the original delimiter values.
    """
    og_delimiters = {
        "block_start_string": env.block_start_string,
        "block_end_string": env.block_end_string,
        "variable_start_string": env.variable_start_string,
        "variable_end_string": env.variable_end_string,
    }
    env.block_start_string = new_delimiter.get("block_start_string", env.block_start_string)
    env.block_end_string = new_delimiter.get("block_end_string", env.block_end_string)
    env.variable_start_string = new_delimiter.get("variable_start_string", env.variable_start_string)
    env.variable_end_string = new_delimiter.get("variable_end_string", env.variable_end_string)
    return og_delimiters


def print_configuration(configuration):
    """
    :brief: Print the configuration.
    :param configuration: The configuration to print.
    """
    print("\nConfiguration:\n")
    for key, value in configuration.items():
        if isinstance(value, dict) and len(value) > 0:
            print(f"  {key.capitalize()} template(s):")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                    print(f"    - {sub_key.replace('_', ' ').capitalize()}: {', '.join(sub_value)}")
                else:
                    print(f"    - {sub_key.replace('_', ' ').capitalize()}: {sub_value}")
            print()


if __name__ == "__main__":
    run_wizard()
