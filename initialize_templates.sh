#!/bin/bash

set -euo pipefail

script_dir=".init_wizard/"
container_template_sources="/ws"
container_template_target_dir="/source"
host_sources="."

pip_requirements="questionary==2.0.1 jinja2==3.0.0"

docker run --rm -it \
    -e IN_DOCKER=1 \
    -e UID="$(id -u)" \
    -e GID="$(id -g)" \
    -e TEMPLATE_SOURCES="${container_template_sources}" \
    -e TEMPLATE_TARGET_DIR="${container_template_target_dir}" \
    -v ./"${script_dir}:${container_template_sources}" \
    -v ./"${host_sources}:${container_template_target_dir}" \
    python:3.12-slim \
    bash -c "
      echo 'Installing wizard dependencies in the container...' && \
      pip install ${pip_requirements} >> /dev/null 2>&1 && \
      python3 /${container_template_sources}/initialize_package.py
    "
