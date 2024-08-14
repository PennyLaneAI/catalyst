#!/usr/bin/env bash

export CATALYST_DIR=$PWD
export CATALYST_PYTHON_ENV_DIR=$CATALYST_DIR/.catalyst_python_env

bash setup_dev_from_wheel.sh
python3 -m venv $CATALYST_PYTHON_ENV_DIR
source $CATALYST_PYTHON_ENV_DIR/bin/activate
