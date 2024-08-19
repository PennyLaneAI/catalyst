#!/usr/bin/env bash

# Python environment path
PYTHON_ENV_PATH=$1

if [ -z "${PYTHON_ENV_PATH}" ]; then
    echo "Error: Please provide a path for the Python virtual environment"
else
    # We call the setup script in a sub-shell session, so in case of an error, the 
    # current session does not get closed.
    bash setup_dev_from_wheel.sh $PYTHON_ENV_PATH

    # Activate the Python virtual environment session
    python3 -m venv $PYTHON_ENV_PATH
    source $PYTHON_ENV_PATH/bin/activate
fi

