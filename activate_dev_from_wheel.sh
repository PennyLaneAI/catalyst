#!/usr/bin/env bash

# Python environment name
PYTHON_ENV_NAME=$1

if [ -z "${PYTHON_ENV_NAME}" ]; then
    echo "Error: Please provide a name for the Python virtual environment"
else
    export CATALYST_PYTHON_ENV_DIR=/tmp/$PYTHON_ENV_NAME

    # We call the setup script in a sub-shell session, so in case of an error, the 
    # current session does not get closed.
    bash setup_dev_from_wheel.sh $PYTHON_ENV_NAME

    # Activate the Python virtual environment session
    python3 -m venv $CATALYST_PYTHON_ENV_DIR
    source $CATALYST_PYTHON_ENV_DIR/bin/activate
fi

