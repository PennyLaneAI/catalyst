#!/usr/bin/env bash

# Python environment path
PYTHON_ENV_PATH=$1

# Exit on any error
set -e

# Turn off detached head advice on Git
git config --local advice.detachedHead false

export CATALYST_DIR=$PWD
export CATALYST_FRONTEND_SRC=$CATALYST_DIR/frontend/catalyst

install_catalyst_wheel(){
    echo "Installing Catalyst Wheel..."

    # Create (if not created yet) and activate the virtual environment 
    python3 -m venv $PYTHON_ENV_PATH
    source $PYTHON_ENV_PATH/bin/activate
    echo "Success: The Python virtual environment located at '$PYTHON_ENV_PATH' was activated"

    # Clean the Catalyst installation
    python -m pip uninstall -y pennylane-catalyst

    # Install Catalyst requirements
    python -m pip install -r requirements.txt

    # Install Catalyst from TestPyPI
    python -m pip install --extra-index-url https://test.pypi.org/simple/ pennylane-catalyst --pre --upgrade
}

checkout_nightly_build(){
    echo "Checking out nightly build..."

    # Get the Python virtual environment site-packages path
    export SITEPKGS=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    export CATALYST_WHEEL=$SITEPKGS/catalyst

    # Update to the latest commit at the main branch
    cd $CATALYST_DIR
    git switch main
    git fetch
    git pull

    # Search for the commit corresponding to latest available Wheel at TestPyPI
    git log --grep="bump nightly version" | grep "commit" | cut -d " " -f 2 | while read -r NIGHTLY_BUMP; do
        # The commit right before the nightly bump must have the same version as the Wheel
        git checkout $NIGHTLY_BUMP^1; 
        export DIFF=$(diff $CATALYST_WHEEL/_version.py $CATALYST_FRONTEND_SRC/_version.py)
        if [ -z "${DIFF}" ]; then
            export CATALYST_WHEEL_COMMIT=$(git log -1 --format="%h")
            echo "Success: The commit $CATALYST_WHEEL_COMMIT corresponding to the Wheel was found"
            break
        fi
        echo "Discarding commit, still searching for the corresponding commit to the Wheel..."
    done

    if [ ! -z "${DIFF}" ]; then
        echo "Error: Could not find the corresponding commit for the Wheel"
        exit
    fi
}

link_repo_to_wheel(){
    echo "Linking Catalyst repository to Catalyst Wheel..."

    export SITEPKGS=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    export CATALYST_WHEEL=$SITEPKGS/catalyst

    # Create hard links to the Wheel Python sources
    cp -lrf $CATALYST_WHEEL $CATALYST_DIR/frontend/
}

restore_catalyst_config(){
    # After linking the Wheel sources, _configuration.py will contain the entry: 'INSTALLED=True'.
    # Hence, we restore the file from the repository.
    cd $CATALYST_DIR
    git checkout frontend/catalyst/_configuration.py
}

report_changed_files(){
    # If everything went well, there should not be any file changes.
    cd $CATALYST_DIR
    export STATUS=$(git status | grep "nothing to commit, working tree clean")
    if [ -z "${STATUS}" ]; then
        echo "Warning: Some files might have changed. Use 'git status' to list them"
    else
        echo "Success: Wheel sources and repository sources match"
    fi
}

overwrite_env(){
    echo "Overwriting the existing environment..."

    install_catalyst_wheel
    checkout_nightly_build
    link_repo_to_wheel
    restore_catalyst_config
    report_changed_files

    echo "Done."
}

setup(){
    echo "Setting up a Catalyst development environment from a Wheel..."

    # Ask before overwriting the repository
    echo "Uncommitted changes in $CATALYST_FRONTEND_SRC will be lost and replaced with the contents from the Wheel."
    read -p "Are you sure? [y/n]" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        overwrite_env
    else
        echo "No changes were made."
    fi
}

setup
