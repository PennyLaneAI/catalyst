#!/usr/bin/env bash

set -e
git config --local advice.detachedHead false
export CATALYST_DIR=$PWD
export CATALYST_FRONTEND_DIR=$CATALYST_DIR/frontend/catalyst
export CATALYST_PYTHON_ENV_DIR=$CATALYST_DIR/.catalyst_python_env

install_CPL_wheels(){
    echo "Installing CPL Wheels..."
    python3 -m venv $CATALYST_PYTHON_ENV_DIR
    source $CATALYST_PYTHON_ENV_DIR/bin/activate
    python -m pip uninstall -y pennylane-catalyst
    python -m pip install --extra-index-url https://test.pypi.org/simple/ \
    pennylane pennylane-lightning pennylane-catalyst --pre --upgrade
}

checkout_nightly_build(){
    echo "Checking out nightly build..."
    cd $CATALYST_DIR
    export SITEPKGS=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    git switch main
    git fetch
    git pull
    # Search for the corresponding commit to the Wheel
    git log --grep="bump nightly version" | grep "commit" | cut -d " " -f 2 | while read -r NIGHTLY_BUMP; do
        git checkout $NIGHTLY_BUMP^1; 
        export DIFF=$(diff $SITEPKGS/catalyst/_version.py $CATALYST_FRONTEND_DIR/_version.py)
        if [ -z "${DIFF}" ]; then
            export WHEEL_COMMIT=$(git log -1 --format="%h")
            echo "Success: the commit $WHEEL_COMMIT corresponding to the Wheel was found"
            break
        fi
        echo "Discarding commit, still searching for the corresponding commit to the Wheel..."
    done
    if [ ! -z "${DIFF}" ]; then
        echo "Could not find the corresponding commit for the Wheel"
        exit
    fi
}

link_repo_to_wheel(){
    echo "Linking Catalyst repository to Catalyst Wheel..."
    export SITEPKGS=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    cp -lrf $SITEPKGS/catalyst/ $CATALYST_DIR/frontend/
}

restore_catalyst_config(){
    cd $CATALYST_DIR
    git checkout frontend/catalyst/_configuration.py
}

report_changed_files(){
    cd $CATALYST_DIR
    git status
}

overwrite_env(){
    echo "Overwriting the existing environment..."
    install_CPL_wheels
    checkout_nightly_build
    link_repo_to_wheel
    restore_catalyst_config
    report_changed_files
    echo "Done."
}

setup(){
    echo "Setting up a Catalyst development environment from a Wheel..."
    echo "Uncommitted changes in $CATALYST_FRONTEND_DIR will be lost and replaced with the contents from the Wheel."
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
