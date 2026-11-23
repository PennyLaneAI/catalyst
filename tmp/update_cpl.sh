#!/bin/bash
pip install --upgrade --index-url https://test.pypi.org/simple/ --pre --no-deps pennylane-catalyst pennylane-lightning
pip install --upgrade --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pennylane-catalyst pennylane-lightning
pip install -e .
