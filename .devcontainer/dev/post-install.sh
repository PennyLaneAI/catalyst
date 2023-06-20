#!/bin/bash

git submodule update --init --depth=1
pip install -r requirements.txt
make all
