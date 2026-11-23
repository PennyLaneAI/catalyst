#!/bin/bash
$(which python3) -m black -t py311 -t py312 -t py313 -l 100 $1
