#!/bin/bash
find pennylane/ -name "*.py" | xargs pyupgrade --py312-plus
