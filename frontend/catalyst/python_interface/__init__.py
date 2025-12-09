# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unified Compiler API for integration of Catalyst with xDSL."""

from importlib.util import find_spec

if not (find_spec("xdsl") and find_spec("xdsl_jax")):  # pragma: no cover
    raise RuntimeError(
        "Using the Unified compiler framework requires xDSL and xDSL-JAX to be installed. "
        "They can be installed by executing 'python -m pip install xdsl xdsl-jax'."
    )

from .compiler import Compiler
from .inspection import QMLCollector, mlir_specs
from .parser import QuantumParser
from .pass_api import compiler_transform

__all__ = [
    "Compiler",
    "compiler_transform",
    "QuantumParser",
    "QMLCollector",
    "mlir_specs",
]
