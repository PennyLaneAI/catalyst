# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This package contains the Catalyst Python interface.
"""

import sys
import types

from catalyst._configuration import INSTALLED
from catalyst._version import __version__

if not INSTALLED:
    import os

    default_bindings_path = os.path.join(
        os.path.dirname(__file__), "../../mlir/build/python_packages/quantum"
    )
    if os.path.exists(default_bindings_path):  # pragma: no cover
        sys.path.insert(0, default_bindings_path)

# Patch certain modules to integrate our MLIR bindings with JAX. This needs to happen before any
# part of 'mlir_quantum' is imported.
# Note that '__import__' does not return the specific submodule, only the parent package.
# pylint: disable=protected-access
sys.modules["mlir_quantum.ir"] = __import__("jaxlib.mlir.ir").mlir.ir
sys.modules["mlir_quantum._mlir_libs"] = __import__("jaxlib.mlir._mlir_libs").mlir._mlir_libs
# C++ extensions to the dialects are mocked out.
sys.modules["mlir_quantum._mlir_libs._quantumDialects.gradient"] = types.ModuleType(
    "mlir_quantum._mlir_libs._quantumDialects.gradient"
)
sys.modules["mlir_quantum._mlir_libs._quantumDialects.quantum"] = types.ModuleType(
    "mlir_quantum._mlir_libs._quantumDialects.quantum"
)


# pylint: disable=wrong-import-position
from catalyst.compilation_pipelines import QJIT, CompileOptions, qjit
from catalyst.pennylane_extensions import (
    adjoint,
    cond,
    for_loop,
    grad,
    jacobian,
    jvp,
    measure,
    vjp,
    while_loop,
)
from catalyst.utils.exceptions import CompileError

__all__ = (
    "qjit",
    "QJIT",
    "for_loop",
    "while_loop",
    "cond",
    "measure",
    "grad",
    "jacobian",
    "vjp",
    "jvp",
    "adjoint",
    "CompileError",
    "CompileOptions",
)
