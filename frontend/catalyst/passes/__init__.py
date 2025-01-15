# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This module provides access to Catalyst's compiler transformation infrastructure, including the
use of Python decorators to configure and schedule individual built-in compiler passes, as well
as load and run external MLIR passes from plugins.

.. note::

    Unlike PennyLane :doc:`circuit transformations <introduction/compiling_circuits>`,
    the QNode itself will not be changed or transformed by applying these
    decorators.

    As a result, circuit inspection tools such as :func:`~.draw` will continue
    to display the circuit as written in Python.

    Instead, these compiler passes are applied at the MLIR level, which occurs
    outside of Python during compile time. To inspect the compiled MLIR from
    Catalyst, use :func:`~.get_compilation_stage` with
    ``stage="QuantumCompilationPass"``.

"""

from catalyst.passes.builtin_passes import (
    cancel_inverses,
    ions_decomposition,
    merge_rotations,
)
from catalyst.passes.pass_api import Pass, PassPlugin, apply_pass, apply_pass_plugin

__all__ = (
    "cancel_inverses",
    "ions_decomposition",
    "merge_rotations",
    "Pass",
    "PassPlugin",
    "apply_pass",
    "apply_pass_plugin",
)
