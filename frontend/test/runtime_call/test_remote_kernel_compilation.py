# Copyright 2026 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for CrossCompileRemoteKernels (CCRK) frontend integration."""

import pathlib

import jax
import numpy as np
import pytest

from catalyst.debug import compile_mlir

_MLIR_FILE = pathlib.Path(__file__).parent / "test_target_module.mlir"
_RESULT_TYPES = [
    jax.ShapeDtypeStruct((), np.float64),
    jax.ShapeDtypeStruct((4,), np.float64),
]


@pytest.mark.skipif(not _MLIR_FILE.is_file(), reason="test_target_module.mlir not found")
class TestRemoteKernelCompilation:

    @pytest.fixture(scope="class")
    def compiled(self):
        fn = compile_mlir(_MLIR_FILE.resolve(), func_name="jit_circuit", result_types=_RESULT_TYPES)
        yield fn
        fn.workspace.cleanup()

    def test_compilation_succeeds(self, compiled):
        assert compiled is not None

    def test_kernel_object_produced(self, compiled):
        ws = pathlib.Path(str(compiled.workspace))
        assert (ws / "module_circuit" / "module_circuit.o").is_file()
