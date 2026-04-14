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
"""Unit test module for the xDSL implementation of the convert_noiseop_to_subroutine pass"""

import re

import numpy as np
import pennylane as qml
import pytest
from pennylane.exceptions import CompileError

from catalyst.python_interface.transforms.qecp import (
    ConvertNoiseOpToSubroutinePass,
    convert_noiseop_to_subroutine_pass,
)

pytestmark = pytest.mark.xdsl


class TestConvertNoiseOpToSubroutinePass:
    """Unit tests for the convert-noiseop-to-subroutine pass."""

    def test_with_pauli_z(self, run_filecheck):
        """Test that a PauliZ observable is not affected by diagonalization"""

        program = """
            builtin.module @module_circuit {
                func.func @test_func() attributes {quantum.node} {
                    %0 = "test.op"() : () -> !qecp.codeblock<1 x 7>

                    // CHECK: func.call @7x1_code_noise_subroutine
                    %1 = qecp.noise %0 : !qecp.codeblock<1 x 7>
                    return
                }
            }
            """

        pipeline = (ConvertNoiseOpToSubroutinePass(number_errors=1),)
        run_filecheck(program, pipeline)
