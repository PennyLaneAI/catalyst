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

import pytest

from catalyst.python_interface.transforms.qecp import (
    ConvertNoiseOpToSubroutinePass,
)

pytestmark = pytest.mark.xdsl


class TestConvertNoiseOpToSubroutinePass:
    """Unit tests for the convert-noiseop-to-subroutine pass."""

    def test_with_pauli_z(self, run_filecheck):
        """Test that a PauliZ observable is not affected by diagonalization"""

        program = """
            builtin.module @module_circuit {
                func.func @test_func() attributes {quantum.node} {
                    // CHECK: [[codeblock:%.*]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
                    %0 = "test.op"() : () -> !qecp.codeblock<1 x 7>

                    // CHECK-NEXT: [[qubit_indices:%.*]] = arith.constant dense<{{.*}}> : tensor<1xi64>
                    // CHECK-NEXT: [[rotation_params:%.*]] = arith.constant dense<[{{.*}}]> : tensor<1x3xf64>
                    // CHECK-NEXT: func.call @noise_subroutine_code1x7([[codeblock]], [[qubit_indices]], [[rotation_params]])
                    %1 = qecp.noise %0 : !qecp.codeblock<1 x 7>
                    return
                }
                // CHECK-LABEL: func.func private @noise_subroutine_code1x7([[codeblock:%.*]]: !qecp.codeblock<1 x 7>, [[qubit_indices:%.*]]: tensor<1xi64>, [[rotation_params:%.*]]: tensor<1x3xf64>)
                // CHECK-SAME: attributes {noise_subroutine_code1x7 = none}
                // CHECK-NEXT: [[num_errors:%.*]] = arith.constant 1 : i64
                // CHECK-NEXT: [[zero:%.*]] = arith.constant 0 : i64
                // CHECK-NEXT: [[one:%.*]] = arith.constant 1 : i64
                // CHECK-NEXT: [[two:%.*]] = arith.constant 2 : i64
                // CHECK-NEXT: [[noisy_codeblock:%.*]] = scf.for [[index:%.*]] = [[zero]] to [[num_errors]] step [[one]] iter_args([[current_codeblock:%.*]] = [[codeblock]]) -> (!qecp.codeblock<1 x 7>)
                // CHECK-NEXT: [[index_int:%.*]] = arith.index_cast [[index]] : index to i64
                // CHECK-NEXT: [[qubit_index:%.*]] = tensor.extract [[qubit_indices]][[[index_int]]] : tensor<1xi64>
                // CHECK-NEXT: [[phi:%.*]] = tensor.extract [[rotation_params]][[[index_int]], [[zero]]] : tensor<1x3xf64>
                // CHECK-NEXT: [[theta:%.*]] = tensor.extract [[rotation_params]][[[index_int]], [[one]]] : tensor<1x3xf64>
                // CHECK-NEXT: [[omega:%.*]] = tensor.extract [[rotation_params]][[[index_int]], [[two]]] : tensor<1x3xf64>
                // CHECK-NEXT: [[qubit:%.*]] = qecp.extract [[current_codeblock]][[[qubit_index]]] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[updated_qubit:%.*]] = qecp.rot([[phi]], [[theta]], [[omega]]) [[qubit]] : !qecp.qubit<data>
                // CHECK-NEXT: [[updated_codeblock:%.*]] = qecp.insert [[current_codeblock]][[[qubit_index]]], [[updated_qubit]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: scf.yield [[updated_codeblock]] : !qecp.codeblock<1 x 7>
                // CHECK-NEXT: }
                // CHECK: func.return [[noisy_codeblock]] : !qecp.codeblock<1 x 7>
            }
            """

        pipeline = (ConvertNoiseOpToSubroutinePass(number_errors=1),)
        run_filecheck(program, pipeline)
