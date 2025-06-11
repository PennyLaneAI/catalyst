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

# RUN: %PYTHON %s | FileCheck %s

import pennylane as qml

from catalyst.jax_primitives import subroutine

def test_quantum_subroutine_wire_param():


    # CHECK: func.func private @Hadamard0([[REG:%.+]]: !quantum.reg, [[WIRE_IDX_TENSOR:%.+]]: tensor<i64>) -> !quantum.reg 
    # CHECK-NEXT: [[WIRE_IDX:%.+]] = tensor.extract [[WIRE_IDX_TENSOR]][] : tensor<i64>
    # CHECK-NEXT: [[QUBIT:%.+]] = quantum.extract [[REG]][[[WIRE_IDX]]] : !quantum.reg -> !quantum.bit
    # CHECK-NEXT: [[QUBIT_1:%.+]] = quantum.custom "Hadamard"() [[QUBIT]] : !quantum.bit
    # CHECK-NEXT: [[WIRE_IDX:%.+]] = tensor.extract [[WIRE_IDX_TENSOR]][] : tensor<i64>
    # CHECK-NEXT: [[REG_1:%.+]] = quantum.insert [[REG]][[[WIRE_IDX]]], [[QUBIT_1]] : !quantum.reg, !quantum.bit
    # CHECK-NEXT: return [[REG_1]] : !quantum.reg

    @subroutine
    def Hadamard0(wire):
        qml.Hadamard(wire)

    qml.capture.enable()

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def subroutine_test(c: int):
        Hadamard0(c)
        return qml.state()

    print(subroutine_test.mlir)

    qml.capture.disable()

test_quantum_subroutine_wire_param()
