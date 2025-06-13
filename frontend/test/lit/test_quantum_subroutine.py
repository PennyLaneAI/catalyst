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


def test_quantum_subroutine_identity():

    # CHECK: func.func private @identity([[REG:%.+]]: !quantum.reg) -> !quantum.reg
    # CHECK-NEXT: return [[REG]] : !quantum.reg

    @subroutine
    def identity(): ...

    qml.capture.enable()

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def foo():
        identity()
        return qml.probs()

    print(foo.mlir)
    qml.capture.disable()


test_quantum_subroutine_identity()


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
        return qml.probs()

    print(subroutine_test.mlir)

    qml.capture.disable()


test_quantum_subroutine_wire_param()


def test_quantum_subroutine_gate_param_param():

    # CHECK: func.func private @RX_on_wire_0([[REG:%.+]]: !quantum.reg, [[PARAM_TENSOR:%.+]]: tensor<f64>) -> !quantum.reg
    # CHECK-NEXT: [[QUBIT:%.+]] = quantum.extract [[REG]][ 0] : !quantum.reg -> !quantum.bit
    # CHECK-NEXT: [[PARAM:%.+]] = tensor.extract [[PARAM_TENSOR]][] : tensor<f64>
    # CHECK-NEXT: [[QUBIT_1:%.+]] = quantum.custom "RX"([[PARAM]]) [[QUBIT]] : !quantum.bit
    # CHECK-NEXT: [[REG_1:%.+]] = quantum.insert [[REG]][ 0], [[QUBIT_1]] : !quantum.reg, !quantum.bit
    # CHECK-NEXT: return [[REG_1]] : !quantum.reg
    @subroutine
    def RX_on_wire_0(param):
        qml.RX(param, wires=[0])

    qml.capture.enable()

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def subroutine_test_2():
        RX_on_wire_0(3.14)
        return qml.probs()

    print(subroutine_test_2.mlir)

    qml.capture.disable()


test_quantum_subroutine_gate_param_param()

def test_quantum_subroutine_with_control_flow():

    import catalyst

    qml.capture.enable()

    @subroutine
    def conditional_RX(param: float):

        def true_path():
            qml.RX(param, wires=[0])

        def false_path():
            ...

        qml.cond(param != 0., true_path, false_path)()


    @qml.qjit(autograph=False)
    @qml.qnode(qml.device("lightning.qubit", wires=1), autograph=False)
    def subroutine_test_3():
        conditional_RX(3.14)
        return qml.probs()

    print(subroutine_test_3.mlir)
    qml.capture.disable()

test_quantum_subroutine_with_control_flow()

