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

# pylint: disable=line-too-long

"""Lit tests for the PLxPR to JAXPR with quantum primitives pipeline"""

import pennylane as qml


def test_conditional_capture():
    """Test an if statement"""

    qml.capture.enable()

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def captured_circuit():
        m = qml.measure(0)
        # CHECK: [[QREG:%.+]] = quantum.insert
        # CHECK: [[QREG_3:%.+]] = scf.if
        # CHECK:    [[QREG_2:%.+]] = quantum.insert [[QREG]][ 0]
        # CHECK:    scf.yield [[QREG_2]]
        # CHECK: else
        # CHECK:     scf.yield [[QREG]]
        # CHECK: quantum.compbasis qreg [[QREG_3]] : !quantum.obs
        qml.cond(m, lambda: (qml.X(0), None)[1])()
        return qml.state()

    @qml.qjit
    def main():
        return captured_circuit()

    print(main.mlir)

    qml.capture.enable()


test_conditional_capture()


def test_loop_capture():
    """Test a for loop"""

    qml.capture.enable()

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def captured_circuit():
        _ = qml.measure(0)

        # CHECK: [[QREG:%.+]] = quantum.insert
        # CHECK: [[QREG_4:%.+]] = scf.for {{.*}} iter_args([[QREG_2:%.+]] = [[QREG]]
        # CHECK:   [[QREG_3:%.+]] = quantum.insert [[QREG_2]]
        # CHECK:   scf.yield [[QREG_3]]
        # CHECK: quantum.compbasis qreg [[QREG_4]]
        @qml.for_loop(0, 2, 1)
        def loop_fn(_):
            qml.Hadamard(0)

        loop_fn()

        return qml.state()

    @qml.qjit
    def main():
        return captured_circuit()

    print(main.mlir)

    qml.capture.disable()


test_loop_capture()


def test_while_capture():
    """Test a while loop"""

    qml.capture.enable()

    # CHECK: [[QREG:%.+]] = quantum.insert
    # CHECK: [[PAIR:%.+]]:2 = scf.while {{.*}} [[QREG2:%.+]] = [[QREG]]
    # CHECK:     [[QREG3:%.+]] = quantum.insert [[QREG2:%.+]]
    # CHECK:     scf.yield {{.*}} [[QREG3]]
    # CHECK: quantum.compbasis qreg [[PAIR]]#1
    @qml.qnode(qml.device("null.qubit", wires=1))
    def captured_circuit():
        _ = qml.measure(0)

        def less_than_10(x):
            return x[0] < 10

        @qml.while_loop(less_than_10)
        def loop(v):
            qml.Hadamard(0)
            return v[0] + 1, v[1]

        loop((0, 1))
        return qml.state()

    @qml.qjit
    def main():
        return captured_circuit()

    print(main.mlir)

    qml.capture.disable()


test_while_capture()


def test_dynamic_wire():
    """Test dynamic wires no re-insertion"""

    qml.capture.enable()
    dev = qml.device("null.qubit", wires=3)

    @qml.qjit(target="mlir")
    @qml.qnode(dev)
    def circuit(w1: int):
        # CHECK: [[QREG:%.+]] = quantum.insert
        # CHECK-NEXT: [[SCALAR:%.+]] = tensor.extract %arg0
        # CHECK-NEXT: [[QBIT:%.+]] = quantum.extract [[QREG]][[[SCALAR]]]
        # CHECK-NEXT: [[QBIT_1:%.+]] = quantum.custom "PauliY"() [[QBIT]]
        # CHECK-NEXT: [[QBIT_2:%.+]] = quantum.custom "PauliZ"() [[QBIT_1]]
        qml.X(0)
        qml.Y(w1)
        qml.Z(w1)
        qml.X(0)
        return qml.state()

    print(circuit.mlir)
    qml.capture.disable()


test_dynamic_wire()


def test_dynamic_wire_reinsertion():
    """Test dynamic wires re-insertion"""

    qml.capture.enable()
    dev = qml.device("null.qubit", wires=3)

    @qml.qjit(target="mlir")
    @qml.qnode(dev)
    def circuit(w1: int):

        # CHECK: [[QUBIT:%.+]] = quantum.custom "PauliX"() %1 : !quantum.bit
        # CHECK-NEXT: [[QREG:%.+]] = quantum.insert %0[ 0], [[QUBIT]] : !quantum.reg, !quantum.bit
        # CHECK-NEXT: [[SCALAR:%.+]] = tensor.extract %arg0[] : tensor<i64>
        # CHECK-NEXT: [[QUBIT_1:%.+]] = quantum.extract [[QREG]][[[SCALAR]]] : !quantum.reg -> !quantum.bit
        # CHECK-NEXT: [[QUBIT_2:%.+]] = quantum.custom "PauliY"() [[QUBIT_1]]
        # CHECK-NEXT: [[SCALAR:%.+]] = tensor.extract %arg0[] : tensor<i64>
        # CHECK-NEXT: [[QREG_1:%.+]] = quantum.insert [[QREG]][[[SCALAR]]], [[QUBIT_2]] : !quantum.reg, !quantum.bit
        # CHECK-NEXT: [[QUBIT_3:%.+]] = quantum.extract [[QREG_1]][ 0]
        # CHECK-NEXT: [[QUBIT_4:%.+]] = quantum.custom "PauliX"() [[QUBIT_3]]
        # CHECK-NEXT: [[QREG_2:%.+]] = quantum.insert [[QREG_1]][ 0], [[QUBIT_4]]
        # CHECK-NEXT: [[SCALAR:%.+]] = tensor.extract %arg0[] : tensor<i64>
        # CHECK-NEXT: [[QUBIT_5:%.+]] = quantum.extract [[QREG_2]][[[SCALAR]]] : !quantum.reg -> !quantum.bit
        # CHECK-NEXT: [[QUBIT_6:%.+]] = quantum.custom "PauliZ"() [[QUBIT_5]]
        # CHECK-NEXT: [[SCALAR:%.+]] = tensor.extract %arg0[] : tensor<i64>
        # CHECK-NEXT: [[QREG_3:%.+]] = quantum.insert [[QREG_2]][[[SCALAR]]], [[QUBIT_6]]
        # CHECK-NEXT: [[QUBIT_7:%.+]] = quantum.extract [[QREG_3]][ 0]
        # CHECK-NEXT: [[QUBIT_8:%.+]] = quantum.custom "PauliX"() [[QUBIT_7]]
        qml.X(0)
        qml.Y(w1)
        qml.X(0)
        qml.Z(w1)
        qml.X(0)
        return qml.state()

    print(circuit.mlir)
    qml.capture.disable()


test_dynamic_wire_reinsertion()


def test_two_dynamic_CNOTs():
    """Test two dynamic CNOTs"""

    dev = qml.device("null.qubit", wires=3)

    qml.capture.enable()

    @qml.qjit(target="mlir")
    @qml.qnode(dev)
    def circuit(w1: int, w2: int):
        # CHECK: [[SCALAR:%.+]] = tensor.extract %arg0
        # CHECK-NEXT: [[QUBIT_0:%.+]] = quantum.extract {{%.+}}[[[SCALAR]]]
        # CHECK-NEXT: [[SCALAR_2:%.+]] = tensor.extract %arg1
        # CHECK-NEXT: [[QUBIT_1:%.+]] = quantum.extract {{%.+}}[[[SCALAR_2]]]
        # CHECK-NEXT: [[QUBITS:%.+]]:2 = quantum.custom "CNOT"() [[QUBIT_0]], [[QUBIT_1]]
        # CHECK-NEXT: quantum.custom "CNOT"() [[QUBITS]]#0, [[QUBITS]]#1
        qml.CNOT(wires=[w1, w2])
        qml.CNOT(wires=[w1, w2])
        return qml.state()

    print(circuit.mlir)
    qml.capture.disable()


test_two_dynamic_CNOTs()


def test_single_qubit_dynamic():
    """single qubit gates on two different dynamic wires one after the other"""

    dev = qml.device("null.qubit", wires=3)

    qml.capture.enable()

    @qml.qjit(target="mlir")
    @qml.qnode(dev)
    def circuit(w1: int, w2: int):

        # CHECK: [[REG:%.+]] = quantum.alloc
        # CHECK-NEXT: [[QUBIT_0:%.+]] = quantum.extract [[REG]][ 0]
        # CHECK-NEXT: [[QUBIT_0_1:%.+]] = quantum.custom "Hadamard"() [[QUBIT_0]]
        # CHECK-NEXT: [[REG_1:%.+]] = quantum.insert [[REG]][ 0], [[QUBIT_0_1]]
        # CHECK-NEXT: [[SCALAR:%.+]] = tensor.extract %arg0[]
        # CHECK-NEXT: [[QUBIT_W1:%.+]] = quantum.extract [[REG_1]][[[SCALAR]]]
        # CHECK-NEXT: [[QUBIT_W1_1:%.+]] = quantum.custom "Hadamard"() [[QUBIT_W1]]
        # CHECK-NEXT: [[SCALAR:%.+]] = tensor.extract %arg0[]
        # CHECK-NEXT: [[REG_2:%.+]] = quantum.insert [[REG_1]][[[SCALAR]]], [[QUBIT_W1_1]]
        # CHECK-NEXT: [[SCALAR_1:%.+]] = tensor.extract %arg1[]
        # CHECK-NEXT: [[QUBIT_W2:%.+]] = quantum.extract [[REG_2]][[[SCALAR_1]]]
        # CHECK-NEXT: [[QUBIT_W2_1:%.+]] = quantum.custom "Hadamard"() [[QUBIT_W2]]
        # CHECK-NEXT: [[SCALAR_1:%.+]] = tensor.extract %arg1[]
        # CHECK-NEXT: [[REG_3:%.+]] = quantum.insert [[REG_2]][[[SCALAR_1]]], [[QUBIT_W2_1]]
        # CHECK-NEXT: [[QUBIT_0_2:%.+]] = quantum.extract [[REG_3]][ 0]
        # CHECK-NEXT: [[QUBIT_0_3:%.+]] = quantum.custom "Hadamard"() [[QUBIT_0_2]]
        # CHECK-NEXT: quantum.insert [[REG_3]][ 0], [[QUBIT_0_3]]
        qml.Hadamard(0)
        qml.Hadamard(w1)
        qml.Hadamard(w2)
        qml.Hadamard(0)
        return qml.state()

    print(circuit.mlir)
    qml.capture.disable()


test_single_qubit_dynamic()


def test_single_qubit_dynamic_static_interleaved():
    """Single qubit gates on two different dynamic wires interleaved with static wires"""

    dev = qml.device("null.qubit", wires=3)

    qml.capture.enable()

    @qml.qjit(target="mlir")
    @qml.qnode(dev)
    def circuit(w1: int, w2: int):

        # CHECK: [[REG:%.+]] = quantum.alloc
        # CHECK-NEXT: [[QUBIT_0:%.+]] = quantum.extract [[REG]][ 0]
        # CHECK-NEXT: [[QUBIT_0_1:%.+]] = quantum.custom "Hadamard"() [[QUBIT_0]]
        # CHECK-NEXT: [[REG_1:%.+]] = quantum.insert [[REG]][ 0], [[QUBIT_0_1]]
        # CHECK-NEXT: [[SCALAR:%.+]] = tensor.extract %arg0[]
        # CHECK-NEXT: [[QUBIT_W1:%.+]] = quantum.extract [[REG_1]][[[SCALAR]]]
        # CHECK-NEXT: [[QUBIT_W1_1:%.+]] = quantum.custom "Hadamard"() [[QUBIT_W1]]
        # CHECK-NEXT: [[SCALAR:%.+]] = tensor.extract %arg0[]
        # CHECK-NEXT: [[REG_2:%.+]] = quantum.insert [[REG_1]][[[SCALAR]]], [[QUBIT_W1_1]]

        # CHECK-NEXT: [[QUBIT_0_2:%.+]] = quantum.extract [[REG_2]][ 0]
        # CHECK-NEXT: [[QUBIT_0_3:%.+]] = quantum.custom "Hadamard"() [[QUBIT_0_2]]
        # CHECK-NEXT: [[REG_3:%.+]] = quantum.insert [[REG_2]][ 0], [[QUBIT_0_1]]

        # CHECK-NEXT: [[SCALAR_1:%.+]] = tensor.extract %arg1[]
        # CHECK-NEXT: [[QUBIT_W2:%.+]] = quantum.extract [[REG_3]][[[SCALAR_1]]]
        # CHECK-NEXT: [[QUBIT_W2_1:%.+]] = quantum.custom "Hadamard"() [[QUBIT_W2]]
        # CHECK-NEXT: [[SCALAR_1:%.+]] = tensor.extract %arg1[]
        # CHECK-NEXT: [[REG_4:%.+]] = quantum.insert [[REG_3]][[[SCALAR_1]]], [[QUBIT_W2_1]]
        # CHECK-NEXT: [[QUBIT_0_2:%.+]] = quantum.extract [[REG_4]][ 0]
        # CHECK-NEXT: [[QUBIT_0_3:%.+]] = quantum.custom "Hadamard"() [[QUBIT_0_2]]
        # CHECK-NEXT: quantum.insert [[REG_4]][ 0], [[QUBIT_0_3]]
        qml.Hadamard(0)
        qml.Hadamard(w1)
        qml.Hadamard(0)
        qml.Hadamard(w2)
        qml.Hadamard(0)
        return qml.state()

    print(circuit.mlir)
    qml.capture.disable()


test_single_qubit_dynamic_static_interleaved()


def test_multi_qubit_gates_on_different_dynamic_wires():
    """Multi-qubit gates on different dynamic wires."""

    dev = qml.device("null.qubit", wires=3)

    qml.capture.enable()

    @qml.qjit(target="mlir")
    @qml.qnode(dev)
    def circuit(w1: int, w2: int, w3: int):
        # CHECK: [[SCALAR:%.+]] = tensor.extract %arg0
        # CHECK-NEXT: [[QUBIT_0:%.+]] = quantum.extract {{%.+}}[[[SCALAR]]]
        # CHECK-NEXT: [[SCALAR_2:%.+]] = tensor.extract %arg1
        # CHECK-NEXT: [[QUBIT_1:%.+]] = quantum.extract {{%.+}}[[[SCALAR_2]]]
        # CHECK-NEXT: [[QUBITS:%.+]]:2 = quantum.custom "CNOT"() [[QUBIT_0]], [[QUBIT_1]]
        qml.CNOT(wires=[w1, w2])
        # CHECK-NEXT: [[SCALAR:%.+]] = tensor.extract %arg0
        # CHECK-NEXT: [[QREG:%.+]] = quantum.insert {{%.+}}[[[SCALAR]]]
        # CHECK-NEXT: [[SCALAR_1:%.+]] = tensor.extract %arg1
        # CHECK-NEXT: [[QREG_1:%.+]] = quantum.insert [[QREG]][[[SCALAR_1]]]
        # CHECK-NEXT: [[SCALAR_1:%.+]] = tensor.extract %arg1
        # CHECK-NEXT: [[QUBIT_2:%.+]] = quantum.extract [[QREG_1]][[[SCALAR_1]]]
        # CHECK-NEXT: [[SCALAR_2:%.+]] = tensor.extract %arg2
        # CHECK-NEXT: [[QUBIT_3:%.+]] = quantum.extract [[QREG_1]][[[SCALAR_2]]]
        # CHECK-NEXT: [[QUBITS:%.+]]:2 = quantum.custom "CNOT"() [[QUBIT_2]], [[QUBIT_3]]
        qml.CNOT(wires=[w2, w3])
        return qml.state()

    print(circuit.mlir)
    qml.capture.disable()


test_multi_qubit_gates_on_different_dynamic_wires()
