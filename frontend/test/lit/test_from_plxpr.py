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

from functools import partial

import pennylane as qp


def test_conditional_capture():
    """Test an if statement"""

    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def captured_circuit():
        # CHECK: [[QREG:%.+]] = qref.alloc( 1) : !qref.reg<1>
        # CHECK: [[QUBIT:%.+]] = qref.get [[QREG]][ 0] : !qref.reg<1> -> !qref.bit
        # CHECK: [[m:%.+]] = qref.measure [[QUBIT]] : i1
        m = qp.measure(0)

        # CHECK: scf.if
        # CHECK:   [[QUBIT:%.+]] = qref.get [[QREG]][ 0] : !qref.reg<1> -> !qref.bit
        # CHECK:   qref.custom "PauliX"() [[QUBIT]] : !qref.bit
        # CHECK: }
        # CHECK: qref.compbasis(qreg [[QREG]] : !qref.reg<1>) : !quantum.obs
        qp.cond(m, lambda: (qp.X(0), None)[1])()
        return qp.state()

    @qp.qjit(capture=True)
    def main():
        return captured_circuit()

    print(main.mlir)


test_conditional_capture()


def test_loop_capture():
    """Test a for loop"""

    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def captured_circuit():
        # CHECK: [[QREG:%.+]] = qref.alloc( 1) : !qref.reg<1>
        # CHECK: [[QUBIT:%.+]] = qref.get [[QREG]][ 0] : !qref.reg<1> -> !qref.bit
        # CHECK: [[m:%.+]] = qref.measure [[QUBIT]] : i1
        _ = qp.measure(0)

        # CHECK: scf.for
        # CHECK:   [[QUBIT:%.+]] = qref.get [[QREG]][ 0] : !qref.reg<1> -> !qref.bit
        # CHECK:   qref.custom "Hadamard"() [[QUBIT]] : !qref.bit
        # CHECK: qref.compbasis(qreg [[QREG]] : !qref.reg<1>) : !quantum.obs
        @qp.for_loop(0, 2, 1)
        def loop_fn(_):
            qp.Hadamard(0)

        loop_fn()  # pylint: disable=no-value-for-parameter

        return qp.state()

    @qp.qjit(capture=True)
    def main():
        return captured_circuit()

    print(main.mlir)


test_loop_capture()


def test_while_capture():
    """Test a while loop"""

    @qp.qnode(qp.device("null.qubit", wires=1))
    def captured_circuit():
        # CHECK: [[QREG:%.+]] = qref.alloc( 1) : !qref.reg<1>
        # CHECK: [[QUBIT:%.+]] = qref.get [[QREG]][ 0] : !qref.reg<1> -> !qref.bit
        # CHECK: [[m:%.+]] = qref.measure [[QUBIT]] : i1
        _ = qp.measure(0)

        # CHECK: {{%.+}} = scf.while (%arg0 = {{%.+}}) : (tensor<i64>) -> tensor<i64> {
        # CHECK:   stablehlo.compare  LT
        # CHECK:   scf.condition({{%.+}}) %arg0 : tensor<i64>
        # CHECK: } do {
        # CHECK: ^bb0(%arg0: tensor<i64>):
        # CHECK:   [[QUBIT:%.+]] = qref.get [[QREG]][ 0] : !qref.reg<1> -> !qref.bit
        # CHECK:   qref.custom "Hadamard"() [[QUBIT]] : !qref.bit
        # CHECK:   [[add:%.+]] = stablehlo.add %arg0, {{%.+}} : tensor<i64>
        # CHECK:   scf.yield [[add]] : tensor<i64>
        # CHECK: }
        # CHECK: qref.compbasis(qreg [[QREG]] : !qref.reg<1>) : !quantum.obs
        def less_than_10(x):
            return x[0] < 10

        @qp.while_loop(less_than_10)
        def loop(v):
            qp.Hadamard(0)
            return v[0] + 1, v[1]

        loop((0, 1))
        return qp.state()

    @qp.qjit(capture=True)
    def main():
        return captured_circuit()

    print(main.mlir)


test_while_capture()


def test_dynamic_wire():
    """Test dynamic wires no re-insertion"""

    dev = qp.device("null.qubit", wires=3)

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(dev)
    def circuit(w1: int):

        # CHECK: [[QREG:%.+]] = qref.alloc( 3) : !qref.reg<3>
        # CHECK: [[q0:%.+]] = qref.get [[QREG]][ 0] : !qref.reg<3> -> !qref.bit
        # CHECK: qref.custom "PauliX"() [[q0]] : !qref.bit
        qp.X(0)

        # CHECK: [[SCALAR:%.+]] = tensor.extract %arg0[] : tensor<i64>
        # CHECK: [[q_w1:%.+]] = qref.get [[QREG]][[[SCALAR]]] : !qref.reg<3>, i64 -> !qref.bit
        # CHECK: qref.custom "PauliY"() [[q_w1]] : !qref.bit
        qp.Y(w1)

        # CHECK: [[SCALAR:%.+]] = tensor.extract %arg0[] : tensor<i64>
        # CHECK: [[q_w1:%.+]] = qref.get [[QREG]][[[SCALAR]]] : !qref.reg<3>, i64 -> !qref.bit
        # CHECK: qref.custom "PauliZ"() [[q_w1]] : !qref.bit
        qp.Z(w1)

        # CHECK: [[SCALAR:%.+]] = tensor.extract %arg0[] : tensor<i64>
        # CHECK: [[q_w1:%.+]] = qref.get [[QREG]][[[SCALAR]]] : !qref.reg<3>, i64 -> !qref.bit
        # CHECK: qref.custom "PauliX"() [[q_w1]] : !qref.bit
        qp.X(w1)

        # CHECK: [[SCALAR:%.+]] = tensor.extract %arg0[] : tensor<i64>
        # CHECK: [[q_w1:%.+]] = qref.get [[QREG]][[[SCALAR]]] : !qref.reg<3>, i64 -> !qref.bit
        # CHECK: qref.pcphase({{%.+}}, {{%.+}}) [[q_w1]] : !qref.bit
        qp.PCPhase(0.5, wires=w1, dim=1)

        return qp.state()

    print(circuit.mlir)


test_dynamic_wire()


def test_two_dynamic_CNOTs():
    """Test two dynamic CNOTs"""

    dev = qp.device("null.qubit", wires=3)

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(dev)
    def circuit(w1: int, w2: int):
        # CHECK: [[QREG:%.+]] = qref.alloc( 3) : !qref.reg<3>

        # CHECK: [[w1:%.+]] = tensor.extract %arg0[] : tensor<i64>
        # CHECK: [[q_w1:%.+]] = qref.get [[QREG]][[[w1]]] : !qref.reg<3>, i64 -> !qref.bit
        # CHECK: [[w2:%.+]] = tensor.extract %arg1[] : tensor<i64>
        # CHECK: [[q_w2:%.+]] = qref.get [[QREG]][[[w2]]] : !qref.reg<3>, i64 -> !qref.bit
        # CHECK: qref.custom "CNOT"() [[q_w1]], [[q_w2]] : !qref.bit, !qref.bit
        qp.CNOT(wires=[w1, w2])

        # CHECK: [[w1:%.+]] = tensor.extract %arg0[] : tensor<i64>
        # CHECK: [[q_w1:%.+]] = qref.get [[QREG]][[[w1]]] : !qref.reg<3>, i64 -> !qref.bit
        # CHECK: [[w2:%.+]] = tensor.extract %arg1[] : tensor<i64>
        # CHECK: [[q_w2:%.+]] = qref.get [[QREG]][[[w2]]] : !qref.reg<3>, i64 -> !qref.bit
        # CHECK: qref.custom "CNOT"() [[q_w1]], [[q_w2]] : !qref.bit, !qref.bit
        qp.CNOT(wires=[w1, w2])
        return qp.state()

    print(circuit.mlir)


test_two_dynamic_CNOTs()


def test_pass_application():
    """Application of pass decorators."""

    dev = qp.device("null.qubit", wires=1)

    @qp.qjit(capture=True, target="mlir")
    @qp.transforms.cancel_inverses
    @qp.transforms.merge_rotations
    @qp.qnode(dev)
    def circuit():
        return qp.probs()

    # CHECK: [[first_pass:%.+]] = transform.apply_registered_pass "merge-rotations"
    # CHECK-NEXT: transform.apply_registered_pass "cancel-inverses" to [[first_pass]]

    print(circuit.mlir)


test_pass_application()


def test_pass_decomposition():
    """Application of pass decorator with decomposition."""

    dev = qp.device("null.qubit", wires=1)

    qp.decomposition.enable_graph()

    @qp.qjit(capture=True, target="mlir")
    @qp.transforms.cancel_inverses
    @qp.transforms.merge_rotations
    @partial(qp.transforms.decompose, gate_set={"RX", "RZ"})
    @qp.qnode(dev)
    def circuit1():
        return qp.probs()

    # CHECK: [[first_pass:%.+]] = transform.apply_registered_pass "decompose-lowering"
    # CHECK-NEXT: [[second_pass:%.+]] = transform.apply_registered_pass "merge-rotations"
    # CHECK-NEXT: transform.apply_registered_pass "cancel-inverses" to [[second_pass]]

    print(circuit1.mlir)

    @qp.qjit(capture=True, target="mlir")
    @qp.transforms.cancel_inverses
    @partial(qp.transforms.decompose, gate_set={"RX", "RZ"})
    @qp.transforms.merge_rotations
    @qp.qnode(dev)
    def circuit2():
        return qp.probs()

    # CHECK: [[first_pass:%.+]] = transform.apply_registered_pass "merge-rotations"
    # CHECK-NEXT: [[second_pass:%.+]] = transform.apply_registered_pass "decompose-lowering"
    # CHECK-NEXT: transform.apply_registered_pass "cancel-inverses" to [[second_pass]]

    print(circuit2.mlir)

    @qp.qjit(capture=True, target="mlir")
    @partial(qp.transforms.decompose, gate_set={"RX", "RZ"})
    @qp.transforms.cancel_inverses
    @qp.transforms.merge_rotations
    @qp.qnode(dev)
    def circuit3():
        return qp.probs()

    # CHECK: [[first_pass:%.+]] = transform.apply_registered_pass "merge-rotations"
    # CHECK-NEXT: [[second_pass:%.+]] = transform.apply_registered_pass "cancel-inverses"
    # CHECK-NEXT: transform.apply_registered_pass "decompose-lowering" to [[second_pass]]

    print(circuit3.mlir)

    qp.decomposition.disable_graph()


test_pass_decomposition()


def test_two_qnodes_with_different_passes_in_one_workflow():
    """Two qnodes with different passes in one workflow."""

    dev = qp.device("null.qubit", wires=1)

    @qp.qjit(capture=True, target="mlir")
    def workflow():
        @qp.transforms.merge_rotations
        @qp.qnode(dev)
        def circuit1():
            return qp.probs()

        @qp.transforms.cancel_inverses
        @qp.qnode(dev)
        def circuit2():
            return qp.probs()

        return circuit1() + circuit2()

    # CHECK: module @module_circuit1 {
    # CHECK: transform.apply_registered_pass "merge-rotations"
    # CHECK: module @module_circuit2 {
    # CHECK: transform.apply_registered_pass "cancel-inverses"

    print(workflow.mlir)


test_two_qnodes_with_different_passes_in_one_workflow()
