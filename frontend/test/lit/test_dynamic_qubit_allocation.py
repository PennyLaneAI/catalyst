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

"""
Unit tests for the dynamic qubit allocation.
"""

# RUN: %PYTHON %s | FileCheck %s

import pennylane as qp

from catalyst import qjit


@qjit(target="mlir", capture=True)
@qp.qnode(qp.device("lightning.qubit", wires=3))
def test_basic_dynalloc():
    """
    Test basic qp.allocate and qp.deallocate.

    Test both the explicit call API and the context manager API.
    """

    # CHECK: [[device_init_qreg:%.+]] = qref.alloc( 3)

    # CHECK: [[dyn_qreg:%.+]] = qref.alloc( 2)
    # CHECK: [[dyn_bit0:%.+]] = qref.get [[dyn_qreg]][ 0]
    # CHECK: [[dyn_bit1:%.+]] = qref.get [[dyn_qreg]][ 1]
    # CHECK: qref.custom "PauliX"() [[dyn_bit0]]
    # CHECK: [[dev_bit2:%.+]] = qref.get [[device_init_qreg]][ 2]
    # CHECK: qref.custom "CNOT"() [[dyn_bit1]], [[dev_bit2]]
    # CHECK: qref.dealloc [[dyn_qreg]]

    qs = qp.allocate(2)
    qp.X(qs[0])
    qp.CNOT(wires=[qs[1], 2])
    qp.deallocate(qs[:])

    # CHECK: [[dyn_qreg:%.+]] = qref.alloc( 4)
    # CHECK: [[dyn_bit1:%.+]] = qref.get [[dyn_qreg]][ 1]
    # CHECK: [[dyn_bit2:%.+]] = qref.get [[dyn_qreg]][ 2]
    # CHECK: qref.custom "PauliX"() [[dyn_bit1]]
    # CHECK: [[dev_bit1:%.+]] = qref.get [[device_init_qreg]][ 1]
    # CHECK: qref.custom "CNOT"() [[dyn_bit2]], [[dev_bit1]]
    # CHECK: qref.dealloc [[dyn_qreg]]

    with qp.allocate(4) as qs1:
        qp.X(qs1[1])
        qp.CNOT(wires=[qs1[2], 1])

    return qp.probs(wires=[0])


print(test_basic_dynalloc.mlir)


@qjit(autograph=True, target="mlir", capture=True)
@qp.qnode(qp.device("lightning.qubit", wires=3))
def test_measure_with_reset():
    """
    Test qp.allocate with qp.measure with a reset.
    """

    # CHECK: [[device_init_qreg:%.+]] = qref.alloc( 3)

    # CHECK: [[dyn_qreg:%.+]] = qref.alloc( 1)
    # CHECK: [[dyn_qubit:%.+]] = qref.get [[dyn_qreg]][ 0]
    # CHECK: [[mres:%.+]] = qref.measure [[dyn_qubit]] postselect 1
    # CHECK: scf.if [[mres]] {
    # CHECK:    qref.custom "PauliX"() [[dyn_qubit]]
    # CHECK:  }
    # CHECK:  qref.dealloc [[dyn_qreg]]

    with qp.allocate(1) as q:
        qp.measure(wires=q[0], reset=True, postselect=1)

    return qp.probs(wires=[0, 1, 2])


print(test_measure_with_reset.mlir)


@qjit(autograph=True, target="mlir", capture=True)
@qp.qnode(qp.device("lightning.qubit", wires=2))
def test_pass_reg_into_forloop():
    """
    Test using a dynamically allocated resgister from inside a subscope.
    """

    # CHECK: [[global_reg:%.+]] = qref.alloc( 2)
    # CHECK: [[dyn_reg:%.+]] = qref.alloc( 1)
    # CHECK: [[dyn_qubit:%.+]] = qref.get [[dyn_reg]][ 0]
    # CHECK: scf.for %arg0 = {{.+}} to {{.+}} step {{.+}} {
    # CHECK:    qref.custom "PauliX"() [[dyn_qubit]]
    # CHECK:    [[q0:%.+]] = qref.get [[global_reg]][ 0]
    # CHECK:    qref.custom "CNOT"() [[dyn_qubit]], [[q0]]
    # CHECK: qref.dealloc [[dyn_reg]] : !qref.reg<1>

    with qp.allocate(1) as q:
        for _ in range(3):
            qp.X(wires=q[0])
            qp.CNOT(wires=[q[0], 0])

    # CHECK: [[global_bit0:%.+]] = qref.get [[global_reg]][ 0]
    # CHECK: [[global_bit1:%.+]] = qref.get [[global_reg]][ 1]
    # CHECK: [[obs:%.+]] = qref.compbasis qubits [[global_bit0]], [[global_bit1]] : !quantum.obs
    # CHECK: {{.+}} = quantum.probs [[obs]] : tensor<4xf64>
    return qp.probs(wires=[0, 1])


print(test_pass_reg_into_forloop.mlir)


@qjit(autograph=True, target="mlir", capture=True)
@qp.qnode(qp.device("lightning.qubit", wires=3))
def test_pass_multiple_regs_into_forloop():
    """
    Test using multiple dynamically allocated resgisters from inside a subscope.
    """

    # CHECK: [[global_reg:%.+]] = qref.alloc( 3)
    # CHECK: [[reg1:%.+]] = qref.alloc( 1)
    # CHECK: [[q1_0:%.+]] = qref.get [[reg1]][ 0]
    # CHECK: [[reg2:%.+]] = qref.alloc( 2)
    # CHECK: [[q2_1:%.+]] = qref.get [[reg2]][ 1]
    # CHECK: scf.for %arg0 = {{.+}} to {{.+}} step {{.+}} {
    # CHECK:    [[glob_0:%.+]] = qref.get [[global_reg]][ 0]
    # CHECK:    qref.custom "CNOT"() [[q1_0]], [[glob_0]]
    # CHECK:    [[glob_1:%.+]] = qref.get [[global_reg]][ 1]
    # CHECK:    qref.custom "CNOT"() [[q2_1]], [[glob_1]]
    # CHECK:  qref.dealloc [[reg2]] : !qref.reg<2>
    # CHECK:  qref.dealloc [[reg1]] : !qref.reg<1>

    with qp.allocate(1) as q1:
        with qp.allocate(2) as q2:
            for _ in range(3):
                qp.CNOT(wires=[q1[0], 0])
                qp.CNOT(wires=[q2[1], 1])

    return qp.probs(wires=[0, 1])


print(test_pass_multiple_regs_into_forloop.mlir)


@qjit(autograph=True, target="mlir", capture=True)
@qp.qnode(qp.device("lightning.qubit", wires=2))
def test_pass_multiple_regs_into_whileloop(N: int):
    """
    Test using multiple dynamically allocated resgisters from inside a while loop.
    """

    # CHECK:  [[global_reg:%.+]] = qref.alloc( 2)
    # CHECK:  [[reg1:%.+]] = qref.alloc( 1)
    # CHECK:  [[q1_0:%.+]] = qref.get [[reg1]][ 0]
    # CHECK:  [[reg2:%.+]] = qref.alloc( 4)
    # CHECK:  [[q2_0:%.+]] = qref.get [[reg2]][ 0]
    # CHECK:  [[i:%.+]] = scf.while (%arg1 = {{%.+}}) : (tensor<i64>) -> tensor<i64> {
    # CHECK:    stablehlo.compare  LT, %arg1, %arg0
    # CHECK:    scf.condition({{%.+}}) %arg1 : tensor<i64>
    # CHECK:  } do {
    # CHECK:  ^bb0(%arg1: tensor<i64>):
    # CHECK:    [[glob_1:%.+]] = qref.get [[global_reg]][ 1]
    # CHECK:    qref.custom "CNOT"() [[q1_0]], [[glob_1]]
    # CHECK:    [[glob_1:%.+]] = qref.get [[global_reg]][ 1]
    # CHECK:    qref.custom "CNOT"() [[q2_0]], [[glob_1]]
    # CHECK:    [[i:%.+]] = stablehlo.add %arg1, {{%.+}}
    # CHECK:    scf.yield [[i]] : tensor<i64>
    # CHECK:  }
    # CHECK:  qref.dealloc [[reg2]] : !qref.reg<4>
    # CHECK:  qref.dealloc [[reg1]] : !qref.reg<1>

    i = 0
    with qp.allocate(1) as q1:
        with qp.allocate(4) as q2:
            while i < N:
                qp.CNOT(wires=[q1[0], 1])
                qp.CNOT(wires=[q2[0], 1])
                i += 1

    return qp.probs(wires=[0, 1])


print(test_pass_multiple_regs_into_whileloop.mlir)


@qjit(target="mlir", capture=True)
@qp.qnode(qp.device("lightning.qubit", wires=2))
def test_magic_state_fabricate():
    """
    Test that magic state allocation lowers to pbc.ref.fabricate.
    """

    # CHECK: [[magic:%.+]] = pbc.ref.fabricate magic : !qref.bit
    # CHECK: qref.custom "PauliX"() [[magic]]
    # CHECK: qref.dealloc_qb [[magic]]

    with qp.allocate(1, state="magic") as q:
        qp.X(q[0])

    return qp.probs(wires=[0])


print(test_magic_state_fabricate.mlir)


# pylint: disable=line-too-long
def test_quantum_subroutine():
    """
    Test passing dynamically allocated wires into a quantum subroutine.
    """

    @qp.capture.subroutine
    def flip(w1, w2, w3, theta):
        qp.X(w1)
        qp.Y(w2)
        qp.Z(w3)
        qp.ctrl(qp.RX, (w1, w2))(theta, wires=0)

    # CHECK:  [[angle:%.+]] = stablehlo.constant dense<1.230000e+00>
    # CHECK:  [[global_qreg:%.+]] = qref.alloc( 1)
    # CHECK:  [[reg1:%.+]] = qref.alloc( 2)
    # CHECK:  [[q1_0:%.+]] = qref.get [[reg1]][ 0] : !qref.reg<2> -> !qref.bit
    # CHECK:  [[q1_1:%.+]] = qref.get [[reg1]][ 1] : !qref.reg<2> -> !qref.bit
    # CHECK:  [[reg2:%.+]] = qref.alloc( 3)
    # CHECK:  [[q2_2:%.+]] = qref.get [[reg2]][ 2] : !qref.reg<3> -> !qref.bit
    # CHECK:  call @flip([[global_qreg]], [[q1_0]], [[q1_1]], [[q2_2]], [[angle]])
    # CHECK-SAME: (!qref.reg<1>, !qref.bit, !qref.bit, !qref.bit, tensor<f64>) -> ()

    @qjit(target="mlir", capture=True)
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def circuit():
        with qp.allocate(2) as q1:
            with qp.allocate(3) as q2:
                flip(q1[0], q1[1], q2[2], 1.23)
        return qp.probs(wires=[0])

    # CHECK: func.func private @flip(%arg0: !qref.reg<1>, %arg1: !qref.bit, %arg2: !qref.bit, %arg3: !qref.bit, %arg4: tensor<f64>)
    # CHECK:   [[true:%.+]] = arith.constant true
    # CHECK:   qref.custom "PauliX"() %arg1 : !qref.bit
    # CHECK:   qref.custom "PauliY"() %arg2 : !qref.bit
    # CHECK:   qref.custom "PauliZ"() %arg3 : !qref.bit
    # CHECK:   [[glob_0:%.+]] = qref.get %arg0[ 0] : !qref.reg<1> -> !qref.bit
    # CHECK:   [[angle:%.+]] = tensor.extract %arg4[] : tensor<f64>
    # CHECK:   qref.custom "RX"([[angle]]) [[glob_0]] ctrls(%arg1, %arg2) ctrlvals([[true]], [[true]]) : !qref.bit ctrls !qref.bit, !qref.bit
    # CHECK:   return

    print(circuit.mlir)


test_quantum_subroutine()
