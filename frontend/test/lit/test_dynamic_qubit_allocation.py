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

import pennylane as qml

from catalyst import qjit
from catalyst.jax_primitives import qalloc_p, qdealloc_qb_p, qextract_p


@qjit(target="mlir")
def test_single_qubit_dealloc():
    """
    Unit test for the single qubit dealloc primitive's lowerings.
    """

    # CHECK: [[qubit:.]]:AbstractQbit() = qextract {{.+}} 3
    # CHECK: qdealloc_qb [[qubit]]

    # CHECK: [[qubit:%.+]] = quantum.extract {{.+}} 3
    # CHECK: quantum.dealloc_qb [[qubit]] : !quantum.bit

    qreg = qalloc_p.bind(10)
    qubit = qextract_p.bind(qreg, 3)
    qdealloc_qb_p.bind(qubit)


print(test_single_qubit_dealloc.jaxpr)
print(test_single_qubit_dealloc.mlir)


qml.capture.enable()


@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def test_basic_dynalloc():
    """
    Test basic qml.allocate and qml.deallocate.

    Test both the explicit call API and the context manager API.
    """

    # CHECK: [[device_init_qreg:%.+]] = quantum.alloc( 3)

    # CHECK: [[dyn_qreg:%.+]] = quantum.alloc( 2)
    # CHECK: [[dyn_bit0:%.+]] = quantum.extract [[dyn_qreg]][ 0]
    # CHECK: [[dyn_bit1:%.+]] = quantum.extract [[dyn_qreg]][ 1]
    # CHECK: [[Xout:%.+]] = quantum.custom "PauliX"() [[dyn_bit0]]
    # CHECK: [[dev_bit2:%.+]] = quantum.extract [[device_init_qreg]][ 2]
    # CHECK: [[CNOTout:%.+]]:2 = quantum.custom "CNOT"() [[dyn_bit1]], [[dev_bit2]]
    # CHECK: [[insert0:%.+]] = quantum.insert [[dyn_qreg]][ 0], [[Xout]]
    # CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[CNOTout]]#0
    # CHECK: quantum.dealloc [[insert1]]

    qs = qml.allocate(2)
    qml.X(qs[0])
    qml.CNOT(wires=[qs[1], 2])
    qml.deallocate(qs[:])

    # CHECK: [[dyn_qreg:%.+]] = quantum.alloc( 4)
    # CHECK: [[dyn_bit1:%.+]] = quantum.extract [[dyn_qreg]][ 1]
    # CHECK: [[dyn_bit2:%.+]] = quantum.extract [[dyn_qreg]][ 2]
    # CHECK: [[Xout:%.+]] = quantum.custom "PauliX"() [[dyn_bit1]]
    # CHECK: [[dev_bit1:%.+]] = quantum.extract [[device_init_qreg]][ 1]
    # CHECK: [[CNOTout:%.+]]:2 = quantum.custom "CNOT"() [[dyn_bit2]], [[dev_bit1]]
    # CHECK: [[insert0:%.+]] = quantum.insert [[dyn_qreg]][ 1], [[Xout]]
    # CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 2], [[CNOTout]]#0
    # CHECK: [[insert2:%.+]] = quantum.insert [[insert1]][ 3]
    # CHECK: quantum.dealloc [[insert2]]

    with qml.allocate(4) as qs1:
        qml.X(qs1[1])
        qml.CNOT(wires=[qs1[2], 1])

    return qml.probs(wires=[0])


print(test_basic_dynalloc.mlir)


@qjit(autograph=True, target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def test_measure_with_reset():
    """
    Test qml.allocate with qml.measure with a reset.
    """

    # CHECK: [[device_init_qreg:%.+]] = quantum.alloc( 3)

    # CHECK: [[dyn_qreg:%.+]] = quantum.alloc( 1)
    # CHECK: [[dyn_qubit:%.+]] = quantum.extract [[dyn_qreg]][ 0]
    # CHECK: [[mres:%.+]], [[mout_qubit:%.+]] = quantum.measure [[dyn_qubit]] postselect 1
    # CHECK: [[reset_qubit:%.+]] = scf.if [[mres]] -> (!quantum.bit) {
    # CHECK:    [[x_out_qubit:%.+]] = quantum.custom "PauliX"() [[mout_qubit]]
    # CHECK:    scf.yield [[x_out_qubit]] : !quantum.bit
    # CHECK:  } else {
    # CHECK:    scf.yield [[mout_qubit]] : !quantum.bit
    # CHECK:  }
    # CHECK:  [[dealloc_qreg:%.+]] = quantum.insert [[dyn_qreg]][ 0], [[reset_qubit]]
    # CHECK:  quantum.dealloc [[dealloc_qreg]]

    with qml.allocate(1) as q:
        qml.measure(wires=q[0], reset=True, postselect=1)

    return qml.probs(wires=[0, 1, 2])


print(test_measure_with_reset.mlir)


@qjit(autograph=True, target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def test_pass_reg_into_forloop():
    """
    Test using a dynamically allocated resgister from inside a subscope.
    """

    # CHECK: [[global_reg:%.+]] = quantum.alloc( 2)
    # CHECK: [[dyn_reg:%.+]] = quantum.alloc( 1)
    # CHECK: [[for_out:%.+]]:2 = scf.for %arg0 = {{.+}} to {{.+}} step {{.+}} iter_args
    # CHECK-SAME: (%arg1 = [[dyn_reg]], %arg2 = [[global_reg]]) -> (!quantum.reg, !quantum.reg) {
    # CHECK:    [[x_in:%.+]] = quantum.extract %arg1[ 0]
    # CHECK:    [[x_out:%.+]] = quantum.custom "PauliX"() [[x_in]]
    # CHECK:    [[cnot_in:%.+]] = quantum.extract %arg2[ 0]
    # CHECK:    [[cnot_out:%.+]]:2 = quantum.custom "CNOT"() [[x_out]], [[cnot_in]]
    # CHECK:    [[global_reg_yield:%.+]] = quantum.insert %arg2[ 0], [[cnot_out]]#1
    # CHECK:    [[dyn_reg_yield:%.+]] = quantum.insert %arg1[ 0], [[cnot_out]]#0
    # CHECK:    scf.yield [[dyn_reg_yield]], [[global_reg_yield]] : !quantum.reg, !quantum.reg
    # CHECK: quantum.dealloc [[for_out]]#0 : !quantum.reg

    with qml.allocate(1) as q:
        for _ in range(3):
            qml.X(wires=q[0])
            qml.CNOT(wires=[q[0], 0])

    # CHECK: [[global_bit0:%.+]] = quantum.extract [[for_out]]#1[ 0]
    # CHECK: [[global_bit1:%.+]] = quantum.extract [[for_out]]#1[ 1]
    # CHECK: [[obs:%.+]] = quantum.compbasis qubits [[global_bit0]], [[global_bit1]] : !quantum.obs
    # CHECK: {{.+}} = quantum.probs [[obs]] : tensor<4xf64>
    return qml.probs(wires=[0, 1])


print(test_pass_reg_into_forloop.mlir)


@qjit(autograph=True, target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def test_pass_multiple_regs_into_forloop():
    """
    Test using multiple dynamically allocated resgisters from inside a subscope.
    """

    # CHECK: [[global_reg:%.+]] = quantum.alloc( 3)
    # CHECK: [[q1:%.+]] = quantum.alloc( 1)
    # CHECK: [[q2:%.+]] = quantum.alloc( 2)
    # CHECK: [[for_out:%.+]]:3 = scf.for %arg0 = {{.+}} to {{.+}} step {{.+}} iter_args
    # CHECK-SAME: (%arg1 = [[q1]], %arg2 = [[q2]], %arg3 = [[global_reg]]) -> (!quantum.reg, !quantum.reg, !quantum.reg) {
    # CHECK:    [[q1_0:%.+]] = quantum.extract %arg1[ 0]
    # CHECK:    [[glob_0:%.+]] = quantum.extract %arg3[ 0]
    # CHECK:    [[cnot_out0:%.+]]:2 = quantum.custom "CNOT"() [[q1_0]], [[glob_0]]
    # CHECK:    [[q2_1:%.+]] = quantum.extract %arg2[ 1]
    # CHECK:    [[glob_1:%.+]] = quantum.extract %arg3[ 1]
    # CHECK:    [[cnot_out1:%.+]]:2 = quantum.custom "CNOT"() [[q2_1]], [[glob_1]]
    # CHECK:    [[glob_ins:%.+]] = quantum.insert %arg3[ 0], [[cnot_out0]]#1
    # CHECK:    [[glob_yield:%.+]] = quantum.insert [[glob_ins]][ 1], [[cnot_out1]]#1
    # CHECK:    [[q1_yield:%.+]] = quantum.insert %arg1[ 0], [[cnot_out0]]#0
    # CHECK:    [[q2_yield:%.+]] = quantum.insert %arg2[ 1], [[cnot_out1]]#0
    # CHECK:    scf.yield [[q1_yield]], [[q2_yield]], [[glob_yield]] : !quantum.reg, !quantum.reg, !quantum.reg
    # CHECK:  quantum.dealloc [[for_out]]#1 : !quantum.reg
    # CHECK:  quantum.dealloc [[for_out]]#0 : !quantum.reg

    with qml.allocate(1) as q1:
        with qml.allocate(2) as q2:
            for _ in range(3):
                qml.CNOT(wires=[q1[0], 0])
                qml.CNOT(wires=[q2[1], 1])

    return qml.probs(wires=[0, 1])


print(test_pass_multiple_regs_into_forloop.mlir)


qml.capture.disable()
