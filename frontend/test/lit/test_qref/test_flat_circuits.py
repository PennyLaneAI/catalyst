# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Unit tests for lowering gate-like primitives to reference semantics MLIR during PLxPR conversion.
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable=line-too-long

import numpy as np
import pennylane as qp


# CHECK: func.func public @test_custom_op(%arg0: tensor<i64>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_custom_op(i: int):
    """
    Test custom op.
    """
    # CHECK-DAG: [[false:%.+]] = arith.constant false
    # CHECK-DAG: [[true:%.+]] = arith.constant true
    # CHECK-DAG: [[angle:%.+]] = arith.constant 1.000000e-01 : f64

    # CHECK: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: qref.custom "PauliX"() [[q0]] : !qref.bit
    qp.X(0)

    # CHECK: [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<3> -> !qref.bit
    # CHECK: qref.custom "RZ"([[angle]]) [[q1]] : !qref.bit
    qp.RZ(0.1, wires=1)

    # CHECK: [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<3> -> !qref.bit
    # CHECK: [[q2:%.+]] = qref.get [[reg]][ 2] : !qref.reg<3> -> !qref.bit
    # CHECK: qref.custom "SWAP"() [[q1]], [[q2]] : !qref.bit, !qref.bit
    qp.SWAP([1, 2])

    # CHECK: [[i:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK: [[qi:%.+]] = qref.get [[reg]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: qref.custom "RX"([[angle]]) [[qi]] ctrls([[q0]]) ctrlvals([[true]]) : !qref.bit ctrls !qref.bit
    qp.ctrl(qp.RX, control=0)(0.1, wires=[i])

    # CHECK: [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<3> -> !qref.bit
    # CHECK: [[q2:%.+]] = qref.get [[reg]][ 2] : !qref.reg<3> -> !qref.bit
    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: qref.custom "CNOT"() [[q1]], [[q2]] ctrls([[q0]]) ctrlvals([[false]]) : !qref.bit, !qref.bit ctrls !qref.bit
    qp.ctrl(qp.CNOT, control=0, control_values=False)(wires=[1, 2])

    return qp.expval(qp.X(0))


print(test_custom_op.mlir)


# CHECK: func.func public @test_measure() -> tensor<2xf64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_measure():
    """
    Test measure.
    """
    # CHECK: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[mres:%.+]] = qref.measure [[q0]] : i1
    # CHECK: [[mres_tensori1:%.+]] = tensor.from_elements [[mres]] : tensor<i1>
    m = qp.measure(0)

    # CHECK: [[mres_tensori64:%.+]] = stablehlo.convert [[mres_tensori1]] : (tensor<i1>) -> tensor<i64>
    # CHECK: [[mres_tensori1:%.+]] = stablehlo.convert [[mres_tensori64]] : (tensor<i64>) -> tensor<i1>
    # CHECK: [[mres_i1:%.+]] = tensor.extract [[mres_tensori1]][] : tensor<i1>
    # CHECK: [[mcmobs:%.+]] = quantum.mcmobs [[mres_i1]] : !quantum.obs
    # CHECK: [[probs:%.+]] = quantum.probs [[mcmobs]] : tensor<2xf64>
    return qp.probs(op=m)


print(test_measure.mlir)


# CHECK: func.func public @test_dynamic_qubit_allocation(%arg0: tensor<2x2xf64>, %arg1: tensor<i64>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_dynamic_qubit_allocation(i: int):
    """
    Test dynamic qubit allocation.
    """
    # CHECK-DAG: [[false:%.+]] = arith.constant false

    # CHECK: [[reg_device:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: [[reg_alloc:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK: [[alloc_q0:%.+]] = qref.get [[reg_alloc]][ 0] : !qref.reg<2> -> !qref.bit
    # CHECK: [[alloc_q1:%.+]] = qref.get [[reg_alloc]][ 1] : !qref.reg<2> -> !qref.bit
    with qp.allocate(2) as q:
        # CHECK: qref.custom "PauliX"() [[alloc_q0]] : !qref.bit
        qp.X(q[0])

        # CHECK: [[i:%.+]] = tensor.extract %arg1[] : tensor<i64>
        # CHECK: [[qi:%.+]] = qref.get [[reg_device]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
        # CHECK: [[q2:%.+]] = qref.get [[reg_device]][ 2] : !qref.reg<3> -> !qref.bit
        # CHECK: qref.custom "CNOT"() [[qi]], [[q2]] ctrls([[alloc_q1]]) ctrlvals([[false]]) : !qref.bit, !qref.bit ctrls !qref.bit
        qp.ctrl(qp.CNOT, control=q[1], control_values=False)(wires=[i, 2])

        # CHECK: qref.multirz({{%.+}}) [[alloc_q1]] : !qref.bit
        qp.MultiRZ(0.1, wires=q[1])

        # CHECK: qref.pcphase({{%.+}}, {{%.+}}) [[alloc_q0]], [[alloc_q1]] : !qref.bit, !qref.bit
        qp.PCPhase(0.1, dim=0, wires=[q[0], q[1]])

        # CHECK: qref.gphase({{%.+}}) ctrls([[alloc_q0]]) ctrlvals({{%.+}}) : ctrls !qref.bit
        qp.ctrl(qp.GlobalPhase(np.pi / 4), control=[q[0]])

        # CHECK: qref.paulirot ["X"]({{%.+}}) [[alloc_q0]] : !qref.bit
        qp.PauliRot(0.1, ["X"], wires=q[0])

        # CHECK: qref.unitary({{%.+}} : tensor<2x2xcomplex<f64>>) [[alloc_q1]] : !qref.bit
        qp.QubitUnitary(np.identity(2), wires=q[1])

        # CHECK: {{%.+}} = qref.measure [[alloc_q0]] : i1
        qp.measure(q[0])

    # CHECK: qref.dealloc [[reg_alloc]] : !qref.reg<2>

    # CHECK: [[q0:%.+]] = qref.get [[reg_device]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: qref.custom "PauliX"() [[q0]] : !qref.bit
    qp.X(0)

    return qp.expval(qp.X(0))


print(test_dynamic_qubit_allocation.mlir)


# CHECK: func.func public @test_multirz() -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_multirz():
    """
    Test multirz.
    """
    # CHECK-DAG: [[true:%.+]] = arith.constant true
    # CHECK-DAG: [[angle:%.+]] = arith.constant 1.000000e-01 : f64

    # CHECK: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<3> -> !qref.bit
    # CHECK: qref.multirz([[angle]]) [[q1]] : !qref.bit
    qp.MultiRZ(0.1, wires=1)

    # CHECK: [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<3> -> !qref.bit
    # CHECK: [[q2:%.+]] = qref.get [[reg]][ 2] : !qref.reg<3> -> !qref.bit
    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: qref.multirz([[angle]]) [[q1]], [[q2]] ctrls([[q0]]) ctrlvals([[true]]) : !qref.bit, !qref.bit ctrls !qref.bit
    qp.ctrl(qp.MultiRZ, control=0, control_values=True)(0.1, wires=[1, 2])

    return qp.expval(qp.X(0))


print(test_multirz.mlir)


# CHECK: func.func public @test_pcphase() -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_pcphase():
    """
    Test pcphase.
    """
    # CHECK-DAG: [[true:%.+]] = arith.constant true
    # CHECK-DAG: [[stablehlo_two:%.+]] = stablehlo.constant dense<2> : tensor<i64>
    # CHECK-DAG: [[stablehlo_one:%.+]] = stablehlo.constant dense<1> : tensor<i64>
    # CHECK-DAG: [[angle:%.+]] = arith.constant 1.000000e-01 : f64

    # CHECK: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<3> -> !qref.bit
    # CHECK: [[q2:%.+]] = qref.get [[reg]][ 2] : !qref.reg<3> -> !qref.bit
    # CHECK: [[_two:%.+]] = stablehlo.convert [[stablehlo_two]] : (tensor<i64>) -> tensor<f64>
    # CHECK: [[dim_2:%.+]] = tensor.extract [[_two]][] : tensor<f64>
    # CHECK: qref.pcphase([[angle]], [[dim_2]]) [[q0]], [[q1]], [[q2]] : !qref.bit, !qref.bit, !qref.bit
    qp.PCPhase(0.1, dim=2, wires=[0, 1, 2])

    # CHECK: [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<3> -> !qref.bit
    # CHECK: [[q2:%.+]] = qref.get [[reg]][ 2] : !qref.reg<3> -> !qref.bit
    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[_one:%.+]] = stablehlo.convert [[stablehlo_one]] : (tensor<i64>) -> tensor<f64>
    # CHECK: [[dim_1:%.+]] = tensor.extract [[_one]][] : tensor<f64>
    # CHECK: qref.pcphase([[angle]], [[dim_1]]) [[q1]], [[q2]] ctrls([[q0]]) ctrlvals([[true]]) : !qref.bit, !qref.bit ctrls !qref.bit
    qp.ctrl(qp.PCPhase, control=0, control_values=True)(0.1, dim=1, wires=[1, 2])

    return qp.expval(qp.X(0))


print(test_pcphase.mlir)


# CHECK: func.func public @test_pauli_rot() -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=4))
def test_pauli_rot():
    """
    Test paulirot.
    """
    # CHECK-DAG: [[true:%.+]] = arith.constant true
    # CHECK-DAG: [[false:%.+]] = arith.constant false
    # CHECK-DAG: [[angle:%.+]] = arith.constant 1.000000e-01 : f64

    # CHECK: [[reg:%.+]] = qref.alloc( 4) : !qref.reg<4>

    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<4> -> !qref.bit
    # CHECK: qref.paulirot ["X"]([[angle]]) [[q0]] : !qref.bit
    qp.PauliRot(0.1, ["X"], wires=[0])

    # CHECK: [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<4> -> !qref.bit
    # CHECK: [[q2:%.+]] = qref.get [[reg]][ 2] : !qref.reg<4> -> !qref.bit
    # CHECK: qref.paulirot ["Y", "I"]([[angle]]) [[q1]], [[q2]] : !qref.bit, !qref.bit
    qp.PauliRot(0.1, ["Y", "I"], wires=[1, 2])

    # CHECK: [[q2:%.+]] = qref.get [[reg]][ 2] : !qref.reg<4> -> !qref.bit
    # CHECK: [[q3:%.+]] = qref.get [[reg]][ 3] : !qref.reg<4> -> !qref.bit
    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<4> -> !qref.bit
    # CHECK: [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<4> -> !qref.bit
    # CHECK: qref.paulirot ["Z", "X"]([[angle]]) [[q2]], [[q3]] ctrls([[q0]], [[q1]]) ctrlvals([[true]], [[false]]) : !qref.bit, !qref.bit ctrls !qref.bit, !qref.bit
    qp.ctrl(
        qp.PauliRot(0.1, ["Z", "X"], wires=[2, 3]), control=[0, 1], control_values=[True, False]
    )

    return qp.expval(qp.X(0))


print(test_pauli_rot.mlir)


# CHECK: func.func public @test_global_phase() -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=4))
def test_global_phase():
    """
    Test global phase.
    """
    # CHECK-DAG: [[true:%.+]] = arith.constant true
    # CHECK-DAG: [[angle:%.+]] = arith.constant 0.78539816339744828 : f64

    # CHECK: [[reg:%.+]] = qref.alloc( 4) : !qref.reg<4>

    # CHECK: qref.gphase([[angle]])
    qp.GlobalPhase(np.pi / 4)

    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<4> -> !qref.bit
    # CHECK: qref.gphase([[angle]]) ctrls([[q0]]) ctrlvals([[true]]) : ctrls !qref.bit
    qp.ctrl(qp.GlobalPhase(np.pi / 4), control=[0])

    return qp.expval(qp.X(0))


print(test_global_phase.mlir)


# CHECK: func.func public @test_unitary(%arg0: tensor<2x2xf64>, %arg1: tensor<4x4xf64>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=4))
def test_unitary():
    """
    Test unitary.
    """
    # CHECK-DAG: [[true:%.+]] = arith.constant true

    # CHECK: [[reg:%.+]] = qref.alloc( 4) : !qref.reg<4>

    # CHECK: [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<4> -> !qref.bit
    # CHECK: [[mat2:%.+]] = stablehlo.convert %arg0 : (tensor<2x2xf64>) -> tensor<2x2xcomplex<f64>>
    # CHECK: qref.unitary([[mat2]] : tensor<2x2xcomplex<f64>>) [[q1]] : !qref.bit
    qp.QubitUnitary(np.identity(2), wires=[1])

    # CHECK: [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<4> -> !qref.bit
    # CHECK: [[q2:%.+]] = qref.get [[reg]][ 2] : !qref.reg<4> -> !qref.bit
    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<4> -> !qref.bit
    # CHECK: [[mat4:%.+]] = stablehlo.convert %arg1 : (tensor<4x4xf64>) -> tensor<4x4xcomplex<f64>>
    # CHECK: qref.unitary([[mat4]] : tensor<4x4xcomplex<f64>>) [[q1]], [[q2]] ctrls([[q0]]) ctrlvals([[true]]) : !qref.bit, !qref.bit ctrls !qref.bit
    qp.ctrl(qp.QubitUnitary(np.identity(4), wires=[1, 2]), control=[0])

    return qp.expval(qp.X(0))


print(test_unitary.mlir)


# CHECK: func.func public @test_set_state(%arg0: tensor<4xi64>, %arg1: tensor<4xi64>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=4))
def test_set_state():
    """
    Test set_state.
    """
    # CHECK: [[reg:%.+]] = qref.alloc( 4) : !qref.reg<4>

    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<4> -> !qref.bit
    # CHECK: [[q2:%.+]] = qref.get [[reg]][ 2] : !qref.reg<4> -> !qref.bit
    # CHECK: [[state:%.+]] = stablehlo.convert %arg0 : (tensor<4xi64>) -> tensor<4xcomplex<f64>>
    # CHECK: qref.set_state([[state]]) [[q0]], [[q2]] : tensor<4xcomplex<f64>>, !qref.bit, !qref.bit
    qp.StatePrep(np.array([0, 0, 1, 0]), wires=[0, 2])

    # CHECK: [[reg_alloc:%.+]] = qref.alloc( 1) : !qref.reg<1>
    # CHECK: [[q0_alloc:%.+]] = qref.get [[reg_alloc]][ 0] : !qref.reg<1> -> !qref.bit
    # CHECK: [[q2:%.+]] = qref.get [[reg]][ 2] : !qref.reg<4> -> !qref.bit
    # CHECK: [[state:%.+]] = stablehlo.convert %arg1 : (tensor<4xi64>) -> tensor<4xcomplex<f64>>
    # CHECK: qref.set_state([[state]]) [[q0_alloc]], [[q2]] : tensor<4xcomplex<f64>>, !qref.bit, !qref.bit
    # CHECK: qref.dealloc [[reg_alloc]] : !qref.reg<1>
    with qp.allocate(1) as q:
        qp.StatePrep(np.array([0, 0, 1, 0]), wires=[q[0], 2])

    return qp.expval(qp.X(0))


print(test_set_state.mlir)


# CHECK: func.func public @test_set_basis_state(%arg0: tensor<3xi64>, %arg1: tensor<2xi64>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=4))
def test_set_basis_state():
    """
    Test set_basis_state.
    """
    # CHECK: [[reg:%.+]] = qref.alloc( 4) : !qref.reg<4>

    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<4> -> !qref.bit
    # CHECK: [[q2:%.+]] = qref.get [[reg]][ 2] : !qref.reg<4> -> !qref.bit
    # CHECK: [[q3:%.+]] = qref.get [[reg]][ 3] : !qref.reg<4> -> !qref.bit
    # CHECK: [[state:%.+]] = stablehlo.convert {{%.+}} : tensor<3xi1>
    # CHECK: qref.set_basis_state([[state]]) [[q0]], [[q2]], [[q3]] : tensor<3xi1>, !qref.bit, !qref.bit, !qref.bit
    qp.BasisState(np.array([0, 0, 1]), wires=[0, 2, 3])

    # CHECK: [[reg_alloc:%.+]] = qref.alloc( 1) : !qref.reg<1>
    # CHECK: [[q0_alloc:%.+]] = qref.get [[reg_alloc]][ 0] : !qref.reg<1> -> !qref.bit
    # CHECK: [[q2:%.+]] = qref.get [[reg]][ 2] : !qref.reg<4> -> !qref.bit
    # CHECK: [[state:%.+]] = stablehlo.convert {{%.+}} : tensor<2xi1>
    # CHECK: qref.set_basis_state([[state]]) [[q0_alloc]], [[q2]] : tensor<2xi1>, !qref.bit, !qref.bit
    # CHECK: qref.dealloc [[reg_alloc]] : !qref.reg<1>
    with qp.allocate(1) as q:
        qp.BasisState(np.array([0, 1]), wires=[q[0], 2])

    return qp.expval(qp.X(0))


print(test_set_basis_state.mlir)


# CHECK: func.func public @test_adjoint(%arg0: tensor<i64>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=4))
def test_adjoint(i: int):
    """
    Test adjoint
    """
    # CHECK-DAG: [[angle:%.+]] = arith.constant 1.000000e-01 : f64
    # CHECK-DAG: [[angle_adj:%.+]] = arith.constant -1.000000e-01 : f64

    # CHECK: [[reg:%.+]] = qref.alloc( 4) : !qref.reg<4>

    # CHECK: qref.adjoint {
    # CHECK:   [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<4> -> !qref.bit
    # CHECK:   [[i:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK:   [[qi:%.+]] = qref.get [[reg]][[[i]]] : !qref.reg<4>, i64 -> !qref.bit
    # CHECK:   qref.custom "CNOT"() [[q0]], [[qi]] : !qref.bit, !qref.bit
    # CHECK: }
    qp.adjoint(qp.CNOT)(wires=[0, i])

    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<4> -> !qref.bit
    # CHECK: qref.custom "RX"([[angle_adj]]) [[q0]] : !qref.bit
    qp.adjoint(qp.RX(0.1, wires=0))

    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<4> -> !qref.bit
    # CHECK: qref.paulirot ["X"]([[angle]]) [[q0]] adj : !qref.bit
    qp.adjoint(qp.PauliRot(0.1, ["X"], 0))

    def f(wires):
        qp.X(wires)

    # CHECK: qref.adjoint {
    # CHECK:   [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<4> -> !qref.bit
    # CHECK:   qref.custom "PauliX"() [[q0]] : !qref.bit
    # CHECK: }
    qp.adjoint(f)(0)

    # CHECK: qref.adjoint {
    # CHECK:   [[i:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK:   [[qi:%.+]] = qref.get [[reg]][[[i]]] : !qref.reg<4>, i64 -> !qref.bit
    # CHECK:   qref.custom "PauliX"() [[qi]] : !qref.bit
    # CHECK: }
    qp.adjoint(f)(i)

    return qp.expval(qp.X(0))


print(test_adjoint.mlir)


# CHECK: func.func public @test_adjoint_with_allocation() -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=4))
def test_adjoint_with_allocation():
    """
    Test adjoint with dynamic qubit allocation
    """
    # CHECK-DAG: [[angle:%.+]] = arith.constant 1.000000e-01 : f64

    # CHECK-DAG: [[reg_device:%.+]] = qref.alloc( 4) : !qref.reg<4>

    def f(wires):
        qp.RX(0.1, wires)

    # CHECK: [[reg_alloc:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK: [[alloc_q0:%.+]] = qref.get [[reg_alloc]][ 0] : !qref.reg<2> -> !qref.bit
    # CHECK: [[alloc_q1:%.+]] = qref.get [[reg_alloc]][ 1] : !qref.reg<2> -> !qref.bit
    with qp.allocate(2) as q:

        # CHECK:  qref.adjoint {
        # CHECK:    qref.custom "PauliX"() [[alloc_q0]] : !qref.bit
        # CHECK:  }
        qp.adjoint(qp.X)(q[0])

        # CHECK:  qref.adjoint {
        # CHECK:    qref.custom "RX"([[angle]]) [[alloc_q1]] : !qref.bit
        # CHECK:  }
        qp.adjoint(f)(q[1])
    # CHECK: qref.dealloc [[reg_alloc]] : !qref.reg<2>

    # CHECK: qref.adjoint {
    # CHECK:   [[reg_alloc:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK:   [[alloc_q0:%.+]] = qref.get [[reg_alloc]][ 0] : !qref.reg<2> -> !qref.bit
    # CHECK:   [[alloc_q1:%.+]] = qref.get [[reg_alloc]][ 1] : !qref.reg<2> -> !qref.bit
    # CHECK:   qref.custom "PauliX"() [[alloc_q0]] : !qref.bit
    # CHECK:   [[q0:%.+]] = qref.get [[reg_device]][ 0] : !qref.reg<4> -> !qref.bit
    # CHECK:   qref.custom "CNOT"() [[q0]], [[alloc_q1]] : !qref.bit, !qref.bit
    # CHECK:   qref.dealloc [[reg_alloc]] : !qref.reg<2>
    # CHECK: }
    def g():
        with qp.allocate(2) as q:
            qp.X(q[0])
            qp.CNOT(wires=[0, q[1]])

    qp.adjoint(g)()

    return qp.expval(qp.X(0))


print(test_adjoint_with_allocation.mlir)


# CHECK: func.func public @test_adjoint_with_ctrl() -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=4))
def test_adjoint_with_ctrl():
    """
    Test adjoint and ctrl used together
    """

    # CHECK: [[true:%.+]] = arith.constant true
    # CHECK: [[reg:%.+]] = qref.alloc( 4) : !qref.reg<4>

    # CHECK: qref.adjoint {
    # CHECK:   [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<4> -> !qref.bit
    # CHECK:   [[q2:%.+]] = qref.get [[reg]][ 2] : !qref.reg<4> -> !qref.bit
    # CHECK:   [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<4> -> !qref.bit
    # CHECK:   qref.custom "SWAP"() [[q1]], [[q2]] ctrls([[q0]]) ctrlvals([[true]]) : !qref.bit, !qref.bit ctrls !qref.bit
    # CHECK: }
    qp.ctrl(qp.adjoint(qp.SWAP), control=0)(wires=[1, 2])

    # CHECK: qref.adjoint {
    # CHECK:   [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<4> -> !qref.bit
    # CHECK:   [[q2:%.+]] = qref.get [[reg]][ 2] : !qref.reg<4> -> !qref.bit
    # CHECK:   [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<4> -> !qref.bit
    # CHECK:   qref.custom "SWAP"() [[q1]], [[q2]] ctrls([[q0]]) ctrlvals([[true]]) : !qref.bit, !qref.bit ctrls !qref.bit
    # CHECK: }
    qp.adjoint(qp.ctrl(qp.SWAP, control=0))(wires=[1, 2])

    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<4> -> !qref.bit
    # CHECK: [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<4> -> !qref.bit
    # CHECK: qref.custom "S"() [[q0]] adj ctrls([[q1]]) ctrlvals([[true]]) : !qref.bit ctrls !qref.bit
    qp.ctrl(qp.adjoint(qp.S(wires=0)), control=1)

    return qp.expval(qp.X(0))


print(test_adjoint_with_ctrl.mlir)
