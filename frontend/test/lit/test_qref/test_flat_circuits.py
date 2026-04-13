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
    # CHECK-DAG: [[two:%.+]] = arith.constant 2 : i64
    # CHECK-DAG: [[one:%.+]] = arith.constant 1 : i64
    # CHECK-DAG: [[zero:%.+]] = arith.constant 0 : i64
    # CHECK-DAG: [[angle:%.+]] = arith.constant 1.000000e-01 : f64

    # CHECK: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: [[q0:%.+]] = qref.get [[reg]][[[zero]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: qref.custom "PauliX"() [[q0]] : !qref.bit
    qp.X(0)

    # CHECK: [[q1:%.+]] = qref.get [[reg]][[[one]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: qref.custom "RZ"([[angle]]) [[q1]] : !qref.bit
    qp.RZ(0.1, wires=1)

    # CHECK: [[q1:%.+]] = qref.get [[reg]][[[one]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[q2:%.+]] = qref.get [[reg]][[[two]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: qref.custom "SWAP"() [[q1]], [[q2]] : !qref.bit, !qref.bit
    qp.SWAP([1, 2])

    # CHECK: [[i:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK: [[qi:%.+]] = qref.get [[reg]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[q0:%.+]] = qref.get [[reg]][[[zero]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: qref.custom "RX"([[angle]]) [[qi]] ctrls([[q0]]) ctrlvals([[true]]) : !qref.bit ctrls !qref.bit
    qp.ctrl(qp.RX, control=0)(0.1, wires=[i])

    # CHECK: [[q1:%.+]] = qref.get [[reg]][[[one]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[q2:%.+]] = qref.get [[reg]][[[two]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[q0:%.+]] = qref.get [[reg]][[[zero]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: qref.custom "CNOT"() [[q1]], [[q2]] ctrls([[q0]]) ctrlvals([[false]]) : !qref.bit, !qref.bit ctrls !qref.bit
    qp.ctrl(qp.CNOT, control=0, control_values=False)(wires=[1, 2])

    return qp.expval(qp.X(0))


print(test_custom_op.mlir)


# CHECK: func.func public @test_dynamic_qubit_allocation(%arg0: tensor<i64>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_dynamic_qubit_allocation(i: int):
    """
    Test dynamic qubit allocation.
    """
    # CHECK-DAG: [[false:%.+]] = arith.constant false
    # CHECK-DAG: [[two:%.+]] = arith.constant 2 : i64
    # CHECK-DAG: [[one:%.+]] = arith.constant 1 : i64
    # CHECK-DAG: [[zero:%.+]] = arith.constant 0 : i64

    # CHECK: [[reg_device:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: [[reg_alloc:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK: [[alloc_q0:%.+]] = qref.get [[reg_alloc]][[[zero]]] : !qref.reg<2>, i64 -> !qref.bit
    # CHECK: [[alloc_q1:%.+]] = qref.get [[reg_alloc]][[[one]]] : !qref.reg<2>, i64 -> !qref.bit
    with qp.allocate(2) as q:
        # CHECK: qref.custom "PauliX"() [[alloc_q0]] : !qref.bit
        qp.X(q[0])

        # CHECK: [[i:%.+]] = tensor.extract %arg0[] : tensor<i64>
        # CHECK: [[qi:%.+]] = qref.get [[reg_device]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
        # CHECK: [[q2:%.+]] = qref.get [[reg_device]][[[two]]] : !qref.reg<3>, i64 -> !qref.bit
        # CHECK: qref.custom "CNOT"() [[qi]], [[q2]] ctrls([[alloc_q1]]) ctrlvals([[false]]) : !qref.bit, !qref.bit ctrls !qref.bit
        qp.ctrl(qp.CNOT, control=q[1], control_values=False)(wires=[i, 2])

    # CHECK: qref.dealloc [[reg_alloc]] : !qref.reg<2>

    # CHECK: [[q0:%.+]] = qref.get [[reg_device]][[[zero]]] : !qref.reg<3>, i64 -> !qref.bit
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
    # CHECK-DAG: [[two:%.+]] = arith.constant 2 : i64
    # CHECK-DAG: [[one:%.+]] = arith.constant 1 : i64
    # CHECK-DAG: [[zero:%.+]] = arith.constant 0 : i64
    # CHECK-DAG: [[angle:%.+]] = arith.constant 1.000000e-01 : f64

    # CHECK: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: [[q1:%.+]] = qref.get [[reg]][[[one]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: qref.multirz([[angle]]) [[q1]] : !qref.bit
    qp.MultiRZ(0.1, wires=1)

    # CHECK: [[q1:%.+]] = qref.get [[reg]][[[one]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[q2:%.+]] = qref.get [[reg]][[[two]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[q0:%.+]] = qref.get [[reg]][[[zero]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: qref.multirz([[angle]]) [[q1]], [[q2]] ctrls([[q0]]) ctrlvals([[true]]) : !qref.bit, !qref.bit ctrls !qref.bit
    qp.ctrl(qp.MultiRZ, control=0, control_values=True)(0.1, wires=[1, 2])

    return qp.expval(qp.X(0))


print(test_multirz.mlir)
