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
Unit tests for lowering control flow primitives to reference semantics MLIR during
PLxPR conversion.
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable=line-too-long

import numpy as np
import pennylane as qp


# CHECK: func.func public @test_for_loop_basic(%arg0: tensor<i64>, %arg1: tensor<f64>) -> tensor<8xcomplex<f64>>
@qp.qjit(capture=True, autograph=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_for_loop_basic(size: int, angle: float):
    """
    Test basic for loop
    """

    # CHECK-DAG: [[one_index:%.+]] = arith.constant 1 : index
    # CHECK-DAG: [[zero_index:%.+]] = arith.constant 0 : index
    # CHECK-DAG: [[size:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK-DAG: [[size_index:%.+]] = arith.index_cast [[size]] : i64 to index
    # CHECK-DAG: [[zero:%.+]] = arith.constant 0 : i64

    # CHECK-DAG: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: scf.for %arg2 = [[zero_index]] to [[size_index]] step [[one_index]] {
    # CHECK:   [[i:%.+]] = arith.index_cast %arg2 : index to i64
    # CHECK:   [[q0:%.+]] = qref.get [[reg]][[[zero]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK:   [[angle:%.+]] = tensor.extract %arg1[] : tensor<f64>
    # CHECK:   qref.custom "RX"([[angle]]) [[q0]] : !qref.bit
    # CHECK:   [[qi:%.+]] = qref.get [[reg]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK:   {{%.+}} = qref.measure [[qi]] : i1
    # CHECK: }

    for i in range(size):
        qp.RX(angle, wires=0)
        qp.measure(i)

    return qp.state()


print(test_for_loop_basic.mlir)


# CHECK: func.func public @test_for_loop_with_dynamic_allocation() -> tensor<f64>
@qp.qjit(capture=True, autograph=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_for_loop_with_dynamic_allocation():
    """
    Test for loop with dynamic qubit allocation
    """

    # CHECK-DAG: [[zero_index:%.+]] = arith.constant 0 : index
    # CHECK-DAG: [[one_index:%.+]] = arith.constant 1 : index
    # CHECK-DAG: [[three_index:%.+]] = arith.constant 3 : index
    # CHECK-DAG: [[zero:%.+]] = arith.constant 0 : i64

    # CHECK-DAG: [[reg_device:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: scf.for %arg0 = [[zero_index]] to [[three_index]] step [[one_index]] {
    # CHECK:   [[i:%.+]] = arith.index_cast %arg0 : index to i64
    # CHECK:   [[reg_loop:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK:   [[q0_loop:%.+]] = qref.get [[reg_loop]][[[zero]]] : !qref.reg<2>, i64 -> !qref.bit
    # CHECK:   [[qi:%.+]] = qref.get [[reg_device]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK:   qref.custom "CNOT"() [[q0_loop]], [[qi]] : !qref.bit, !qref.bit
    # CHECK:   qref.dealloc [[reg_loop]] : !qref.reg<2>
    # CHECK: }

    for i in range(3):
        with qp.allocate(2) as q:
            qp.CNOT(wires=[q[0], i])

    # CHECK: [[reg_loop:%.+]] = qref.alloc( 1) : !qref.reg<1>
    # CHECK: [[q0_loop:%.+]] = qref.get [[reg_loop]][[[zero]]] : !qref.reg<1>, i64 -> !qref.bit
    # CHECK: scf.for %arg0 = [[zero_index]] to [[three_index]] step [[one_index]] {
    # CHECK:   [[i:%.+]] = arith.index_cast %arg0 : index to i64
    # CHECK:   [[qi:%.+]] = qref.get [[reg_device]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK:   qref.custom "CNOT"() [[q0_loop]], [[qi]] : !qref.bit, !qref.bit
    # CHECK: }
    # CHECK: qref.dealloc [[reg_loop]] : !qref.reg<1>

    with qp.allocate(1) as q:
        for i in range(3):
            qp.CNOT(wires=[q[0], i])

    return qp.expval(qp.X(0))


print(test_for_loop_with_dynamic_allocation.mlir)
