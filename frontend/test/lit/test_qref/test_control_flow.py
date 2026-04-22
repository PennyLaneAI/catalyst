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

import pennylane as qp
from jax import numpy as jnp


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

    # CHECK-DAG: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: scf.for %arg2 = [[zero_index]] to [[size_index]] step [[one_index]] {
    # CHECK:   [[i:%.+]] = arith.index_cast %arg2 : index to i64
    # CHECK:   [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
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


# CHECK: func.func public @test_for_loop_nested(%arg0: tensor<i64>) -> tensor<8xcomplex<f64>>
@qp.qjit(capture=True, autograph=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_for_loop_nested(size: int):
    """
    Test nested for loop
    """

    # CHECK-DAG: [[one_index:%.+]] = arith.constant 1 : index
    # CHECK-DAG: [[zero_index:%.+]] = arith.constant 0 : index
    # CHECK-DAG: [[size:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK-DAG: [[size_index:%.+]] = arith.index_cast [[size]] : i64 to index

    # CHECK-DAG: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: scf.for %arg1 = [[zero_index]] to [[size_index]] step [[one_index]] {
    # CHECK:   [[i:%.+]] = arith.index_cast %arg1 : index to i64
    # CHECK:   [[size:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK:   [[size_index:%.+]] = arith.index_cast [[size]] : i64 to index
    # CHECK:   scf.for %arg2 = [[zero_index]] to [[size_index]] step [[one_index]] {
    # CHECK:     [[j:%.+]] = arith.index_cast %arg2 : index to i64
    # CHECK:     [[qi:%.+]] = qref.get [[reg]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK:     [[qj:%.+]] = qref.get [[reg]][[[j]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK:     qref.custom "CNOT"() [[qi]], [[qj]] : !qref.bit, !qref.bit
    # CHECK:   }
    # CHECK: }

    for i in range(size):
        for j in range(size):
            qp.CNOT(wires=[i, j])

    return qp.state()


print(test_for_loop_nested.mlir)


# CHECK: func.func public @test_for_loop_with_result(%arg0: tensor<i64>) -> tensor<8xcomplex<f64>>
@qp.qjit(capture=True, autograph=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_for_loop_with_result(size: int):
    """
    Test for loop with results
    """

    # CHECK-DAG: [[one_index:%.+]] = arith.constant 1 : index
    # CHECK-DAG: [[zero_index:%.+]] = arith.constant 0 : index
    # CHECK-DAG: [[size:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK-DAG: [[size_index:%.+]] = arith.index_cast [[size]] : i64 to index

    # CHECK-DAG: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK-DAG: [[sum:%.+]] = stablehlo.constant dense<0> : tensor<i64>

    # CHECK: [[loopOut:%.+]] = scf.for %arg1 = [[zero_index]] to [[size_index]] step [[one_index]]
    # CHECK-SAME:    iter_args(%arg2 = [[sum]]) -> (tensor<i64>) {
    # CHECK:   [[i:%.+]] = arith.index_cast %arg1 : index to i64
    # CHECK:   [[qi:%.+]] = qref.get [[reg]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK:   [[mres_i1:%.+]] = qref.measure [[qi]] : i1
    # CHECK:   [[mres_tensori1:%.+]] = tensor.from_elements [[mres_i1]] : tensor<i1>
    # CHECK:   [[mres_tensori64:%.+]] = stablehlo.convert [[mres_tensori1]] : (tensor<i1>) -> tensor<i64>
    # CHECK:   [[sum_looparg:%.+]] = stablehlo.convert %arg2 : tensor<i64>
    # CHECK:   [[add:%.+]] = stablehlo.add [[sum_looparg]], [[mres_tensori64]] : tensor<i64>
    # CHECK:   scf.yield [[add]] : tensor<i64>
    # CHECK: }

    sum = 0
    for i in range(size):
        m = qp.measure(i)
        sum += m

    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[loopOut_tensorf64:%.+]] = stablehlo.convert [[loopOut]] : (tensor<i64>) -> tensor<f64>
    # CHECK: [[angle:%.+]] = tensor.extract [[loopOut_tensorf64]][] : tensor<f64>
    # CHECK: qref.custom "RX"([[angle]]) [[q0]] : !qref.bit
    qp.RX(sum, wires=0)

    return qp.state()


print(test_for_loop_with_result.mlir)


# CHECK: func.func public @test_for_loop_with_dynamic_shapes(%arg0: tensor<i64>) -> tensor<8xcomplex<f64>>
@qp.qjit(capture=True, autograph=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_for_loop_with_dynamic_shapes(size: int):
    """
    Test for loop with dynamically shaped results
    """

    # CHECK-DAG: [[one_index:%.+]] = arith.constant 1 : index
    # CHECK-DAG: [[zero_index:%.+]] = arith.constant 0 : index
    # CHECK-DAG: [[size:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK-DAG: [[size_index:%.+]] = arith.index_cast [[size]] : i64 to index

    # CHECK-DAG: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK-DAG: [[sum:%.+]] = stablehlo.dynamic_broadcast_in_dim {{.+}} -> tensor<?xf64>

    # CHECK: [[loopOut:%.+]] = scf.for %arg1 = [[zero_index]] to [[size_index]] step [[one_index]]
    # CHECK-SAME:    iter_args(%arg2 = [[sum]]) -> (tensor<?xf64>) {
    # CHECK:    [[add:%.+]] = stablehlo.add %arg2, %arg2 : tensor<?xf64>
    # CHECK:    scf.yield [[add]] : tensor<?xf64>
    # CHECK: }
    x = jnp.zeros(size)
    for i in range(size):
        x += x

    # CHECK: [[reduce:%.+]] = stablehlo.reduce([[loopOut]]
    # CHECK-SAME:   applies stablehlo.add
    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[reduce_f64:%.+]] = tensor.extract [[reduce]][] : tensor<f64>
    # CHECK: qref.custom "RX"([[reduce_f64]]) [[q0]] : !qref.bit
    qp.RX(jnp.sum(x), wires=0)

    return qp.state()


print(test_for_loop_with_dynamic_shapes.mlir)


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

    # CHECK-DAG: [[reg_device:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: scf.for %arg0 = [[zero_index]] to [[three_index]] step [[one_index]] {
    # CHECK:   [[i:%.+]] = arith.index_cast %arg0 : index to i64
    # CHECK:   [[reg_loop:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK:   [[q0_loop:%.+]] = qref.get [[reg_loop]][ 0] : !qref.reg<2> -> !qref.bit
    # CHECK:   [[qi:%.+]] = qref.get [[reg_device]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK:   qref.custom "CNOT"() [[q0_loop]], [[qi]] : !qref.bit, !qref.bit
    # CHECK:   qref.dealloc [[reg_loop]] : !qref.reg<2>
    # CHECK: }

    for i in range(3):
        with qp.allocate(2) as q:
            qp.CNOT(wires=[q[0], i])

    # CHECK: [[reg_loop:%.+]] = qref.alloc( 1) : !qref.reg<1>
    # CHECK: [[q0_loop:%.+]] = qref.get [[reg_loop]][ 0] : !qref.reg<1> -> !qref.bit
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


# CHECK: func.func public @jit_test_for_loop_classical(%arg0: tensor<i64>) -> tensor<i64> attributes {llvm.emit_c_interface} {
# CHECK:   %c1 = arith.constant 1 : index
# CHECK:   %c = stablehlo.constant dense<0> : tensor<i64>
# CHECK:   %c0 = arith.constant 0 : index
# CHECK:   %extracted = tensor.extract %arg0[] : tensor<i64>
# CHECK:   %0 = arith.index_cast %extracted : i64 to index
# CHECK:   %1 = scf.for %arg1 = %c0 to %0 step %c1 iter_args(%arg2 = %c) -> (tensor<i64>) {
# CHECK:     %2 = arith.index_cast %arg1 : index to i64
# CHECK:     %from_elements = tensor.from_elements %2 : tensor<i64>
# CHECK:     %3 = stablehlo.add %arg2, %from_elements : tensor<i64>
# CHECK:     scf.yield %3 : tensor<i64>
# CHECK:   }
# CHECK:   return %1 : tensor<i64>
# CHECK: }
# !!!! this is just a temporary test to check nothing broke on the `WorkflowInterpreter`
# for classical for loops
# remove before merging to main
# No docstring, so codefactor will pick it up on the main PR
@qp.qjit(capture=True, target="mlir", autograph=True)
def test_for_loop_classical(i: int):
    sum = 0
    for j in range(i):
        sum += j
    return sum


print(test_for_loop_classical.mlir)
