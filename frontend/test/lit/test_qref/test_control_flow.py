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

    x = 0
    for i in range(size):
        m = qp.measure(i)
        x += m

    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[loopOut_tensorf64:%.+]] = stablehlo.convert [[loopOut]] : (tensor<i64>) -> tensor<f64>
    # CHECK: [[angle:%.+]] = tensor.extract [[loopOut_tensorf64]][] : tensor<f64>
    # CHECK: qref.custom "RX"([[angle]]) [[q0]] : !qref.bit
    qp.RX(x, wires=0)

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
    for _ in range(size):
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


# CHECK: func.func public @test_while_loop_basic(%arg0: tensor<i64>, %arg1: tensor<f64>) -> tensor<8xcomplex<f64>>
@qp.qjit(capture=True, autograph=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_while_loop_basic(i: int, angle: float):
    """
    Test basic while loop
    """

    # CHECK-DAG: [[ten:%.+]] = stablehlo.constant dense<10> : tensor<i64>

    # CHECK-DAG: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: scf.while : () -> () {
    # CHECK:   [[cmp:%.+]] = stablehlo.compare  LT, %arg0, [[ten]]
    # CHECK:   [[cmp_i1:%.+]] = tensor.extract [[cmp]][] : tensor<i1>
    # CHECK:   scf.condition([[cmp_i1]])
    # CHECK: } do {
    # CHECK:   [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK:   [[angle:%.+]] = tensor.extract %arg1[] : tensor<f64>
    # CHECK:   qref.custom "RX"([[angle]]) [[q0]] : !qref.bit
    # CHECK:   scf.yield
    # CHECK: }

    while i < 10:
        qp.RX(angle, wires=0)

    return qp.state()


print(test_while_loop_basic.mlir)


# CHECK: func.func public @test_while_loop_nested(%arg0: tensor<i64>, %arg1: tensor<f64>) -> tensor<8xcomplex<f64>>
@qp.qjit(capture=True, autograph=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_while_loop_nested(i: int, angle: float):
    """
    Test nested while loop
    """

    # CHECK-DAG: [[ten:%.+]] = stablehlo.constant dense<10> : tensor<i64>
    # CHECK-DAG: [[twenty:%.+]] = stablehlo.constant dense<20> : tensor<i64>

    # CHECK-DAG: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: scf.while : () -> () {
    # CHECK:   [[cmp:%.+]] = stablehlo.compare  LT, %arg0, [[ten]]
    # CHECK:   [[cmp_i1:%.+]] = tensor.extract [[cmp]][] : tensor<i1>
    # CHECK:   scf.condition([[cmp_i1]])
    # CHECK: } do {
    # CHECK:     scf.while : () -> () {
    # CHECK:       [[cmp:%.+]] = stablehlo.compare  LT, %arg0, [[twenty]]
    # CHECK:       [[cmp_i1:%.+]] = tensor.extract [[cmp]][] : tensor<i1>
    # CHECK:       scf.condition([[cmp_i1]])
    # CHECK:     } do {
    # CHECK:        [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK:        [[angle:%.+]] = tensor.extract %arg1[] : tensor<f64>
    # CHECK:        qref.custom "RX"([[angle]]) [[q0]] : !qref.bit
    # CHECK:        scf.yield
    # CHECK:     }
    # CHECK:     scf.yield
    # CHECK: }

    while i < 10:
        while i < 20:
            qp.RX(angle, wires=0)

    return qp.state()


print(test_while_loop_nested.mlir)


# CHECK: func.func public @test_while_loop_with_dynamic_shapes(%arg0: tensor<i64>) -> tensor<8xcomplex<f64>>
@qp.qjit(capture=True, autograph=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_while_loop_with_dynamic_shapes(i: int):
    """
    Test while loop with dynamically shaped results
    """

    # CHECK-DAG: [[ten:%.+]] = stablehlo.constant dense<10> : tensor<i64>
    # CHECK-DAG: [[sum:%.+]] = stablehlo.dynamic_broadcast_in_dim {{.+}} -> tensor<?xf64>

    # CHECK-DAG: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: [[loopOut:%.+]] = scf.while (%arg1 = [[sum]]) : (tensor<?xf64>) -> tensor<?xf64> {
    # CHECK:   [[cmp:%.+]] = stablehlo.compare  LT, %arg0, [[ten]]
    # CHECK:   [[cmp_i1:%.+]] = tensor.extract [[cmp]][] : tensor<i1>
    # CHECK:   scf.condition([[cmp_i1]]) %arg1 : tensor<?xf64>
    # CHECK: } do {
    # CHECK: ^bb0(%arg1: tensor<?xf64>):
    # CHECK:   [[add:%.+]] = stablehlo.add %arg1, %arg1 : tensor<?xf64>
    # CHECK:   [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK:   qref.custom "PauliX"() [[q0]] : !qref.bit
    # CHECK:   scf.yield [[add]] : tensor<?xf64>
    # CHECK: }
    x = jnp.zeros(i)
    while i < 10:
        x += x
        qp.X(wires=0)

    # CHECK: [[reduce:%.+]] = stablehlo.reduce([[loopOut]]
    # CHECK-SAME:   applies stablehlo.add
    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[reduce_f64:%.+]] = tensor.extract [[reduce]][] : tensor<f64>
    # CHECK: qref.custom "RX"([[reduce_f64]]) [[q0]] : !qref.bit
    qp.RX(jnp.sum(x), wires=0)

    return qp.state()


print(test_while_loop_with_dynamic_shapes.mlir)


# CHECK: func.func public @test_while_loop_with_dynamic_allocation(%arg0: tensor<i64>) -> tensor<f64>
@qp.qjit(capture=True, autograph=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_while_loop_with_dynamic_allocation(i: int):
    """
    Test while loop with dynamic qubit allocation
    """
    # CHECK-DAG: [[reg_device:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: scf.while : () -> () {
    # CHECK:   stablehlo.compare  LT
    # CHECK:   scf.condition
    # CHECK: } do {
    # CHECK:   [[reg_loop:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK:   [[q0_loop:%.+]] = qref.get [[reg_loop]][ 0] : !qref.reg<2> -> !qref.bit
    # CHECK:   [[i:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK:   [[qi:%.+]] = qref.get [[reg_device]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK:   qref.custom "CNOT"() [[q0_loop]], [[qi]] : !qref.bit, !qref.bit
    # CHECK:   qref.dealloc [[reg_loop]] : !qref.reg<2>
    # CHECK:   scf.yield
    # CHECK: }

    while i < 10:
        with qp.allocate(2) as q:
            qp.CNOT(wires=[q[0], i])

    # CHECK: [[reg_loop:%.+]] = qref.alloc( 1) : !qref.reg<1>
    # CHECK: [[q0_loop:%.+]] = qref.get [[reg_loop]][ 0] : !qref.reg<1> -> !qref.bit
    # CHECK: scf.while : () -> () {
    # CHECK:   stablehlo.compare  LT
    # CHECK:   scf.condition
    # CHECK: } do {
    # CHECK:   [[i:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK:   [[qi:%.+]] = qref.get [[reg_device]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK:   qref.custom "CNOT"() [[q0_loop]], [[qi]] : !qref.bit, !qref.bit
    # CHECK:   scf.yield
    # CHECK: }
    # CHECK: qref.dealloc [[reg_loop]] : !qref.reg<1>

    with qp.allocate(1) as q:
        while i < 10:
            qp.CNOT(wires=[q[0], i])

    return qp.expval(qp.X(0))


print(test_while_loop_with_dynamic_allocation.mlir)


# CHECK: func.func public @test_if_basic(%arg0: tensor<i64>, %arg1: tensor<f64>) -> tensor<8xcomplex<f64>>
@qp.qjit(capture=True, autograph=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_if_basic(i: int, angle: float):
    """
    Test basic if statements
    """

    # CHECK-DAG: [[ten:%.+]] = stablehlo.constant dense<10> : tensor<i64>

    # CHECK-DAG: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: [[comp:%.+]] = stablehlo.compare  LT, %arg0, [[ten]]
    # CHECK: [[comp_i1:%.+]] = tensor.extract [[comp]][] : tensor<i1>
    # CHECK: scf.if [[comp_i1]] {
    # CHECK:   [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK:   [[angle:%.+]] = tensor.extract %arg1[] : tensor<f64>
    # CHECK:   qref.custom "RX"([[angle]]) [[q0]] : !qref.bit
    # CHECK: }
    if i < 10:
        qp.RX(angle, wires=0)

    # CHECK: [[comp:%.+]] = stablehlo.compare  GT, %arg0, [[ten]]
    # CHECK: [[comp_i1:%.+]] = tensor.extract [[comp]][] : tensor<i1>
    # CHECK: scf.if [[comp_i1]] {
    # CHECK:   [[i:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK:   [[qi:%.+]] = qref.get [[reg]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK:   [[angle:%.+]] = tensor.extract %arg1[] : tensor<f64>
    # CHECK:   qref.custom "RY"([[angle]]) [[qi]] : !qref.bit
    # CHECK: } else {
    # CHECK:   [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<3> -> !qref.bit
    # CHECK:   [[angle:%.+]] = tensor.extract %arg1[] : tensor<f64>
    # CHECK:   qref.custom "RZ"([[angle]]) [[q1]] : !qref.bit
    # CHECK: }
    if i > 10:
        qp.RY(angle, wires=i)
    else:
        qp.RZ(angle, wires=1)

    return qp.state()


print(test_if_basic.mlir)


# CHECK: func.func public @test_if_nested(%arg0: tensor<i64>, %arg1: tensor<f64>) -> tensor<8xcomplex<f64>>
@qp.qjit(capture=True, autograph=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_if_nested(i: int, angle: float):
    """
    Test nested if statements
    """

    # CHECK-DAG: [[ten:%.+]] = stablehlo.constant dense<10> : tensor<i64>

    # CHECK-DAG: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: [[comp:%.+]] = stablehlo.compare  LT, %arg0, [[ten]]
    # CHECK: [[comp_i1:%.+]] = tensor.extract [[comp]][] : tensor<i1>
    # CHECK: scf.if [[comp_i1]] {
    # CHECK:    [[comp:%.+]] = stablehlo.compare  LT, %arg0, [[ten]]
    # CHECK:    [[comp_i1:%.+]] = tensor.extract [[comp]][] : tensor<i1>
    # CHECK:    scf.if [[comp_i1]] {
    # CHECK:      [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK:      [[angle:%.+]] = tensor.extract %arg1[] : tensor<f64>
    # CHECK:      qref.custom "RX"([[angle]]) [[q0]] : !qref.bit
    # CHECK:    } else {
    # CHECK:      [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK:      qref.custom "PauliX"() [[q0]] : !qref.bit
    # CHECK:    }
    # CHECK: } else {
    # CHECK:    [[comp:%.+]] = stablehlo.compare  LT, %arg0, [[ten]]
    # CHECK:    [[comp_i1:%.+]] = tensor.extract [[comp]][] : tensor<i1>
    # CHECK:    scf.if [[comp_i1]] {
    # CHECK:      [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK:      [[angle:%.+]] = tensor.extract %arg1[] : tensor<f64>
    # CHECK:      qref.custom "RX"([[angle]]) [[q0]] : !qref.bit
    # CHECK:    } else {
    # CHECK:      [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK:      qref.custom "PauliX"() [[q0]] : !qref.bit
    # CHECK:    }
    # CHECK: }
    if i < 10:
        if i < 10:
            qp.RX(angle, wires=0)
        else:
            qp.X(wires=0)
    else:
        if i < 10:
            qp.RX(angle, wires=0)
        else:
            qp.X(wires=0)

    return qp.state()


print(test_if_nested.mlir)


# CHECK: func.func public @test_if_with_dynamic_shapes(%arg0: tensor<i64>) -> tensor<8xcomplex<f64>>
@qp.qjit(capture=True, autograph=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_if_with_dynamic_shapes(i: int):
    """
    Test if statements with dynamically shaped results
    """

    # CHECK-DAG: [[ten:%.+]] = stablehlo.constant dense<10> : tensor<i64>
    # CHECK-DAG: [[sum:%.+]] = stablehlo.dynamic_broadcast_in_dim {{.+}} -> tensor<?xf64>

    # CHECK-DAG: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: [[comp:%.+]] = stablehlo.compare  LT, %arg0, [[ten]]
    # CHECK: [[comp_i1:%.+]] = tensor.extract [[comp]][] : tensor<i1>
    # CHECK: [[if1_out:%.+]] = scf.if [[comp_i1]] -> (tensor<?xf64>) {
    # CHECK:   [[add:%.+]] = stablehlo.add [[sum]], [[sum]] : tensor<?xf64>
    # CHECK:   [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK:   qref.custom "PauliX"() [[q0]] : !qref.bit
    # CHECK:   scf.yield [[add]] : tensor<?xf64>
    # CHECK: } else {
    # CHECK:   scf.yield [[sum:%.+]] : tensor<?xf64>
    # CHECK: }
    x = jnp.zeros(i)
    if i < 10:
        x += x
        qp.X(wires=0)

    # CHECK: [[comp:%.+]] = stablehlo.compare  LT, %arg0, [[ten]]
    # CHECK: [[comp_i1:%.+]] = tensor.extract [[comp]][] : tensor<i1>
    # CHECK: [[if2_out:%.+]] = scf.if [[comp_i1]] -> (tensor<?xf64>) {
    # CHECK:   [[add:%.+]] = stablehlo.add [[if1_out]], [[if1_out]] : tensor<?xf64>
    # CHECK:   scf.yield [[add]] : tensor<?xf64>
    # CHECK: } else {
    # CHECK:   [[mul:%.+]] = stablehlo.multiply [[if1_out]], [[if1_out]] : tensor<?xf64>
    # CHECK:   [[i:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK:   [[qi:%.+]] = qref.get [[reg]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK:   qref.custom "PauliY"() [[qi]] : !qref.bit
    # CHECK:   scf.yield [[mul]] : tensor<?xf64>
    # CHECK: }
    if i < 10:
        x += x
    else:
        x *= x
        qp.Y(wires=i)

    # CHECK: [[reduce:%.+]] = stablehlo.reduce([[if2_out]]
    # CHECK-SAME:   applies stablehlo.add
    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[reduce_f64:%.+]] = tensor.extract [[reduce]][] : tensor<f64>
    # CHECK: qref.custom "RX"([[reduce_f64]]) [[q0]] : !qref.bit
    qp.RX(jnp.sum(x), wires=0)

    return qp.state()


print(test_if_with_dynamic_shapes.mlir)


# CHECK: func.func public @test_if_with_dynamic_allocation(%arg0: tensor<i64>) -> tensor<8xcomplex<f64>>
@qp.qjit(capture=True, autograph=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_if_with_dynamic_allocation(i: int):
    """
    Test if statements with dynamic qubit allocation
    """

    # CHECK-DAG: [[ten:%.+]] = stablehlo.constant dense<10> : tensor<i64>

    # CHECK-DAG: [[reg_device:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: [[reg_alloc:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK: [[q0_alloc:%.+]] = qref.get [[reg_alloc]][ 0] : !qref.reg<2> -> !qref.bit
    # CHECK: [[q1_alloc:%.+]] = qref.get [[reg_alloc]][ 1] : !qref.reg<2> -> !qref.bit
    # CHECK: [[comp:%.+]] = stablehlo.compare  LT, %arg0, [[ten]]
    # CHECK: [[comp_i1:%.+]] = tensor.extract [[comp]][] : tensor<i1>
    # CHECK: scf.if [[comp_i1]] {
    # CHECK:    [[q0:%.+]] = qref.get [[reg_device]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK:    qref.custom "CNOT"() [[q0]], [[q0_alloc]] : !qref.bit, !qref.bit
    # CHECK: } else {
    # CHECK:    [[q1:%.+]] = qref.get [[reg_device]][ 1] : !qref.reg<3> -> !qref.bit
    # CHECK:    qref.custom "SWAP"() [[q1]], [[q1_alloc]] : !qref.bit, !qref.bit
    # CHECK: }
    # CHECK: qref.custom "CNOT"() [[q0_alloc]], [[q1_alloc]] : !qref.bit, !qref.bit
    # CHECK: qref.dealloc [[reg_alloc]] : !qref.reg<2>
    with qp.allocate(2) as q:
        if i < 10:
            qp.CNOT(wires=[0, q[0]])
        else:
            qp.SWAP(wires=[1, q[1]])
        qp.CNOT(wires=[q[0], q[1]])

    # CHECK: [[comp:%.+]] = stablehlo.compare  GT, %arg0, [[ten]]
    # CHECK: [[comp_i1:%.+]] = tensor.extract [[comp]][] : tensor<i1>
    # CHECK: scf.if [[comp_i1]] {
    # CHECK:    [[reg_alloc:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK:    [[q0_alloc:%.+]] = qref.get [[reg_alloc]][ 0] : !qref.reg<2> -> !qref.bit
    # CHECK:    [[q0:%.+]] = qref.get [[reg_device]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK:    qref.custom "CNOT"() [[q0]], [[q0_alloc]] : !qref.bit, !qref.bit
    # CHECK:    qref.dealloc [[reg_alloc]] : !qref.reg<2>
    # CHECK: } else {
    # CHECK:    [[reg_alloc:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK:    [[q1_alloc:%.+]] = qref.get [[reg_alloc]][ 1] : !qref.reg<2> -> !qref.bit
    # CHECK:    [[q1:%.+]] = qref.get [[reg_device]][ 1] : !qref.reg<3> -> !qref.bit
    # CHECK:    qref.custom "SWAP"() [[q1]], [[q1_alloc]] : !qref.bit, !qref.bit
    # CHECK:    qref.dealloc [[reg_alloc]] : !qref.reg<2>
    # CHECK: }
    if i > 10:
        with qp.allocate(2) as q:
            qp.CNOT(wires=[0, q[0]])
    else:
        with qp.allocate(2) as q:
            qp.SWAP(wires=[1, q[1]])

    return qp.state()


print(test_if_with_dynamic_allocation.mlir)


# CHECK: func.func public @test_measurement_result_as_cond() -> tensor<8xcomplex<f64>>
@qp.qjit(capture=True, autograph=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_measurement_result_as_cond():
    """
    Test using measurement result as conditional predicates
    """

    # CHECK: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[mres:%.+]] = qref.measure [[q0]] : i1
    m = qp.measure(0)

    # CHECK: scf.if {{%.+}} {
    # CHECK:   [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK:   qref.custom "PauliX"() [[q0]] : !qref.bit
    # CHECK: } else {
    # CHECK:   [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK:   qref.custom "PauliY"() [[q0]] : !qref.bit
    # CHECK: }
    if m:
        qp.X(0)
    else:
        qp.Y(0)

    return qp.state()


print(test_measurement_result_as_cond.mlir)
