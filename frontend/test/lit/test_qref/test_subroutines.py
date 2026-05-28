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
Unit tests for lowering subroutines to reference semantics MLIR during PLxPR conversion.
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable=line-too-long

import numpy as np
import pennylane as qp


@qp.capture.subroutine
def basic_subroutine(x, y, wires):
    qp.RX(x, wires=wires[0])
    qp.RY(y, wires=wires[1])


# CHECK: func.func public @test_basic_subroutine(%arg0: tensor<i64>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=3))
def test_basic_subroutine(i: int):
    # CHECK-DAG: [[zero:%.+]] = stablehlo.constant dense<0> : tensor<i64>
    # CHECK-DAG: [[one:%.+]] = stablehlo.constant dense<1> : tensor<i64>
    # CHECK-DAG: [[two:%.+]] = stablehlo.constant dense<2> : tensor<i64>
    # CHECK-DAG: [[point_one:%.+]] = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    # CHECK-DAG: [[point_two:%.+]] = stablehlo.constant dense<2.000000e-01> : tensor<f64>

    # CHECK: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: call @basic_subroutine([[reg]], [[point_one]], [[point_two]], [[zero]], [[one]]) : (!qref.reg<3>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<i64>) -> ()
    # CHECK: call @basic_subroutine([[reg]], [[point_one]], [[point_two]], [[one]], [[two]]) : (!qref.reg<3>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<i64>) -> ()
    basic_subroutine(0.1, 0.2, [0, 1])
    basic_subroutine(0.1, 0.2, wires=[1, 2])

    return qp.expval(qp.X(0))


# CHECK: func.func private @basic_subroutine(%arg0: !qref.reg<3>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<i64>, %arg4: tensor<i64>)
# CHECK:   [[wires0:%.+]] = tensor.extract %arg3[] : tensor<i64>
# CHECK:   [[q0:%.+]] = qref.get %arg0[[[wires0]]] : !qref.reg<3>, i64 -> !qref.bit
# CHECK:   [[x:%.+]] = tensor.extract %arg1[] : tensor<f64>
# CHECK:   qref.custom "RX"([[x]]) [[q0]] : !qref.bit
# CHECK:   [[wires1:%.+]] = tensor.extract %arg4[] : tensor<i64>
# CHECK:   [[q1:%.+]] = qref.get %arg0[[[wires1]]] : !qref.reg<3>, i64 -> !qref.bit
# CHECK:   [[y:%.+]] = tensor.extract %arg2[] : tensor<f64>
# CHECK:   qref.custom "RY"([[y]]) [[q1]] : !qref.bit
# CHECK:   return

print(test_basic_subroutine.mlir)


# -----


@qp.capture.subroutine
def subroutine_with_return(x, y, wires):
    qp.RX(x, wires=wires[0])
    qp.RY(y, wires=wires[1])
    return x + y


# CHECK: func.func public @test_subroutine_with_return(%arg0: tensor<i64>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=3))
def test_subroutine_with_return(i: int):
    # CHECK-DAG: [[zero:%.+]] = stablehlo.constant dense<0> : tensor<i64>
    # CHECK-DAG: [[one:%.+]] = stablehlo.constant dense<1> : tensor<i64>
    # CHECK-DAG: [[point_one:%.+]] = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    # CHECK-DAG: [[point_two:%.+]] = stablehlo.constant dense<2.000000e-01> : tensor<f64>

    # CHECK: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[a:%.+]] = call @subroutine_with_return([[reg]], [[point_one]], [[point_two]], [[zero]], [[one]]) : (!qref.reg<3>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<i64>) -> tensor<f64>
    a = subroutine_with_return(0.1, 0.2, wires=[0, 1])

    # CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[angle:%.+]] = tensor.extract [[a]][] : tensor<f64>
    # CHECK: qref.custom "RX"([[angle]]) [[q0]] : !qref.bit
    qp.RX(a, wires=0)

    return qp.expval(qp.X(0))


# CHECK: func.func private @subroutine_with_return(%arg0: !qref.reg<3>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<i64>, %arg4: tensor<i64>)
# CHECK:   [[wires0:%.+]] = tensor.extract %arg3[] : tensor<i64>
# CHECK:   [[q0:%.+]] = qref.get %arg0[[[wires0]]] : !qref.reg<3>, i64 -> !qref.bit
# CHECK:   [[x:%.+]] = tensor.extract %arg1[] : tensor<f64>
# CHECK:   qref.custom "RX"([[x]]) [[q0]] : !qref.bit
# CHECK:   [[wires1:%.+]] = tensor.extract %arg4[] : tensor<i64>
# CHECK:   [[q1:%.+]] = qref.get %arg0[[[wires1]]] : !qref.reg<3>, i64 -> !qref.bit
# CHECK:   [[y:%.+]] = tensor.extract %arg2[] : tensor<f64>
# CHECK:   [[add:%.+]] = stablehlo.add %arg1, %arg2 : tensor<f64>
# CHECK:   return [[add]] : tensor<f64>

print(test_subroutine_with_return.mlir)


# -----


@qp.capture.subroutine
def subroutine_with_allocation(wires):
    with qp.allocate(1) as q:
        qp.Toffoli(wires=[q[0], *wires])


# CHECK: func.func public @test_subroutine_with_allocation(%arg0: tensor<i64>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=3))
def test_subroutine_with_allocation(i: int):
    # CHECK-DAG: [[zero:%.+]] = stablehlo.constant dense<0> : tensor<i64>
    # CHECK-DAG: [[one:%.+]] = stablehlo.constant dense<1> : tensor<i64>

    # CHECK: [[reg_device:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[reg_alloc:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK: [[q0_alloc:%.+]] = qref.get [[reg_alloc]][ 0] : !qref.reg<2> -> !qref.bit
    # CHECK: [[q1_alloc:%.+]] = qref.get [[reg_alloc]][ 1] : !qref.reg<2> -> !qref.bit
    with qp.allocate(2) as q:
        # CHECK: call @subroutine_with_allocation([[reg_device]], [[zero]], [[one]]) : (!qref.reg<3>, tensor<i64>, tensor<i64>) -> ()
        # CHECK: call @subroutine_with_allocation_0([[reg_device]], [[q0_alloc]], [[zero]]) : (!qref.reg<3>, !qref.bit, tensor<i64>) -> ()
        # CHECK: call @subroutine_with_allocation_1([[reg_device]], [[q0_alloc]], [[q1_alloc]]) : (!qref.reg<3>, !qref.bit, !qref.bit) -> ()
        subroutine_with_allocation(wires=[0, 1])
        subroutine_with_allocation(wires=[q[0], 0])
        subroutine_with_allocation(wires=[q[0], q[1]])

    # CHECK: qref.dealloc [[reg_alloc]] : !qref.reg<2>

    return qp.expval(qp.X(0))


# CHECK: func.func private @subroutine_with_allocation(%arg0: !qref.reg<3>, %arg1: tensor<i64>, %arg2: tensor<i64>)
# CHECK:   [[alloc:%.+]] = qref.alloc( 1) : !qref.reg<1>
# CHECK:   [[q_alloc:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<1> -> !qref.bit
# CHECK:   [[extract_arg1:%.+]] = tensor.extract %arg1[] : tensor<i64>
# CHECK:   [[q_arg1:%.+]] = qref.get %arg0[[[extract_arg1]]] : !qref.reg<3>, i64 -> !qref.bit
# CHECK:   [[extract_arg2:%.+]] = tensor.extract %arg2[] : tensor<i64>
# CHECK:   [[q_arg2:%.+]] = qref.get %arg0[[[extract_arg2]]] : !qref.reg<3>, i64 -> !qref.bit
# CHECK:   qref.custom "Toffoli"() [[q_alloc]], [[q_arg1]], [[q_arg2]] : !qref.bit, !qref.bit, !qref.bit
# CHECK:   qref.dealloc [[alloc]] : !qref.reg<1>
# CHECK:   return

# CHECK: func.func private @subroutine_with_allocation_0(%arg0: !qref.reg<3>, %arg1: !qref.bit, %arg2: tensor<i64>)
# CHECK:   [[alloc:%.+]] = qref.alloc( 1) : !qref.reg<1>
# CHECK:   [[q_alloc:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<1> -> !qref.bit
# CHECK:   [[extract_arg2:%.+]] = tensor.extract %arg2[] : tensor<i64>
# CHECK:   [[q_arg2:%.+]] = qref.get %arg0[[[extract_arg2]]] : !qref.reg<3>, i64 -> !qref.bit
# CHECK:   qref.custom "Toffoli"() [[q_alloc]], %arg1, [[q_arg2]] : !qref.bit, !qref.bit, !qref.bit
# CHECK:   qref.dealloc [[alloc]] : !qref.reg<1>
# CHECK:   return

# CHECK: func.func private @subroutine_with_allocation_1(%arg0: !qref.reg<3>, %arg1: !qref.bit, %arg2: !qref.bit)
# CHECK:   [[alloc:%.+]] = qref.alloc( 1) : !qref.reg<1>
# CHECK:   [[q_alloc:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<1> -> !qref.bit
# CHECK:   qref.custom "Toffoli"() [[q_alloc]], %arg1, %arg2 : !qref.bit, !qref.bit, !qref.bit
# CHECK:   qref.dealloc [[alloc]] : !qref.reg<1>
# CHECK:   return

print(test_subroutine_with_allocation.mlir)
