# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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

import pennylane as qp

from catalyst import qjit, while_loop


# CHECK-NOT: Verification failed
# CHECK-LABEL: @jit_circuit
@qjit(target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=1))
def circuit(n: int):
    # CHECK-LABEL: public @circuit(
    # CHECK-DAG: [[one:%.+]] = stablehlo.constant dense<1> : tensor<i64>
    # CHECK-DAG: [[zero:%.+]] = stablehlo.constant dense<0> : tensor<i64>
    # CHECK: [[reg:%.+]] = qref.alloc( 1) : !qref.reg<1>
    # CHECK: scf.while ([[value:%.+]] = [[zero]]) : (tensor<i64>) -> tensor<i64>
    # CHECK: [[comparison:%.+]] = stablehlo.compare LT, [[value]], %arg0, SIGNED
    # CHECK: [[condition:%.+]] = tensor.extract [[comparison]]
    # CHECK: scf.condition([[condition]]) [[value]] : tensor<i64>
    # CHECK: ^bb0([[value:%.+]]: tensor<i64>):
    # CHECK: [[qubit:%.+]] = qref.get [[reg]][ 0]
    # CHECK: qref.custom "PauliX"() [[qubit]]
    # CHECK: [[next:%.+]] = stablehlo.add [[value]], [[one]]
    # CHECK: scf.yield [[next]] : tensor<i64>
    # CHECK: qref.dealloc [[reg]]
    # CHECK: return
    @qp.while_loop(lambda v: v[0] < v[1])
    def loop(v):
        qp.PauliX(wires=0)
        return v[0] + 1, v[1]

    loop((0, n))
    return qp.expval(qp.Z(0))


print(circuit.mlir)


# CHECK-NOT: Verification failed
# CHECK-LABEL: func.func public @jit_circuit_outer_scope_reference
@qjit(target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=1))
def circuit_outer_scope_reference(n: int):
    # CHECK-LABEL: public @circuit_outer_scope_reference(
    # CHECK-DAG: [[one:%.+]] = stablehlo.constant dense<1> : tensor<i64>
    # CHECK-DAG: [[zero:%.+]] = stablehlo.constant dense<0> : tensor<i64>
    # CHECK: [[reg:%.+]] = qref.alloc( 1) : !qref.reg<1>
    # CHECK: scf.while ([[value:%.+]] = [[zero]]) : (tensor<i64>) -> tensor<i64>
    # CHECK: [[comparison:%.+]] = stablehlo.compare LT, [[value]], %arg0, SIGNED
    # CHECK: [[condition:%.+]] = tensor.extract [[comparison]]
    # CHECK: scf.condition([[condition]]) [[value]] : tensor<i64>
    # CHECK: ^bb0([[value:%.+]]: tensor<i64>):
    # CHECK: [[qubit:%.+]] = qref.get [[reg]][ 0]
    # CHECK: qref.custom "PauliX"() [[qubit]]
    # CHECK: [[next:%.+]] = stablehlo.add [[value]], [[one]]
    # CHECK: scf.yield [[next]] : tensor<i64>
    # CHECK: qref.dealloc [[reg]]
    # CHECK: return
    @qp.while_loop(lambda i: i < n)
    def loop(i):
        qp.PauliX(wires=0)
        return i + 1

    loop(0)
    return qp.expval(qp.Z(0))


print(circuit_outer_scope_reference.mlir)


# CHECK-NOT: Verification failed
# CHECK-LABEL: public @jit_circuit_multiple_args
@qjit(target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=1))
def circuit_multiple_args(n: int):
    # CHECK-LABEL: public @circuit_multiple_args(
    # CHECK-DAG: [[one:%.+]] = stablehlo.constant dense<1> : tensor<i64>
    # CHECK-DAG: [[zero:%.+]] = stablehlo.constant dense<0> : tensor<i64>
    # CHECK: [[reg:%.+]] = qref.alloc( 1) : !qref.reg<1>
    # CHECK: scf.while ([[value:%.+]] = [[zero]]) : (tensor<i64>) -> tensor<i64>
    # CHECK: [[comparison:%.+]] = stablehlo.compare LT, [[value]], %arg0, SIGNED
    # CHECK: [[condition:%.+]] = tensor.extract [[comparison]]
    # CHECK: scf.condition([[condition]]) [[value]] : tensor<i64>
    # CHECK: ^bb0([[value:%.+]]: tensor<i64>):
    # CHECK: [[qubit:%.+]] = qref.get [[reg]][ 0]
    # CHECK: qref.custom "PauliX"() [[qubit]]
    # CHECK: [[next:%.+]] = stablehlo.add [[value]], [[one]]
    # CHECK: scf.yield [[next]] : tensor<i64>
    # CHECK: qref.dealloc [[reg]]
    # CHECK: return
    @qp.while_loop(lambda v, _: v[0] < v[1])
    def loop(v, inc):
        qp.PauliX(wires=0)
        return (v[0] + inc, v[1]), inc

    loop((0, n), 1)
    return qp.expval(qp.Z(0))


print(circuit_multiple_args.mlir)


# The `estimated_iterations` hint (here a fractional value) is attached as an f64
# `catalyst.estimated_iterations` attribute on the emitted `scf.while`.
# CHECK-NOT: Verification failed
# CHECK-LABEL: public @jit_circuit_estimated_iterations
# old_frontend: control-flow hints are specific to the Catalyst API.
@qjit(capture=False, target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=1))
def circuit_estimated_iterations(n: int):
    # CHECK:   scf.while
    # CHECK:   catalyst.estimated_iterations = 2.500000e+00
    @while_loop(lambda i: i < n, estimated_iterations=2.5)
    def loop(i):
        qp.PauliX(wires=0)
        return i + 1

    loop(0)
    return qp.expval(qp.Z(0))


print(circuit_estimated_iterations.mlir)
