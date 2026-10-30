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

from catalyst import for_loop, qjit


# CHECK-NOT: Verification failed
# CHECK-LABEL: @jit_loop_circuit
@qjit(target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=3))
def loop_circuit(n: int, inc: float):
    # CHECK-LABEL: public @loop_circuit(
    # CHECK-DAG: [[c0:%.+]] = arith.constant 0 : index
    # CHECK-DAG: [[c1:%.+]] = arith.constant 1 : index
    # CHECK-DAG: [[init:%.+]] = stablehlo.constant dense<0.0{{.+}}>
    # CHECK-DAG: [[one:%.+]] = stablehlo.constant dense<1> : tensor<i64>
    # CHECK: [[qreg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[limit:%.+]] = stablehlo.subtract %arg0, [[one]]
    # CHECK: [[limit_scalar:%.+]] = tensor.extract [[limit]]
    # CHECK: [[ub:%.+]] = arith.index_cast [[limit_scalar]]
    # CHECK: scf.for [[i:%.+]] = [[c0]] to [[ub]] step [[c1]] iter_args([[phi:%.+]] = [[init]])
    # CHECK: [[wire:%.+]] = arith.index_cast [[i]]
    # CHECK: [[qubit:%.+]] = qref.get [[qreg]][[[wire]]]
    # CHECK: [[angle:%.+]] = tensor.extract [[phi]]
    # CHECK: qref.custom "RY"([[angle]]) [[qubit]]
    # CHECK: [[next:%.+]] = stablehlo.add [[phi]], %arg1
    # CHECK: scf.yield [[next]]
    # CHECK: qref.dealloc [[qreg]]
    # CHECK: return
    def loop_fn(i, phi):
        qp.RY(phi, wires=i)
        return phi + inc

    qp.for_loop(0, n - 1, 1)(loop_fn)(0.0)
    return qp.state()


print(loop_circuit.mlir)


# The `estimated_iterations` hint is attached as an f64 `catalyst.estimated_iterations`
# attribute on the emitted `scf.for`.
# CHECK-LABEL: @jit_loop_estimated_iterations
# old_frontend: control-flow hints are specific to the Catalyst API.
@qjit(capture=False, target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=1))
def loop_estimated_iterations(n: int):
    # CHECK:       scf.for
    # CHECK:       catalyst.estimated_iterations = 5.000000e+00
    @for_loop(0, n, 1, estimated_iterations=5)
    def loop_fn(i):
        qp.PauliX(wires=0)

    loop_fn()
    return qp.state()


print(loop_estimated_iterations.mlir)
