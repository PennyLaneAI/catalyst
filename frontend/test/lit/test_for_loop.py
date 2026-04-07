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

import pennylane as qml

from catalyst import for_loop, qjit


# CHECK-NOT: Verification failed
# CHECK-LABEL: @jit_loop_circuit
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def loop_circuit(n: int, inc: float):
    # CHECK-DAG:   [[qreg:%.+]] = quantum.alloc
    # CHECK-DAG:   [[c0:%.+]] = arith.constant 0 : index
    # CHECK-DAG:   [[c1:%.+]] = arith.constant 1 : index
    # CHECK-DAG:   [[init:%.+]] = stablehlo.constant dense<0.0{{.+}}>

    # CHECK-DAG:   [[c1_t:%.+]] = stablehlo.constant dense<1>
    # CHECK-DAG:   [[n_1_t:%.+]] = stablehlo.subtract %arg0, [[c1_t]]
    # CHECK-DAG:   [[n_1:%.+]] = tensor.extract [[n_1_t]]
    # CHECK-DAG:   [[ub:%.+]] = arith.index_cast [[n_1]]

    # CHECK:       [[newqreg:%.+]]:2 = scf.for [[i:%.+]] = [[c0]] to [[ub]] step [[c1]] iter_args([[phi0:%.+]] = [[init]], [[r0:%.+]] = [[qreg]])
    @for_loop(0, n - 1, 1)
    def loop_fn(i, phi):
        # CHECK:       [[i_cast:%.+]] = arith.index_cast [[i]]
        # CHECK:       [[phi1:%.+]] = stablehlo.add %arg3, %arg1

        # CHECK:       [[q0:%.+]] = quantum.extract [[r0]][[[i_cast]]]
        # CHECK:       [[phi_e:%.+]] = tensor.extract [[phi0]]
        # CHECK:       [[q1:%.+]] = quantum.custom "RY"([[phi_e]]) [[q0]]
        # CHECK:       [[r1:%.+]] = quantum.insert [[r0]][[[i_cast]]], [[q1]]
        qml.RY(phi, wires=i)

        # CHECK:       scf.yield [[phi1]], [[r1]]
        return phi + inc

    loop_fn(0.0)
    # CHECK:       quantum.dealloc [[newqreg]]#1
    # CHECK:       return
    return qml.state()


print(loop_circuit.mlir)
