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

import subprocess
from catalyst import qjit, for_loop
import pennylane as qml


# CHECK-NOT: Verification failed
# CHECK-LABEL: @jit_loop_circuit
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def loop_circuit(n: int, inc: float):
    # CHECK-DAG:   [[qreg:%.+]] = "quantum.alloc"
    # CHECK-DAG:   [[c0:%.+]] = arith.constant 0 : index
    # CHECK-DAG:   [[c1:%.+]] = arith.constant 1 : index
    # CHECK-DAG:   [[init:%.+]] = mhlo.constant dense<0.0{{.+}}>

    # CHECK-DAG:   [[c1_t:%.+]] = mhlo.constant dense<1>
    # CHECK-DAG:   [[n_1_t:%.+]] = mhlo.subtract %arg0, [[c1_t]]
    # CHECK-DAG:   [[n_1:%.+]] = tensor.extract [[n_1_t]]
    # CHECK-DAG:   [[ub:%.+]] = arith.index_cast [[n_1]]

    # CHECK:       scf.for [[i:%.+]] = [[c0]] to [[ub]] step [[c1]] iter_args([[phi0:%.+]] = [[init]], [[r0:%.+]] = [[qreg]])
    @for_loop(0, n - 1, 1)
    def loop_fn(i, phi):
        # CHECK:       [[i_cast:%.+]] = arith.index_cast [[i]]
        # CHECK:       [[phi1:%.+]] = mhlo.add [[phi0]], %arg1

        # CHECK:       [[q0:%.+]] = "quantum.extract"([[r0]], [[i_cast]])
        # CHECK:       [[phi_e:%.+]] = tensor.extract [[phi0]]
        # CHECK:       [[q1:%.+]] = "quantum.custom"([[phi_e]], [[q0]]) {gate_name = "RY"
        # CHECK:       [[r1:%.+]] = "quantum.insert"([[r0]], [[i_cast]], [[q1]])
        qml.RY(phi, wires=i)

        # CHECK:       scf.yield [[phi1]], [[r1]]
        return phi + inc

    loop_fn(0.0)
    # CHECK:       "quantum.dealloc"([[qreg]])
    # CHECK:       return
    return qml.state()


# TODO: replace with internally applied canonicalization (#241)
subprocess.run(
    ["mlir-hlo-opt", "--canonicalize", "--allow-unregistered-dialect"],
    input=loop_circuit.mlir,
    text=True,
)
