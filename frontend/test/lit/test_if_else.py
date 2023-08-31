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

from catalyst import cond, qjit


# CHECK-NOT: Verification failed
# CHECK-LABEL: public @jit_circuit
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit(n: int):
    # CHECK-DAG:   [[qreg_0:%[a-zA-Z0-9_]+]] = "quantum.alloc"
    # CHECK-DAG:   [[c5:%[a-zA-Z0-9_]+]] = stablehlo.constant dense<5> : tensor<i64>
    # CHECK:       [[b_t:%[a-zA-Z0-9_]+]] = stablehlo.compare  LE, %arg0, [[c5]], SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    # CHECK:       [[b:%[a-zA-Z0-9_]+]] = "tensor.extract"([[b_t]])
    @cond(n <= 5)
    # CHECK:       scf.if [[b]]
    def cond_fn():
        # CHECK-DAG:   [[q0:%[a-zA-Z0-9_]+]] = "quantum.extract"
        # CHECK-DAG:   [[q1:%[a-zA-Z0-9_]+]] = "quantum.custom"([[q0]]) {gate_name = "PauliX"
        # CHECK-DAG:   [[qreg_1:%[a-zA-Z0-9_]+]] = "quantum.insert"([[qreg_0]], {{%[a-zA-Z0-9_]+}}, [[q1]])
        # CHECK:       scf.yield %arg0, [[qreg_1]]
        qml.PauliX(wires=0)
        return n

    @cond_fn.otherwise
    def otherwise():
        # CHECK:       [[r0:%[a-zA-Z0-9_a-z]+]] = stablehlo.multiply %arg0, %arg0
        # CHECK:       [[r1:%[a-zA-Z0-9_a-z]+]] = stablehlo.multiply %arg0, [[r0]]
        # CHECK:       scf.yield [[r1]], [[qreg_0]]
        return n**3

    out = cond_fn()
    # CHECK:       "quantum.dealloc"([[qreg_0]])
    # CHECK:       return
    return out


print(circuit.mlir)
