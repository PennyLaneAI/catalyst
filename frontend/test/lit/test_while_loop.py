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

from catalyst import qjit, while_loop


# CHECK-NOT: Verification failed
# CHECK-LABEL: @jit_circuit
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit(n: int):
    # CHECK:   scf.while ([[v0:%.+]] = {{%.+}}, [[v1:%.+]] = {{%.+}}, [[array0:%.+]] = {{%.+}})
    # CHECK:       [[ct:%.+]] = stablehlo.compare LT, [[v0]], [[v1]], SIGNED
    # CHECK:       [[cond:%.+]] = "tensor.extract"([[ct]])
    # CHECK:       scf.condition([[cond]]) [[v0]], [[v1]], [[array0]]

    # CHECK:   ^bb0([[v0:%.+]]: tensor<i64>, [[v1:%.+]]: tensor<i64>, [[array0:%.+]]: !quantum.reg):
    # CHECK:       [[v0p:%.+]] = stablehlo.add [[v0]]
    # CHECK:       [[q0:%.+]] = "quantum.extract"([[array0]], {{%.+}})
    # CHECK:       [[q1:%[a-zA-Z0-9_]]] = "quantum.custom"([[q0]]) {gate_name = "PauliX"
    # CHECK:       [[array1:%.+]] = "quantum.insert"([[array0]], {{%.+}}, [[q1]])
    # CHECK:       scf.yield [[v0p]], [[v1]], [[array1]]
    @while_loop(lambda v: v[0] < v[1])
    def loop(v):
        qml.PauliX(wires=0)
        return v[0] + 1, v[1]

    out = loop((0, n))
    return out[0]


print(circuit.mlir)


# CHECK-NOT: Verification failed
# CHECK-LABEL: func.func public @jit_circuit_outer_scope_reference
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit_outer_scope_reference(n: int):
    # CHECK:   [[array0:%.+]] = "quantum.alloc"

    # CHECK:   scf.while ([[v0:%.+]] = {{%.+}}, [[array_inner:%.+]] = {{%.+}})
    # CHECK:       [[ct:%.+]] = stablehlo.compare LT, [[v0]], %arg0, SIGNED
    # CHECK:       [[cond:%.+]] = "tensor.extract"([[ct]])
    # CHECK:       scf.condition([[cond]]) [[v0]], [[array_inner]]

    # CHECK:   ^bb0([[v0:%.+]]: tensor<i64>, [[array_inner:%.+]]: !quantum.reg):
    # CHECK:       [[v0p:%[a-zA-Z0-9_]]] = stablehlo.add [[v0]]
    # CHECK:       [[q0:%.+]] = "quantum.extract"([[array_inner]], {{%.+}})
    # CHECK:       [[q1:%[a-zA-Z0-9_]]] = "quantum.custom"([[q0]]) {gate_name = "PauliX"
    # CHECK:       [[array_inner_2:%.+]] = "quantum.insert"([[array_inner]], {{%.+}}, [[q1]])
    # CHECK:       scf.yield [[v0p]], [[array_inner_2]]
    @while_loop(lambda i: i < n)
    def loop(i):
        qml.PauliX(wires=0)
        return i + 1

    # CHECK:   "quantum.dealloc"([[array0]])
    # CHECK:   return
    return loop(0)


print(circuit_outer_scope_reference.mlir)


# CHECK-NOT: Verification failed
# CHECK-LABEL: public @jit_circuit_multiple_args
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit_multiple_args(n: int):
    # CHECK:   [[R0:%.+]] = "quantum.alloc"() {{.+}} -> !quantum.reg
    # CHECK:   [[C0:%.+]] = stablehlo.constant dense<0> : tensor<i64>
    # CHECK:   [[C1:%.+]] = stablehlo.constant dense<1> : tensor<i64>

    # CHECK:   scf.while ([[w0:%.+]] = [[C0]], [[w1:%.+]] = %arg0, [[w2:%.+]] = [[C1]], [[w3:%.+]] = [[R0]])
    # CHECK:       [[LT:%.+]] = stablehlo.compare LT, [[w0]], [[w1]], SIGNED
    # CHECK:       [[COND:%.+]] = "tensor.extract"([[LT]])
    # CHECK:       scf.condition([[COND]]) [[w0]], [[w1]], [[w2]], [[w3]]

    # CHECK:   ^bb0([[w0:%.+]]: tensor<i64>, [[w1:%.+]]: tensor<i64>, [[w2:%.+]]: tensor<i64>, [[w3:%.+]]: !quantum.reg):
    # CHECK:       [[V0p:%.+]] = stablehlo.add [[w0]], [[w2]]
    # CHECK:       [[Q0:%.+]] = "quantum.extract"([[w3]]
    # CHECK:       [[Q1:%.+]] = "quantum.custom"([[Q0]]) {gate_name = "PauliX"
    # CHECK:       [[QREGp:%.+]] = "quantum.insert"([[w3]], {{%.+}}, [[Q1]])
    # CHECK:       scf.yield [[V0p]], [[w1]], [[w2]], [[QREGp]]
    @while_loop(lambda v, _: v[0] < v[1])
    def loop(v, inc):
        qml.PauliX(wires=0)
        return (v[0] + inc, v[1]), inc

    out = loop((0, n), 1)
    # CHECK:   "quantum.dealloc"([[R0]])
    # CHECK:   return
    return out[0]


print(circuit_multiple_args.mlir)
