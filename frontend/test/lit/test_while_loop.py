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
    # CHECK:   "scf.while"({{%[a-zA-Z0-9_]+}}, {{%[a-zA-Z0-9_]+}}, {{%[a-zA-Z0-9_]+}})
    # CHECK:   [[a1p:%[a-zA-Z0-9_]+]] = mhlo.convert %arg1
    # CHECK:   [[ct:%[a-zA-Z0-9_]+]] = mhlo.compare  LT, [[a1p]], %arg2,  SIGNED
    # CHECK:   [[cond:%[a-zA-Z0-9_]+]] = tensor.extract [[ct]]
    # CHECK:   "scf.condition"([[cond]], %arg1, %arg2, %arg3)
    @while_loop(lambda v: v[0] < v[1])
    # CHECK:   ^bb0([[v0:%[a-zA-Z0-9_]+]]: tensor<i64>, [[v1:%[a-zA-Z0-9_]+]]: tensor<i64>, [[array0:%[a-zA-Z0-9_]+]]: !quantum.reg):
    def loop(v):
        # CHECK:   [[v0p:%[a-zA-Z0-9_]+]] = mhlo.add [[v0]]
        # CHECK:   [[q0:%[a-zA-Z0-9_]+]] = "quantum.extract"(%arg3, {{%[a-zA-Z0-9_]+}})
        # CHECK:   [[q1:%[a-zA-Z0-9_]]] = "quantum.custom"([[q0]]) {gate_name = "PauliX"
        qml.PauliX(wires=0)
        # CHECK:   [[array1:%[a-zA-Z0-9_]+]] = "quantum.insert"(%arg3, {{%[a-zA-Z0-9_]+}}, [[q1]])
        # CHECK:   "scf.yield"([[v0p]], [[v1]], [[array1]])
        return v[0] + 1, v[1]

    out = loop((0, n))
    return out[0]


print(circuit.mlir)


# CHECK-NOT: Verification failed
# CHECK-LABEL: func.func public @jit_circuit_outer_scope_reference
# CHECK-SAME: ([[c1:%[a-zA-Z0-9_]+]]
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit_outer_scope_reference(n: int):
    # CHECK:   [[array0:%[a-zA-Z0-9_]+]] = "quantum.alloc"
    # CHECK:   "scf.while"({{%[a-zA-Z0-9_]+}}, {{%[a-zA-Z0-9_]+}})
    # CHECK:   [[a1p:%[a-zA-Z0-9_]+]] = mhlo.convert %arg1
    # CHECK:   [[ct:%[a-zA-Z0-9_]+]] = mhlo.compare  LT, [[a1p]], [[c1]],  SIGNED
    # CHECK:   [[cond:%[a-zA-Z0-9_]+]] = tensor.extract [[ct]]
    # CHECK:   "scf.condition"([[cond]], %arg1, %arg2)
    @while_loop(lambda i: i < n)
    # CHECK:   ^bb0([[v0:%[a-zA-Z0-9_]+]]: tensor<i64>, [[array_inner:%[a-zA-Z0-9_]+]]: !quantum.reg):
    def loop(i):
        # CHECK:   [[v0p:%[a-zA-Z0-9_]]] = mhlo.add [[v0]]
        # CHECK:   [[q0:%[a-zA-Z0-9_]+]] = "quantum.extract"([[array_inner]], {{%[a-zA-Z0-9_]+}})
        # CHECK:   [[q1:%[a-zA-Z0-9_]]] = "quantum.custom"([[q0]]) {gate_name = "PauliX"
        # CHECK:   [[array_inner_2:%[a-zA-Z0-9_]+]] = "quantum.insert"([[array_inner]], {{%[a-zA-Z0-9_]+}}, [[q1]])
        qml.PauliX(wires=0)
        # CHECK:   "scf.yield"([[v0p]], [[array_inner_2]])
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
    # CHECK:   [[C0:%.+]] = mhlo.constant dense<0> : tensor<i64>
    # CHECK:   [[C1:%.+]] = mhlo.constant dense<1> : tensor<i64>

    # CHECK:   "scf.while"([[C0]], %arg0, [[C1]], [[R0]])
    # CHECK:       [[V0p:%.+]] = mhlo.convert %arg1
    # CHECK:       [[LT:%.+]] = mhlo.compare  LT, [[V0p]], %arg2,  SIGNED
    # CHECK:       [[COND:%.+]] = tensor.extract [[LT]]
    # CHECK:       "scf.condition"([[COND]], %arg1, %arg2, %arg3, %arg4)

    # CHECK:   ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<i64>, %arg4: !quantum.reg):
    # CHECK:       [[V0p:%.+]] = mhlo.add %arg1, %arg3
    # CHECK:       [[Q0:%.+]] = "quantum.extract"(%arg4
    # CHECK:       [[Q1:%.+]] = "quantum.custom"([[Q0]]) {gate_name = "PauliX"
    # CHECK:       [[QREGp:%.+]] = "quantum.insert"(%arg4, {{%[a-zA-Z0-9_]+}}, [[Q1]])
    # CHECK:       "scf.yield"([[V0p]], %arg2, %arg3, [[QREGp]])
    @while_loop(lambda v, _: v[0] < v[1])
    def loop(v, inc):
        qml.PauliX(wires=0)
        return (v[0] + inc, v[1]), inc

    out = loop((0, n), 1)
    # CHECK:   "quantum.dealloc"([[R0]])
    # CHECK:   return
    return out[0]


print(circuit_multiple_args.mlir)
