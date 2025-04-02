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

from catalyst import measure, qjit


@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
# CHECK-LABEL @f.jit
def f(arg0: float, arg1: int, arg2: int):
    # CHECK:   [[reg0:%.+]] = quantum.alloc( 2)
    # CHECK:   [[w0_0:%.+]] = tensor.extract %arg1
    # CHECK:   [[q_w0_0:%.+]] = quantum.extract [[reg0]][[[w0_0]]]
    # CHECK:   [[q_w0_1:%.+]] = quantum.custom "RZ"({{%.+}}) [[q_w0_0]]
    qml.RZ(arg0, wires=[arg1])
    # CHECK:   [[w0_1:%.+]] = tensor.extract %arg1
    # CHECK:   [[reg1:%.+]] = quantum.insert [[reg0]][[[w0_1]]], [[q_w0_1]]
    # CHECK:   [[w1_0:%.+]] = tensor.extract %arg2
    # CHECK:   [[q_w1_0:%.+]] = quantum.extract [[reg1]][[[w1_0]]]
    # CHECK:   [[mres:%.+]], [[out_qubit:%.+]] = quantum.measure [[q_w1_0]]
    m = measure(wires=[arg2])
    # CHECK:   [[w2:%.+]] = tensor.extract %arg2
    # CHECK:   [[reg2:%.+]] = quantum.insert [[reg1]][[[w2]]], [[out_qubit]] : !quantum.reg, !quantum.bit
    # CHECK:   quantum.dealloc [[reg2]]
    # CHECK:   return
    return m


print(f.mlir)
