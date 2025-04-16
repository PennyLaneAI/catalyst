# Copyright 2024 Xanadu Quantum Technologies Inc.

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

from catalyst import qjit


# CHECK-LABEL: @adjoint_adjoint
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=1))
def adjoint_adjoint():
    qml.adjoint(qml.adjoint(qml.S(0)))
    return qml.probs()


# CHECK:   quantum.custom "S"() %{{[^\s]+}} : !quantum.bit
print(adjoint_adjoint.mlir)


# -----


# CHECK-LABEL: @adjoint_ctrl_adjoint
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def adjoint_ctrl_adjoint():
    qml.adjoint(qml.ctrl(qml.adjoint(qml.S(0)), control=1))
    return qml.probs()


# CHECK:   quantum.custom "S"() %{{[^\s]+}} ctrls(%{{[^\s]+}}) ctrlvals(%{{[^\s]+}}) : !quantum.bit
print(adjoint_ctrl_adjoint.mlir)
