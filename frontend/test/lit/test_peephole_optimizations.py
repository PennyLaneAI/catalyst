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

# pylint: disable=line-too-long

import jax
import pennylane as qml

from catalyst import qjit, cancel_inverses

# CHECK-LABEL: public @jit_cancel_inverses_not_applied
@qjit(target="mlir")
def cancel_inverses_not_applied(x: float):
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float):
        qml.RX(x, wires=0)
        qml.PauliX(wires=0)
        qml.PauliX(wires=0)
        return qml.expval(qml.PauliY(0))

    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def g(x: float):
        qml.RX(x, wires=1)
        qml.CNOT(wires=[0,1])
        qml.CNOT(wires=[0,1])
        return qml.expval(qml.PauliY(0))

    # CHECK: {{%.+}} = quantum.custom "PauliX"() {{%.+}} : !quantum.bit
    # CHECK: {{%.+}} = quantum.custom "PauliX"() {{%.+}} : !quantum.bit
    ff = f(42.42)

    # CHECK: {{%.+}} = quantum.custom "CNOT"() {{%.+}}, {{%.+}} : !quantum.bit, !quantum.bit
    # CHECK: {{%.+}} = quantum.custom "CNOT"() {{%.+}}, {{%.+}} : !quantum.bit, !quantum.bit
    gg = g(42.42)
    return ff, gg


print(cancel_inverses_not_applied.mlir)


# CHECK-LABEL: public @jit_cancel_inverses_workflow
@qjit(target="mlir")
def cancel_inverses_workflow(x: float):
    @cancel_inverses
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float):
        qml.RX(x, wires=0)
        qml.PauliX(wires=0)
        qml.PauliX(wires=0)
        return qml.expval(qml.PauliY(0))

    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def g(x: float):
        qml.RX(x, wires=1)
        qml.CNOT(wires=[0,1])
        qml.CNOT(wires=[0,1])
        return qml.expval(qml.PauliY(0))

    # CHECK-NOT: {{%.+}} = quantum.custom "PauliX"() {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "PauliX"() {{%.+}} : !quantum.bit
    ff = f(42.42)

    # CHECK: {{%.+}} = quantum.custom "CNOT"() {{%.+}}, {{%.+}} : !quantum.bit, !quantum.bit
    # CHECK: {{%.+}} = quantum.custom "CNOT"() {{%.+}}, {{%.+}} : !quantum.bit, !quantum.bit
    gg = g(42.42)
    return ff, gg

print(cancel_inverses_workflow.mlir)