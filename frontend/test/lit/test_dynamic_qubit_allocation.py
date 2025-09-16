# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for the dynamic qubit allocation.
"""

# RUN: %PYTHON %s | FileCheck %s

import pennylane as qml
from pennylane.allocation import allocate, deallocate

from catalyst import qjit
from catalyst.jax_primitives import qalloc_p, qdealloc_qb_p, qextract_p


@qjit
def test_single_qubit_dealloc():
    """
    Unit test for the single qubit dealloc primitive's lowerings.
    """

    # CHECK: [[qubit:.]]:AbstractQbit() = qextract {{.+}} 3
    # CHECK: qdealloc_qb [[qubit]]

    # CHECK: [[qubit:%.+]] = quantum.extract {{.+}} 3
    # CHECK: quantum.dealloc_qb [[qubit]] : !quantum.bit

    qreg = qalloc_p.bind(10)
    qubit = qextract_p.bind(qreg, 3)
    qdealloc_qb_p.bind(qubit)


print(test_single_qubit_dealloc.jaxpr)
print(test_single_qubit_dealloc.mlir)


qml.capture.enable()


@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def test_basic_dynalloc():
    qml.X(1)
    qml.X(1)
    wires = allocate(1)
    qml.X(wires[0])
    qml.Z(wires[0])
    deallocate(wires[0])

    qml.X(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)

    wires = allocate(2)
    qml.Y(wires[0])
    qml.Z(wires[1])
    deallocate(wires[:])

    return qml.probs()


print(test_basic_dynalloc.mlir)


qml.capture.disable()
