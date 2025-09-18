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
    """
    Test basic qml.allocate and qml.deallocate.

    Test both the explicit call API and the context manager API.
    """

    # CHECK: [[device_init_qreg:%.+]] = quantum.alloc( 3)

    # CHECK: [[dyn_qreg:%.+]] = quantum.alloc( 2)
    # CHECK: [[dyn_bit0:%.+]] = quantum.extract [[dyn_qreg]][ 0]
    # CHECK: [[dyn_bit1:%.+]] = quantum.extract [[dyn_qreg]][ 1]
    # CHECK: [[Xout:%.+]] = quantum.custom "PauliX"() [[dyn_bit0]]
    # CHECK: [[dev_bit2:%.+]] = quantum.extract [[device_init_qreg]][ 2]
    # CHECK: [[CNOTout:%.+]]:2 = quantum.custom "CNOT"() [[dyn_bit1]], [[dev_bit2]]
    # COM: dealloc blocked by https://github.com/PennyLaneAI/catalyst/pull/1996
    # COM: CHECK: [[insert0:%.+]] = quantum.insert [[dyn_qreg]][ 0], [[Xout]]
    # COM: CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[CNOTout]]#0
    # COM: CHECK: quantum.dealloc [[insert1]]

    qs = qml.allocate(2)
    qml.X(qs[0])
    qml.CNOT(wires=[qs[1], 2])
    qml.deallocate(qs[:])

    # CHECK: [[dyn_qreg:%.+]] = quantum.alloc( 4)
    # CHECK: [[dyn_bit1:%.+]] = quantum.extract [[dyn_qreg]][ 1]
    # CHECK: [[dyn_bit2:%.+]] = quantum.extract [[dyn_qreg]][ 2]
    # CHECK: [[Xout:%.+]] = quantum.custom "PauliX"() [[dyn_bit1]]
    # CHECK: [[dev_bit1:%.+]] = quantum.extract [[device_init_qreg]][ 1]
    # CHECK: [[CNOTout:%.+]]:2 = quantum.custom "CNOT"() [[dyn_bit2]], [[dev_bit1]]
    # COM: dealloc blocked by https://github.com/PennyLaneAI/catalyst/pull/1996
    # COM: CHECK: [[insert0:%.+]] = quantum.insert [[dyn_qreg]][ 1], [[Xout]]
    # COM: CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 2], [[CNOTout]]#0
    # COM: CHECK: quantum.dealloc [[insert1]]

    with qml.allocate(4) as qs1:
        qml.X(qs1[1])
        qml.CNOT(wires=[qs1[2], 1])

    return qml.probs()


print(test_basic_dynalloc.mlir)


qml.capture.disable()
