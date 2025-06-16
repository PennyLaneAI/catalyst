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
Unit tests for the dynamic work wire allocation.
Note that this feature is only available under the plxpr pipeline.
"""

import numpy as np

import pennylane as qml
from pennylane.allocation import allocate, deallocate
from catalyst import qjit


def test_basic_dynamic_wire_alloc():

    qml.capture.enable()

    @qjit
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def circuit():
        qml.X(1)  # |010>

        wires = allocate(1)  # |010> and |0>
        qml.X(wires[0])  # |010> and |1>
        qml.CNOT(wires=[wires[0], 2])  # |011> and |1>
        deallocate(wires[0])  # |011>

        return qml.probs(wires=[0, 1, 2])

    observed = circuit()
    qml.capture.disable()

    expected = [0, 0, 0, 1, 0, 0, 0, 0]
    assert np.allclose(expected, observed)
