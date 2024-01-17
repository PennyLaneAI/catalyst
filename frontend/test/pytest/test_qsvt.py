# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test PennyLane's QSVT operations and workflows"""

import numpy as np
import pennylane as qml

from catalyst import qjit


def test_BlockEncode(backend):
    """Test the decomposition of BlockEncode in a QJIT decorated workflow."""
    dev = qml.device(backend, wires=2)

    A = np.array([[0.1]])
    block_encode = qml.BlockEncode(A, wires=[0, 1])
    shifts = [qml.PCPhase(i + 0.1, dim=1, wires=[0, 1]) for i in range(3)]

    @qjit
    @qml.qnode(dev)
    def QSVT_example():
        qml.QSVT(block_encode, shifts)
        return qml.expval(qml.PauliZ(wires=0))

    assert np.allclose(QSVT_example(), [1.0])
