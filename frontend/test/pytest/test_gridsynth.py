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

"""Test cases for the gridsynth discretization/decomposition pass."""

import pennylane as qml
import pytest


@pytest.mark.parametrize(
    "param",
    [-11.1, -7.7, -4.4, -2.2, -1.1, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 1.1, 2.2, 4.4, 7.7, 11.1],
)
@pytest.mark.parametrize("op", [qml.RZ, qml.PhaseShift])
@pytest.mark.parametrize("eps", [1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
def test_PhaseShift_gridsynth(param, op, eps):
    """Test that PhaseShift gates are correctly decomposed using the gridsynth pass."""

    qml.capture.enable()

    dev = qml.device("lightning.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(x: float):
        qml.Hadamard(0)
        op(x, wires=0)
        return qml.state()

    expected = circuit(param)
    gridsynthed_circuit = qml.transforms.gridsynth(circuit, epsilon=eps)
    qjitted_circuit = qml.qjit(gridsynthed_circuit)
    result = qjitted_circuit(param)
    qml.capture.disable()

    assert qml.math.allclose(result, expected, atol=eps)
