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
"""Tests for seeded qjit runs in Catalyst"""

import numpy as np
import pennylane as qml
import pytest

import catalyst
from catalyst import cond, measure, qjit


@pytest.mark.parametrize(
    "seed",
    [
        "qwerty",
        "This string is just as random as the nest one",
        "bn980 2y9t'K(*^  jq42)",
        "... and his music was electric",
    ],
)
@pytest.mark.parametrize("device", ["lightning.qubit", "lightning.kokkos"])
def test_seeded_measurement(seed, device):
    """Test that different calls to qjits with the same seed produce the same measurement results"""
    dev = qml.device(device, wires=1)

    @qjit(seed=seed)
    def workflow():
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            m = measure(0)

            @cond(m)
            def cfun0():
                qml.Hadamard(0)

            cfun0()
            return qml.probs()

        return circuit(), circuit(), circuit(), circuit()

    @qjit(seed=seed)
    def workflow1():
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            m = measure(0)

            @cond(m)
            def cfun0():
                qml.Hadamard(0)

            cfun0()
            return qml.probs()

        return circuit(), circuit(), circuit(), circuit()

    # Calls to qjits with the same seed should return the same results
    # TODO: However, each measurement within a qjit should be random.
    for i in range(300):
        results0 = workflow()
        results1 = workflow()
        results2 = workflow1()
        assert np.allclose(results0, results1)
        assert np.allclose(results0, results2)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
