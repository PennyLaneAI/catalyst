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

"""Unit tests for Global Phase"""

import numpy as np
import pennylane as qml
import pytest

from catalyst import cond, qjit


def test_global_phase(backend):
    """Test vanilla global phase"""
    dev = qml.device(backend, wires=1)

    @qml.qnode(dev)
    def qnn():
        qml.RX(np.pi / 4, wires=[0])
        qml.GlobalPhase(np.pi / 4)
        return qml.state()

    expected = qnn()
    observed = qjit(qnn)()
    assert np.allclose(expected, observed)


@pytest.mark.parametrize("inp", [True, False])
def test_global_phase_in_region(backend, inp):
    """Test global phase in region"""
    dev = qml.device(backend, wires=1)

    @qml.qnode(dev)
    def qnn(c):
        qml.RX(np.pi / 4, wires=[0])

        @cond(c)
        def cir():
            qml.GlobalPhase(np.pi / 4)

        cir()
        return qml.state()

    expected = qnn(inp)
    observed = qjit(qnn)(inp)
    assert np.allclose(expected, observed)


def test_global_phase_control(backend):
    """Test global phase controlled"""

    if backend == "lightning.kokkos":
        pytest.skip("control phase is unsupported in kokkos or at least its toml file.")

    dev = qml.device(backend, wires=2)

    @qml.qnode(dev)
    def qnn():
        qml.RX(np.pi / 4, wires=[0])
        qml.ctrl(qml.GlobalPhase(np.pi / 4), control=[0])
        return qml.state()

    expected = qnn()
    observed = qjit(qnn)()
    assert np.allclose(expected, observed)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
