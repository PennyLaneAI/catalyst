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
import pennylane as qp
import pytest
from pennylane import cond, qjit


def test_global_phase(backend, capture_mode):
    """Test vanilla global phase"""
    dev = qp.device(backend, wires=1)

    @qp.qnode(dev)
    def qnn():
        qp.RX(np.pi / 4, wires=[0])
        qp.GlobalPhase(np.pi / 4)
        return qp.state()

    expected = qnn()
    observed = qjit(qnn, capture=capture_mode)()
    assert np.allclose(expected, observed)


@pytest.mark.parametrize("inp", [True, False])
def test_global_phase_in_region(backend, inp, capture_mode):
    """Test global phase in region"""
    dev = qp.device(backend, wires=1)

    @qp.qnode(dev)
    def qnn(c):
        qp.RX(np.pi / 4, wires=[0])

        @cond(c)
        def cir():
            qp.GlobalPhase(np.pi / 4)

        cir()
        return qp.state()

    observed = qjit(qnn, capture=capture_mode)(inp)
    pass
    expected = qnn(inp)
    assert np.allclose(expected, observed)


def test_global_phase_control(backend, capture_mode):
    """Test global phase controlled"""

    if backend in ("lightning.kokkos", "lightning.gpu"):
        pytest.skip("control phase is unsupported in kokkos or at least its toml file.")

    dev = qp.device(backend, wires=2)

    @qp.qnode(dev)
    def qnn():
        qp.RX(np.pi / 4, wires=[0])
        qp.ctrl(qp.GlobalPhase(np.pi / 4), control=[0])
        return qp.state()

    expected = qnn()
    observed = qjit(qnn, capture=capture_mode)()
    assert np.allclose(expected, observed)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
