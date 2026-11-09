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

import json

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


@pytest.mark.parametrize(
    "ops, expected_depth",
    [
        (lambda: qp.GlobalPhase(0.1), 1),
        (lambda: (qp.RX(0.1, 0), qp.GlobalPhase(0.1), qp.RX(0.1, 1)), 3),
        (lambda: (qp.GlobalPhase(0.1), qp.X(0), qp.X(1)), 2),
        (lambda: (qp.RX(0.1, 0), qp.ctrl(qp.GlobalPhase(0.1), control=1), qp.RX(0.1, 0)), 2),
    ],
)
def test_global_phase_tracked_depth(ops, expected_depth, capture_mode, tmp_path):
    """Test that the circuit depth tracked by null.qubit for global phases matches the depth
    PennyLane computes for the tape, in which a global phase without control wires acts on all
    wires."""
    filename = tmp_path / "resources.json"
    dev = qp.device(
        "null.qubit",
        wires=2,
        track_resources=True,
        resources_filename=str(filename),
        compute_depth=True,
    )

    @qp.qnode(dev)
    def circuit():
        ops()
        return qp.probs(wires=0)

    tape = qp.workflow.construct_tape(circuit)()
    assert qp.resource.resources_from_tape(tape, compute_depth=True).depth == expected_depth

    qjit(circuit, capture=capture_mode)()
    with filename.open("r", encoding="utf-8") as f:
        assert json.load(f)["depth"] == expected_depth


if __name__ == "__main__":
    pytest.main(["-x", __file__])
