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

from catalyst import CompileError, cond, measure, qjit


def test_seeded_async():
    """Test that seeding and async cannot be simultaneously used"""
    with pytest.raises(CompileError, match="Seeding has no effect on asyncronous qnodes"):

        @qjit(async_qnodes=True, seed=37)
        def _():
            return

        _()


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_out_of_range(seed):
    """Test that a seed that is not a unsigned 32-bit int raises an error"""
    with pytest.raises(ValueError, match="Seed must be an unsigned 32-bit integer!"):

        @qjit(seed=seed)
        def _():
            return

        _()


@pytest.mark.parametrize(
    "seed",
    [
        42,
        37,
        1337,
        2**32 - 1,
        0,
    ],
)
def test_seeded_measurement(seed, backend):
    """Test that different calls to qjits with the same seed produce the same measurement results"""

    dev = qml.device(backend, wires=1)

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
    for _ in range(5):
        results0 = workflow()
        results1 = workflow()
        results2 = workflow1()
        assert np.allclose(results0, results1)
        assert np.allclose(results0, results2)


@pytest.mark.parametrize(
    "seed",
    [
        42,
        37,
        1337,
        2**32 - 1,
        0,
    ],
)
@pytest.mark.parametrize("shots", [10])
@pytest.mark.parametrize("readout", [qml.sample, qml.counts])
def test_seeded_sample(seed, shots, readout, backend):
    """Test that different calls to qjits with the same seed produce the same sample results"""

    if backend not in ["lightning.qubit", "lightning.kokkos", "lightning.gpu"]:
        pytest.skip("Sample seeding is only supported on lightning.qubit, lightning.kokkos and lightning.gpu")

    dev = qml.device(backend, wires=2, shots=shots)

    @qjit(seed=seed)
    def workflow():
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=[0])
            qml.RX(12.34, wires=[1])
            return readout()

        return circuit(), circuit(), circuit(), circuit()

    @qjit(seed=seed)
    def workflow1():
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=[0])
            qml.RX(12.34, wires=[1])
            return readout()

        return circuit(), circuit(), circuit(), circuit()

    # Calls to qjits with the same seed should return the same samples
    for _ in range(5):
        results0 = workflow()
        results1 = workflow()
        results2 = workflow1()
        assert np.allclose(results0, results1)
        assert np.allclose(results0, results2)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
