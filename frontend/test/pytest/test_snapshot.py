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

"""
This file performs the frontend pytest checking for qml.Snapshot support in Catalyst.
"""

import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as np

from catalyst import qjit


class TestSnapshot:
    """Test if Snapshots are captured correctly."""

    @pytest.mark.parametrize(
        "operation_passed_to_snapshot",
        (
            (qml.probs()),
            (qml.expval(qml.X(0))),
            (qml.counts()),
            (qml.var(qml.X(0))),
        ),
    )
    def test_not_implemented_snapshots(self, operation_passed_to_snapshot):
        """Make sure only qml.state is allowed to be in qml.Snapshot"""
        with pytest.raises(
            NotImplementedError,
            match=f"Snapshot of type {type(operation_passed_to_snapshot)} is not implemented",
        ):
            dev = qml.device("lightning.qubit", wires=1, shots=5)

            @qjit
            @qml.qnode(dev)
            def circuit():
                qml.Snapshot(measurement=operation_passed_to_snapshot)
                return qml.probs()

            circuit()

    def test_snapshot_on_single_wire(self):
        """Test all six single qubit basis states in qml.Snapshot without shots"""
        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        @qml.qnode(dev)
        def circuit():
            qml.Snapshot()  # |0>
            qml.X(wires=0)
            qml.Snapshot()  # |1>
            qml.Hadamard(wires=0)
            qml.Snapshot()  # |->
            qml.PhaseShift(np.pi / 2, wires=0)
            qml.Snapshot()  # |-i>
            qml.Z(wires=0)
            qml.Snapshot()  # |+i>
            qml.PhaseShift(-np.pi / 2, wires=0)
            qml.Snapshot()  # |+>
            return qml.state(), qml.probs(), qml.expval(qml.X(0)), qml.var(qml.Z(0))

        expected_output = (
            [
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex128),
                jnp.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=jnp.complex128),
                jnp.array([0.70710678 + 0.0j, -0.70710678 + 0.0j], dtype=jnp.complex128),
                jnp.array([7.07106781e-01 + 0.0j, 0.0 - 0.70710678j], dtype=jnp.complex128),
                jnp.array([7.07106781e-01 + 0.0j, 0.0 + 0.70710678j], dtype=jnp.complex128),
                jnp.array([0.70710678 + 0.0j, 0.70710678 + 0.0j], dtype=jnp.complex128),
            ],
            jnp.array([0.70710678 + 0.0j, 0.70710678 + 0.0j], dtype=jnp.complex128),
            jnp.array([0.5, 0.5], dtype=jnp.float64),
            jnp.array(1.0, dtype=jnp.float64),
            jnp.array(1, dtype=jnp.float64),
        )
        returned_output = circuit()
        expected_snapshot_states, returned_snapshot_states = expected_output[0], returned_output[0]
        expected_measurement_results, returned_measurement_results = (
            expected_output[1:],
            returned_output[1:],
        )
        assert all(
            jnp.allclose(expected_snapshot_states[i], returned_snapshot_states[i])
            for i in range(len(returned_snapshot_states))
        )
        assert all(
            jnp.allclose(expected_measurement_results[i], returned_measurement_results[i])
            for i in range(len(returned_measurement_results))
        )

    def test_snapshot_on_two_wire(self):
        """Test qml.Snapshot on two qubits with shots"""
        dev = qml.device("lightning.qubit", wires=2, shots=5)

        @qjit
        @qml.qnode(dev)
        def circuit():
            qml.Snapshot()  # |00>
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.Snapshot()  # |++>

            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)  # |00>
            qml.X(wires=0)
            qml.X(wires=1)  # |11> to measure in comp-basis
            return qml.counts(), qml.sample()

        expected_output = (
            [
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex128),
                jnp.array([0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j], dtype=jnp.complex128),
            ],
            (jnp.array([0, 1, 2, 3], dtype=jnp.int64), jnp.array([0, 0, 0, 5], dtype=jnp.int64)),
            jnp.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]], dtype=jnp.int64),
        )
        returned_output = circuit()
        expected_snapshot_states, returned_snapshot_states = expected_output[0], returned_output[0]
        expected_counts, returned_counts = expected_output[1], returned_output[1]
        expected_samples, returned_samples = expected_output[2], returned_output[2]
        assert all(
            jnp.allclose(expected_snapshot_states[i], returned_snapshot_states[i])
            for i in range(len(returned_snapshot_states))
        )
        assert all(
            jnp.allclose(expected_counts[i], returned_counts[i])
            for i in range(len(returned_counts))
        )
        assert all(
            jnp.allclose(expected_samples[i], returned_samples[i])
            for i in range(len(returned_samples))
        )
