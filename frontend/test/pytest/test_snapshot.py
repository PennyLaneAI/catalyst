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
            (qml.sample()),
        ),
    )
    def test_not_implemented_snapshots(self, operation_passed_to_snapshot):
        """Make sure only qml.state is allowed to be in qml.Snapshot"""
        with pytest.raises(
            NotImplementedError,
            match=r"qml.Snapshot\(\) only supports qml.state\(\) when used from within Catalyst,"
            + f" but encountered {type(operation_passed_to_snapshot)}",
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

        expected_measurement_results = circuit()

        expected_snapshot_results = list(
            qml.snapshots(circuit)().values()
        )  # get the snapshot result values
        expected_snapshot_results = expected_snapshot_results[:-1]  # remove 'execution_results' key

        jitted_results = qjit(circuit)()
        jitted_snapshot_results = jitted_results[0]
        jitted_measurement_results = jitted_results[1]

        assert np.allclose(jitted_snapshot_results, expected_snapshot_results)
        assert all(
            np.allclose(expected_measurement_results[i], jitted_measurement_results[i])
            for i in range(len(jitted_measurement_results))
        )

    def test_snapshot_on_two_wire(self):
        """Test qml.Snapshot on two qubits with shots"""
        dev = qml.device("lightning.qubit", wires=2, shots=5)

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
            return {0: qml.counts(), 1: qml.sample()}

        expected_measurement_results = circuit()

        expected_snapshot_results = list(
            qml.snapshots(circuit)().values()
        )  # get the snapshot result values
        expected_snapshot_results = expected_snapshot_results[:-1]  # remove 'execution_results' key

        jitted_results = qjit(circuit)()
        jitted_snapshot_results = jitted_results[0]
        jitted_measurement_results = jitted_results[1]

        assert np.allclose(jitted_snapshot_results, expected_snapshot_results)
        assert expected_measurement_results.keys() == jitted_measurement_results.keys()

        assert expected_measurement_results[0]["11"] == jitted_measurement_results[0][1][3]
        assert np.allclose(expected_measurement_results[1], jitted_measurement_results[1])

    def test_snapshots_with_dynamic_wires(self):
        """Test if qml.Snapshot captures dynamic shaped states"""

        @qjit
        def workflow(num_qubits):
            @qml.qnode(qml.device("lightning.qubit", wires=num_qubits))
            def circuit():
                qml.X(wires=0)
                qml.Snapshot()
                return qml.probs()

            return circuit()

        returned_results = workflow(2)
        assert np.allclose(
            returned_results[0],
            [jnp.array([0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex128)],
        )
        assert np.allclose(
            returned_results[1], [jnp.array([0.0, 0.0, 1.0, 0.0], dtype=jnp.float64)]
        )
