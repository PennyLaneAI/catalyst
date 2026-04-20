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
This file performs the frontend pytest checking for qp.Snapshot support in Catalyst.
"""

import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pytest
from pennylane import numpy as np

from catalyst import qjit


class TestSnapshot:
    """Test if Snapshots are captured correctly."""

    @pytest.mark.parametrize(
        "operation_passed_to_snapshot",
        (
            (qp.probs()),
            (qp.expval(qp.X(0))),
            (qp.counts()),
            (qp.var(qp.X(0))),
            (qp.sample()),
        ),
    )
    def test_not_implemented_snapshots(self, operation_passed_to_snapshot):
        """Make sure only qp.state is allowed to be in qp.Snapshot"""
        with pytest.raises(
            NotImplementedError,
            match=r"qml.Snapshot\(\) only supports qml.state\(\) when used from within Catalyst,"
            + f" but encountered {type(operation_passed_to_snapshot)}",
        ):
            dev = qp.device("lightning.qubit", wires=1)

            @qjit
            @qp.set_shots(5)
            @qp.qnode(dev)
            def circuit():
                qp.Snapshot(measurement=operation_passed_to_snapshot)
                return qp.probs()

            circuit()

    def test_snapshot_on_single_wire(self, backend):
        """Test all six single qubit basis states in qp.Snapshot without shots"""

        pl_dev = qp.device("default.qubit", wires=1)
        cat_dev = qp.device(backend, wires=1)

        def circuit():
            qp.Snapshot()  # |0>
            qp.X(wires=0)
            qp.Snapshot()  # |1>
            qp.Hadamard(wires=0)
            qp.Snapshot()  # |->
            qp.PhaseShift(np.pi / 2, wires=0)
            qp.Snapshot()  # |-i>
            qp.Z(wires=0)
            qp.Snapshot()  # |+i>
            qp.PhaseShift(-np.pi / 2, wires=0)
            qp.Snapshot()  # |+>
            return qp.state(), qp.probs(), qp.expval(qp.X(0)), qp.var(qp.Z(0))

        pl_circuit = qp.qnode(pl_dev)(circuit)
        expected_measurement_results = pl_circuit()

        expected_snapshot_results = list(qp.snapshots(pl_circuit)().values())
        expected_snapshot_results = expected_snapshot_results[:-1]  # remove 'execution_results' key

        cat_circuit = qjit(qp.qnode(cat_dev)(circuit))
        jitted_results = cat_circuit()
        jitted_snapshot_results = jitted_results[0]
        jitted_measurement_results = jitted_results[1]

        assert np.allclose(jitted_snapshot_results, expected_snapshot_results)
        assert all(
            np.allclose(expected_measurement_results[i], jitted_measurement_results[i])
            for i in range(len(jitted_measurement_results))
        )

    def test_snapshot_on_two_wire(self, backend):
        """Test qp.Snapshot on two qubits with shots"""
        pl_dev = qp.device("default.qubit", wires=2)
        cat_dev = qp.device(backend, wires=2)

        def circuit():
            qp.Snapshot()  # |00>
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=1)
            qp.Snapshot()  # |++>

            qp.Hadamard(wires=0)
            qp.Hadamard(wires=1)  # |00>
            qp.X(wires=0)
            qp.X(wires=1)  # |11> to measure in comp-basis
            return {0: qp.counts(), 1: qp.sample()}

        pl_circuit = qp.qnode(pl_dev, shots=5)(circuit)
        expected_measurement_results = pl_circuit()

        expected_snapshot_results = list(qp.snapshots(pl_circuit)().values())
        expected_snapshot_results = expected_snapshot_results[:-1]  # remove 'execution_results' key

        cat_circuit = qjit(qp.qnode(cat_dev, shots=5)(circuit))
        jitted_results = cat_circuit()
        jitted_snapshot_results = jitted_results[0]
        jitted_measurement_results = jitted_results[1]

        assert np.allclose(jitted_snapshot_results, expected_snapshot_results)
        assert expected_measurement_results.keys() == jitted_measurement_results.keys()

        assert expected_measurement_results[0]["11"] == jitted_measurement_results[0][1][3]
        assert np.allclose(expected_measurement_results[1], jitted_measurement_results[1])

    def test_snapshots_with_dynamic_wires(self):
        """Test if qp.Snapshot captures dynamic shaped states"""

        @qjit
        def workflow(num_qubits):
            @qp.qnode(qp.device("lightning.qubit", wires=num_qubits))
            def circuit():
                qp.X(wires=0)
                qp.Snapshot()
                return qp.probs()

            return circuit()

        returned_results = workflow(2)
        assert np.allclose(
            returned_results[0],
            [jnp.array([0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex128)],
        )
        assert np.allclose(
            returned_results[1], [jnp.array([0.0, 0.0, 1.0, 0.0], dtype=jnp.float64)]
        )
