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

import pytest

pytest.importorskip("braket")

import numpy as np
import pennylane as qml

from catalyst import qjit


def test_no_parameters_braket():
    """Test no-param operations on braket.aws.qubit."""

    def circuit():
        qml.PauliX(wires=1)
        qml.PauliY(wires=2)
        qml.PauliZ(wires=0)

        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)

        qml.S(wires=0)
        qml.T(wires=0)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 0])

        qml.CZ(wires=[0, 1])
        qml.CZ(wires=[0, 2])

        qml.SWAP(wires=[0, 1])
        qml.SWAP(wires=[0, 2])
        qml.SWAP(wires=[1, 2])

        qml.CSWAP(wires=[0, 1, 2])

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    qjit_fn = qjit()(
        qml.qnode(
            qml.device(
                "braket.aws.qubit",
                device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                wires=3,
            )
        )(circuit)
    )
    qml_fn = qml.qnode(qml.device("default.qubit", wires=3))(circuit)

    assert np.allclose(qjit_fn(), qml_fn())


def test_param_braket():
    """Test param operations on braket.aws.qubit."""

    def circuit(x: float, y: float):
        qml.Rot(x, y, x + y, wires=0)

        qml.RX(x, wires=0)
        qml.RY(y, wires=1)
        qml.RZ(x, wires=2)

        qml.RZ(y, wires=0)
        qml.RY(x, wires=1)
        qml.RX(y, wires=2)

        qml.PhaseShift(x, wires=0)
        qml.PhaseShift(y, wires=1)

        return qml.var(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

    qjit_fn = qjit()(
        qml.qnode(
            qml.device(
                "braket.aws.qubit",
                device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                wires=3,
            )
        )(circuit)
    )
    qml_fn = qml.qnode(qml.device("default.qubit", wires=4))(circuit)

    assert np.allclose(qjit_fn(3.14, 0.6), qml_fn(3.14, 0.6))


def test_sample_on_1qbit_braket():
    """Test sample on 1 qubit on braket.aws.qubit."""

    device = qml.device(
        "braket.aws.qubit",
        device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
        wires=1,
        shots=1000,
    )

    @qjit()
    @qml.qnode(device)
    def sample_1qbit(x: float):
        qml.RX(x, wires=0)
        return qml.sample()

    expected = np.array([[0.0]] * 1000)
    observed = sample_1qbit(0.0)
    assert np.array_equal(observed, expected)

    expected = np.array([[1.0]] * 1000)
    observed = sample_1qbit(np.pi)
    assert np.array_equal(observed, expected)


def test_sample_on_2qbits_braket():
    """Test sample on 2 qubits on braket.aws.qubit."""

    device = qml.device(
        "braket.aws.qubit",
        device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
        wires=2,
        shots=1000,
    )

    @qjit()
    @qml.qnode(device)
    def sample_2qbits(x: float):
        qml.RX(x, wires=0)
        qml.RY(x, wires=1)
        return qml.sample()

    expected = np.array([[0.0, 0.0]] * 1000)
    observed = sample_2qbits(0.0)
    assert np.array_equal(observed, expected)
    expected = np.array([[1.0, 1.0]] * 1000)
    observed = sample_2qbits(np.pi)
    assert np.array_equal(observed, expected)


def test_count_on_1qbit_braket():
    """Test counts on 1 qubits on braket.aws.qubit."""

    device = qml.device(
        "braket.aws.qubit",
        device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
        wires=1,
        shots=1000,
    )

    @qjit()
    @qml.qnode(device)
    def counts_1qbit(x: float):
        qml.RX(x, wires=0)
        return qml.counts()

    expected = [np.array([0, 1]), np.array([1000, 0])]
    observed = counts_1qbit(0.0)
    assert np.array_equal(observed, expected)

    expected = [np.array([0, 1]), np.array([0, 1000])]
    observed = counts_1qbit(np.pi)
    assert np.array_equal(observed, expected)


def test_count_on_2qbits_braket(backend):
    """Test counts on 2 qubits on braket.aws.qubit."""

    device = qml.device(
        "braket.aws.qubit",
        device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
        wires=2,
        shots=1000,
    )

    @qjit()
    @qml.qnode(device)
    def counts_2qbit(x: float):
        qml.RX(x, wires=0)
        qml.RY(x, wires=1)
        return qml.counts()

    expected = [np.array([0, 1, 2, 3]), np.array([1000, 0, 0, 0])]
    observed = counts_2qbit(0.0)
    assert np.array_equal(observed, expected)

    expected = [np.array([0, 1, 2, 3]), np.array([0, 0, 0, 1000])]
    observed = counts_2qbit(np.pi)
    assert np.array_equal(observed, expected)


class TestBraketExpval:
    def test_named(self):
        """Test expval for named observables on braket.aws.qubit."""

        device = qml.device(
            "braket.aws.qubit",
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            wires=1,
        )

        @qjit()
        @qml.qnode(device)
        def expval(x: float):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        expected = np.array(1.0)
        observed = expval(0.0)
        assert np.isclose(observed, expected)

        expected = np.array(-1.0)
        observed = expval(np.pi)
        assert np.isclose(observed, expected)

    def test_tensor(self):
        """Test expval for Tensor observable on braket.aws.qubit."""

        device = qml.device(
            "braket.aws.qubit",
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            wires=2,
        )

        @qjit()
        @qml.qnode(device)
        def expval(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(0) @ qml.PauliY(1))

        expected = np.array(-0.35355339)
        observed = expval(np.pi / 4, np.pi / 3)
        assert np.isclose(observed, expected)

        expected = np.array(-0.49999788)
        observed = expval(np.pi / 2, np.pi / 3)
        assert np.isclose(observed, expected)


class TestBraketVar:
    def test_rx(self):
        """Test var with RX on braket.aws.qubit."""

        device = qml.device(
            "braket.aws.qubit",
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            wires=1,
        )

        @qjit()
        @qml.qnode(device)
        def var(x: float):
            qml.RX(x, wires=0)
            return qml.var(qml.PauliZ(0))

        expected = np.array(0.0)
        observed = var(0.0)
        assert np.isclose(observed, expected)
        observed = var(np.pi)
        assert np.isclose(observed, expected)

    def test_hadamard(self, backend):
        """Test var with Hadamard on braket.aws.qubit."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=1))
        def var(x: float):
            qml.Hadamard(wires=0)
            return qml.var(qml.PauliZ(0))

        expected = np.array(1.0)
        observed = var(0.0)
        assert np.isclose(observed, expected)
        observed = var(np.pi)
        assert np.isclose(observed, expected)

    def test_tensor(self):
        """Test variance for Tensor observable on braket.aws.qubit."""

        device = qml.device(
            "braket.aws.qubit",
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            wires=2,
        )

        @qjit()
        @qml.qnode(device)
        def circuit(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliX(0) @ qml.PauliY(1))

        expected = np.array(0.875)
        observed = circuit(np.pi / 4, np.pi / 3)
        assert np.isclose(observed, expected)

        expected = np.array(1.0)
        observed = circuit(np.pi / 2, np.pi / 2)
        assert np.isclose(observed, expected)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
