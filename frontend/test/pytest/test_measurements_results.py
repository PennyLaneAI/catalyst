# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pennylane as qml
import pytest

from catalyst import qjit


def test_sample_on_1qbit(backend):
    """Test sample on 1 qubit."""

    @qjit()
    @qml.qnode(qml.device(backend, wires=1, shots=1000))
    def sample_1qbit(x: float):
        qml.RX(x, wires=0)
        return qml.sample()

    expected = np.array([[0.0]] * 1000)
    observed = sample_1qbit(0.0)
    assert np.array_equal(observed, expected)

    expected = np.array([[1.0]] * 1000)
    observed = sample_1qbit(np.pi)
    assert np.array_equal(observed, expected)


def test_sample_on_2qbits(backend):
    """Test sample on 2 qubits."""

    @qjit()
    @qml.qnode(qml.device(backend, wires=2, shots=1000))
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


def test_count_on_1qbit(backend):
    """Test counts on 1 qubits."""

    @qjit()
    @qml.qnode(qml.device(backend, wires=1, shots=1000))
    def counts_1qbit(x: float):
        qml.RX(x, wires=0)
        return qml.counts()

    expected = [np.array([0, 1]), np.array([1000, 0])]
    observed = counts_1qbit(0.0)
    assert np.array_equal(observed, expected)

    expected = [np.array([0, 1]), np.array([0, 1000])]
    observed = counts_1qbit(np.pi)
    assert np.array_equal(observed, expected)


def test_count_on_2qbits(backend):
    """Test counts on 2 qubits."""

    @qjit()
    @qml.qnode(qml.device(backend, wires=2, shots=1000))
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


class TestExpval:
    def test_named(self, backend):
        """Test expval for named observables."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=1))
        def expval1(x: float):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        expected = np.array(1.0)
        observed = expval1(0.0)
        assert np.isclose(observed, expected)

        expected = np.array(-1.0)
        observed = expval1(np.pi)
        assert np.isclose(observed, expected)

    def test_hermitian_1(self, backend):
        """Test expval for Hermitian observable."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=1))
        def expval2(x: float):
            qml.RY(x, wires=0)
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qml.expval(qml.Hermitian(A, wires=0))

        expected = np.array(1.0)
        observed = expval2(0.0)
        assert np.isclose(observed, expected)

        expected = np.array(3.0)
        observed = expval2(np.pi / 2)
        assert np.isclose(observed, expected)

    def test_hermitian_2(self, backend):
        """Test expval for Hermitian observable."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=2))
        def expval3(x: float):
            qml.RX(x, wires=1)
            B = np.array(
                [
                    [complex(1.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
                    [complex(2.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
                    [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
                    [complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0)],
                ]
            )
            return qml.expval(qml.Hermitian(B, wires=[0, 1]))

        expected = np.array(1.0)
        observed = expval3(0.0)
        assert np.isclose(observed, expected)

        expected = np.array(2.0)
        observed = expval3(np.pi)
        assert np.isclose(observed, expected)

    def test_tensor_1(self, backend):
        """Test expval for Tensor observable."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=2))
        def expval4(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(0) @ qml.PauliY(1))

        expected = np.array(-0.35355339)
        observed = expval4(np.pi / 4, np.pi / 3)
        assert np.isclose(observed, expected)

        expected = np.array(0.0)
        observed = expval4(np.pi / 2, np.pi / 2)
        assert np.isclose(observed, expected)

    def test_tensor_2(self, backend):
        """Test expval for Tensor observable."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=3))
        def expval5(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.CNOT(wires=[0, 2])
            qml.CNOT(wires=[1, 2])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(-1.0, 0.0)]]
            )
            return qml.expval(qml.PauliX(0) @ qml.Hadamard(1) @ qml.Hermitian(A, wires=2))

        expected = np.array(-0.4330127)
        observed = expval5(np.pi / 4, np.pi / 3)
        assert np.isclose(observed, expected)

        expected = np.array(-0.70710678)
        observed = expval5(np.pi / 2, np.pi / 2)
        assert np.isclose(observed, expected)

    def test_hamiltonian_1(self, backend):
        """Test expval for Hamiltonian observable."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=3))
        def expval6(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(0.1, wires=2)

            coeffs = np.array([0.2, -0.543])
            obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]

            return qml.expval(qml.Hamiltonian(coeffs, obs))

        expected = np.array(-0.2715)
        observed = expval6(np.pi / 4, np.pi / 3)
        assert np.isclose(observed, expected)

        expected = np.array(-0.33695571)
        observed = expval6(0.5, 0.8)
        assert np.isclose(observed, expected)

    def test_hamiltonian_2(self, backend):
        """Test expval for Hamiltonian observable."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=2))
        def expval6(x: float):
            qml.RX(x, wires=0)

            coeff = np.array([0.8, 0.2])
            obs_matrix = np.array(
                [
                    [0.5, 1.0j, 0.0, -3j],
                    [-1.0j, -1.1, 0.0, -0.1],
                    [0.0, 0.0, -0.9, 12.0],
                    [3j, -0.1, 12.0, 0.0],
                ]
            )

            obs = qml.Hermitian(obs_matrix, wires=[0, 1])
            return qml.expval(qml.Hamiltonian(coeff, [obs, qml.PauliX(0)]))

        expected = np.array(0.2359798)
        observed = expval6(np.pi / 4)
        assert np.isclose(observed, expected)

        expected = np.array(-0.16)
        observed = expval6(np.pi / 2)
        assert np.isclose(observed, expected)


class TestVar:
    def test_rx(self, backend):
        """Test var with RX."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=1))
        def var1(x: float):
            qml.RX(x, wires=0)
            return qml.var(qml.PauliZ(0))

        expected = np.array(0.0)
        observed = var1(0.0)
        assert np.isclose(observed, expected)
        observed = var1(np.pi)
        assert np.isclose(observed, expected)

    def test_hadamard(self, backend):
        """Test var with Hadamard."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=1))
        def var2(x: float):
            qml.Hadamard(wires=0)
            return qml.var(qml.PauliZ(0))

        expected = np.array(1.0)
        observed = var2(0.0)
        assert np.isclose(observed, expected)
        observed = var2(np.pi)
        assert np.isclose(observed, expected)

    def test_hermitian_1(self, backend):
        """Test variance for Hermitian observable."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):
            qml.RY(x, wires=0)
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qml.var(qml.Hermitian(A, wires=0))

        expected = np.array(4.0)
        observed = circuit(0.0)
        assert np.isclose(observed, expected)

        expected = np.array(0.0)
        observed = circuit(np.pi / 2)
        assert np.isclose(observed, expected)

    def test_hermitian_2(self, backend):
        """Test variance for Hermitian observable."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x: float):
            qml.RX(x, wires=1)
            B = np.array(
                [
                    [complex(1.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
                    [complex(2.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
                    [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
                    [complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0)],
                ]
            )
            return qml.var(qml.Hermitian(B, wires=[0, 1]))

        expected = np.array(9.0)
        observed = circuit(0.0)
        assert np.isclose(observed, expected)

        expected = np.array(9.0)
        observed = circuit(np.pi)
        assert np.isclose(observed, expected)

    def test_tensor_1(self, backend):
        """Test variance for Tensor observable."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=2))
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

    def test_tensor_2(self, backend):
        """Test variance for Tensor observable."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=3))
        def circuit(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.CNOT(wires=[0, 2])
            qml.CNOT(wires=[1, 2])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(-1.0, 0.0)]]
            )
            return qml.var(qml.PauliX(0) @ qml.Hadamard(1) @ qml.Hermitian(A, wires=2))

        expected = np.array(4.8125)
        observed = circuit(np.pi / 4, np.pi / 3)
        assert np.isclose(observed, expected)

        expected = np.array(4.5)
        observed = circuit(np.pi / 2, np.pi / 2)
        assert np.isclose(observed, expected)

    def test_hamiltonian_1(self, backend):
        """Test variance for Hamiltonian observable."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=3))
        def circuit(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(0.1, wires=2)

            coeffs = np.array([0.2, -0.543])
            obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]

            return qml.var(qml.Hamiltonian(coeffs, obs))

        expected = np.array(0.26113675)
        observed = circuit(np.pi / 4, np.pi / 3)
        assert np.isclose(observed, expected)

        expected = np.array(0.22130985)
        observed = circuit(0.5, 0.8)
        assert np.isclose(observed, expected)

    def test_hamiltonian_2(self, backend):
        """Test variance for Hamiltonian observable."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x: float):
            qml.RX(x, wires=0)

            coeff = np.array([0.8, 0.2])
            obs_matrix = np.array(
                [
                    [0.5, 1.0j, 0.0, -3j],
                    [-1.0j, -1.1, 0.0, -0.1],
                    [0.0, 0.0, -0.9, 12.0],
                    [3j, -0.1, 12.0, 0.0],
                ]
            )

            obs = qml.Hermitian(obs_matrix, wires=[0, 1])
            return qml.var(qml.Hamiltonian(coeff, [obs, qml.PauliX(0)]))

        expected = np.array(2.86432098)
        observed = circuit(np.pi / 4)
        assert np.isclose(observed, expected)

        expected = np.array(26.5936)
        observed = circuit(np.pi / 2)
        assert np.isclose(observed, expected)


def test_state(backend):
    """Test state."""

    @qjit()
    @qml.qnode(qml.device(backend, wires=1))
    def state(x: float):
        qml.RX(x, wires=0)
        return qml.state()

    expected = np.array([complex(1.0, 0.0), complex(0.0, 0.0)])
    observed = state(0.0)
    assert np.array_equal(observed, expected)


def test_multiple_return_values(backend):
    """Test multiple return values."""

    @qjit()
    @qml.qnode(qml.device(backend, wires=2, shots=100))
    def all_measurements(x):
        qml.RY(x, wires=0)
        return (
            qml.sample(),
            qml.counts(),
            qml.expval(qml.PauliZ(0)),
            qml.var(qml.PauliZ(0)),
            qml.probs(wires=[0, 1]),
            qml.state(),
        )

    @qml.qnode(qml.device("default.qubit", wires=2))
    def expected(x, measurement):
        qml.RY(x, wires=0)
        return qml.apply(measurement)

    x = 0.7
    result = all_measurements(x)

    # qml.sample
    assert result[0].shape == expected(x, qml.sample(wires=[0, 1]), shots=100).shape

    # qml.counts
    for r, e in zip(result[1][0], expected(x, qml.counts(all_outcomes=True), shots=100).keys()):
        assert format(int(r), "02b") == e
    assert sum(result[1][1]) == 100

    # qml.expval
    assert np.allclose(result[2], expected(x, qml.expval(qml.PauliZ(0))))

    # qml.var
    assert np.allclose(result[3], expected(x, qml.var(qml.PauliZ(0))))

    # qml.probs
    assert np.allclose(result[4], expected(x, qml.probs(wires=[0, 1])))

    # qml.state
    assert np.allclose(result[5], expected(x, qml.state()))


if __name__ == "__main__":
    pytest.main(["-x", __file__])
