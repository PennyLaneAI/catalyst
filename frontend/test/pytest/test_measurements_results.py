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
from jax import numpy as jnp

from catalyst import qjit


class TestSample:
    """Test sample."""

    def test_sample_on_0qbits(self):
        """Test sample on 0 qubits."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=0, shots=10))
        def sample_0qbit():
            return qml.sample()

        expected = np.empty(shape=(10, 0), dtype=int)
        observed = sample_0qbit()
        assert np.array_equal(observed, expected)

    def test_sample_on_1qbit(self, backend):
        """Test sample on 1 qubit."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1, shots=1000))
        def sample_1qbit(x: float):
            qml.RX(x, wires=0)
            return qml.sample()

        expected = np.array([[0]] * 1000)
        observed = sample_1qbit(0.0)
        assert np.array_equal(observed, expected)

        expected = np.array([[1]] * 1000)
        observed = sample_1qbit(np.pi)
        assert np.array_equal(observed, expected)

    def test_sample_on_2qbits(self, backend):
        """Test sample on 2 qubits."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2, shots=1000))
        def sample_2qbits(x: float):
            qml.RX(x, wires=0)
            qml.RY(x, wires=1)
            return qml.sample()

        expected = np.array([[0, 0]] * 1000)
        observed = sample_2qbits(0.0)
        assert np.array_equal(observed, expected)
        expected = np.array([[1, 1]] * 1000)
        observed = sample_2qbits(np.pi)
        assert np.array_equal(observed, expected)


class TestCounts:
    """Test counts."""

    def test_counts_on_0qbits(self):
        """Test counts on 0 qubits."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=0, shots=10))
        def counts_0qbit():
            return qml.counts()

        expected = [np.array([0]), np.array([10])]
        observed = counts_0qbit()
        assert np.array_equal(observed, expected)

    def test_count_on_1qbit(self, backend):
        """Test counts on 1 qubits."""

        @qjit
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

    def test_count_on_2qbits(self, backend):
        """Test counts on 2 qubits."""

        @qjit
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

    def test_count_on_2qbits_endianness(self, backend):
        """Test counts on 2 qubits with check for endianness."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2, shots=1000))
        def counts_2qbit(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            return qml.counts()

        expected = [np.array([0, 1, 2, 3]), np.array([0, 0, 1000, 0])]
        observed = counts_2qbit(np.pi, 0)
        assert np.array_equal(observed, expected)

        expected = [np.array([0, 1, 2, 3]), np.array([0, 1000, 0, 0])]
        observed = counts_2qbit(0, np.pi)
        assert np.array_equal(observed, expected)


    @pytest.mark.xfail(reason="Not supported by Catalyst")
    def test_counts_all_outcomes(self, backend):
        """Test counts with all_outcomes=True."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2, shots=1000))
        def counts_2qbit(x: float):
            qml.RX(x, wires=0)
            qml.RY(x, wires=1)
            return qml.counts(all_outcomes=True)

        expected = {'00': 1000, '01': 0, '10': 0, '11': 0}
        observed = counts_2qbit(0.0)
        assert np.array_equal(observed, expected)

        expected = {'00': 0, '01': 0, '10': 0, '11': 1000}
        observed = counts_2qbit(np.pi)
        assert np.array_equal(observed, expected)



class TestExpval:
    def test_named(self, backend):
        """Test expval for named observables."""

        @qjit
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

        @qjit
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

        @qjit
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

        @qjit
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

        @qjit
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

        @qjit
        @qml.qnode(qml.device(backend, wires=3))
        def expval(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(0.1, wires=2)

            coeffs = np.array([0.2, -0.543])
            obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]

            return qml.expval(qml.Hamiltonian(coeffs, obs))

        expected = np.array(-0.2715)
        observed = expval(np.pi / 4, np.pi / 3)
        assert np.isclose(observed, expected)

        expected = np.array(-0.33695571)
        observed = expval(0.5, 0.8)
        assert np.isclose(observed, expected)

    def test_hamiltonian_2(self, backend):
        """Test expval for Hamiltonian observable."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def expval(x: float):
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
        observed = expval(np.pi / 4)
        assert np.isclose(observed, expected)

        expected = np.array(-0.16)
        observed = expval(np.pi / 2)
        assert np.isclose(observed, expected)

    def test_hamiltonian_3(self, backend):
        """Test expval for nested Hamiltonian observable."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def expval(x: float):
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
            return qml.expval(
                qml.Hamiltonian(
                    coeff, [obs, qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(1)])]
                )
            )

        expected = np.array(0.4359798)
        observed = expval(np.pi / 4)
        assert np.isclose(observed, expected)

        expected = np.array(0.04)
        observed = expval(np.pi / 2)
        assert np.isclose(observed, expected)

    def test_hamiltonian_4(self, backend):
        """Test expval with TensorObs and nested Hamiltonian observables."""

        @qjit
        @qml.qnode(qml.device(backend, wires=3))
        def expval(x: float):
            qml.RX(x, wires=0)
            qml.RX(x + 1.0, wires=2)

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
            return qml.expval(
                qml.Hamiltonian(
                    coeff, [obs, qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(1)])]
                )
                @ qml.PauliZ(2)
            )

        expected = np.array(-0.09284557)
        observed = expval(np.pi / 4)
        assert np.isclose(observed, expected)

        expected = np.array(-0.03365884)
        observed = expval(np.pi / 2)
        assert np.isclose(observed, expected)


class TestVar:
    def test_rx(self, backend):
        """Test var with RX."""

        @qjit
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

        @qjit
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

        @qjit
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

        @qjit
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

        @qjit
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

        @qjit
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

        @qjit
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

        @qjit
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

    @pytest.mark.parametrize(
        "coeffs",
        [
            [1, 1],
            np.array([1, 1], dtype=np.int64),
            jnp.array([1, 1], dtype=np.int64),
        ],
    )
    def test_hamiltonian_3(self, coeffs, backend):
        """Test variance for Hamiltonian observable with integer coefficients."""

        @qjit
        @qml.qnode(qml.device(backend, wires=3))
        def circuit(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(0.1, wires=2)

            obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]

            return qml.var(qml.Hamiltonian(coeffs, obs))

        expected = np.array(1.75)
        observed = circuit(np.pi / 4, np.pi / 3)
        assert np.isclose(observed, expected)

        expected = np.array(1.61492442)
        observed = circuit(0.5, 0.8)
        assert np.isclose(observed, expected)

    @pytest.mark.parametrize(
        "coeffs",
        [
            np.array([1, 1], dtype=np.int64),
            jnp.array([1, 1], dtype=np.int64),
        ],
    )
    def test_hamiltonian_4(self, coeffs, backend):
        """Test variance for Hamiltonian observable with integer coefficients
        as the circuit parameters."""

        @qjit
        @qml.qnode(qml.device(backend, wires=3))
        def circuit(x, y, coeffs):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(0.1, wires=2)

            obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]

            return qml.var(qml.Hamiltonian(coeffs, obs))

        expected = np.array(1.75)
        observed = circuit(np.pi / 4, np.pi / 3, coeffs)
        assert np.isclose(
            observed,
            expected,
        )

    def test_hamiltonian_5(self, backend):
        """Test variance with nested Hamiltonian observable."""

        @qjit
        @qml.qnode(qml.device(backend, wires=3))
        def circuit(x, y, coeffs):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(0.1, wires=2)

            obs = [
                qml.PauliX(0) @ qml.PauliZ(1),
                qml.Hamiltonian(np.array([0.5]), [qml.PauliZ(0) @ qml.Hadamard(2)]),
            ]

            return qml.var(qml.Hamiltonian(coeffs, obs))

        expected = np.array(0.1075)
        coeffs = np.array([0.2, 0.6])
        observed = circuit(np.pi / 4, np.pi / 3, coeffs)
        assert np.isclose(
            observed,
            expected,
        )

    def test_hamiltonian_6(self, backend):
        """Test variance with TensorObs and nested Hamiltonian observables."""

        @qjit
        @qml.qnode(qml.device(backend, wires=3))
        def circuit(x, y, coeffs):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(0.1, wires=2)

            return qml.var(
                qml.Hamiltonian(
                    coeffs,
                    [
                        qml.PauliX(0)
                        @ qml.PauliZ(1)
                        @ qml.Hamiltonian(np.array([0.5]), [qml.Hadamard(2)])
                    ],
                )
            )

        expected = np.array(0.01)
        coeffs = np.array([0.2])
        observed = circuit(np.pi / 4, np.pi / 3, coeffs)
        assert np.isclose(
            observed,
            expected,
        )


class TestState:
    """Test state measurement processes."""

    def test_state_on_0qbits(self):
        """Test state on 0 qubits."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=0))
        def state_0qbit():
            return qml.state()

        expected = np.array([complex(1.0, 0.0)])
        observed = state_0qbit()
        assert np.array_equal(observed, expected)

    def test_state_on_1qubit(self, backend):
        """Test state on 1 qubit."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def state(x: float):
            qml.RX(x, wires=0)
            return qml.state()

        expected = np.array([complex(1.0, 0.0), complex(0.0, -1.0)]) / np.sqrt(2)
        observed = state(np.pi / 2)
        assert np.allclose(observed, expected)


class TestProbs:
    """Test probabilities measurement processes."""

    def test_probs_on_0qbits(self):
        """Test probs on 0 qubits."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=0))
        def probs_0qbit():
            return qml.probs()

        expected = np.array([1.0])
        observed = probs_0qbit()
        assert np.array_equal(observed, expected)

    def test_probs_on_1qubit(self, backend):
        """Test probs on 1 qubit."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def probs(x: float):
            qml.RX(x, wires=0)
            return qml.probs()

        expected = np.array([0.5, 0.5])
        observed = probs(np.pi / 2)
        assert np.allclose(observed, expected)

class TestNewArithmeticOps:
    "Test PennyLane new arithmetic operators"

    @pytest.mark.parametrize(
        "meas, expected",
        [[qml.expval, np.array(-0.70710678)], [qml.var, np.array(0.5)]],
    )
    def test_prod_xzi(self, meas, expected, backend):
        """Test ``qml.ops.op_math.Prod`` converting to TensorObs."""

        @qjit
        @qml.qnode(qml.device(backend, wires=3))
        def circuit(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.RX(x + y, wires=2)
            qml.CNOT(wires=[0, 1])
            return meas(
                qml.ops.op_math.Prod(
                    qml.PauliX(wires=0), qml.PauliZ(wires=1), qml.Identity(wires=2)
                )
            )

        result = circuit(np.pi / 4, np.pi / 2)
        assert np.allclose(expected, result)

    @pytest.mark.parametrize(
        "meas, expected",
        [
            [
                qml.expval(
                    qml.ops.op_math.Sum(
                        qml.PauliX(wires=0), qml.PauliY(wires=1), qml.PauliZ(wires=2)
                    )
                ),
                np.array(-1.41421356),
            ],
            [
                qml.var(
                    qml.ops.op_math.Sum(
                        qml.PauliX(wires=0), qml.PauliY(wires=1), qml.PauliZ(wires=2)
                    )
                ),
                np.array(2.0),
            ],
            [
                qml.expval(qml.PauliX(wires=0) + qml.PauliY(wires=1) + qml.PauliZ(wires=2)),
                np.array(-1.41421356),
            ],
            [
                qml.var(qml.PauliX(wires=0) + qml.PauliY(wires=1) + qml.PauliZ(wires=2)),
                np.array(2.0),
            ],
        ],
    )
    def test_sum_xyz(self, meas, expected, backend):
        """Test ``qml.ops.op_math.Sum`` and ``+`` converting to HamiltonianObs.
        with integer coefficients."""

        @qjit
        @qml.qnode(qml.device(backend, wires=3))
        def circuit(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.RX(x + y, wires=2)
            qml.CNOT(wires=[0, 1])
            return meas

        result = circuit(np.pi / 4, np.pi / 2)
        assert np.allclose(expected, result)

    @pytest.mark.parametrize(
        "meas, expected",
        [
            [
                qml.expval(
                    qml.ops.op_math.Sum(
                        qml.PauliX(wires=0),
                        qml.PauliY(wires=1),
                        qml.ops.op_math.SProd(0.5, qml.PauliZ(2)),
                    )
                ),
                np.array(-1.06066017),
            ],
            [
                qml.var(
                    qml.ops.op_math.Sum(
                        qml.ops.op_math.SProd(0.2, qml.PauliX(wires=0)),
                        qml.ops.op_math.SProd(0.4, qml.PauliY(wires=1)),
                        qml.ops.op_math.SProd(0.5, qml.PauliZ(wires=2)),
                    )
                ),
                np.array(0.245),
            ],
            [
                qml.expval(qml.PauliX(wires=0) + qml.PauliY(wires=1) + 0.5 * qml.PauliZ(wires=2)),
                np.array(-1.06066017),
            ],
            [
                qml.var(
                    0.2 * qml.PauliX(wires=0)
                    + 0.4 * qml.PauliY(wires=1)
                    + 0.5 * qml.PauliZ(wires=2)
                ),
                np.array(0.245),
            ],
        ],
    )
    def test_sum_sprod_xyz(self, meas, expected, backend):
        """Test ``qml.ops.op_math.Sum`` (``+``) and ``qml.ops.op_math.SProd`` (``*``)."""

        @qjit
        @qml.qnode(qml.device(backend, wires=3))
        def circuit(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.RX(x + y, wires=2)
            qml.CNOT(wires=[0, 1])
            return meas

        result = circuit(np.pi / 4, np.pi / 2)
        assert np.allclose(expected, result)

    def test_mix_dunder(self, backend):
        """Test ``*`` and ``@`` dunder methods."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(-0.5 * qml.PauliX(0) @ qml.PauliY(1))

        result = circuit(np.pi / 4, np.pi / 4)
        expected = np.array(0.25)
        assert np.allclose(expected, result)

    def test_sum_hermitian(self, backend):
        """Test ``+`` with Hermitian observables."""

        @qjit
        @qml.qnode(qml.device(backend, wires=3))
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.CNOT(wires=[0, 1])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qml.expval(qml.Hermitian(A, wires=2) + qml.PauliX(0) + qml.Hermitian(A, wires=1))

        result = circuit(np.pi / 4, np.pi / 4)
        expected = np.array(2.0)
        assert np.allclose(expected, result)

    def test_prod_hermitian(self, backend):
        """Test ``@`` with Hermitian observables."""

        @qjit
        @qml.qnode(qml.device(backend, wires=4))
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.RY(x + y, wires=2)
            qml.RY(x - y, wires=3)
            qml.CNOT(wires=[0, 1])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qml.expval(
                qml.PauliZ(2)
                @ qml.Hermitian(A, wires=1)
                @ qml.PauliZ(0)
                @ qml.Hermitian(A, wires=3)
            )

        result = circuit(np.pi / 4, np.pi / 2)
        expected = np.array(0.20710678)
        assert np.allclose(expected, result)

    def test_sprod_hermitian(self, backend):
        """Test ``*`` and ``@`` with Hermitian observable."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.CNOT(wires=[0, 1])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qml.expval(0.2 * qml.Hermitian(A, wires=1) @ qml.PauliZ(0))

        result = circuit(np.pi / 4, np.pi / 2)
        expected = np.array(0.14142136)
        assert np.allclose(expected, result)

    def test_sum_sprod_prod_hermitian(self, backend):
        """Test ``+`` of ``@`` with Hermitian observable."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.CNOT(wires=[0, 1])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qml.expval(qml.Hermitian(A, wires=1) + qml.PauliZ(0) @ qml.PauliY(1))

        result = circuit(np.pi / 4, np.pi)
        expected = np.array(1.0)
        assert np.allclose(expected, result)

    def test_dunder_hermitian_1(self, backend):
        """Test dunder methods with Hermitian observable to a HamiltonianObs."""

        @qjit
        @qml.qnode(qml.device(backend, wires=4))
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)

            qml.RX(x + y, wires=2)
            qml.RX(y - x, wires=3)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])

            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qml.expval(
                0.2 * (qml.Hermitian(A, wires=1) + qml.PauliZ(0))
                + 0.4 * (qml.PauliX(2) @ qml.PauliZ(3))
            )

        result = circuit(np.pi / 4, np.pi / 2)
        expected = np.array(0.34142136)
        assert np.allclose(expected, result)

    def test_dunder_hermitian_2(self, backend):
        """Test dunder methods with Hermitian observable to a TensorObs."""

        @qjit
        @qml.qnode(qml.device(backend, wires=4))
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)

            qml.RX(x + y, wires=2)
            qml.RX(y - x, wires=3)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])

            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qml.var(
                (qml.Hermitian(A, wires=1) + qml.PauliZ(0))
                @ (0.5 * (qml.PauliX(2) @ qml.PauliZ(3)))
            )

        result = circuit(np.pi, np.pi)
        expected = np.array(1.0)
        assert np.allclose(expected, result)


class TestDensityMatrixMP:
    """Tests for density_matrix"""

    def test_error(self, backend):
        """Test that tracing density matrix produces an error"""

        err_msg = "Measurement .* is not implemented"
        with pytest.raises(NotImplementedError, match=err_msg):

            @qml.qjit
            @qml.qnode(qml.device(backend, wires=1))
            def circuit():
                return qml.density_matrix([0])

class TestVnEntropy:
    """Test vnentropy."""

    @pytest.mark.xfail(reason="Not supported on lightning.")
    def test_vn_entropy(self):
        """Test that VnEntropy can be used with Catalyst."""

        dev = qml.device("lightning.qubit", wires=2)

        @qjit
        @qml.qnode(dev)
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=[0])

        expected = 0.6931471805599453
        assert circuit_entropy(np.pi/2) == expected

class TestMutualInfo:
    """Test mutualinfo."""

    @pytest.mark.xfail(reason="Not supported on lightning.")
    def test_mutual_info(self):
        """Test that MutualInfo can be used with Catalyst."""

        dev = qml.device("lightning.qubit", wires=range(2))

        @qjit
        @qml.qnode(dev)
        def mutual_info_circuit():
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            qml.RX(0, wires=0)
            return qml.mutual_info(0, 1)

        expected = 1.3862943611198906
        assert mutual_info_circuit() == expected


class TestShadow:
    """Test shadow."""

    @pytest.mark.xfail(reason="Not supported on lightning.")
    def test_shadow(self):
        """Test that Shadow can be used with Catalyst."""

        dev = qml.device("lightning.qubit", wires=range(2))

        @qjit
        @qml.qnode(dev)
        def classical_shadow_circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.classical_shadow(wires=[0, 1])

        expected_bits = [[1, 1], [0, 1]]
        expected_recipes = [[0, 1], [0, 2]]
        actual_bits, actual_recipes = classical_shadow_circuit()
        assert expected_bits == actual_bits
        assert expected_recipes == actual_recipes

class TestShadowExpval:
    """Test shadowexpval."""

    @pytest.mark.xfail(reason="Not supported on lightning.")
    def test_shadow_expval(self):
        """Test that ShadowExpVal can be used with Catalyst."""

        dev = qml.device("lightning.qubit", wires=range(2))

        @qjit
        @qml.qnode(dev)
        def shadow_expval_circuit(x, obs):
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            qml.RX(x, wires=0)
            return qml.shadow_expval(obs)

        H = qml.Hamiltonian([1., 1.], [qml.Z(0) @ qml.Z(1), qml.X(0) @ qml.X(1)])
        expected = 1.917
        assert shadow_expval_circuit(0, H) == expected

class TestPurity:
    """Test purity."""

    @pytest.mark.xfail(reason="Not supported on lightning.")
    def test_purity(self):
        """Test that Purity can be used with Catalyst."""

        dev = qml.device("lightning.qubit", wires=[0])

        @qjit
        @qml.qnode(dev)
        def purity_circuit():
            return qml.purity(wires=[0])

        expected = 1.0
        assert purity_circuit() == expected


if __name__ == "__main__":
    pytest.main(["-x", __file__])
