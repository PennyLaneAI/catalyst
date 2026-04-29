# Copyright 2022-2023 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform
from copy import deepcopy

import numpy as np
import pennylane as qp
import pytest
from jax import numpy as jnp
from utils import CONFIG_CUSTOM_DEVICE

from catalyst import CompileError, qjit
from catalyst.device import get_device_capabilities
from catalyst.utils.runtime_environment import get_lib_path

# pylint: disable=too-many-lines


@pytest.mark.usefixtures("use_both_frontend")
class TestSample:
    """Test sample."""

    def test_sample_on_0qbits(self):
        """Test sample on 0 qubits."""

        device = qp.device("lightning.qubit", wires=0)

        @qjit
        @qp.qnode(device, shots=10)
        def sample_0qbit():
            return qp.sample()

        expected = np.empty(shape=(10, 0), dtype=int)
        observed = sample_0qbit()
        assert np.array_equal(observed, expected)

    def test_sample_on_1qbit(self, backend):
        """Test sample on 1 qubit."""

        device = qp.device(backend, wires=1)

        @qjit
        @qp.qnode(device, shots=1000)
        def sample_1qbit(x: float):
            qp.RX(x, wires=0)
            return qp.sample()

        expected = np.array([[0]] * 1000)
        observed = sample_1qbit(0.0)
        assert np.array_equal(observed, expected)

        expected = np.array([[1]] * 1000)
        observed = sample_1qbit(np.pi)
        assert np.array_equal(observed, expected)

    def test_sample_on_2qbits(self, backend):
        """Test sample on 2 qubits."""

        device = qp.device(backend, wires=2)

        @qjit
        @qp.qnode(device, shots=1000)
        def sample_2qbits(x: float):
            qp.RX(x, wires=0)
            qp.RY(x, wires=1)
            return qp.sample()

        expected = np.array([[0, 0]] * 1000)
        observed = sample_2qbits(0.0)
        assert np.array_equal(observed, expected)
        expected = np.array([[1, 1]] * 1000)
        observed = sample_2qbits(np.pi)
        assert np.array_equal(observed, expected)

    @pytest.mark.parametrize("mcm_method", ["single-branch-statistics", "one-shot"])
    def test_sample_on_empty_wires(self, mcm_method):
        """Test sample on dynamic wires."""

        # Devices must specify wires for integration with program capture
        # Since this test is used to test dynamic wires, we skip it if capture is enabled
        if qp.capture.enabled():
            return

        @qp.set_shots(10)
        @qp.qnode(qp.device("lightning.qubit"), mcm_method=mcm_method)
        def sample_dynamic_wires():
            qp.Hadamard(wires=1)
            return qp.sample()

        if mcm_method == "one-shot":
            with pytest.raises(
                NotImplementedError,
                match="cannot be used without wires and a dynamic number of device wires",
            ):
                qjit(sample_dynamic_wires)()
        else:
            qjit(sample_dynamic_wires)()


@pytest.mark.usefixtures("use_both_frontend")
class TestCounts:
    """Test counts."""

    def test_counts_on_0qbits(self):
        """Test counts on 0 qubits."""

        @qjit
        @qp.set_shots(10)
        @qp.qnode(qp.device("lightning.qubit", wires=0))
        def counts_0qbit():
            return qp.counts(all_outcomes=True)

        expected = [np.array([0]), np.array([10])]
        observed = counts_0qbit()
        assert np.array_equal(observed, expected)

    @pytest.mark.parametrize("mcm_method", ["single-branch-statistics", "one-shot"])
    def test_count_on_1qbit(self, backend, mcm_method):
        """Test counts on 1 qubits."""

        @qjit
        @qp.set_shots(1000)
        @qp.qnode(qp.device(backend, wires=1), mcm_method=mcm_method)
        def counts_1qbit(x: float):
            qp.RX(x, wires=0)
            return qp.counts(all_outcomes=True)

        expected = [np.array([0, 1]), np.array([1000, 0])]
        observed = counts_1qbit(0.0)
        assert np.array_equal(observed, expected)

        expected = [np.array([0, 1]), np.array([0, 1000])]
        observed = counts_1qbit(np.pi)
        assert np.array_equal(observed, expected)

    @pytest.mark.parametrize("mcm_method", ["single-branch-statistics", "one-shot"])
    def test_count_on_2qbits(self, backend, mcm_method):
        """Test counts on 2 qubits."""

        @qjit
        @qp.set_shots(1000)
        @qp.qnode(qp.device(backend, wires=2), mcm_method=mcm_method)
        def counts_2qbit(x: float):
            qp.RX(x, wires=0)
            qp.RY(x, wires=1)
            return qp.counts(all_outcomes=True)

        expected = [np.array([0, 1, 2, 3]), np.array([1000, 0, 0, 0])]
        observed = counts_2qbit(0.0)
        assert np.array_equal(observed, expected)

        expected = [np.array([0, 1, 2, 3]), np.array([0, 0, 0, 1000])]
        observed = counts_2qbit(np.pi)
        assert np.array_equal(observed, expected)

    def test_count_on_2qbits_endianness(self, backend):
        """Test counts on 2 qubits with check for endianness."""

        @qjit
        @qp.set_shots(1000)
        @qp.qnode(qp.device(backend, wires=2))
        def counts_2qbit(x: float, y: float):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)
            return qp.counts(all_outcomes=True)

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
        @qp.set_shots(1000)
        @qp.qnode(qp.device(backend, wires=2))
        def counts_2qbit(x: float):
            qp.RX(x, wires=0)
            qp.RY(x, wires=1)
            return qp.counts(all_outcomes=True)

        expected = {"00": 1000, "01": 0, "10": 0, "11": 0}
        observed = counts_2qbit(0.0)
        assert np.array_equal(observed, expected)

        expected = {"00": 0, "01": 0, "10": 0, "11": 1000}
        observed = counts_2qbit(np.pi)
        assert np.array_equal(observed, expected)

    @pytest.mark.parametrize("mcm_method", ["single-branch-statistics", "one-shot"])
    def test_counts_on_empty_wires(self, mcm_method):
        """Test counts on dynamic wires."""

        @qp.set_shots(10)
        @qp.qnode(qp.device("lightning.qubit"), mcm_method=mcm_method)
        def counts_dynamic_wires():
            qp.Hadamard(wires=1)
            return qp.counts(all_outcomes=True)

        if qp.capture.enabled():
            with pytest.raises(
                NotImplementedError,
                match="devices must specify wires for integration with program capture",
            ):
                qjit(counts_dynamic_wires)()
        else:
            if mcm_method == "one-shot":
                with pytest.raises(
                    NotImplementedError,
                    match="cannot be used without wires and a dynamic number of device wires",
                ):
                    qjit(counts_dynamic_wires)()
            else:
                qjit(counts_dynamic_wires)()


@pytest.mark.usefixtures("use_both_frontend")
class TestExpval:

    def test_named(self, backend):
        """Test expval for named observables."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def expval1(x: float):
            qp.RX(x, wires=0)
            return qp.expval(qp.PauliZ(0))

        expected = np.array(1.0)
        observed = expval1(0.0)
        assert np.isclose(observed, expected)

        expected = np.array(-1.0)
        observed = expval1(np.pi)
        assert np.isclose(observed, expected)

    def test_named_identity(self, backend):
        """
        Test expval for identity named observable on multiple wires.
        """

        @qjit
        @qp.qnode(qp.device(backend, wires=3))
        def expval():
            return qp.expval(qp.Identity(wires=[1, 2])), qp.expval(qp.Identity(wires=[0, 1, 2]))

        expected = np.array([1.0, 1.0])
        observed = expval()
        assert np.allclose(observed, expected)

    def test_hermitian_1(self, backend):
        """Test expval for Hermitian observable."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def expval2(x: float):
            qp.RY(x, wires=0)
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qp.expval(qp.Hermitian(A, wires=0))

        expected = np.array(1.0)
        observed = expval2(0.0)
        assert np.isclose(observed, expected)

        expected = np.array(3.0)
        observed = expval2(np.pi / 2)
        assert np.isclose(observed, expected)

    def test_hermitian_2(self, backend):
        """Test expval for Hermitian observable."""

        @qjit
        @qp.qnode(qp.device(backend, wires=2))
        def expval3(x: float):
            qp.RX(x, wires=1)
            B = np.array(
                [
                    [complex(1.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
                    [complex(2.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
                    [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
                    [complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0)],
                ]
            )
            return qp.expval(qp.Hermitian(B, wires=[0, 1]))

        expected = np.array(1.0)
        observed = expval3(0.0)
        assert np.isclose(observed, expected)

        expected = np.array(2.0)
        observed = expval3(np.pi)
        assert np.isclose(observed, expected)

    def test_tensor_1(self, backend):
        """Test expval for Tensor observable."""

        @qjit
        @qp.qnode(qp.device(backend, wires=2))
        def expval4(x: float, y: float):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliX(0) @ qp.PauliY(1))

        expected = np.array(-0.35355339)
        observed = expval4(np.pi / 4, np.pi / 3)
        assert np.isclose(observed, expected)

        expected = np.array(0.0)
        observed = expval4(np.pi / 2, np.pi / 2)
        assert np.isclose(observed, expected)

    def test_tensor_2(self, backend):
        """Test expval for Tensor observable."""

        @qjit
        @qp.qnode(qp.device(backend, wires=3))
        def expval5(x: float, y: float):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)
            qp.CNOT(wires=[0, 2])
            qp.CNOT(wires=[1, 2])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(-1.0, 0.0)]]
            )
            return qp.expval(qp.PauliX(0) @ qp.Hadamard(1) @ qp.Hermitian(A, wires=2))

        expected = np.array(-0.4330127)
        observed = expval5(np.pi / 4, np.pi / 3)
        assert np.isclose(observed, expected)

        expected = np.array(-0.70710678)
        observed = expval5(np.pi / 2, np.pi / 2)
        assert np.isclose(observed, expected)

    def test_hamiltonian_1(self, backend):
        """Test expval for Hamiltonian observable."""

        @qjit
        @qp.qnode(qp.device(backend, wires=3))
        def expval(x: float, y: float):
            qp.RX(x, wires=0)
            qp.RY(y, wires=1)
            qp.RZ(0.1, wires=2)

            coeffs = np.array([0.2, -0.543])
            obs = [qp.PauliX(0) @ qp.PauliZ(1), qp.PauliZ(0) @ qp.Hadamard(2)]

            return qp.expval(qp.Hamiltonian(coeffs, obs))

        expected = np.array(-0.2715)
        observed = expval(np.pi / 4, np.pi / 3)
        assert np.isclose(observed, expected)

        expected = np.array(-0.33695571)
        observed = expval(0.5, 0.8)
        assert np.isclose(observed, expected)

    def test_hamiltonian_2(self, backend):
        """Test expval for Hamiltonian observable."""

        @qjit
        @qp.qnode(qp.device(backend, wires=2))
        def expval(x: float):
            qp.RX(x, wires=0)

            coeff = np.array([0.8, 0.2])
            obs_matrix = np.array(
                [
                    [0.5, 1.0j, 0.0, -3j],
                    [-1.0j, -1.1, 0.0, -0.1],
                    [0.0, 0.0, -0.9, 12.0],
                    [3j, -0.1, 12.0, 0.0],
                ]
            )

            obs = qp.Hermitian(obs_matrix, wires=[0, 1])
            return qp.expval(qp.Hamiltonian(coeff, [obs, qp.PauliX(0)]))

        expected = np.array(0.2359798)
        observed = expval(np.pi / 4)
        assert np.isclose(observed, expected)

        expected = np.array(-0.16)
        observed = expval(np.pi / 2)
        assert np.isclose(observed, expected)

    def test_hamiltonian_3(self, backend):
        """Test expval for nested Hamiltonian observable."""

        @qjit
        @qp.qnode(qp.device(backend, wires=2))
        def expval(x: float):
            qp.RX(x, wires=0)

            coeff = np.array([0.8, 0.2])
            obs_matrix = np.array(
                [
                    [0.5, 1.0j, 0.0, -3j],
                    [-1.0j, -1.1, 0.0, -0.1],
                    [0.0, 0.0, -0.9, 12.0],
                    [3j, -0.1, 12.0, 0.0],
                ]
            )

            obs = qp.Hermitian(obs_matrix, wires=[0, 1])
            return qp.expval(
                qp.Hamiltonian(coeff, [obs, qp.Hamiltonian([1, 1], [qp.PauliX(0), qp.PauliZ(1)])])
            )

        expected = np.array(0.4359798)
        observed = expval(np.pi / 4)
        assert np.isclose(observed, expected)

        expected = np.array(0.04)
        observed = expval(np.pi / 2)
        assert np.isclose(observed, expected)

    def test_hamiltonian_4(self, backend):
        """Test expval with TensorObs and nested Hamiltonian observables."""

        obs_matrix = np.array(
            [
                [0.5, 1.0j, 0.0, -3j],
                [-1.0j, -1.1, 0.0, -0.1],
                [0.0, 0.0, -0.9, 12.0],
                [3j, -0.1, 12.0, 0.0],
            ]
        )
        obs = qp.Hermitian(obs_matrix, wires=[0, 1])
        coeff = np.array([0.8, 0.2])
        obs2 = qp.Hamiltonian(coeff, [obs, qp.Hamiltonian([1, 1], [qp.X(0), qp.Z(1)])])
        obs3 = obs2 @ qp.Z(2)

        @qjit
        @qp.qnode(qp.device(backend, wires=3))
        def expval(x: float):
            qp.RX(x, wires=0)
            qp.RX(x + 1.0, wires=2)

            return qp.expval(obs3)

        expected = np.array(-0.09284557)
        observed = expval(np.pi / 4)
        assert np.isclose(observed, expected)

        expected = np.array(-0.03365884)
        observed = expval(np.pi / 2)
        assert np.isclose(observed, expected)


@pytest.mark.usefixtures("use_both_frontend")
class TestVar:

    def test_rx(self, backend):
        """Test var with RX."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def var1(x: float):
            qp.RX(x, wires=0)
            return qp.var(qp.PauliZ(0))

        expected = np.array(0.0)
        observed = var1(0.0)
        assert np.isclose(observed, expected)
        observed = var1(np.pi)
        assert np.isclose(observed, expected)

    def test_hadamard(self, backend):
        """Test var with Hadamard."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def var2(x: float):
            qp.Hadamard(wires=0)
            return qp.var(qp.PauliZ(0))

        expected = np.array(1.0)
        observed = var2(0.0)
        assert np.isclose(observed, expected)
        observed = var2(np.pi)
        assert np.isclose(observed, expected)

    def test_hermitian_1(self, backend):
        """Test variance for Hermitian observable."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(x: float):
            qp.RY(x, wires=0)
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qp.var(qp.Hermitian(A, wires=0))

        expected = np.array(4.0)
        observed = circuit(0.0)
        assert np.isclose(observed, expected)

        expected = np.array(0.0)
        observed = circuit(np.pi / 2)
        assert np.isclose(observed, expected)

    def test_hermitian_2(self, backend):
        """Test variance for Hermitian observable."""

        @qjit
        @qp.qnode(qp.device(backend, wires=2))
        def circuit(x: float):
            qp.RX(x, wires=1)
            B = np.array(
                [
                    [complex(1.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
                    [complex(2.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
                    [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
                    [complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0)],
                ]
            )
            return qp.var(qp.Hermitian(B, wires=[0, 1]))

        expected = np.array(9.0)
        observed = circuit(0.0)
        assert np.isclose(observed, expected)

        expected = np.array(9.0)
        observed = circuit(np.pi)
        assert np.isclose(observed, expected)

    def test_tensor_1(self, backend):
        """Test variance for Tensor observable."""

        @qjit
        @qp.qnode(qp.device(backend, wires=2))
        def circuit(x: float, y: float):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)
            qp.CNOT(wires=[0, 1])
            return qp.var(qp.PauliX(0) @ qp.PauliY(1))

        expected = np.array(0.875)
        observed = circuit(np.pi / 4, np.pi / 3)
        assert np.isclose(observed, expected)

        expected = np.array(1.0)
        observed = circuit(np.pi / 2, np.pi / 2)
        assert np.isclose(observed, expected)

    def test_tensor_2(self, backend):
        """Test variance for Tensor observable."""

        @qjit
        @qp.qnode(qp.device(backend, wires=3))
        def circuit(x: float, y: float):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)
            qp.CNOT(wires=[0, 2])
            qp.CNOT(wires=[1, 2])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(-1.0, 0.0)]]
            )
            return qp.var(qp.PauliX(0) @ qp.Hadamard(1) @ qp.Hermitian(A, wires=2))

        expected = np.array(4.8125)
        observed = circuit(np.pi / 4, np.pi / 3)
        assert np.isclose(observed, expected)

        expected = np.array(4.5)
        observed = circuit(np.pi / 2, np.pi / 2)
        assert np.isclose(observed, expected)

    def test_hamiltonian_1(self, backend):
        """Test variance for Hamiltonian observable."""

        @qjit
        @qp.qnode(qp.device(backend, wires=3))
        def circuit(x: float, y: float):
            qp.RX(x, wires=0)
            qp.RY(y, wires=1)
            qp.RZ(0.1, wires=2)

            coeffs = np.array([0.2, -0.543])
            obs = [qp.PauliX(0) @ qp.PauliZ(1), qp.PauliZ(0) @ qp.Hadamard(2)]

            return qp.var(qp.Hamiltonian(coeffs, obs))

        expected = np.array(0.26113675)
        observed = circuit(np.pi / 4, np.pi / 3)
        assert np.isclose(observed, expected)

        expected = np.array(0.22130985)
        observed = circuit(0.5, 0.8)
        assert np.isclose(observed, expected)

    def test_hamiltonian_2(self, backend):
        """Test variance for Hamiltonian observable."""

        @qjit
        @qp.qnode(qp.device(backend, wires=2))
        def circuit(x: float):
            qp.RX(x, wires=0)

            coeff = np.array([0.8, 0.2])
            obs_matrix = np.array(
                [
                    [0.5, 1.0j, 0.0, -3j],
                    [-1.0j, -1.1, 0.0, -0.1],
                    [0.0, 0.0, -0.9, 12.0],
                    [3j, -0.1, 12.0, 0.0],
                ]
            )

            obs = qp.Hermitian(obs_matrix, wires=[0, 1])
            return qp.var(qp.Hamiltonian(coeff, [obs, qp.PauliX(0)]))

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
        @qp.qnode(qp.device(backend, wires=3))
        def circuit(x: float, y: float):
            qp.RX(x, wires=0)
            qp.RY(y, wires=1)
            qp.RZ(0.1, wires=2)

            obs = [qp.PauliX(0) @ qp.PauliZ(1), qp.PauliZ(0) @ qp.Hadamard(2)]

            return qp.var(qp.Hamiltonian(coeffs, obs))

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
        @qp.qnode(qp.device(backend, wires=3))
        def circuit(x, y, coeffs):
            qp.RX(x, wires=0)
            qp.RY(y, wires=1)
            qp.RZ(0.1, wires=2)

            obs = [qp.PauliX(0) @ qp.PauliZ(1), qp.PauliZ(0) @ qp.Hadamard(2)]

            return qp.var(qp.Hamiltonian(coeffs, obs))

        expected = np.array(1.75)
        observed = circuit(np.pi / 4, np.pi / 3, coeffs)
        assert np.isclose(
            observed,
            expected,
        )

    def test_hamiltonian_5(self, backend):
        """Test variance with nested Hamiltonian observable."""

        @qjit
        @qp.qnode(qp.device(backend, wires=3))
        def circuit(x, y, coeffs):
            qp.RX(x, wires=0)
            qp.RY(y, wires=1)
            qp.RZ(0.1, wires=2)

            obs = [
                qp.PauliX(0) @ qp.PauliZ(1),
                qp.Hamiltonian(np.array([0.5]), [qp.PauliZ(0) @ qp.Hadamard(2)]),
            ]

            return qp.var(qp.Hamiltonian(coeffs, obs))

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
        @qp.qnode(qp.device(backend, wires=3))
        def circuit(x, y, coeffs):
            qp.RX(x, wires=0)
            qp.RY(y, wires=1)
            qp.RZ(0.1, wires=2)

            return qp.var(
                qp.Hamiltonian(
                    coeffs,
                    [
                        qp.PauliX(0)
                        @ qp.PauliZ(1)
                        @ qp.Hamiltonian(np.array([0.5]), [qp.Hadamard(2)])
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


@pytest.mark.usefixtures("use_both_frontend")
class TestState:
    """Test state measurement processes."""

    def test_state_on_0qbits(self):
        """Test state on 0 qubits."""

        if qp.capture.enabled():
            pytest.xfail("capture doesn't currently support 0 wires.")

        @qjit
        @qp.qnode(qp.device("lightning.qubit", wires=0))
        def state_0qbit():
            return qp.state()

        expected = np.array([complex(1.0, 0.0)])
        observed = state_0qbit()
        assert np.array_equal(observed, expected)

    def test_state_on_1qubit(self, backend):
        """Test state on 1 qubit."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def state(x: float):
            qp.RX(x, wires=0)
            return qp.state()

        expected = np.array([complex(1.0, 0.0), complex(0.0, -1.0)]) / np.sqrt(2)
        observed = state(np.pi / 2)
        assert np.allclose(observed, expected)


@pytest.mark.usefixtures("use_both_frontend")
class TestProbs:
    """Test probabilities measurement processes."""

    def test_probs_on_0qbits(self):
        """Test probs on 0 qubits."""

        if qp.capture.enabled():
            pytest.xfail("capture doesn't currently support 0 wires.")

        @qjit
        @qp.qnode(qp.device("lightning.qubit", wires=0))
        def probs_0qbit():
            return qp.probs()

        expected = np.array([1.0])
        observed = probs_0qbit()
        assert np.array_equal(observed, expected)

    def test_probs_on_1qubit(self, backend):
        """Test probs on 1 qubit."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def probs(x: float):
            qp.RX(x, wires=0)
            return qp.probs()

        expected = np.array([0.5, 0.5])
        observed = probs(np.pi / 2)
        assert np.allclose(observed, expected)


@pytest.mark.usefixtures("use_both_frontend")
class TestNewArithmeticOps:
    "Test PennyLane new arithmetic operators"

    @pytest.mark.parametrize(
        "meas, expected",
        [[qp.expval, np.array(-0.70710678)], [qp.var, np.array(0.5)]],
    )
    def test_prod_xzi(self, meas, expected, backend):
        """Test ``qp.ops.op_math.Prod`` converting to TensorObs."""

        @qjit
        @qp.qnode(qp.device(backend, wires=3))
        def circuit(x: float, y: float):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)
            qp.RX(x + y, wires=2)
            qp.CNOT(wires=[0, 1])
            return meas(
                qp.ops.op_math.Prod(qp.PauliX(wires=0), qp.PauliZ(wires=1), qp.Identity(wires=2))
            )

        result = circuit(np.pi / 4, np.pi / 2)
        assert np.allclose(expected, result)

    @pytest.mark.parametrize(
        "meas_fn, expected",
        [
            [
                lambda: qp.expval(
                    qp.ops.op_math.Sum(qp.PauliX(wires=0), qp.PauliY(wires=1), qp.PauliZ(wires=2))
                ),
                np.array(-1.41421356),
            ],
            [
                lambda: qp.var(
                    qp.ops.op_math.Sum(qp.PauliX(wires=0), qp.PauliY(wires=1), qp.PauliZ(wires=2))
                ),
                np.array(2.0),
            ],
            [
                lambda: qp.expval(qp.PauliX(wires=0) + qp.PauliY(wires=1) + qp.PauliZ(wires=2)),
                np.array(-1.41421356),
            ],
            [
                lambda: qp.var(qp.PauliX(wires=0) + qp.PauliY(wires=1) + qp.PauliZ(wires=2)),
                np.array(2.0),
            ],
        ],
    )
    def test_sum_xyz(self, meas_fn, expected, backend):
        """Test ``qp.ops.op_math.Sum`` and ``+`` converting to HamiltonianObs.
        with integer coefficients."""

        @qjit
        @qp.qnode(qp.device(backend, wires=3))
        def circuit(x: float, y: float):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)
            qp.RX(x + y, wires=2)
            qp.CNOT(wires=[0, 1])
            return meas_fn()

        result = circuit(np.pi / 4, np.pi / 2)
        assert np.allclose(expected, result)

    @pytest.mark.parametrize(
        "meas_fn, expected",
        [
            [
                lambda: qp.expval(
                    qp.ops.op_math.Sum(
                        qp.PauliX(wires=0),
                        qp.PauliY(wires=1),
                        qp.ops.op_math.SProd(0.5, qp.PauliZ(2)),
                    )
                ),
                np.array(-1.06066017),
            ],
            [
                lambda: qp.var(
                    qp.ops.op_math.Sum(
                        qp.ops.op_math.SProd(0.2, qp.PauliX(wires=0)),
                        qp.ops.op_math.SProd(0.4, qp.PauliY(wires=1)),
                        qp.ops.op_math.SProd(0.5, qp.PauliZ(wires=2)),
                    )
                ),
                np.array(0.245),
            ],
            [
                lambda: qp.expval(
                    qp.PauliX(wires=0) + qp.PauliY(wires=1) + 0.5 * qp.PauliZ(wires=2)
                ),
                np.array(-1.06066017),
            ],
            [
                lambda: qp.var(
                    0.2 * qp.PauliX(wires=0) + 0.4 * qp.PauliY(wires=1) + 0.5 * qp.PauliZ(wires=2)
                ),
                np.array(0.245),
            ],
        ],
    )
    def test_sum_sprod_xyz(self, meas_fn, expected, backend):
        """Test ``qp.ops.op_math.Sum`` (``+``) and ``qp.ops.op_math.SProd`` (``*``)."""

        @qjit
        @qp.qnode(qp.device(backend, wires=3))
        def circuit(x: float, y: float):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)
            qp.RX(x + y, wires=2)
            qp.CNOT(wires=[0, 1])
            return meas_fn()

        result = circuit(np.pi / 4, np.pi / 2)
        assert np.allclose(expected, result)

    def test_mix_dunder(self, backend):
        """Test ``*`` and ``@`` dunder methods."""

        @qjit
        @qp.qnode(qp.device(backend, wires=2))
        def circuit(x, y):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)
            qp.CNOT(wires=[0, 1])
            return qp.expval(-0.5 * qp.PauliX(0) @ qp.PauliY(1))

        result = circuit(np.pi / 4, np.pi / 4)
        expected = np.array(0.25)
        assert np.allclose(expected, result)

    def test_sum_hermitian(self, backend):
        """Test ``+`` with Hermitian observables."""

        @qjit
        @qp.qnode(qp.device(backend, wires=3))
        def circuit(x, y):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)
            qp.CNOT(wires=[0, 1])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qp.expval(qp.Hermitian(A, wires=2) + qp.PauliX(0) + qp.Hermitian(A, wires=1))

        result = circuit(np.pi / 4, np.pi / 4)
        expected = np.array(2.0)
        assert np.allclose(expected, result)

    def test_prod_hermitian(self, backend):
        """Test ``@`` with Hermitian observables."""

        @qjit
        @qp.qnode(qp.device(backend, wires=4))
        def circuit(x, y):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)
            qp.RY(x + y, wires=2)
            qp.RY(x - y, wires=3)
            qp.CNOT(wires=[0, 1])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qp.expval(
                qp.PauliZ(2) @ qp.Hermitian(A, wires=1) @ qp.PauliZ(0) @ qp.Hermitian(A, wires=3)
            )

        result = circuit(np.pi / 4, np.pi / 2)
        expected = np.array(0.20710678)
        assert np.allclose(expected, result)

    def test_sprod_hermitian(self, backend):
        """Test ``*`` and ``@`` with Hermitian observable."""

        @qjit
        @qp.qnode(qp.device(backend, wires=2))
        def circuit(x, y):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)
            qp.CNOT(wires=[0, 1])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qp.expval(0.2 * qp.Hermitian(A, wires=1) @ qp.PauliZ(0))

        result = circuit(np.pi / 4, np.pi / 2)
        expected = np.array(0.14142136)
        assert np.allclose(expected, result)

    def test_sum_sprod_prod_hermitian(self, backend):
        """Test ``+`` of ``@`` with Hermitian observable."""

        @qjit
        @qp.qnode(qp.device(backend, wires=2))
        def circuit(x, y):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)
            qp.CNOT(wires=[0, 1])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qp.expval(qp.Hermitian(A, wires=1) + qp.PauliZ(0) @ qp.PauliY(1))

        result = circuit(np.pi / 4, np.pi)
        expected = np.array(1.0)
        assert np.allclose(expected, result)

    def test_dunder_hermitian_1(self, backend):
        """Test dunder methods with Hermitian observable to a HamiltonianObs."""

        @qjit
        @qp.qnode(qp.device(backend, wires=4))
        def circuit(x, y):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)

            qp.RX(x + y, wires=2)
            qp.RX(y - x, wires=3)

            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[0, 2])

            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qp.expval(
                0.2 * (qp.Hermitian(A, wires=1) + qp.PauliZ(0))
                + 0.4 * (qp.PauliX(2) @ qp.PauliZ(3))
            )

        result = circuit(np.pi / 4, np.pi / 2)
        expected = np.array(0.34142136)
        assert np.allclose(expected, result)

    def test_dunder_hermitian_2(self, backend):
        """Test dunder methods with Hermitian observable to a TensorObs."""

        @qjit
        @qp.qnode(qp.device(backend, wires=4))
        def circuit(x, y):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)

            qp.RX(x + y, wires=2)
            qp.RX(y - x, wires=3)

            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[0, 2])

            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qp.var(
                (qp.Hermitian(A, wires=1) + qp.PauliZ(0)) @ (0.5 * (qp.PauliX(2) @ qp.PauliZ(3)))
            )

        result = circuit(np.pi, np.pi)
        expected = np.array(1.0)
        assert np.allclose(expected, result)


class CustomDevice(qp.devices.Device):
    """Custom Gate Set Device"""

    name = "Custom Device"
    config_filepath = CONFIG_CUSTOM_DEVICE

    _to_matrix_ops = {}

    def __init__(self, shots=None, wires=None):
        super().__init__(wires=wires, shots=shots)
        self.qjit_capabilities = deepcopy(get_device_capabilities(self))
        self.qjit_capabilities.measurement_processes["DensityMatrixMP"] = []

    def apply(self, operations, **kwargs):
        """Unused"""
        raise RuntimeError("Only C/C++ interface is defined")

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """
        system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
        lib_path = (
            get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_null_qubit" + system_extension
        )
        return "NullQubit", lib_path

    def execute(self, circuits, execution_config):
        """Execution."""
        return circuits, execution_config


class TestDensityMatrixMP:
    """Tests for density_matrix"""

    def test_error(self):
        """Test that tracing density matrix produces an error"""

        err_msg = "DensityMatrixMP is not a supported measurement process"
        with pytest.raises(CompileError, match=err_msg):

            @qjit
            @qp.qnode(CustomDevice(wires=1))
            def circuit():
                return qp.density_matrix([0])


@pytest.mark.usefixtures("use_both_frontend")
class TestVnEntropy:
    """Test vnentropy."""

    @pytest.mark.xfail(reason="Not supported on lightning.")
    def test_vn_entropy(self):
        """Test that VnEntropy can be used with Catalyst."""

        dev = qp.device("lightning.qubit", wires=2)

        @qjit
        @qp.qnode(dev)
        def circuit_entropy(x):
            qp.IsingXX(x, wires=[0, 1])
            return qp.vn_entropy(wires=[0])

        expected = 0.6931471805599453
        assert circuit_entropy(np.pi / 2) == expected


@pytest.mark.usefixtures("use_both_frontend")
class TestMutualInfo:
    """Test mutualinfo."""

    @pytest.mark.xfail(reason="Not supported on lightning.")
    def test_mutual_info(self):
        """Test that MutualInfo can be used with Catalyst."""

        dev = qp.device("lightning.qubit", wires=range(2))

        @qjit
        @qp.qnode(dev)
        def mutual_info_circuit():
            qp.Hadamard(0)
            qp.CNOT((0, 1))
            qp.RX(0, wires=0)
            return qp.mutual_info(0, 1)

        expected = 1.3862943611198906
        assert mutual_info_circuit() == expected


@pytest.mark.usefixtures("use_both_frontend")
class TestShadow:
    """Test shadow."""

    @pytest.mark.xfail(reason="Not supported on lightning.")
    def test_shadow(self):
        """Test that Shadow can be used with Catalyst."""

        dev = qp.device("lightning.qubit", wires=range(2))

        @qjit
        @qp.qnode(dev)
        def classical_shadow_circuit():
            qp.Hadamard(0)
            qp.CNOT(wires=[0, 1])
            return qp.classical_shadow(wires=[0, 1])

        expected_bits = [[1, 1], [0, 1]]
        expected_recipes = [[0, 1], [0, 2]]
        actual_bits, actual_recipes = classical_shadow_circuit()
        assert expected_bits == actual_bits
        assert expected_recipes == actual_recipes


@pytest.mark.usefixtures("use_both_frontend")
class TestShadowExpval:
    """Test shadowexpval."""

    @pytest.mark.xfail(reason="Not supported on lightning.")
    def test_shadow_expval(self):
        """Test that ShadowExpVal can be used with Catalyst."""

        dev = qp.device("lightning.qubit", wires=range(2))

        @qjit
        @qp.qnode(dev)
        def shadow_expval_circuit(x, obs):
            qp.Hadamard(0)
            qp.CNOT((0, 1))
            qp.RX(x, wires=0)
            return qp.shadow_expval(obs)

        H = qp.Hamiltonian([1.0, 1.0], [qp.Z(0) @ qp.Z(1), qp.X(0) @ qp.X(1)])
        expected = 1.917
        assert shadow_expval_circuit(0, H) == expected


@pytest.mark.usefixtures("use_both_frontend")
class TestPurity:
    """Test purity."""

    @pytest.mark.xfail(reason="Not supported on lightning.")
    def test_purity(self):
        """Test that Purity can be used with Catalyst."""

        dev = qp.device("lightning.qubit", wires=[0])

        @qjit
        @qp.qnode(dev)
        def purity_circuit():
            return qp.purity(wires=[0])

        expected = 1.0
        assert purity_circuit() == expected


class TestNullQubitMeasurements:
    """Test measurement results with null.qubit."""

    n_shots = 100

    @pytest.mark.parametrize("n_qubits", [0, 1, 2])
    def test_nullq_sample(self, n_qubits):
        """Test qp.sample() on null.qubit device."""

        @qjit
        @qp.set_shots(self.n_shots)
        @qp.qnode(qp.device("null.qubit", wires=n_qubits))
        def circuit_sample():
            for i in range(n_qubits):
                qp.Hadamard(wires=i)
            return qp.sample()

        # Explicitly define expected result for sample since qjit outputs results in different
        # format than native PennyLane
        expected = np.zeros(shape=(self.n_shots, n_qubits), dtype=np.int64)
        observed = circuit_sample()
        assert np.array_equal(observed, expected)

    def test_nullq_sample_per_wire(self):
        """Test qp.sample() on null.qubit device, returning results per wire."""

        @qjit
        @qp.set_shots(self.n_shots)
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit_sample():
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=1)
            return qp.sample(wires=0), qp.sample(wires=1)

        # Explicitly define expected result for sample since qjit outputs results in different
        # format than native PennyLane
        expected = np.zeros(shape=(self.n_shots, 1), dtype=np.int64)
        observed_0, observed_1 = circuit_sample()
        assert np.array_equal(observed_0, expected)
        assert np.array_equal(observed_1, expected)

    @pytest.mark.parametrize("n_qubits", [0, 1, 2])
    def test_nullq_counts(self, n_qubits):
        """Test qp.counts() on null.qubit device."""

        @qjit
        @qp.set_shots(self.n_shots)
        @qp.qnode(qp.device("null.qubit", wires=n_qubits))
        def circuit_counts():
            for i in range(n_qubits):
                qp.Hadamard(wires=i)
            return qp.counts(all_outcomes=True)

        # Explicitly define expected result for counts since qjit outputs results in different
        # format than native PennyLane
        expected = [
            np.arange(0, 2**n_qubits, dtype=np.int64),
            np.zeros(shape=2**n_qubits, dtype=np.int64),
        ]
        expected[1][0] = self.n_shots
        observed = circuit_counts()
        assert np.array_equal(observed, expected)

    def test_nullq_counts_per_wire(self):
        """Test qp.counts() on null.qubit device, returning results per wire."""

        @qjit
        @qp.set_shots(self.n_shots)
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit_counts():
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=1)
            return qp.counts(wires=0, all_outcomes=True), qp.counts(wires=1, all_outcomes=True)

        # Explicitly define expected result for counts since qjit outputs results in different
        # format than native PennyLane
        expected = [
            np.arange(0, 2, dtype=np.int64),
            np.zeros(shape=2, dtype=np.int64),
        ]
        expected[1][0] = self.n_shots
        observed_0, observed_1 = circuit_counts()
        assert np.array_equal(observed_0, expected)
        assert np.array_equal(observed_1, expected)

    @pytest.mark.parametrize("n_qubits", [0, 1, 2])
    def test_nullq_probs(self, n_qubits):
        """Test qp.probs() on null.qubit device."""

        @qp.set_shots(self.n_shots)
        @qp.qnode(qp.device("null.qubit", wires=n_qubits))
        def circuit_probs():
            for i in range(n_qubits):
                qp.Hadamard(wires=i)
            return qp.probs()

        expected = circuit_probs()
        observed = qjit(circuit_probs)()
        assert np.array_equal(observed, expected)

    def test_nullq_probs_per_wire(self):
        """Test qp.probs() on null.qubit device, returning results per wire."""

        @qp.set_shots(self.n_shots)
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit_probs():
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=1)
            return qp.probs(wires=0), qp.probs(wires=1)

        expected = circuit_probs()
        observed = qjit(circuit_probs)()
        assert np.array_equal(observed, expected)

    @pytest.mark.parametrize("n_qubits", [0, 1, 2])
    def test_nullq_state(self, n_qubits):
        """Test qp.state() on null.qubit device."""

        @qp.set_shots(None)
        @qp.qnode(qp.device("null.qubit", wires=n_qubits))
        def circuit_state():
            for i in range(n_qubits):
                qp.Hadamard(wires=i)
            return qp.state()

        expected = circuit_state()
        observed = qjit(circuit_state)()
        assert np.array_equal(observed, expected)

    @pytest.mark.parametrize("n_qubits", [1, 2])
    def test_nullq_expval(self, n_qubits):
        """Test qp.expval() on null.qubit device."""

        @qp.set_shots(self.n_shots)
        @qp.qnode(qp.device("null.qubit", wires=n_qubits))
        def circuit_expval():
            for i in range(n_qubits):
                qp.Hadamard(wires=i)

            return qp.expval(qp.X(0)), qp.expval(qp.Y(0)), qp.expval(qp.Z(0))

        expected = circuit_expval()
        observed = qjit(circuit_expval)()
        assert np.array_equal(observed, expected)

    @pytest.mark.parametrize("n_qubits", [1, 2])
    def test_nullq_var(self, n_qubits):
        """Test qp.var() on null.qubit device."""

        @qp.set_shots(self.n_shots)
        @qp.qnode(qp.device("null.qubit", wires=n_qubits))
        def circuit_var():
            for i in range(n_qubits):
                qp.Hadamard(wires=i)

            return qp.var(qp.X(0)), qp.var(qp.Y(0)), qp.var(qp.Z(0))

        expected = circuit_var()
        observed = qjit(circuit_var)()
        assert np.array_equal(observed, expected)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
