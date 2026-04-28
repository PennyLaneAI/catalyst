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

"""Test measurement results with finite-shots."""

import numpy as np
import pennylane as qp
import pytest

from catalyst import CompileError, qjit


class TestExpval:
    "Test expval with shots > 0"

    def test_identity(self, backend, tol_stochastic, capture_mode):
        """Test that identity expectation value (i.e. the trace) is 1."""
        n_wires = 2
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        theta = 0.432
        phi = 0.123

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.Identity(wires=0)), qp.expval(qp.Identity(wires=1))

        result = qjit(circuit, seed=37, capture=capture_mode)()
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit()

        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_pauliz(self, backend, tol_stochastic, capture_mode):
        """Test that PauliZ expectation value is correct"""
        n_wires = 2
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        theta = 0.432
        phi = 0.123

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(wires=0)), qp.expval(qp.PauliZ(wires=1))

        result = qjit(circuit, seed=37, capture=capture_mode)()
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_paulix(self, backend, tol_stochastic, capture_mode):
        """Test that PauliX expectation value is correct"""
        n_wires = 2
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        theta = 0.432
        phi = 0.123

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit():
            qp.RY(theta, wires=[0])
            qp.RY(phi, wires=[1])
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliX(wires=0)), qp.expval(qp.PauliX(wires=1))

        result = qjit(circuit, seed=37, capture=capture_mode)()
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_pauliy(self, backend, tol_stochastic, capture_mode):
        """Test that PauliY expectation value is correct"""
        n_wires = 2
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        theta = 0.432
        phi = 0.123

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliY(wires=0)), qp.expval(qp.PauliY(wires=1))

        result = qjit(circuit, seed=37, capture=capture_mode)()
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_hadamard(self, backend, tol_stochastic, capture_mode):
        """Test that Hadamard expectation value is correct"""
        n_wires = 2
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        theta = 0.432
        phi = 0.123

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit():
            qp.RY(theta, wires=[0])
            qp.RY(phi, wires=[1])
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.Hadamard(wires=0)), qp.expval(qp.Hadamard(wires=1))

        result = qjit(circuit, seed=37, capture=capture_mode)()
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit()

        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_hermitian(self, backend, tol_stochastic, capture_mode):
        """Test expval Hermitian observables with shots."""
        n_wires = 3
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit(x, y):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)
            qp.CNOT(wires=[0, 1])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qp.expval(qp.Hermitian(A, wires=2))

        result = qjit(circuit, seed=37, capture=capture_mode)(np.pi / 4, np.pi / 4)
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit(np.pi / 4, np.pi / 4)

        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_paulix_pauliy(self, backend, tol_stochastic, capture_mode):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        n_wires = 3
        n_shots = 100000
        dev = qp.device(backend, wires=n_wires)

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.expval(qp.PauliX(wires=0) @ qp.PauliY(wires=2))

        result = qjit(circuit, seed=37, capture=capture_mode)()
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_pauliz_pauliy_prod(self, backend, tol_stochastic, capture_mode):
        """Test that a tensor product involving PauliZ and PauliY works correctly"""
        n_wires = 3
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit(theta, phi, varphi):
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.expval(qp.PauliX(2) @ qp.PauliY(1) @ qp.PauliZ(0))

        result = qjit(circuit, seed=37, capture=capture_mode)(0.432, 0.123, -0.543)
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit(0.432, 0.123, -0.543)

        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_pauliz_hamiltonian(self, backend, tol_stochastic, capture_mode):
        """Test that a hamiltonian involving PauliZ and PauliY and hadamard works correctly"""
        n_wires = 3
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit(theta, phi, varphi):
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.expval(
                0.2 * qp.PauliZ(wires=0) + 0.5 * qp.Hadamard(wires=1) + qp.PauliY(wires=2)
            )

        result = qjit(circuit, seed=37, capture=capture_mode)(0.432, 0.123, -0.543)
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit(0.432, 0.123, -0.543)

        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_prod_hamiltonian(self, backend, tol_stochastic, capture_mode):
        """Test that a hamiltonian involving PauliZ and Hadamard @ PauliX works correctly"""
        n_wires = 3
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit(theta, phi, varphi):
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.expval(0.2 * qp.PauliZ(wires=0) + 0.5 * qp.Hadamard(wires=1) @ qp.PauliX(2))

        result = qjit(circuit, seed=37, capture=capture_mode)(0.432, 0.123, -0.543)
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit(0.432, 0.123, -0.543)

        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)


class TestVar:
    "Test var with shots > 0"

    def test_identity(self, backend, tol_stochastic, capture_mode):
        """Test that identity variance value (i.e. the trace) is 1."""
        n_wires = 2
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        theta = 0.432
        phi = 0.123

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.CNOT(wires=[0, 1])
            return qp.var(qp.Identity(wires=0)), qp.var(qp.Identity(wires=1))

        result = qjit(circuit, seed=37, capture=capture_mode)()
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_pauliz(self, backend, tol_stochastic, capture_mode):
        """Test that PauliZ variance value is correct"""
        n_wires = 2
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        theta = 0.432
        phi = 0.123

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.CNOT(wires=[0, 1])
            return qp.var(qp.PauliZ(wires=0)), qp.var(qp.PauliZ(wires=1))

        result = qjit(circuit, seed=37, capture=capture_mode)()
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit()

        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_paulix(self, backend, tol_stochastic, capture_mode):
        """Test that PauliX variance value is correct"""
        n_wires = 2
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        theta = 0.432
        phi = 0.123

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit():
            qp.RY(theta, wires=[0])
            qp.RY(phi, wires=[1])
            qp.CNOT(wires=[0, 1])
            return qp.var(qp.PauliX(wires=0)), qp.var(qp.PauliX(wires=1))

        result = qjit(circuit, seed=37, capture=capture_mode)()
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_pauliy(self, backend, tol_stochastic, capture_mode):
        """Test that PauliY variance value is correct"""
        n_wires = 2
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        theta = 0.432
        phi = 0.123

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.CNOT(wires=[0, 1])
            return qp.var(qp.PauliY(wires=0)), qp.var(qp.PauliY(wires=1))

        result = qjit(circuit, seed=37, capture=capture_mode)()
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit()

        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_hadamard(self, backend, tol_stochastic, capture_mode):
        """Test that Hadamard variance value is correct"""
        n_wires = 2
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        theta = 0.432
        phi = 0.123

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit():
            qp.RY(theta, wires=[0])
            qp.RY(phi, wires=[1])
            qp.CNOT(wires=[0, 1])
            return qp.var(qp.Hadamard(wires=0)), qp.var(qp.Hadamard(wires=1))

        result = qjit(circuit, seed=37, capture=capture_mode)()
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_hermitian_shots(self, backend, tol_stochastic, capture_mode):
        """Test var Hermitian observables with shots."""

        n_wires = 3
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit(x, y):
            qp.RX(x, wires=0)
            qp.RX(y, wires=1)
            qp.CNOT(wires=[0, 1])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qp.var(qp.Hermitian(A, wires=2))

        result = qjit(circuit, seed=37, capture=capture_mode)(np.pi / 4, np.pi / 4)
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit(np.pi / 4, np.pi / 4)

        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_paulix_pauliy(self, backend, tol_stochastic, capture_mode):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        n_wires = 3
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.var(qp.PauliX(wires=0) @ qp.PauliY(wires=2))

        result = qjit(circuit, seed=37, capture=capture_mode)()
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit()

        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_hadamard_pauliy_prod(self, backend, tol_stochastic, capture_mode):
        """Test that a tensor product involving Hadamard and PauliY works correctly"""
        n_wires = 3
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit(theta, phi, varphi):
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.var(qp.Hadamard(wires=1) @ qp.PauliY(wires=2))

        result = qjit(circuit, seed=37, capture=capture_mode)(0.432, 0.123, -0.543)
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit(0.432, 0.123, -0.543)

        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_pauliz_pauliy_prod(self, backend, tol_stochastic, capture_mode):
        """Test that a tensor product involving PauliZ and PauliY works correctly"""
        n_wires = 3
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit(theta, phi, varphi):
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.var(qp.PauliX(2) @ qp.PauliY(1) @ qp.PauliZ(0))

        result = qjit(circuit, seed=37, capture=capture_mode)(0.432, 0.123, -0.543)
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit(0.432, 0.123, -0.543)

        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    @pytest.mark.xfail(
        reason="error disappeared when I added qjit. Should be investigated. sc-95950"
    )
    def test_pauliz_hamiltonian(self, backend, capture_mode):
        """Test that a hamiltonian involving PauliZ and PauliY and hadamard works correctly"""

        n_wires = 3
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        @qp.qjit(capture=capture_mode)
        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit(theta, phi, varphi):
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.var(0.2 * qp.PauliZ(wires=0) + 0.5 * qp.Hadamard(wires=1))

        if isinstance(dev, qp.devices.LegacyDeviceFacade):
            with pytest.raises(
                RuntimeError,
                match=r"Cannot split up terms in sums for MeasurementProcess <class 'pennylane.measurements.var.VarianceMP'>",
            ):
                circuit(0.432, 0.123, -0.543)
        else:
            # TODO: only raises with the new API, Kokkos should also raise an error.
            with pytest.raises(
                TypeError,
                match=r"VarianceMP\(Sum\) cannot be computed with samples.",
            ):
                circuit(0.432, 0.123, -0.543)


class TestProbs:
    "Test var with shots > 0"

    def test_probs(self, backend, tol_stochastic, capture_mode):
        """Test probs on all wires"""

        n_wires = 2
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit(theta):
            qp.RX(theta, wires=[0])
            qp.Hadamard(wires=[1])
            return qp.probs()

        result = qjit(circuit, seed=37, capture=capture_mode)(0.432)
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit(0.432)

        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_probs_wire(self, backend, tol_stochastic, capture_mode):
        """Test probs on subset of wires"""

        n_wires = 2
        n_shots = 10000
        dev = qp.device(backend, wires=n_wires)

        @qp.set_shots(n_shots)
        @qp.qnode(dev)
        def circuit(theta):
            qp.RX(theta, wires=[0])
            qp.Hadamard(wires=[1])
            return qp.probs(wires=[0])

        result = qjit(circuit, seed=37, capture=capture_mode)(0.432)
        qp.capture.disable()  # capture execution unmaintained
        expected = circuit(0.432)

        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)


class TestShadow:
    """Test shadow."""

    @pytest.mark.xfail(reason="Not supported on lightning.")
    def test_shadow(self, capture_mode):
        """Test that Shadow can be used with Catalyst."""

        dev = qp.device("lightning.qubit", wires=range(2))

        @qjit(capture=capture_mode)
        @qp.set_shots(10000)
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


class TestShadowExpval:
    """Test shadowexpval."""

    @pytest.mark.xfail(reason="TypeError in Catalyst")
    def test_shadow_expval(self, capture_mode):
        """Test that ShadowExpVal can be used with Catalyst."""

        dev = qp.device("lightning.qubit", wires=range(2))

        @qjit(capture=capture_mode)
        @qp.set_shots(10000)
        @qp.qnode(dev)
        def shadow_expval_circuit(x, obs):
            qp.Hadamard(0)
            qp.CNOT((0, 1))
            qp.RX(x, wires=0)
            return qp.shadow_expval(obs)

        H = qp.Hamiltonian([1.0, 1.0], [qp.Z(0) @ qp.Z(1), qp.X(0) @ qp.X(1)])
        expected = 1.9917
        assert shadow_expval_circuit(0, H) == expected


class TestOtherMeasurements:
    """Test other measurement processes."""

    @pytest.mark.parametrize("meas_fun", (qp.sample, qp.counts))
    def test_missing_shots_value(self, backend, meas_fun, capture_mode):
        """Test error for missing shots value."""

        dev = qp.device(backend, wires=1)

        @qp.qnode(dev)
        def circuit():
            return meas_fun(wires=0, **({"all_outcomes": True} if meas_fun is qp.counts else {}))

        if capture_mode:
            with pytest.raises(ValueError, match="finite shots are required"):
                qjit(circuit, capture=capture_mode)
        else:

            with pytest.raises(CompileError, match="cannot work with shots=None"):
                qjit(circuit, capture=capture_mode)

    def test_multiple_return_values(self, backend, tol_stochastic, capture_mode):
        """Test multiple return values."""

        @qjit(capture=capture_mode)
        @qp.set_shots(shots=10000)
        @qp.qnode(qp.device(backend, wires=2))
        def all_measurements(x):
            qp.RY(x, wires=0)
            return (
                qp.sample(),
                qp.counts(all_outcomes=True),
                qp.expval(qp.PauliZ(0)),
                qp.var(qp.PauliZ(0)),
                qp.probs(wires=[0, 1]),
            )

        @qp.set_shots(shots=10000)
        @qp.qnode(qp.device("lightning.qubit", wires=2), static_argnums=(1,))
        def expected(x, measurement_fn):
            qp.RY(x, wires=0)
            return measurement_fn()

        x = 0.7
        result = all_measurements(x)

        # qp.sample
        assert result[0].shape == expected(x, lambda: qp.sample(wires=[0, 1])).shape
        assert result[0].dtype == np.int64

        # qp.counts
        qp.capture.disable()  # cant execute with counts with program capture
        for r, e in zip(result[1][0], expected(x, lambda: qp.counts(all_outcomes=True))):
            assert format(int(r), "02b") == e
        assert sum(result[1][1]) == 10000
        assert result[1][0].dtype == np.int64

        # qp.expval
        assert np.allclose(
            result[2],
            expected(x, lambda: qp.expval(qp.PauliZ(0))),
            atol=tol_stochastic,
            rtol=tol_stochastic,
        )

        # qp.var
        assert np.allclose(
            result[3],
            expected(x, lambda: qp.var(qp.PauliZ(0))),
            atol=tol_stochastic,
            rtol=tol_stochastic,
        )

        # qp.probs
        assert np.allclose(
            result[4],
            expected(x, lambda: qp.probs(wires=[0, 1])),
            atol=tol_stochastic,
            rtol=tol_stochastic,
        )


if __name__ == "__main__":
    pytest.main(["-x", __file__])
