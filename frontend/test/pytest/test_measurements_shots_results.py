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
import pennylane as qml
import pytest

from catalyst import CompileError, qjit


class TestExpval:
    "Test expval with shots > 0"

    def test_identity(self, backend, tol_stochastic):
        """Test that identity expectation value (i.e. the trace) is 1."""
        n_wires = 2
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Identity(wires=0)), qml.expval(qml.Identity(wires=1))

        expected = circuit()
        result = qjit(circuit)()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_pauliz(self, backend, tol_stochastic):
        """Test that PauliZ expectation value is correct"""
        n_wires = 2
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

        expected = circuit()
        result = qjit(circuit)()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_paulix(self, backend, tol_stochastic):
        """Test that PauliX expectation value is correct"""
        n_wires = 2
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(wires=0)), qml.expval(qml.PauliX(wires=1))

        expected = circuit()
        result = qjit(circuit)()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_pauliy(self, backend, tol_stochastic):
        """Test that PauliY expectation value is correct"""
        n_wires = 2
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(wires=0)), qml.expval(qml.PauliY(wires=1))

        expected = circuit()
        result = qjit(circuit)()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_hadamard(self, backend, tol_stochastic):
        """Test that Hadamard expectation value is correct"""
        n_wires = 2
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hadamard(wires=0)), qml.expval(qml.Hadamard(wires=1))

        expected = circuit()
        result = qjit(circuit)()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_hermitian(self, backend):
        """Test expval Hermitian observables with shots."""

        @qjit
        @qml.qnode(qml.device(backend, wires=3, shots=10000))
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.CNOT(wires=[0, 1])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qml.expval(qml.Hermitian(A, wires=2) + qml.PauliX(0) + qml.Hermitian(A, wires=1))

        with pytest.raises(
            RuntimeError,
            match="Hermitian observables with shot measurement are not supported",
        ):
            circuit(np.pi / 4, np.pi / 4)

    def test_paulix_pauliy(self, backend, tol_stochastic):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        n_wires = 3
        n_shots = 100000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliX(wires=0) @ qml.PauliY(wires=2))

        expected = circuit()
        result = qjit(circuit)()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_pauliz_pauliy_prod(self, backend, tol_stochastic):
        """Test that a tensor product involving PauliZ and PauliY works correctly"""
        n_wires = 3
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliX(2) @ qml.PauliY(1) @ qml.PauliZ(0))

        expected = circuit(0.432, 0.123, -0.543)
        result = qjit(circuit)(0.432, 0.123, -0.543)
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_pauliz_hamiltonian(self, backend, tol_stochastic):
        """Test that a hamiltonian involving PauliZ and PauliY and hadamard works correctly"""
        n_wires = 3
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(
                0.2 * qml.PauliZ(wires=0) + 0.5 * qml.Hadamard(wires=1) + qml.PauliY(wires=2)
            )

        expected = circuit(0.432, 0.123, -0.543)
        result = qjit(circuit)(0.432, 0.123, -0.543)
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_prod_hamiltonian(self, backend, tol_stochastic):
        """Test that a hamiltonian involving PauliZ and Hadamard @ PauliX works correctly"""
        n_wires = 3
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(
                0.2 * qml.PauliZ(wires=0) + 0.5 * qml.Hadamard(wires=1) @ qml.PauliX(2)
            )

        expected = circuit(0.432, 0.123, -0.543)
        result = qjit(circuit)(0.432, 0.123, -0.543)
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)


class TestVar:
    "Test var with shots > 0"

    def test_identity(self, backend, tol_stochastic):
        """Test that identity variance value (i.e. the trace) is 1."""
        n_wires = 2
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.Identity(wires=0)), qml.var(qml.Identity(wires=1))

        expected = circuit()
        result = qjit(circuit)()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_pauliz(self, backend, tol_stochastic):
        """Test that PauliZ variance value is correct"""
        n_wires = 2
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(wires=0)), qml.var(qml.PauliZ(wires=1))

        expected = circuit()
        result = qjit(circuit)()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_paulix(self, backend, tol_stochastic):
        """Test that PauliX variance value is correct"""
        n_wires = 2
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliX(wires=0)), qml.var(qml.PauliX(wires=1))

        expected = circuit()
        result = qjit(circuit)()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_pauliy(self, backend, tol_stochastic):
        """Test that PauliY variance value is correct"""
        n_wires = 2
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliY(wires=0)), qml.var(qml.PauliY(wires=1))

        expected = circuit()
        result = qjit(circuit)()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_hadamard(self, backend, tol_stochastic):
        """Test that Hadamard variance value is correct"""
        n_wires = 2
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.Hadamard(wires=0)), qml.var(qml.Hadamard(wires=1))

        expected = circuit()
        result = qjit(circuit)()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_hermitian_shots(self, backend):
        """Test var Hermitian observables with shots."""

        @qjit
        @qml.qnode(qml.device(backend, wires=3, shots=10000))
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.CNOT(wires=[0, 1])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qml.var(qml.Hermitian(A, wires=2) + qml.PauliX(0) + qml.Hermitian(A, wires=1))

        with pytest.raises(
            RuntimeError,
            match="Hermitian observables with shot measurement are not supported",
        ):
            circuit(np.pi / 4, np.pi / 4)

    def test_paulix_pauliy(self, backend, tol_stochastic):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        n_wires = 3
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.var(qml.PauliX(wires=0) @ qml.PauliY(wires=2))

        expected = circuit()
        result = qjit(circuit)()
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_hadamard_pauliy_prod(self, backend, tol_stochastic):
        """Test that a tensor product involving Hadamard and PauliY works correctly"""
        n_wires = 3
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.var(qml.Hadamard(wires=1) @ qml.PauliY(wires=2))

        expected = circuit(0.432, 0.123, -0.543)
        result = qjit(circuit)(0.432, 0.123, -0.543)
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_pauliz_pauliy_prod(self, backend, tol_stochastic):
        """Test that a tensor product involving PauliZ and PauliY works correctly"""
        n_wires = 3
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.var(qml.PauliX(2) @ qml.PauliY(1) @ qml.PauliZ(0))

        expected = circuit(0.432, 0.123, -0.543)
        result = qjit(circuit)(0.432, 0.123, -0.543)
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_pauliz_hamiltonian(self, backend):
        """Test that a hamiltonian involving PauliZ and PauliY and hadamard works correctly"""
        n_wires = 3
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.var(0.2 * qml.PauliZ(wires=0) + 0.5 * qml.Hadamard(wires=1))

        if isinstance(dev, qml.devices.LegacyDeviceFacade):
            with pytest.raises(
                RuntimeError,
                match=r"Cannot split up terms in sums for MeasurementProcess <class 'pennylane.measurements.var.VarianceMP'>",
            ):
                circuit(0.432, 0.123, -0.543)
        else:
            # TODO: only raises with the new API, Kokkos should also raise an error.
            with pytest.raises(
                TypeError,
                match=r"VarianceMP(Sum) cannot be computed with samples",
            ):
                circuit(0.432, 0.123, -0.543)


class TestProbs:
    "Test var with shots > 0"

    def test_probs(self, backend, tol_stochastic):
        """Test probs on all wires"""

        n_wires = 2
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        @qml.qnode(dev)
        def circuit(theta):
            qml.RX(theta, wires=[0])
            qml.Hadamard(wires=[1])
            return qml.probs()

        expected = circuit(0.432)
        result = qjit(circuit)(0.432)
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)

    def test_probs_wire(self, backend, tol_stochastic):
        """Test probs on subset of wires"""

        n_wires = 2
        n_shots = 10000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        @qml.qnode(dev)
        def circuit(theta):
            qml.RX(theta, wires=[0])
            qml.Hadamard(wires=[1])
            return qml.probs(wires=[0])

        expected = circuit(0.432)
        result = qjit(circuit)(0.432)
        assert np.allclose(result, expected, atol=tol_stochastic, rtol=tol_stochastic)


class TestShadow:
    """Test shadow."""

    @pytest.mark.xfail(reason="Not supported on lightning.")
    def test_shadow(self):
        """Test that Shadow can be used with Catalyst."""

        dev = qml.device("lightning.qubit", wires=range(2), shots=10000)

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

    @pytest.mark.xfail(reason="TypeError in Catalyst")
    def test_shadow_expval(self):
        """Test that ShadowExpVal can be used with Catalyst."""

        dev = qml.device("lightning.qubit", wires=range(2), shots=10000)

        @qjit
        @qml.qnode(dev)
        def shadow_expval_circuit(x, obs):
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            qml.RX(x, wires=0)
            return qml.shadow_expval(obs)

        H = qml.Hamiltonian([1.0, 1.0], [qml.Z(0) @ qml.Z(1), qml.X(0) @ qml.X(1)])
        expected = 1.9917
        assert shadow_expval_circuit(0, H) == expected


class TestOtherMeasurements:
    """Test other measurement processes."""

    @pytest.mark.parametrize("meas_fun", (qml.sample, qml.counts))
    def test_missing_shots_value(self, backend, meas_fun):
        """Test error for missing shots value."""

        dev = qml.device(backend, wires=1)

        @qml.qnode(dev)
        def circuit():
            return meas_fun(wires=0)

        with pytest.raises(CompileError, match="cannot work with shots=None"):
            qjit(circuit)

    def test_multiple_return_values(self, backend, tol_stochastic):
        """Test multiple return values."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2, shots=10000))
        def all_measurements(x):
            qml.RY(x, wires=0)
            return (
                qml.sample(),
                qml.counts(),
                qml.expval(qml.PauliZ(0)),
                qml.var(qml.PauliZ(0)),
                qml.probs(wires=[0, 1]),
            )

        @qml.qnode(qml.device("lightning.qubit", wires=2, shots=10000))
        def expected(x, measurement):
            qml.RY(x, wires=0)
            return qml.apply(measurement)

        x = 0.7
        result = all_measurements(x)

        # qml.sample
        assert result[0].shape == expected(x, qml.sample(wires=[0, 1]), shots=10000).shape
        assert result[0].dtype == np.int64

        # qml.counts
        for r, e in zip(
            result[1][0], expected(x, qml.counts(all_outcomes=True), shots=10000).keys()
        ):
            assert format(int(r), "02b") == e
        assert sum(result[1][1]) == 10000
        assert result[1][0].dtype == np.int64

        # qml.expval
        assert np.allclose(
            result[2],
            expected(x, qml.expval(qml.PauliZ(0))),
            atol=tol_stochastic,
            rtol=tol_stochastic,
        )

        # qml.var
        assert np.allclose(
            result[3], expected(x, qml.var(qml.PauliZ(0))), atol=tol_stochastic, rtol=tol_stochastic
        )

        # qml.probs
        assert np.allclose(
            result[4],
            expected(x, qml.probs(wires=[0, 1])),
            atol=tol_stochastic,
            rtol=tol_stochastic,
        )


if __name__ == "__main__":
    pytest.main(["-x", __file__])
