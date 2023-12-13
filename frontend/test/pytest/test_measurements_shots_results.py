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

from catalyst import qjit


class TestExpval:
    "Test expval with shots > 0"

    def test_identity(self, backend):
        """Test that identity expectation value (i.e. the trace) is 1."""
        n_wires = 2
        n_shots = 5000
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
        assert np.allclose(result, expected, atol=0.05)

    def test_pauliz(self, backend):
        """Test that PauliZ expectation value is correct"""
        n_wires = 2
        n_shots = 5000
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
        assert np.allclose(result, expected, atol=0.05)

    def test_paulix(self, backend):
        """Test that PauliX expectation value is correct"""
        n_wires = 2
        n_shots = 5000
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
        assert np.allclose(result, expected, atol=0.05)

    def test_pauliy(self, backend):
        """Test that PauliY expectation value is correct"""
        n_wires = 2
        n_shots = 5000
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
        assert np.allclose(result, expected, atol=0.05)

    def test_hadamard(self, backend):
        """Test that Hadamard expectation value is correct"""
        n_wires = 2
        n_shots = 5000
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
        assert np.allclose(result, expected, atol=0.05)

    def test_hermitian(self, backend):
        """Test expval Hermitian observables with shots."""

        @qjit
        @qml.qnode(qml.device(backend, wires=3, shots=5000))
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
            match="Hermitian observables do not support shot measurement",
        ):
            circuit(np.pi / 4, np.pi / 4)

    def test_paulix_pauliy(self, backend):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        n_wires = 3
        n_shots = 50000
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
        assert np.allclose(result, expected, atol=0.05)

    # TODO: This test should be uncommented after fixing the issue in Lightning
    # def test_pauliz_pauliy_prod(self, backend):
    #     """Test that a tensor product involving PauliZ and PauliY works correctly"""
    #     n_wires = 3
    #     n_shots = 5000
    #     dev = qml.device(backend, wires=n_wires, shots=n_shots)

    #     @qml.qnode(dev)
    #     def circuit(theta, phi, varphi):
    #         qml.RX(theta, wires=[0])
    #         qml.RX(phi, wires=[1])
    #         qml.RX(varphi, wires=[2])
    #         qml.CNOT(wires=[0, 1])
    #         qml.CNOT(wires=[1, 2])
    #         return qml.expval(qml.PauliZ(wires=1) @ qml.PauliY(wires=2))

    #     expected = circuit(0.432, 0.123, -0.543)
    #     result = qjit(circuit)(0.432, 0.123, -0.543)
    #     assert np.allclose(result, expected, atol=0.05)

    def test_pauliz_hamiltonian(self, backend):
        """Test that a hamiltonian involving PauliZ and PauliY and hadamard works correctly"""
        n_wires = 3
        n_shots = 5000
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
        assert np.allclose(result, expected, atol=0.05)


class TestVar:
    "Test var with shots > 0"

    def test_identity(self, backend):
        """Test that identity variance value (i.e. the trace) is 1."""
        n_wires = 2
        n_shots = 5000
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
        assert np.allclose(result, expected, atol=0.05)

    def test_pauliz(self, backend):
        """Test that PauliZ variance value is correct"""
        n_wires = 2
        n_shots = 5000
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
        assert np.allclose(result, expected, atol=0.05)

    def test_paulix(self, backend):
        """Test that PauliX variance value is correct"""
        n_wires = 2
        n_shots = 5000
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
        assert np.allclose(result, expected, atol=0.05)

    def test_pauliy(self, backend):
        """Test that PauliY variance value is correct"""
        n_wires = 2
        n_shots = 5000
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
        assert np.allclose(result, expected, atol=0.05)

    def test_hadamard(self, backend):
        """Test that Hadamard variance value is correct"""
        n_wires = 2
        n_shots = 5000
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
        assert np.allclose(result, expected, atol=0.05)

    def test_hermitian_shots(self, backend):
        """Test var Hermitian observables with shots."""

        @qjit
        @qml.qnode(qml.device(backend, wires=3, shots=5000))
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
            match="Hermitian observables do not support shot measurement",
        ):
            circuit(np.pi / 4, np.pi / 4)

    def test_paulix_pauliy(self, backend):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        n_wires = 3
        n_shots = 5000
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
        assert np.allclose(result, expected, atol=0.05)

    def test_paulix_pauliy_prod(self, backend):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        n_wires = 3
        n_shots = 5000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.var(qml.PauliX(wires=1) @ qml.PauliY(wires=2))

        expected = circuit(0.432, 0.123, -0.543)
        result = qjit(circuit)(0.432, 0.123, -0.543)
        assert np.allclose(result, expected, atol=0.05)

    def test_pauliz_hamiltonian(self, backend):
        """Test that a hamiltonian involving PauliZ and PauliY and hadamard works correctly"""
        n_wires = 3
        n_shots = 5000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.var(0.2 * qml.PauliZ(wires=0) + 0.5 * qml.Hadamard(wires=1))

        with pytest.raises(
            ValueError,
            match="Can only return the expectation of a single Hamiltonian observable",
        ):
            circuit(0.432, 0.123, -0.543)


class TestProbs:
    "Test var with shots > 0"

    def test_probs(self, backend):
        """Test probs on all wires"""

        n_wires = 2
        n_shots = 5000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        @qml.qnode(dev)
        def circuit(theta):
            qml.RX(theta, wires=[0])
            qml.Hadamard(wires=[1])
            return qml.probs()

        expected = circuit(0.432)
        result = qjit(circuit)(0.432)
        assert np.allclose(result, expected, atol=0.05)

    def test_probs_wire(self, backend):
        """Test probs on subset of wires"""

        n_wires = 2
        n_shots = 5000
        dev = qml.device(backend, wires=n_wires, shots=n_shots)

        @qml.qnode(dev)
        def circuit(theta):
            qml.RX(theta, wires=[0])
            qml.Hadamard(wires=[1])
            return qml.probs(wires=[0])

        expected = circuit(0.432)
        result = qjit(circuit)(0.432)
        assert np.allclose(result, expected, atol=0.05)
