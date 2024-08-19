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

"""Unit tests for `OpenQasmDevice` on "local" Amazon Braket devices
"""
import numpy as np
import pennylane as qml
import pytest
from numpy.testing import assert_allclose

from catalyst import grad, qjit

try:
    qml.device("braket.local.qubit", backend="default", wires=1)
except qml._device.DeviceError:
    pytest.skip(
        "skipping Braket local tests because ``amazon-braket-pennylane-plugin`` is not installed",
        allow_module_level=True,
    )


@pytest.mark.braketlocal
class TestBraketGates:
    """Unit tests for quantum gates."""

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="default",
                wires=3,
            ),
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=3,
            ),
        ],
    )
    def test_no_parameters_braket(self, device):
        """Test no-param operations on braket devices."""

        def circuit():
            qml.Identity(wires=1)
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

            qml.CY(wires=[0, 1])
            qml.CY(wires=[0, 2])

            qml.CZ(wires=[0, 1])
            qml.CZ(wires=[0, 2])

            qml.SWAP(wires=[0, 1])
            qml.SWAP(wires=[0, 2])
            qml.SWAP(wires=[1, 2])

            U1 = 1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
            qml.QubitUnitary(U1, wires=0)

            U2 = np.array(
                [
                    [0.99500417 - 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 - 0.09983342j],
                ]
            )
            qml.QubitUnitary(U2, wires=[1, 2])

            qml.ISWAP(wires=[0, 1])
            qml.CSWAP(wires=[0, 1, 2])
            qml.SX(wires=0)
            qml.MultiControlledX(wires=[0, 1, 2])
            qml.Toffoli(wires=[0, 1, 2])

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        qjit_fn = qjit()(qml.qnode(device)(circuit))
        qml_fn = qml.qnode(qml.device("default.qubit", wires=3))(circuit)

        assert np.allclose(qjit_fn(), qml_fn())

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=3,
            ),
        ],
    )
    def test_param_braket(self, device):
        """Test param operations on braket devices."""

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

            qml.PSWAP(x, wires=[0, 2])

            return qml.var(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

        qjit_fn = qjit()(qml.qnode(device)(circuit))
        qml_fn = qml.qnode(qml.device("default.qubit", wires=3))(circuit)

        assert np.allclose(qjit_fn(3.14, 0.6), qml_fn(3.14, 0.6))


@pytest.mark.braketlocal
class TestBraketSample:
    """Unit tests for ``qml.sample``."""

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=1,
                shots=1000,
            ),
        ],
    )
    def test_sample_on_1qbit_braket(self, device):
        """Test sample on 1 qubit on braket devices."""

        @qjit
        @qml.qnode(device)
        def sample_1qbit(x: float):
            qml.RX(x, wires=0)
            return qml.sample()

        expected = np.array([[0]] * 1000)
        observed = sample_1qbit(0.0)
        assert np.array_equal(observed, expected)

        expected = np.array([[1]] * 1000)
        observed = sample_1qbit(np.pi)
        assert np.array_equal(observed, expected)

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=2,
                shots=1000,
            ),
        ],
    )
    def test_sample_on_2qbits_braket(self, device):
        """Test sample on 2 qubits on braket devices."""

        @qjit
        @qml.qnode(device)
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


@pytest.mark.braketlocal
class TestBraketProbs:
    """Unit tests for ``qml.probs``."""

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=1,
                shots=1000,
            ),
        ],
    )
    def test_probs_on_1qbit_braket(self, device):
        """Test probs on 1 qubit on braket devices."""

        @qjit
        @qml.qnode(device)
        def probs_1qbit(x: float):
            qml.RX(x, wires=0)
            return qml.probs()

        observed = probs_1qbit(np.pi / 2)
        assert np.allclose(np.sum(observed), 1.0)

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=2,
                shots=1000,
            ),
        ],
    )
    def test_probs_on_2qbits_braket(self, device):
        """Test probs on 2 qubits on braket devices."""

        @qjit
        @qml.qnode(device)
        def probs_2qbits(x: float):
            qml.RX(x, wires=0)
            qml.RY(x, wires=1)
            return qml.probs()

        observed = probs_2qbits(np.pi / 3)
        assert np.allclose(np.sum(observed), 1.0)


@pytest.mark.braketlocal
class TestBraketCounts:
    """Unit tests for ``qml.counts``."""

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=1,
                shots=1000,
            ),
        ],
    )
    def test_count_on_1qbit_braket(self, device):
        """Test counts on 1 qubits on braket devices."""

        @qjit
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

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=2,
                shots=1000,
            ),
        ],
    )
    def test_count_on_2qbits_braket(self, device):
        """Test counts on 2 qubits on braket devices."""

        @qjit
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


@pytest.mark.braketlocal
class TestBraketExpval:
    """Unit tests for ``qml.expval``."""

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=1,
            ),
        ],
    )
    def test_named(self, device):
        """Test expval for named observables on braket devices."""

        @qjit
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

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=1,
            ),
        ],
    )
    def test_hermitian_1(self, device):
        """Test expval for Hermitian observable on braket devices."""

        @qjit
        @qml.qnode(device)
        def expval(x: float):
            qml.RY(x, wires=0)
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qml.expval(qml.Hermitian(A, wires=0))

        expected = np.array(1.0)
        observed = expval(0.0)
        assert np.isclose(observed, expected)

        expected = np.array(3.0)
        observed = expval(np.pi / 2)
        assert np.isclose(observed, expected)

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=2,
            ),
        ],
    )
    def test_hermitian_2(self, device):
        """Test expval for Hermitian observable on braket devices."""

        @qjit
        @qml.qnode(device)
        def expval(x: float):
            qml.RX(x, wires=0)
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
        observed = expval(0.0)
        assert np.isclose(observed, expected)

        expected = np.array(1.0)
        observed = expval(np.pi)
        assert np.isclose(observed, expected)

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=2,
            ),
        ],
    )
    def test_tensor_1(self, device):
        """Test expval for Tensor observable on braket devices."""

        @qjit
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

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=3,
            ),
        ],
    )
    def test_tensor_2(self, device):
        """Test expval for Tensor observable including hermitian observable on braket devices."""

        @qjit
        @qml.qnode(device)
        def expval(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.CNOT(wires=[0, 2])
            qml.CNOT(wires=[1, 2])
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(-1.0, 0.0)]]
            )
            return qml.expval(qml.PauliX(0) @ qml.Hadamard(1) @ qml.Hermitian(A, wires=2))

        expected = np.array(-0.4330127)
        observed = expval(np.pi / 4, np.pi / 3)
        assert np.isclose(observed, expected)

        expected = np.array(-0.70710678)
        observed = expval(np.pi / 2, np.pi / 2)
        assert np.isclose(observed, expected)


@pytest.mark.braketlocal
class TestBraketVar:
    """Unit tests for ``qml.var``."""

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=1,
            ),
        ],
    )
    def test_rx(self, device):
        """Test var with RX on braket devices."""

        @qjit
        @qml.qnode(device)
        def var(x: float):
            qml.RX(x, wires=0)
            return qml.var(qml.PauliZ(0))

        expected = np.array(0.0)
        observed = var(0.0)
        assert np.isclose(observed, expected)
        observed = var(np.pi)
        assert np.isclose(observed, expected)

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=1,
            ),
        ],
    )
    def test_hadamard(self, device):
        """Test var with Hadamard on braket devices."""

        @qjit
        @qml.qnode(device)
        def var(x: float):
            qml.Hadamard(wires=0)
            qml.RX(x, wires=0)
            return qml.var(qml.PauliZ(0))

        expected = np.array(1.0)
        observed = var(0.0)
        assert np.isclose(observed, expected)
        observed = var(np.pi)
        assert np.isclose(observed, expected)

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=2,
            ),
        ],
    )
    def test_tensor_1(self, device):
        """Test variance for Tensor observable on braket devices."""

        @qjit
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

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=3,
            ),
        ],
    )
    def test_tensor_2(self, device):
        """Test variance for Tensor observable including hermitian observable on braket devices."""

        @qjit
        @qml.qnode(device)
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


@pytest.mark.braketlocal
class TestBraketMeasurementsProcess:
    """Unit tests for mixing measurement processes."""

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                wires=2,
            ),
        ],
    )
    def test_multiple_return_values_braket1(self, device):
        """Test multiple return values."""

        @qjit
        @qml.qnode(device)
        def all_measurements(x):
            qml.RY(x, wires=0)
            return (
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
                qml.var(qml.PauliZ(1) @ qml.PauliZ(0)),
            )

        @qml.qnode(qml.device("default.qubit", wires=2))
        def expected(x, measurement):
            qml.RY(x, wires=0)
            return qml.apply(measurement)

        x = 0.7
        result = all_measurements(x)

        # qml.expval
        assert np.allclose(result[0], expected(x, qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))))

        # qml.var
        assert np.allclose(result[1], expected(x, qml.var(qml.PauliZ(1) @ qml.PauliZ(0))))

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                wires=1,
                shots=100,
            ),
        ],
    )
    def test_multiple_return_values_braket2(self, device):
        """Test multiple return values with shots > 0."""

        @qjit
        @qml.qnode(device)
        def all_measurements(x):
            qml.RY(x, wires=0)
            return (
                qml.sample(),
                qml.counts(),
                qml.probs(wires=[0]),
                qml.expval(qml.PauliZ(0)),
                qml.var(qml.PauliZ(0)),
            )

        @qml.qnode(qml.device("default.qubit", wires=2))
        def expected(x, measurement):
            qml.RY(x, wires=0)
            return qml.apply(measurement)

        x = 0.7
        result = all_measurements(x)

        # qml.sample
        assert result[0].shape[0] == 100
        assert result[0].dtype == np.int64

        # qml.counts
        assert sum(result[1][1]) == 100
        assert result[1][0].dtype == np.int64

        # qml.probs
        assert result[2][0] > result[2][1]

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=2,
            ),
        ],
    )
    def test_unsupported_measurement_braket(self, device):
        """Test unsupported measurement on braket devices."""

        @qjit
        @qml.qnode(device)
        def circuit(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        with pytest.raises(RuntimeError, match="Not implemented method"):
            circuit(np.pi / 4, np.pi / 3)


@pytest.mark.braketlocal
class TestBraketGradient:
    """Unit tests for gradient methods."""

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=1,
            ),
        ],
    )
    @pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
    def test_fd(self, inp, device):
        """Test the finite-diff method on braket devices."""

        def f(x):
            qml.RX(x * 2, wires=0)
            return qml.expval(qml.PauliY(0))

        @qjit
        def compiled(x: float):
            g = qml.qnode(device)(f)
            h = grad(g, method="fd", h=1e-4)
            return h(x)

        def interpreted(x):
            device = qml.device("default.qubit", wires=1)
            g = qml.QNode(f, device, diff_method="finite-diff")
            h = qml.grad(g, argnums=0)
            return h(x)

        assert np.allclose(compiled(inp), interpreted(inp), rtol=1e-3)

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=2,
            ),
        ],
    )
    @pytest.mark.parametrize("inp", [(1.0, 2.0), (2.0, 3.0)])
    def test_fd_2qubits(self, inp, device):
        """Test the finite-diff method on braket devices."""

        def f(x, y):
            qml.RX(y * x, wires=0)
            qml.RX(x * 2, wires=1)
            return qml.expval(qml.PauliY(0) @ qml.PauliZ(1))

        @qjit
        def compiled(x: float, y: float):
            g = qml.qnode(device)(f)
            h = grad(g, method="fd", h=1e-4)
            return h(x, y)

        def interpreted(x, y):
            device = qml.device("default.qubit", wires=2)
            g = qml.QNode(f, device, diff_method="finite-diff")
            h = qml.grad(g, argnums=0)
            return h(x, y)

        assert np.allclose(compiled(*inp), interpreted(*inp), rtol=1e-3)

    @pytest.mark.parametrize(
        "device",
        [
            qml.device(
                "braket.local.qubit",
                backend="braket_sv",
                wires=2,
            ),
        ],
    )
    @pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
    def test_fd_higher_order(self, inp, device):
        """Test finite diff method on braket devices."""

        def f(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        @qjit
        def compiled_grad_default(x: float):
            g = qml.qnode(device)(f)
            h = grad(g, method="fd", h=1e-4)
            i = grad(h, method="fd", h=1e-4)
            return i(x)

        def interpretted_grad_default(x):
            device = qml.device("default.qubit", wires=1)
            g = qml.QNode(f, device, diff_method="backprop", max_diff=2)
            h = qml.grad(g, argnums=0)
            i = qml.grad(h, argnums=0)
            return i(x)

        assert_allclose(compiled_grad_default(inp), interpretted_grad_default(inp), rtol=0.1)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
