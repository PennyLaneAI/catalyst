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


@pytest.mark.parametrize(
    "device",
    [
        qml.device(
            "braket.aws.qubit",
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            wires=3,
        ),
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
def test_no_parameters_braket(device):
    """Test no-param operations on braket devices."""

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

    qjit_fn = qjit()(qml.qnode(device)(circuit))
    qml_fn = qml.qnode(qml.device("default.qubit", wires=3))(circuit)

    assert np.allclose(qjit_fn(), qml_fn())


# @pytest.mark.parametrize(
#     "device",
#     [
#         qml.device(
#             "braket.local.qubit",
#             backend="braket_sv",
#             wires=2,
#         ),
#     ],
# )
# def test_unsupported_gate_braket(device):
#     """Test unsupported gates on braket devices."""

#     def circuit():
#         U1 = 1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
#         qml.QubitUnitary(U1, wires=0)
#         return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

#     qjit_fn = qjit()(qml.qnode(device)(circuit))

#     with pytest.raises(RuntimeError):
#         # QubitUnitary is not supported!
#         qjit_fn()


# @pytest.mark.parametrize(
#     "device",
#     [
#         qml.device(
#             "braket.aws.qubit",
#             device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
#             wires=3,
#         ),
#         qml.device(
#             "braket.local.qubit",
#             backend="braket_sv",
#             wires=3,
#         ),
#     ],
# )
# def test_param_braket(device):
#     """Test param operations on braket devices."""

#     def circuit(x: float, y: float):
#         qml.Rot(x, y, x + y, wires=0)

#         qml.RX(x, wires=0)
#         qml.RY(y, wires=1)
#         qml.RZ(x, wires=2)

#         qml.RZ(y, wires=0)
#         qml.RY(x, wires=1)
#         qml.RX(y, wires=2)

#         qml.PhaseShift(x, wires=0)
#         qml.PhaseShift(y, wires=1)

#         return qml.var(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

#     qjit_fn = qjit()(qml.qnode(device)(circuit))
#     qml_fn = qml.qnode(qml.device("default.qubit", wires=3))(circuit)

#     assert np.allclose(qjit_fn(3.14, 0.6), qml_fn(3.14, 0.6))


# @pytest.mark.parametrize(
#     "device",
#     [
#         qml.device(
#             "braket.aws.qubit",
#             device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
#             wires=1,
#             shots=1000,
#         ),
#         qml.device(
#             "braket.local.qubit",
#             backend="braket_sv",
#             wires=1,
#             shots=1000,
#         ),
#     ],
# )
# def test_sample_on_1qbit_braket(device):
#     """Test sample on 1 qubit on braket devices."""

#     @qjit()
#     @qml.qnode(device)
#     def sample_1qbit(x: float):
#         qml.RX(x, wires=0)
#         return qml.sample()

#     expected = np.array([[0.0]] * 1000)
#     observed = sample_1qbit(0.0)
#     assert np.array_equal(observed, expected)

#     expected = np.array([[1.0]] * 1000)
#     observed = sample_1qbit(np.pi)
#     assert np.array_equal(observed, expected)


# @pytest.mark.parametrize(
#     "device",
#     [
#         qml.device(
#             "braket.local.qubit",
#             backend="braket_sv",
#             wires=2,
#             shots=1000,
#         ),
#     ],
# )
# def test_sample_on_2qbits_braket(device):
#     """Test sample on 2 qubits on braket devices."""

#     @qjit()
#     @qml.qnode(device)
#     def sample_2qbits(x: float):
#         qml.RX(x, wires=0)
#         qml.RY(x, wires=1)
#         return qml.sample()

#     expected = np.array([[0.0, 0.0]] * 1000)
#     observed = sample_2qbits(0.0)
#     assert np.array_equal(observed, expected)
#     expected = np.array([[1.0, 1.0]] * 1000)
#     observed = sample_2qbits(np.pi)
#     assert np.array_equal(observed, expected)


# @pytest.mark.parametrize(
#     "device",
#     [
#         qml.device(
#             "braket.aws.qubit",
#             device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
#             wires=1,
#             shots=1000,
#         ),
#         qml.device(
#             "braket.local.qubit",
#             backend="braket_sv",
#             wires=1,
#             shots=1000,
#         ),
#     ],
# )
# def test_probs_on_1qbit_braket(device):
#     """Test probs on 1 qubit on braket devices."""

#     @qjit()
#     @qml.qnode(device)
#     def probs_1qbit(x: float):
#         qml.RX(x, wires=0)
#         return qml.probs()

#     observed = probs_1qbit(np.pi / 2)
#     assert np.allclose(np.sum(observed), 1.0)


# @pytest.mark.parametrize(
#     "device",
#     [
#         qml.device(
#             "braket.local.qubit",
#             backend="braket_sv",
#             wires=2,
#             shots=1000,
#         ),
#     ],
# )
# def test_probs_on_2qbits_braket(device):
#     """Test probs on 2 qubits on braket devices."""

#     @qjit()
#     @qml.qnode(device)
#     def probs_2qbits(x: float):
#         qml.RX(x, wires=0)
#         qml.RY(x, wires=1)
#         return qml.probs()

#     observed = probs_2qbits(np.pi / 3)
#     assert np.allclose(np.sum(observed), 1.0)


# @pytest.mark.parametrize(
#     "device",
#     [
#         qml.device(
#             "braket.aws.qubit",
#             device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
#             wires=1,
#             shots=1000,
#         ),
#         qml.device(
#             "braket.local.qubit",
#             backend="braket_sv",
#             wires=1,
#             shots=1000,
#         ),
#     ],
# )
# def test_count_on_1qbit_braket(device):
#     """Test counts on 1 qubits on braket devices."""

#     @qjit()
#     @qml.qnode(device)
#     def counts_1qbit(x: float):
#         qml.RX(x, wires=0)
#         return qml.counts()

#     expected = [np.array([0, 1]), np.array([1000, 0])]
#     observed = counts_1qbit(0.0)
#     assert np.array_equal(observed, expected)

#     expected = [np.array([0, 1]), np.array([0, 1000])]
#     observed = counts_1qbit(np.pi)
#     assert np.array_equal(observed, expected)


# @pytest.mark.parametrize(
#     "device",
#     [
#         qml.device(
#             "braket.local.qubit",
#             backend="braket_sv",
#             wires=2,
#             shots=1000,
#         ),
#     ],
# )
# def test_count_on_2qbits_braket(device):
#     """Test counts on 2 qubits on braket devices."""

#     @qjit()
#     @qml.qnode(device)
#     def counts_2qbit(x: float):
#         qml.RX(x, wires=0)
#         qml.RY(x, wires=1)
#         return qml.counts()

#     expected = [np.array([0, 1, 2, 3]), np.array([1000, 0, 0, 0])]
#     observed = counts_2qbit(0.0)
#     assert np.array_equal(observed, expected)

#     expected = [np.array([0, 1, 2, 3]), np.array([0, 0, 0, 1000])]
#     observed = counts_2qbit(np.pi)
#     assert np.array_equal(observed, expected)


# class TestBraketExpval:
#     @pytest.mark.parametrize(
#         "device",
#         [
#             qml.device(
#                 "braket.local.qubit",
#                 backend="braket_sv",
#                 wires=1,
#             ),
#         ],
#     )
#     def test_named(self, device):
#         """Test expval for named observables on braket devices."""

#         @qjit()
#         @qml.qnode(device)
#         def expval(x: float):
#             qml.RX(x, wires=0)
#             return qml.expval(qml.PauliZ(0))

#         expected = np.array(1.0)
#         observed = expval(0.0)
#         assert np.isclose(observed, expected)

#         expected = np.array(-1.0)
#         observed = expval(np.pi)
#         assert np.isclose(observed, expected)

#     @pytest.mark.parametrize(
#         "device",
#         [
#             qml.device(
#                 "braket.local.qubit",
#                 backend="braket_sv",
#                 wires=2,
#             ),
#         ],
#     )
#     def test_tensor(self, device):
#         """Test expval for Tensor observable on braket devices."""

#         @qjit()
#         @qml.qnode(device)
#         def expval(x: float, y: float):
#             qml.RX(x, wires=0)
#             qml.RX(y, wires=1)
#             qml.CNOT(wires=[0, 1])
#             return qml.expval(qml.PauliX(0) @ qml.PauliY(1))

#         expected = np.array(-0.35355339)
#         observed = expval(np.pi / 4, np.pi / 3)
#         assert np.isclose(observed, expected)

#         expected = np.array(-0.49999788)
#         observed = expval(np.pi / 2, np.pi / 3)
#         assert np.isclose(observed, expected)


# class TestBraketVar:
#     @pytest.mark.parametrize(
#         "device",
#         [
#             qml.device(
#                 "braket.aws.qubit",
#                 device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
#                 wires=1,
#             ),
#             qml.device(
#                 "braket.local.qubit",
#                 backend="braket_sv",
#                 wires=1,
#             ),
#         ],
#     )
#     def test_rx(self, device):
#         """Test var with RX on braket devices."""

#         @qjit()
#         @qml.qnode(device)
#         def var(x: float):
#             qml.RX(x, wires=0)
#             return qml.var(qml.PauliZ(0))

#         expected = np.array(0.0)
#         observed = var(0.0)
#         assert np.isclose(observed, expected)
#         observed = var(np.pi)
#         assert np.isclose(observed, expected)

#     @pytest.mark.parametrize(
#         "device",
#         [
#             qml.device(
#                 "braket.local.qubit",
#                 backend="braket_sv",
#                 wires=1,
#             ),
#         ],
#     )
#     def test_hadamard(self, device):
#         """Test var with Hadamard on braket devices."""

#         @qjit()
#         @qml.qnode(device)
#         def var(x: float):
#             qml.Hadamard(wires=0)
#             return qml.var(qml.PauliZ(0))

#         expected = np.array(1.0)
#         observed = var(0.0)
#         assert np.isclose(observed, expected)
#         observed = var(np.pi)
#         assert np.isclose(observed, expected)

#     @pytest.mark.parametrize(
#         "device",
#         [
#             qml.device(
#                 "braket.local.qubit",
#                 backend="braket_sv",
#                 wires=2,
#             ),
#         ],
#     )
#     def test_tensor(self, device):
#         """Test variance for Tensor observable on braket devices."""

#         @qjit()
#         @qml.qnode(device)
#         def circuit(x: float, y: float):
#             qml.RX(x, wires=0)
#             qml.RX(y, wires=1)
#             qml.CNOT(wires=[0, 1])
#             return qml.var(qml.PauliX(0) @ qml.PauliY(1))

#         expected = np.array(0.875)
#         observed = circuit(np.pi / 4, np.pi / 3)
#         assert np.isclose(observed, expected)

#         expected = np.array(1.0)
#         observed = circuit(np.pi / 2, np.pi / 2)
#         assert np.isclose(observed, expected)


# @pytest.mark.parametrize(
#     "device",
#     [
#         qml.device(
#             "braket.local.qubit",
#             wires=2,
#         ),
#     ],
# )
# def test_multiple_return_values_braket1(device):
#     """Test multiple return values."""

#     @qjit()
#     @qml.qnode(device)
#     def all_measurements(x):
#         qml.RY(x, wires=0)
#         return (
#             qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
#             qml.var(qml.PauliZ(1) @ qml.PauliZ(0)),
#         )

#     @qml.qnode(qml.device("default.qubit", wires=2))
#     def expected(x, measurement):
#         qml.RY(x, wires=0)
#         return qml.apply(measurement)

#     x = 0.7
#     result = all_measurements(x)

#     # qml.expval
#     assert np.allclose(result[0], expected(x, qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))))

#     # qml.var
#     assert np.allclose(result[1], expected(x, qml.var(qml.PauliZ(1) @ qml.PauliZ(0))))


# @pytest.mark.parametrize(
#     "device",
#     [
#         qml.device(
#             "braket.local.qubit",
#             wires=1,
#             shots=100,
#         ),
#     ],
# )
# def test_multiple_return_values_braket2(device):
#     """Test multiple return values with shots > 0."""

#     @qjit()
#     @qml.qnode(device)
#     def all_measurements(x):
#         qml.RY(x, wires=0)
#         return (
#             qml.sample(),
#             qml.counts(),
#             qml.probs(wires=[0]),
#             qml.expval(qml.PauliZ(0)),
#             qml.var(qml.PauliZ(0)),
#         )

#     @qml.qnode(qml.device("default.qubit", wires=2))
#     def expected(x, measurement):
#         qml.RY(x, wires=0)
#         return qml.apply(measurement)

#     x = 0.7
#     result = all_measurements(x)

#     # qml.sample
#     assert result[0].shape[0] == 100

#     # qml.counts
#     assert sum(result[2]) == 100

#     # qml.probs
#     assert result[3][0] > result[3][1]


# @pytest.mark.parametrize(
#     "device",
#     [
#         qml.device(
#             "braket.local.qubit",
#             backend="braket_sv",
#             wires=2,
#         ),
#     ],
# )
# def test_unsupported_measurement_braket(device):
#     """Test unsupported measurement on braket devices."""

#     @qjit()
#     @qml.qnode(device)
#     def circuit(x: float, y: float):
#         qml.RX(x, wires=0)
#         qml.RX(y, wires=1)
#         qml.CNOT(wires=[0, 1])
#         return qml.state()

#     with pytest.raises(RuntimeError):
#         # state() is not supported yet!
#         circuit(np.pi / 4, np.pi / 3)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
