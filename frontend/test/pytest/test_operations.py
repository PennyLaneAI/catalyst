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

import pytest

from catalyst import qjit
import pennylane as qml
import numpy as np


def test_no_parameters():
    def circuit():
        qml.Identity(wires=0)

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

        qml.CSWAP(wires=[0, 1, 2])

        U1 = 1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
        qml.QubitUnitary(U1, wires=0)

        # To check if the generated qubit out of `wires=0` can be reused by another gate
        # and `quantum-opt` doesn't fail to materialize conversion for the result
        qml.Hadamard(wires=0)

        U2 = np.array(
            [
                [0.99500417 - 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 - 0.09983342j],
            ]
        )
        qml.QubitUnitary(U2, wires=[1, 2])

        # To check if the generated qubits out of `wires=[1, 2]` can be reused by other gates
        # and `quantum-opt` doesn't fail to materialize conversion for the result
        qml.CZ(wires=[1, 2])

        # Unsupported:
        # qml.SX(wires=0)
        # qml.ISWAP(wires=[0,1])
        # qml.ECR(wires=[0,1])
        # qml.SISWAP(wires=[0,1])
        # qml.Toffoli(wires=[0,1,2])
        # qml.MultiControlledX(wires=[0,1,2,3])

        return qml.state()

    qjit_fn = qjit()(qml.qnode(qml.device("lightning.qubit", wires=3))(circuit))
    qml_fn = qml.qnode(qml.device("default.qubit", wires=3))(circuit)

    assert np.allclose(qjit_fn(), qml_fn())


def test_param():
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

        qml.IsingXX(x, wires=[0, 1])
        qml.IsingXX(y, wires=[1, 2])

        qml.IsingYY(x, wires=[0, 1])
        qml.IsingYY(y, wires=[1, 2])

        qml.IsingXY(x, wires=[0, 1])
        qml.IsingXY(y, wires=[1, 2])

        qml.IsingZZ(x, wires=[0, 1])
        qml.IsingZZ(y, wires=[1, 2])

        qml.CRX(x, wires=[0, 1])
        qml.CRY(x, wires=[0, 1])
        qml.CRZ(x, wires=[0, 1])

        qml.CRX(y, wires=[1, 2])
        qml.CRY(y, wires=[1, 2])
        qml.CRZ(y, wires=[1, 2])

        qml.MultiRZ(x, wires=[0, 1, 2, 3])

        # Unsupported:
        # qml.PauliRot(x, 'IXYZ', wires=[0,1,2,3])
        # qml.U1(x, wires=0)
        # qml.U2(x, x, wires=0)
        # qml.U3(x, x, x, wires=0)
        # qml.PSWAP(x, wires=[0,1])

        return qml.state()

    qjit_fn = qjit()(qml.qnode(qml.device("lightning.qubit", wires=4))(circuit))
    qml_fn = qml.qnode(qml.device("default.qubit", wires=4))(circuit)

    assert np.allclose(qjit_fn(3.14, 0.6), qml_fn(3.14, 0.6))


if __name__ == "__main__":
    pytest.main(["-x", __file__])
