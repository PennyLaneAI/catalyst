# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Integration tests for qml.StatePrep.

This test suite was introduced following https://github.com/PennyLaneAI/catalyst/issues/1488.
"""

import numpy as np
import pennylane as qml
import pytest

from catalyst import qjit


def generate_random_state(n=1, seed=None):
    """Generate a random n-qubit state in state-vector representation.

    Args:
        n (int, optional): Number of qubits in the state. Defaults to 1.
        seed (int, optional): Seed to the random-number generator. Defaults to None, in which case
            a random seed is used.

    Returns:
        numpy.ndarray: The generated state vector.
    """
    rng = np.random.default_rng(seed=seed)
    input_state = rng.random(2**n) + 1j * rng.random(2**n)
    return input_state / np.linalg.norm(input_state)


class Test2QubitStatePrep:
    """
    Tests two variations of a simple two-qubit circuit beginning with state preparation on one of
    the wires followed by a Hadamard gate on the other wire, and finally a CZ gate. The circuit
    returns the expectation values for each of the Pauli X, Y, Z operations for each wire.

    The results are compared against the analytic solution, which can easily be determined given an
    arbitrary input state

        |psi> = a|0> + b|1>,

    where a, b are complex numbers with |a|^2 + |b|^2 = 1.
    """

    input_states = [
        np.array([1, 0]),  # |0>
        np.array([0, 1]),  # |1>
        np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]),  # |+>
        np.array([1 / np.sqrt(2), -1 / np.sqrt(2)]),  # |->
        np.array([complex(1 / np.sqrt(2), 0.0), complex(0.0, 1 / np.sqrt(2))]),  # |R>
        np.array([complex(1 / np.sqrt(2), 0.0), complex(0.0, -1 / np.sqrt(2))]),  # |L>
        generate_random_state(seed=42),
    ]

    dev = qml.device("lightning.qubit", wires=2)

    @qjit
    @qml.qnode(dev)
    @staticmethod
    def circuit_01(input_state):
        """A circuit that applies StatePrep on wire 0 and the Hadamard on wire 1.

        The analytic solution for this circuit, given input state |psi>, is:

            |psi'> = (a|00> + a|01> + b|10> - b|11>) / sqrt(2)
        """
        qml.StatePrep(input_state, wires=[0])
        qml.Hadamard(1)
        qml.CZ([1, 0])

        return qml.state()

    @qjit
    @qml.qnode(dev)
    @staticmethod
    def circuit_10(input_state):
        """A circuit that applies StatePrep on wire 1 and the Hadamard on wire 0.

        The analytic solution for this circuit, given input state |psi>, is:

            |psi'> = (a|00> + b|01> + a|10> - b|11>) / sqrt(2)
        """
        qml.StatePrep(input_state, wires=[1])
        qml.Hadamard(0)
        qml.CZ([1, 0])

        return qml.state()

    @pytest.mark.parametrize("input_state", input_states)
    def test_2qubit_state_prep_H_CZ_01(self, input_state):
        """Test circuit_01"""
        result_obs = self.circuit_01(input_state)

        a, b = input_state
        result_exp = (1 / np.sqrt(2)) * np.array([a, a, b, -b])

        assert np.allclose(result_obs, result_exp)

    @pytest.mark.parametrize("input_state", input_states)
    def test_2qubit_state_prep_H_CZ_10(self, input_state):
        """Test circuit_10"""
        result_obs = self.circuit_10(input_state)

        a, b = input_state
        result_exp = (1 / np.sqrt(2)) * np.array([a, b, a, -b])

        assert np.allclose(result_obs, result_exp)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
