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
"""Test for qml.set_shots functionality."""
from functools import partial

import pennylane as qml
import pytest
from pennylane.devices import NullQubit

from catalyst import qjit


def test_simple_circuit_set_shots():
    """Test that a circuit with set_shots is compiling to MLIR."""
    dev = NullQubit(wires=2)

    @qjit(target="mlir")
    @partial(qml.set_shots, shots=2048)
    @qml.qnode(device=dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(wires=0))

    # Check that the MLIR contains the shots constant and device initialization
    mlir_str = str(circuit.mlir)
    assert "2048" in mlir_str


def test_state_with_set_shots_none():
    """Test that qml.set_shots(None) overrides device shots for state measurements."""

    @qml.qjit
    @partial(qml.set_shots, shots=None)
    @qml.qnode(qml.device("lightning.qubit", wires=4, shots=50))
    def f():
        return qml.state()

    result = f()
    assert result.shape == (16,)


def test_sample_with_set_shots_10():
    """Test that qml.set_shots(10) overrides device shots for sample measurements."""

    @qml.qjit
    @partial(qml.set_shots, shots=10)
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f():
        return qml.sample(wires=0)

    result = f()
    assert result.shape == (10, 1)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
