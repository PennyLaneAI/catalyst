# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test the quantum peephole passes' UI"""

import numpy as np
import pennylane as qml
import pytest

from catalyst import cancel_inverses, qjit
from catalyst.api_extensions.quantum_passes import QUANTUM_PASSES_TABLE

### Test passes are correctly added to and queried from the quantum pass table ###

### SETUP ###
dev = qml.device("lightning.qubit", wires=10)


@qml.qnode(dev)
def circuit():
    qml.Identity(wires=np.arange(10))
    return qml.probs()


@qml.qnode(dev)
def circuit2():
    qml.Identity(wires=np.arange(10))
    return qml.probs()


pass_table = QUANTUM_PASSES_TABLE()
# Used as a global, the same usage as in the original file api_extensions/quantum_passes.py


### TESTS ###
def test_add_pass_to_table():
    """
    Tests that passes can be added correctly to the table.
    """
    pass_table.add_pass_on_qnode(circuit, "some-pass")
    assert pass_table.table == {circuit: ["some-pass"]}

    pass_table.add_pass_on_qnode(circuit, "another-pass")
    assert pass_table.table == {circuit: ["some-pass", "another-pass"]}

    pass_table.add_pass_on_qnode(circuit, "another-pass")
    assert pass_table.table == {circuit: ["some-pass", "another-pass", "another-pass"]}

    pass_table.add_pass_on_qnode(circuit2, "some-pass")
    assert pass_table.table == {
        circuit: ["some-pass", "another-pass", "another-pass"],
        circuit2: ["some-pass"],
    }


def test_query_from_table():
    """
    Tests that passes can be queried correctly from the table.
    """
    assert pass_table.query(circuit) == ["some-pass", "another-pass", "another-pass"]
    assert pass_table.query(circuit2) == ["some-pass"]


def test_reset_table():
    """
    Tests that the table can be reset correctly.
    """
    pass_table.reset()
    assert pass_table.table == {}


### Test pass decorators preserve functionality of circuits ###
def test_cancel_inverses_functionality():
    dev = qml.device("lightning.qubit", wires=10)

    @qjit
    def workflow():
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.probs()

        circuit_opted = cancel_inverses(circuit)
        return circuit(42.42), circuit_opted(42.42)

    assert np.allclose(workflow()[0], workflow()[1])


### Test bad usages of pass decorators ###
def test_cancel_inverses_bad_usages():
    """
    Tests that an error is raised when catalyst.cancel_inverses is not used properly
    """

    def test_cancel_inverses_not_on_qnode():
        def classical_func():
            return 42.42

        with pytest.raises(
            TypeError,
            match="A QNode is expected, got the classical function",
        ):
            cancel_inverses(classical_func)

    test_cancel_inverses_not_on_qnode()

    def test_cancel_inverses_not_in_qjit():
        @qml.qnode(qml.device("lightning.qubit", wires=10))
        def circuit():
            qml.Identity(wires=np.arange(10))
            return qml.probs()

        with pytest.raises(
            RuntimeError,
            match="catalyst.cancel_inverses can only be used on a qnode inside a qjit context!",
        ):
            cancel_inverses(circuit)

    test_cancel_inverses_not_in_qjit()


if __name__ == "__main__":
    pytest.main(["-x", __file__])
