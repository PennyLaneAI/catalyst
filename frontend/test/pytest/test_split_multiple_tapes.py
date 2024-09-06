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

"""
This file performs the frontend pytest checking that multi-tape transforms retain 
correct funcitonality after splitting each tape into a separate function in mlir.
"""

from typing import Callable, Sequence

import numpy as np
import pennylane as qml
import pytest

from catalyst import qjit


def test_split_multiple_tapes():
    """
    Test that multi-tape qnodes have the same functionality as core PL.
    """
    dev = qml.device("lightning.qubit", wires=2)

    def my_quantum_transform(
        tape: qml.tape.QuantumTape,
    ) -> (Sequence[qml.tape.QuantumTape], Callable):
        tape1 = tape
        tape2 = qml.tape.QuantumTape(
            [qml.RY(tape1.operations[1].parameters[0] + 0.4, wires=0)], [qml.expval(qml.X(0))]
        )

        def post_processing_fn(results):
            return results[0] + results[1]

        return [tape1, tape2], post_processing_fn

    dispatched_transform = qml.transform(my_quantum_transform)

    @dispatched_transform
    @qml.qnode(dev)
    def circuit(x):
        qml.adjoint(qml.RY)(x[0], wires=0)
        qml.RX(x[1] + 0.8, wires=1)
        return qml.expval(qml.X(0))

    expected = circuit([0.1, 0.2])

    circuit = qjit(circuit)
    qjit_results = circuit([0.1, 0.2])

    assert np.allclose(expected, qjit_results)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
