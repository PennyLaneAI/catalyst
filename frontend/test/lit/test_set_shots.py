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

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable=line-too-long

"""Test for qml.set_shots functionality."""
from functools import partial

import pennylane as qml

from catalyst import qjit


def test_simple_circuit_set_shots():
    """Test that a circuit with qml.set_shots is compiling to MLIR."""
    dev = qml.device("lightning.qubit", wires=2)

    @qjit(target="mlir")
    @partial(qml.set_shots, shots=2048)
    @qml.qnode(device=dev)
    def circuit():
        return qml.expval(qml.PauliZ(wires=0))

    # CHECK: [[shots:%.+]] = arith.constant 2048 : i64
    # CHECK: quantum.device shots([[shots]]) {{.*}}
    print(circuit.mlir)


test_simple_circuit_set_shots()
