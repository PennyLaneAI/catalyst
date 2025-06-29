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
"""Test for the device API."""
import platform

import pennylane as qml
import pytest
from pennylane.devices import NullQubit

from catalyst import qjit
from catalyst.device import QJITDevice, get_device_capabilities, qjit_device
from catalyst.tracing.contexts import EvaluationContext, EvaluationMode

from functools import partial
def test_simple_circuit_set_shots():
    """Test that a circuit with the new device API is compiling to MLIR."""
    dev = NullQubit(wires=2)

    @qjit(target="mlir")
    @partial(qml.set_shots, shots=2048)
    @qml.qnode(device=dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.sample()
    
    print(f"QNode type: {circuit.__class__}")
    print(f"QNode shots: {circuit._shots}")
    print(f"QNode shots type: {type(circuit._shots)}")
    if hasattr(circuit._shots, 'total_shots'):
        print(f"QNode shots.total_shots: {circuit._shots.total_shots}")
    
    # Check device shots too
    print(f"Device shots: {getattr(dev, '_shots', 'No _shots attr')}")

    # Check that the MLIR contains the shots constant and device initialization
    mlir_str = str(circuit.mlir)
    print("=== MLIR OUTPUT ===")
    print(mlir_str)

test_simple_circuit_set_shots()


@qjit(keep_intermediate=True)
def workflow_dyn_sample(shots):  # pylint: disable=unused-argument
    # qml.device still needs concrete shots
    device = qml.device("lightning.qubit", wires=1)

    @partial(qml.set_shots, shots=shots)
    @qml.qnode(device)
    def circuit():
        qml.RX(1.5, 0)
        return qml.sample()

    return circuit()

workflow_dyn_sample(10)
res = workflow_dyn_sample(37)
print(len(res))