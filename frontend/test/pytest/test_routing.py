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

"""Integration tests for routing at runtime"""

from functools import partial

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms.transpile import transpile


def qfunc_ops(wires, x, y, z):
    qml.Hadamard(wires=wires[0])
    qml.RZ(z, wires=wires[2])
    qml.CNOT(wires=[wires[2], wires[0]])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RX(x, wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[2]])
    qml.RZ(-z, wires=wires[2])
    qml.RX(y, wires=wires[0])
    qml.PauliY(wires=wires[2])
    qml.CY(wires=[wires[1], wires[2]])


# pylint: disable=too-many-public-methods
class TestRouting:
    """Unit tests for testing routing function at runtime"""
    
    all_to__all_device = qml.device("lightning.qubit")
    linear_device = qml.device("lightning.qubit", wires = [(0,1),(1,2)])

    input_devices = (
        (all_to__all_device, linear_device),
    )
    @pytest.mark.parametrize("all_to__all_device, linear_device", input_devices)
    def test_state_invariance_under_routing(self,all_to__all_device, linear_device):
        def circuit(wires, x, y, z):
            qfunc_ops(wires, x, y, z)
            return qml.state()
        
        all_to_all_qnode = qml.qjit(qml.QNode(circuit, all_to__all_device))
        linear_qnode = qml.qjit(qml.QNode(circuit, linear_device))

        assert np.allclose(all_to_all_qnode([0,1,2], 0.1, 0.2, 0.3), linear_qnode([0,1,2], 0.1, 0.2, 0.3))

    @pytest.mark.parametrize("all_to__all_device, linear_device", input_devices)
    def test_probs_invariance_under_routing(self,all_to__all_device, linear_device):
        def circuit(wires, x, y, z):
            qfunc_ops(wires, x, y, z)
            return qml.probs()
        
        all_to_all_qnode = qml.qjit(qml.QNode(circuit, all_to__all_device))
        linear_qnode = qml.qjit(qml.QNode(circuit, linear_device))

        assert np.allclose(all_to_all_qnode([0,1,2], 0.1, 0.2, 0.3), linear_qnode([0,1,2], 0.1, 0.2, 0.3))

    @pytest.mark.parametrize("all_to__all_device, linear_device", input_devices)
    def test_sample_invariance_under_routing(self,all_to__all_device, linear_device):
        def circuit(wires, x, y, z):
            qfunc_ops(wires, x, y, z)
            return qml.sample()
        
        all_to_all_qnode = qml.qjit(
                partial(qml.set_shots, shots=10)(qml.QNode(circuit, all_to__all_device)),
                seed=37
            )
        linear_qnode = qml.qjit(
                partial(qml.set_shots, shots=10)(qml.QNode(circuit, linear_device)),
                seed=37
            )
        assert np.allclose(all_to_all_qnode([0,1,2], 0.1, 0.2, 0.3), linear_qnode([0,1,2], 0.1, 0.2, 0.3))

    @pytest.mark.parametrize("all_to__all_device, linear_device", input_devices)
    def test_counts_invariance_under_routing(self,all_to__all_device, linear_device):
        def circuit(wires, x, y, z):
            qfunc_ops(wires, x, y, z)
            return qml.counts()
        
        all_to_all_qnode = qml.qjit(
                partial(qml.set_shots, shots=10)(qml.QNode(circuit, all_to__all_device)),
                seed=37
            )
        linear_qnode = qml.qjit(
                partial(qml.set_shots, shots=10)(qml.QNode(circuit, linear_device)),
                seed=37
            )
        assert np.allclose(all_to_all_qnode([0,1,2], 0.1, 0.2, 0.3), linear_qnode([0,1,2], 0.1, 0.2, 0.3))
    
    @pytest.mark.parametrize("all_to__all_device, linear_device", input_devices)
    def test_expvals_invariance_under_routing(self,all_to__all_device, linear_device):
        def circuit(wires, x, y, z):
            qfunc_ops(wires, x, y, z)
            return qml.expval(qml.X(0) @ qml.Y(1)), qml.var(qml.Z(2))
        
        all_to_all_qnode = qml.qjit(qml.QNode(circuit, all_to__all_device))
        linear_qnode = qml.qjit(qml.QNode(circuit, linear_device))
        assert np.allclose(all_to_all_qnode([0,1,2], 0.1, 0.2, 0.3), linear_qnode([0,1,2], 0.1, 0.2, 0.3))

    