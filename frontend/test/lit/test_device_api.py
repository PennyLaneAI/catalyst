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

# RUN: %PYTHON %s | FileCheck %s

"""Test for the device API.
"""
import pathlib
import pennylane as qml
from pennylane.devices import Device
from catalyst.compiler import get_lib_path
from catalyst import qjit
class DummyDevice(Device):

    config = pathlib.Path(__file__).parent.joinpath("dummy_device.toml")

    def __init__(self, wires, shots=1024, **kwargs):
        super().__init__(wires=wires, shots=shots)

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """

        return "dummy.remote", get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/libdummy_device.so"

    def execute(self, circuits, execution_config):
        return super().execute(circuits, execution_config)
    

def test_circuit():

    dev = DummyDevice(wires=2, shots=2048)

    @qjit(target="mlir")
    @qml.qnode(device=dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(wires=0))
    
    print(circuit.mlir)

test_circuit()