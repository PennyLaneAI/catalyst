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

import pennylane as qml
import pytest
from jax import numpy as jnp

from catalyst import for_loop, measure, qjit
from catalyst.compiler import get_lib_path

# This is used just for internal testing
from catalyst.pennylane_extensions import qfunc

lightning = qml.device("lightning.qubit", wires=3)
copy = lightning.operations.copy()
copy.discard("MultiControlledX")
copy.discard("Rot")
copy.discard("S")


class CustomDevice(qml.QubitDevice):
    """Dummy Device"""

    name = "Dummy Device"
    short_name = "dummy.device"
    pennylane_requires = "0.32.0"
    version = "0.0.1"
    author = "Dummy"

    operations = copy
    observables = lightning.observables.copy()

    def __init__(self, shots=None, wires=None):
        self.backend_name = "default"
        self.backend_lib = "default"
        self.backend_kwargs = {}
        super().__init__(wires=wires, shots=shots)

    def apply(self, operations, **kwargs):
        """Unused"""
        raise RuntimeError("Only C/C++ interface is defined")

    @staticmethod
    def get_c_interface():
        """Location to shared object with C/C++ implementation"""
        return get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/libdummy_device.so"


dev = CustomDevice(wires=2)


@pytest.mark.skip(reason="skip this test with the plugin system on CI")
@pytest.mark.parametrize("param,expected", [(0.0, True), (jnp.pi, False)])
def test_decomposition(param, expected):
    @qjit()
    @qfunc(device=dev)
    def mid_circuit(x: float):
        qml.Hadamard(wires=0)
        qml.Rot(0, 0, x, wires=0)
        qml.Hadamard(wires=0)
        m = measure(wires=0)
        b = m ^ 0x1
        qml.Hadamard(wires=1)
        qml.Rot(0, 0, b * jnp.pi, wires=1)
        qml.Hadamard(wires=1)
        return measure(wires=1)

    assert mid_circuit(param) == expected
