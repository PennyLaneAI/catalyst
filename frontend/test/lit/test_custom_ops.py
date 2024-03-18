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

# RUN: %PYTHON %s | FileCheck %s

import pennylane as qml
from pennylane.operation import Operation

from catalyst import measure, qjit
from catalyst.compiler import get_lib_path

# This is used just for internal testing
from catalyst.pennylane_extensions import qfunc


class RXX(Operation):
    num_params = 1
    num_wires = 2
    par_domain = "R"

    def __init__(self, theta, wires=None):
        self.theta = theta
        super().__init__(theta, wires=wires)

    def decomposition(self):
        return [qml.PauliRot(self.theta, pauli_word="XX", wires=self.wires)]


lightning = qml.device("lightning.qubit", wires=3)


class CustomDeviceWithoutSupport(qml.QubitDevice):
    name = "Device without support for RXX gate."
    short_name = "dummy.device"
    pennylane_requires = "0.1.0"
    version = "0.0.1"
    author = "CV quantum"

    operations = lightning.operations.copy()
    observables = lightning.observables.copy()

    # pylint: disable=too-many-arguments
    def __init__(
        self, shots=None, wires=None, backend_name=None, backend_lib=None, backend_kwargs=None
    ):
        self.backend_name = backend_name if backend_name else "default"
        self.backend_lib = backend_lib if backend_lib else "default"
        self.backend_kwargs = backend_kwargs if backend_kwargs else ""
        self.backend_path = CustomDeviceWithoutSupport.get_c_interface()
        super().__init__(shots=shots, wires=wires)

    def apply(self, operations, **kwargs):
        pass

    @staticmethod
    def get_c_interface():
        """Location to shared object with C/C++ implementation"""
        return get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/libdummy_device.so"


operations = lightning.operations.copy()
operations.add("RXX")


class CustomDeviceWithSupport(CustomDeviceWithoutSupport):
    operations = operations
    observables = lightning.observables.copy()

    def __init__(self, shots=None, wires=None):
        super().__init__(wires=wires, shots=shots)
        self.backend_name = "default"
        self.backend_kwargs = ""
        self.backend_path = CustomDeviceWithoutSupport.get_c_interface()


# lightning does not support PauliRot, so it will be decomposed
# Hadamards MultiRZ Hadamard
devMultiRZ = CustomDeviceWithoutSupport(wires=2)
devRXX = CustomDeviceWithSupport(wires=2)


def compile_circuit_with_device(device):
    @qjit(target="mlir")
    @qfunc(device=device)
    def f(x: float):
        RXX(x, wires=[0, 1])
        return measure(wires=0)

    print(f.mlir)


# CHECK-LABEL: public @jit_f
# CHECK-NOT: RXX
# CHECK: multirz
# CHECK-NOT: RXX
compile_circuit_with_device(devMultiRZ)
# CHECK-LABEL: public @jit_f
# CHECK-NOT: multirz
# CHECK: RXX
# CHECK-NOT: multirz
compile_circuit_with_device(devRXX)
