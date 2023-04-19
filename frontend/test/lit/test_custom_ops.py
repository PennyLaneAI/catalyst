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

from catalyst import qjit, measure
from pennylane.operation import Operation
import pennylane as qml

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

    def __init__(self, shots=None, wires=None):
        super().__init__(wires=wires, shots=shots)

    def apply(self, operations, **kwargs):
        pass


operations = lightning.operations.copy()
operations.add("RXX")


class CustomDeviceWithSupport(CustomDeviceWithoutSupport):
    operations = operations
    observables = lightning.observables.copy()

    def __init__(self, shots=None, wires=None):
        super().__init__(wires=wires, shots=shots)


# lightning does not support PauliRot, so it will be decomposed
# Hadamards MultiRZ Hadamard
devMultiRZ = CustomDeviceWithoutSupport(wires=2)
devRXX = CustomDeviceWithSupport(wires=2)


def compile_circuit_with_device(device):
    @qjit(target="mlir")
    @qfunc(2, device=device)
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
