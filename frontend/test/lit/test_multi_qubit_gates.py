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

import os
import pathlib
import platform

import numpy as np
import pennylane as qml
from pennylane.devices.capabilities import OperatorProperties

from catalyst import measure, qjit
from catalyst.compiler import get_lib_path
from catalyst.device import get_device_capabilities

TEST_PATH = os.path.dirname(__file__)
CONFIG_CUSTOM_DEVICE = pathlib.Path(f"{TEST_PATH}/../custom_device/custom_device.toml")


# CHECK-LABEL: public @jit_circuit
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=5))
def circuit(x: float):
    # CHECK: {{%.+}} = quantum.custom "Identity"() {{.+}} : !quantum.bit
    qml.Identity(0)
    # CHECK: {{%.+}} = quantum.custom "CNOT"() {{.+}} : !quantum.bit, !quantum.bit
    qml.CNOT(wires=[0, 1])
    # CHECK: {{%.+}} = quantum.custom "CSWAP"() {{.+}} : !quantum.bit, !quantum.bit, !quantum.bit
    qml.CSWAP(wires=[0, 1, 2])
    # pylint: disable=line-too-long
    # CHECK: {{%.+}} = quantum.multirz({{%.+}}) {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}} : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    qml.MultiRZ(x, wires=[0, 1, 2, 3, 4])
    return measure(wires=0)


print(circuit.mlir)


# CHECK-LABEL: public @jit_circuit
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def circuit():
    U1 = 1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
    # CHECK: {{%.+}} = quantum.unitary({{%.+}} : tensor<2x2xcomplex<f64>>) {{%.+}} : !quantum.bit
    qml.QubitUnitary(U1, wires=0)

    U2 = np.array(
        [
            [0.99500417 - 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 - 0.09983342j],
        ]
    )
    # CHECK: {{%.+}} = quantum.unitary({{%.+}} : tensor<4x4xcomplex<f64>>) {{%.+}}, {{%.+}} : !quantum.bit, !quantum.bit
    qml.QubitUnitary(U2, wires=[0, 2])

    return measure(wires=0), measure(wires=1)


print(circuit.mlir)


def get_custom_qjit_device(num_wires, discards, additions):
    """Generate a custom device without gates in discards."""

    class CustomDevice(qml.devices.Device):
        """Custom Gate Set Device"""

        name = "lightning.qubit"
        config_filepath = CONFIG_CUSTOM_DEVICE

        def __init__(self, shots=None, wires=None):
            super().__init__(wires=wires, shots=shots)
            self.qjit_capabilities = get_device_capabilities(self)
            for gate in discards:
                self.qjit_capabilities.operations.pop(gate, None)
            self.qjit_capabilities.operations.update(additions)

        @staticmethod
        def get_c_interface():
            """Returns a tuple consisting of the device name, and
            the location to the shared object with the C/C device implementation.
            """

            system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
            # Borrowing the NullQubit library:
            lib_path = (
                get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_null_qubit" + system_extension
            )
            return "NullQubit", lib_path

        def execute(self, circuits, execution_config):
            """Exececute the device (no)."""
            raise RuntimeError("No execution for the custom device")

    return CustomDevice(wires=num_wires)


# CHECK-LABEL: public @jit_circuit
@qjit(target="mlir")
@qml.qnode(
    get_custom_qjit_device(2, (), {"ISWAP": OperatorProperties(), "PSWAP": OperatorProperties()})
)
def circuit(x: float):
    # CHECK: {{%.+}} = quantum.custom "ISWAP"() {{.+}} : !quantum.bit, !quantum.bit
    qml.ISWAP(wires=[0, 1])
    # CHECK: {{%.+}} = quantum.custom "PSWAP"({{%.+}}) {{.+}} : !quantum.bit, !quantum.bit
    qml.PSWAP(x, wires=[0, 1])
    return qml.probs()


print(circuit.mlir)


# CHECK-LABEL: public @jit_isingZZ_circuit
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def isingZZ_circuit(x: float):
    """Circuit that applies an IsingZZ gate to a pair of qubits."""
    # CHECK: {{%.+}} = quantum.custom "IsingZZ"({{%.+}}) {{.+}} : !quantum.bit, !quantum.bit
    qml.IsingZZ(x, wires=[0, 1])
    return qml.state()


print(isingZZ_circuit.mlir)
