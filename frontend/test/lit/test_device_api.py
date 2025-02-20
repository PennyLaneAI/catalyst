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

# pylint: disable=line-too-long

"""Test for the device API."""
import os
import pathlib
import platform
from typing import Optional

import pennylane as qml
from pennylane.devices import Device
from pennylane.devices.execution_config import ExecutionConfig
from pennylane.transforms.core import TransformProgram
from utils import qjit_for_tests as qjit

from catalyst.compiler import get_lib_path

TEST_PATH = os.path.dirname(__file__)
CONFIG_CUSTOM_DEVICE = pathlib.Path(f"{TEST_PATH}/../custom_device/custom_device.toml")


class CustomDevice(Device):
    """A custom device that does nothing."""

    config_filepath = CONFIG_CUSTOM_DEVICE

    def __init__(self, wires, shots=1024):
        super().__init__(wires=wires, shots=shots)

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """
        system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
        null_qubit_lib_path = (
            get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_null_qubit" + system_extension
        )

        return "Custom", null_qubit_lib_path

    def execute(self, circuits, execution_config):
        """Execute"""
        return circuits, execution_config

    def preprocess(self, execution_config: Optional[ExecutionConfig] = None):
        """Preprocess"""
        if execution_config is None:
            execution_config = ExecutionConfig()

        transform_program = TransformProgram()
        transform_program.add_transform(qml.transforms.split_non_commuting)
        return transform_program, execution_config


def test_circuit():
    """Test a circuit compilation to MLIR when using the new device API."""

    # CHECK:   [[shots:%.+]] = arith.constant 2048 : i64
    # CHECK:   quantum.device shots([[shots]]) ["[[PATH:.*]]librtd_null_qubit.{{so|dylib}}", "Custom", "{'shots': 2048}"]
    dev = CustomDevice(wires=2, shots=2048)

    @qjit(target="mlir")
    @qml.qnode(device=dev)
    def circuit():
        # CHECK:   quantum.custom "Hadamard"
        qml.Hadamard(wires=0)
        # CHECK:   quantum.custom "CNOT"
        qml.CNOT(wires=[0, 1])
        # CHECK:   quantum.namedobs [[QBIT:.*]][ PauliZ]
        # CHECK:   quantum.expval
        return qml.expval(qml.PauliZ(wires=0))

    print(circuit.mlir)


test_circuit()


def test_preprocess():
    """Test a circuit (with preprocessing transforms) compilation to MLIR when
    using the new device API.
    TODO: we need to readd the two check-not once we accept the device preprocessing."""

    # CHECK:   [[shots:%.+]] = arith.constant 2048 : i64
    # CHECK:   quantum.device shots([[shots]]) ["[[PATH:.*]]librtd_null_qubit.{{so|dylib}}", "Custom", "{'shots': 2048}"]
    dev = CustomDevice(wires=2, shots=2048)

    @qjit(target="mlir")
    @qml.qnode(device=dev)
    def circuit_split():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        # CHECK:   quantum.custom "Hadamard"
        # CHECK:   quantum.custom "CNOT"
        # CHECK:   quantum.namedobs [[QBIT:.*]][ PauliZ]
        # CHECK-NOT:   quantum.custom "Hadamard"
        # CHECK-NOT:   quantum.custom "CNOT"
        # CHECK:   quantum.namedobs [[QBIT:.*]][ PauliY]
        # CHECK:    return [[RETURN:.*]]: tensor<f64>, tensor<f64>
        return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliY(wires=0))

    print(circuit_split.mlir)


test_preprocess()
