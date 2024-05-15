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
"""Test for the device API.
"""
import pathlib
import platform

import pennylane as qml
import pytest
from pennylane.devices import Device
from pennylane.devices.execution_config import DefaultExecutionConfig, ExecutionConfig
from pennylane.transforms import split_non_commuting
from pennylane.transforms.core import TransformProgram

from catalyst import qjit
from catalyst.compiler import get_lib_path
from catalyst.device import QJITDeviceNewAPI
from catalyst.tracing.contexts import EvaluationContext, EvaluationMode
from catalyst.utils.runtime import device_get_toml_config, extract_backend_info


class DummyDevice(Device):
    """A dummy device from the device API."""

    config = get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/backend/dummy_device.toml"

    def __init__(self, wires, shots=1024):
        print(pathlib.Path(__file__).parent.parent.parent.parent)
        super().__init__(wires=wires, shots=shots)

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """
        system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
        lib_path = get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_dummy" + system_extension
        return "dummy.remote", lib_path

    def execute(self, circuits, execution_config):
        """Execution."""
        return circuits, execution_config

    def preprocess(self, execution_config: ExecutionConfig = DefaultExecutionConfig):
        """Preprocessing."""
        transform_program = TransformProgram()
        transform_program.add_transform(split_non_commuting)
        return transform_program, execution_config


class DummyDeviceNoWires(Device):
    """A dummy device from the device API without wires."""

    config = get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/backend/dummy_device.toml"

    def __init__(self, shots=1024):
        super().__init__(shots=shots)

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """

        system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
        lib_path = get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_dummy" + system_extension
        return "dummy.remote", lib_path

    def execute(self, circuits, execution_config):
        """Execution."""
        return circuits, execution_config


def test_qjit_device():
    """Test the qjit device from a device using the new api."""
    device = DummyDevice(wires=10, shots=2032)

    # Create qjit device
    config = device_get_toml_config(device)
    backend_info = extract_backend_info(device, config)
    device_qjit = QJITDeviceNewAPI(device, backend_info)

    # Check attributes of the new device
    assert device_qjit.shots == qml.measurements.Shots(2032)
    assert device_qjit.wires == qml.wires.Wires(range(0, 10))

    # Check the preprocess of the new device
    with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
        transform_program, _ = device_qjit.preprocess(ctx)
    assert transform_program
    assert len(transform_program) == 1

    # TODO: readd when we do not discard device preprocessing
    # t = transform_program[0].transform.__name__
    # assert t == "split_non_commuting"

    t = transform_program[0].transform.__name__
    assert t == "catalyst_decompose"

    # Check that the device cannot execute tapes
    with pytest.raises(RuntimeError, match="QJIT devices cannot execute tapes"):
        device_qjit.execute(10, 2)

    assert isinstance(device_qjit.operations, set)
    assert len(device_qjit.operations) > 0
    assert isinstance(device_qjit.observables, set)
    assert len(device_qjit.observables) > 0


def test_qjit_device_no_wires():
    """Test the qjit device from a device using the new api without wires set."""
    device = DummyDeviceNoWires(shots=2032)

    # Create qjit device
    config = device_get_toml_config(device)
    backend_info = extract_backend_info(device, config)

    with pytest.raises(
        AttributeError, match="Catalyst does not support devices without set wires."
    ):
        QJITDeviceNewAPI(device, backend_info)


def test_simple_circuit():
    """Test that a circuit with the new device API is compiling to MLIR."""
    dev = DummyDevice(wires=2, shots=2048)

    @qjit(target="mlir")
    @qml.qnode(device=dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(wires=0))

    assert circuit.mlir


if __name__ == "__main__":
    pytest.main(["-x", __file__])
