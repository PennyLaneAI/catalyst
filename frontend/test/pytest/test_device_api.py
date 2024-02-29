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
import pytest
import pathlib
import pennylane as qml
from pennylane.devices import Device
from pennylane.devices.execution_config import ExecutionConfig, DefaultExecutionConfig
from pennylane.transforms import split_non_commuting
from pennylane.transforms.core import TransformProgram

from catalyst.compiler import get_lib_path
from catalyst.qjit_device import QJITDeviceNewAPI
from catalyst.utils.runtime import extract_backend_info


class DummyDevice(Device):

    config = pathlib.Path(__file__).parent.parent.joinpath("lit/dummy_device.toml")

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

    def preprocess(self, execution_config: ExecutionConfig = DefaultExecutionConfig):
        transform_program = TransformProgram()
        transform_program.add_transform(split_non_commuting)
        return transform_program, execution_config


def test_initialization():
    device = DummyDevice(wires=10, shots=2032)

    # Create qjit device
    dev_args = extract_backend_info(device)
    config, rest = dev_args[0], dev_args[1:]
    device_qjit = QJITDeviceNewAPI(device, config, *rest)

    # Check attributes of the new device
    assert isinstance(device_qjit.config, dict)
    assert device_qjit.shots == qml.measurements.Shots(2032)
    assert device_qjit.wires == qml.wires.Wires(range(0, 10))

    # Check the preprocess of the new device
    transform_program, _ = device_qjit.preprocess()
    assert transform_program
    assert len(transform_program) == 1

    t = transform_program[0].transform.__name__
    assert t == "split_non_commuting"

    # Check that the device cannot execute tapes
    with pytest.raises(RuntimeError, match="QJIT devices cannot execute tapes"):
        device_qjit.execute(10, 2)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
