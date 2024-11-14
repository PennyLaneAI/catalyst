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
"""Test for the OQC device.
"""
# pylint: disable=unused-argument,import-outside-toplevel,unused-import

import pennylane as qml
import pytest


class TestOQCDevice:
    """Test the OQC device python layer for Catalyst."""

    def test_entrypoint(self, set_dummy_oqc_env):
        """Test the initialization."""

        device = qml.device("oqc.cloud", backend="lucy", shots=1000, wires=8)
        with open(device.config, "r") as file:
            config_data = file.read()
            print(config_data)
        assert device.backend == "lucy"
        assert device.shots == qml.measurements.Shots(1000)
        assert device.wires == qml.wires.Wires(range(0, 8))

    def test_initialization(self, set_dummy_oqc_env):
        """Test the initialization."""
        from catalyst.third_party.oqc import OQCDevice

        device = OQCDevice(backend="lucy", shots=1000, wires=8)

        assert device.backend == "lucy"
        assert device.shots == qml.measurements.Shots(1000)
        assert device.wires == qml.wires.Wires(range(0, 8))

        device = OQCDevice(backend="toshiko", shots=1000, wires=32)

        assert device.backend == "toshiko"
        assert device.shots == qml.measurements.Shots(1000)
        assert device.wires == qml.wires.Wires(range(0, 32))

    def test_wrong_backend(self, set_dummy_oqc_env):
        """Test the backend check."""
        from catalyst.third_party.oqc import OQCDevice

        with pytest.raises(ValueError, match="The backend falcon is not supported."):
            OQCDevice(backend="falcon", shots=1000, wires=8)

    def test_execute_not_implemented(self, set_dummy_oqc_env):
        """Test the python execute is not implemented."""
        from catalyst.third_party.oqc import OQCDevice

        with pytest.raises(NotImplementedError, match="The OQC device only supports Catalyst."):
            dev = OQCDevice(backend="lucy", shots=1000, wires=8)
            dev.execute([], [])

    def test_preprocess(self, set_dummy_oqc_env):
        """Test the device preprocessing"""
        from catalyst.third_party.oqc import OQCDevice

        dev = OQCDevice(backend="lucy", shots=1000, wires=8)
        tranform_program, _ = dev.preprocess()
        assert tranform_program == qml.transforms.core.TransformProgram()

    def test_preprocess_with_config(self, set_dummy_oqc_env):
        """Test the device preprocessing by explicitly passing an execution config"""
        from catalyst.third_party.oqc import OQCDevice

        dev = OQCDevice(backend="lucy", shots=1000, wires=8)
        execution_config = qml.devices.ExecutionConfig()
        tranform_program, config = dev.preprocess(execution_config)
        assert tranform_program == qml.transforms.core.TransformProgram()
        assert config == execution_config

    def test_get_c_interface(self, set_dummy_oqc_env):
        """Test the device get_c_interface method."""
        from catalyst.third_party.oqc import OQCDevice

        dev = OQCDevice(backend="lucy", shots=1000, wires=8)
        name, _ = dev.get_c_interface()
        assert name == "oqc"

    def test_no_envvar(self):
        """Test the device get_c_interface method."""
        from catalyst.third_party.oqc import OQCDevice

        with pytest.raises(
            ValueError, match="You must set url, email and password as environment variables."
        ):
            OQCDevice(backend="lucy", shots=1000, wires=8)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
