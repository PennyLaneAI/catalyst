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

"""Tests for the OQD device.
"""

import pennylane as qml
import pytest
from catalyst.third_party.oqd import OQDDevice

class TestOQDDevice:
    """Test the OQD device python layer for Catalyst."""

    def test_initialization(self):
        """Test the initialization."""

        device = OQDDevice(backend="default", shots=1000, wires=8)

        assert device.backend == "default"
        assert device.shots == qml.measurements.Shots(1000)
        assert device.wires == qml.wires.Wires(range(0, 8))

    def test_wrong_backend(self):
        """Test the backend check."""
        with pytest.raises(ValueError, match="The backend random_backend is not supported."):
            OQDDevice(backend="random_backend", shots=1000, wires=8)

    def test_execute_not_implemented(self):
        """Test the python execute is not implemented."""
        with pytest.raises(NotImplementedError, match="The OQD device only supports Catalyst."):
            dev = OQDDevice(backend="default", shots=1000, wires=8)
            dev.execute([], [])

    def test_preprocess(self):
        """Test the device preprocessing"""
        dev = OQDDevice(backend="default", shots=1000, wires=8)
        tranform_program, _ = dev.preprocess()
        assert tranform_program == qml.transforms.core.TransformProgram()

    def test_preprocess_with_config(self):
        """Test the device preprocessing by explicitly passing an execution config"""
        dev = OQDDevice(backend="default", shots=1000, wires=8)
        execution_config = qml.devices.ExecutionConfig()
        tranform_program, config = dev.preprocess(execution_config)
        assert tranform_program == qml.transforms.core.TransformProgram()
        assert config == execution_config

    def test_get_c_interface(self):
        """Test the device get_c_interface method."""
        dev = OQDDevice(backend="default", shots=1000, wires=8)
        name, _ = dev.get_c_interface()
        assert name == "oqd"


if __name__ == "__main__":
    pytest.main(["-x", __file__])
