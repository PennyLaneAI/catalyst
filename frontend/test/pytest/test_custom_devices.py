# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit test for custom device integration with Catalyst."""
import platform

import pennylane as qml
import pytest
from conftest import CONFIG_CUSTOM_DEVICE

from catalyst import measure, qjit
from catalyst.compiler import get_lib_path
from catalyst.device import QJITDevice, extract_backend_info, get_device_capabilities
from catalyst.utils.exceptions import CompileError

RUNTIME_LIB_PATH = get_lib_path("runtime", "RUNTIME_LIB_DIR")


def test_custom_device_load():
    """Test that custom device can run using Catalyst."""

    class CustomDevice(qml.devices.Device):
        """Custom device"""

        name = "custom.device"
        config_filepath = CONFIG_CUSTOM_DEVICE

        def __init__(self, shots=None, wires=None, options1=42, options2=38):
            super().__init__(wires=wires, shots=shots)
            self.device_kwargs = {
                "option1": options1,
                "option2": options2,
            }

        def apply(self, operations, **kwargs):
            """Unused"""
            raise RuntimeError("Only C/C++ interface is defined")

        @staticmethod
        def get_c_interface():
            """Returns a tuple consisting of the device name, and
            the location to the shared object with the C/C++ device implementation.
            """
            system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
            lib_path = (
                get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_null_qubit" + system_extension
            )
            return "NullQubit", lib_path

        def execute(self, circuits, execution_config):
            """Execution."""
            raise NotImplementedError

    device = CustomDevice(wires=1)
    capabilities = get_device_capabilities(device)
    backend_info = extract_backend_info(device, capabilities)
    assert backend_info.kwargs["option1"] == 42
    assert backend_info.kwargs["option2"] == 38

    @qjit
    @qml.qnode(device)
    def f():
        """This function would normally return False.
        However, NullQubit as defined in librtd_null_qubit.so
        has been implemented to always return True."""
        return measure(0)

    assert f() == True


def test_custom_device_bad_directory():
    """Test that custom device error."""

    class CustomDevice(qml.devices.Device):
        """Custom Device"""

        name = "custom.device"
        config_filepath = CONFIG_CUSTOM_DEVICE

        def __init__(self, shots=None, wires=None):
            super().__init__(wires=wires, shots=shots)

        def apply(self, operations, **kwargs):
            """Unused."""
            raise RuntimeError("Only C/C++ interface is defined")

        @staticmethod
        def get_c_interface():
            """Returns a tuple consisting of the device name, and
            the location to the shared object with the C/C++ device implementation.
            """
            return "CustomDevice", "this-file-does-not-exist.so"

        def execute(self, circuits, execution_config):
            """Execution."""
            raise NotImplementedError

    with pytest.raises(
        CompileError, match="Device at this-file-does-not-exist.so cannot be found!"
    ):

        @qjit
        @qml.qnode(CustomDevice(wires=1))
        def f():
            return measure(0)


def test_custom_device_no_c_interface():
    """Test that custom device error."""

    class CustomDevice(qml.devices.Device):
        """Custom Device"""

        name = "custom.device"
        config_filepath = CONFIG_CUSTOM_DEVICE

        def __init__(self, shots=None, wires=None):
            super().__init__(wires=wires, shots=shots)

        def apply(self, operations, **kwargs):
            """Unused."""
            raise RuntimeError("Custom device")

        def execute(self, circuits, execution_config):
            """Execution."""
            raise NotImplementedError

    with pytest.raises(
        CompileError, match="The custom.device device does not provide C interface for compilation."
    ):

        @qjit
        @qml.qnode(CustomDevice(wires=1))
        def f():
            return measure(0)


def test_error_raised_no_unitary_support_for_matrix_ops():
    """Tests that an error is raised when a device specifies _to_matrix_ops but does not support
    the QubitUnitary operation"""

    class CustomDevice(qml.devices.Device):
        """Custom device for testing."""

        name = "custom.device"
        config_filepath = CONFIG_CUSTOM_DEVICE

        _to_matrix_ops = {
            "DiagonalQubitUnitary": qml.devices.capabilities.OperatorProperties(),
            "BlockEncode": qml.devices.capabilities.OperatorProperties(),
        }

        def __init__(self, wires, shots=None, **kwargs):
            del self.capabilities.operations["QubitUnitary"]
            assert not self.capabilities.supports_operation("QubitUnitary")
            self.qjit_capabilities = self.capabilities
            super().__init__(wires=wires, shots=shots)

        @staticmethod
        def get_c_interface():
            """Returns a tuple consisting of the device name, and
            the location to the shared object with the C/C++ device implementation.
            """
            system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
            lib_path = (
                get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_null_qubit" + system_extension
            )
            return "NullQubit", lib_path

        def execute(self, circuits, execution_config):
            """Execution."""
            return (0,)

    with pytest.raises(
        CompileError,
        match="The device that specifies to_matrix_ops must support QubitUnitary.",
    ):
        QJITDevice(CustomDevice(wires=2, shots=2048))
