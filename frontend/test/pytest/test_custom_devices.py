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
"""Unit test for custom device integration with Catalyst.
"""
import pathlib

import pennylane as qml
import pytest

from catalyst import measure, qjit
from catalyst.compiler import get_lib_path
from catalyst.utils.exceptions import CompileError


@pytest.mark.skipif(
    not pathlib.Path(get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/libdummy_device.so").is_file(),
    reason="lib_dummydevice.so was not found.",
)
def test_custom_device():
    """Test that custom device can run using Catalyst."""

    class DummyDevice(qml.QubitDevice):
        """Dummy Device"""

        name = "Dummy Device"
        short_name = "dummy.device"
        pennylane_requires = "0.33.0"
        version = "0.0.1"
        author = "Dummy"

        # Doesn't matter as at the moment it is dictated by QJITDevice
        operations = []
        observables = []

        def __init__(self, shots=None, wires=None):
            super().__init__(wires=wires, shots=shots)

        def apply(self, operations, **kwargs):
            """Unused"""
            raise RuntimeError("Only C/C++ interface is defined")

        @staticmethod
        def get_c_interface():
            """Returns a tuple consisting of the device name, and
            the location to the shared object with the C/C++ device implementation.
            """

            return "DummyDevice", get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/libdummy_device.so"

    @qjit
    @qml.qnode(DummyDevice(wires=1))
    def f():
        """This function would normally return False.
        However, DummyDevice as defined in libdummy_device.so
        has been implemented to always return True."""
        return measure(0)

    assert True == f()


def test_custom_device_bad_directory():
    """Test that custom device error."""

    class DummyDevice(qml.QubitDevice):
        """Dummy Device"""

        name = "Dummy Device"
        short_name = "dummy.device"
        pennylane_requires = "0.33.0"
        version = "0.0.1"
        author = "Dummy"

        # Doesn't matter as at the moment it is dictated by QJITDevice
        operations = []
        observables = []

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

            return "DummyDevice", "this-file-does-not-exist.so"

    with pytest.raises(CompileError, match="Device at this-file-does-not-exist.so cannot be found!"):

        @qjit
        @qml.qnode(DummyDevice(wires=1))
        def f():
            return measure(0)
