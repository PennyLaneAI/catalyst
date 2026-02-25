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
"""
Pytest utilities for the Catalyst test suite.
"""

import os
import pathlib
import platform

import pennylane as qml

from catalyst.compiler import get_lib_path
from catalyst.device import get_device_capabilities

TEST_PATH = os.path.dirname(__file__)
CONFIG_CUSTOM_DEVICE = pathlib.Path(f"{TEST_PATH}/../custom_device/custom_device.toml")


def get_custom_qjit_device(num_wires, discards, additions, to_matrix_ops={}):
    """Generate a custom device without gates in discards.

    Args:
        num_wires (int): The number of wires the device should have.
        discards (set[str]): The set of gate names to discard from the device capabilities.
        additions (dict[str, OperatorProperties]): A mapping of gate names to their properties
            to add to the device capabilities.
        to_matrix_ops (dict[str, OperatorProperties]): A mapping of gate names to their properties
            to add to the device capabilities as gates that can be decomposed to matrices.

    Returns:
        qml.Device: A custom QJITDevice with the specified capabilities.
    """

    class CustomDevice(qml.devices.Device):
        """Custom Gate Set Device"""

        name = "lightning.qubit"
        config_filepath = CONFIG_CUSTOM_DEVICE

        # A lightning specific quirk is that gates that can be decomposed to matrices
        # must be marked as such in the capabilities, so we allow passing in a separate
        # dictionary for those.
        _to_matrix_ops = to_matrix_ops

        def __init__(self, wires=None):
            super().__init__(wires=wires)
            self.qjit_capabilities = get_device_capabilities(self)
            for gate in discards:
                self.qjit_capabilities.operations.pop(gate, None)
            self.qjit_capabilities.operations.update(additions)

        @staticmethod
        def get_c_interface():
            """Returns a tuple consisting of the device name, and
            the location to the shared object with the C/C++ device implementation.
            """

            system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
            # Borrowing the NullQubit library:
            lib_path = (
                get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_null_qubit" + system_extension
            )
            return "NullQubit", lib_path

        def execute(self):
            """Exececute the device (no)."""
            raise RuntimeError("No execution for the custom device")

    return CustomDevice(wires=num_wires)
