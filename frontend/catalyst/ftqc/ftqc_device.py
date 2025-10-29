# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the dummy FTQC device."""

from pennylane.devices import Device

from catalyst.compiler import get_lib_path


class FTQCDevice(Device):
    """This dummy FTQC device allows to usage PPM/PPR passes using
    Catalyst. This is a temporary device to allow for converting the PauliRot and
    PauliMeasure operations from PennyLane frontend directly to the PPR and PPM
    operations in the QEC dialect.
    """

    config_filepath = get_lib_path("ftqc_runtime", "FTQC_LIB_DIR") + "/ftqc.toml"

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.

        Note: This is a dummy implementation. The second return value should be a 
        path to the library, but we don't actually have a C++ implementation for 
        this, so we just return the config filepath.
        """

        return "ftqc", FTQCDevice.config_filepath

    def __init__(self, wires, **kwargs):
        super().__init__(wires=wires, **kwargs)

    def execute(self, circuits, execution_config):
        """Non-implemented python execution."""
        # Check availability
        raise NotImplementedError("The FTQC device only supports Catalyst.")
