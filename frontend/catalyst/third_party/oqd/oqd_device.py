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

"""
OQD Device
~~~~~~~~~~

This module defines the classes that represent an Open Quantum Design (OQD)
trapped-ion quantum computer device.
"""
from typing import Optional

from pennylane.devices import Device, ExecutionConfig
from pennylane.transforms.core import TransformProgram
from catalyst.compiler import get_lib_path

BACKENDS = ["default"]


class OQDDevice(Device):
    """The OQD device allows to access the hardware devices from OQD using Catalyst."""

    config_filepath = get_lib_path("oqd_runtime", "OQD_LIB_DIR") + "/backend" + "/oqd.toml"

    def __init__(self, wires, backend, shots=1, **kwargs):
        self._backend = backend
        _check_backend(backend=backend)
        super().__init__(wires=wires, shots=shots, **kwargs)

    @property
    def backend(self):
        """Backend property of the device."""
        return self._backend

    def preprocess(
        self,
        execution_config: Optional[ExecutionConfig] = None,
    ):
        """This function defines the device transform program to be applied and
        an updated device configuration.

        TODO: This function is boilerplate only
        """
        if execution_config is None:
            execution_config = ExecutionConfig()

        transform_program = TransformProgram()

        return transform_program, execution_config


def _check_backend(backend):
    """Helper function to check the backend."""
    if backend not in BACKENDS:
        raise ValueError(f"The backend {backend} is not supported. Valid devices are {BACKENDS}")
