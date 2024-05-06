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

"""This module contains the OQC device."""

import os

from pennylane.devices import DefaultExecutionConfig, Device, ExecutionConfig
from pennylane.transforms.core import TransformProgram

try:
    from qcaas_client.client import OQCClient  # pylint: disable=unused-import
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Oqc qcaas client not found. Please install: pip install oqc-qcaas-client"
    ) from e

from catalyst.compiler import get_lib_path

BACKENDS = ["lucy", "toshiko"]


class OQCDevice(Device):
    """The OQC device allows to access the hardware devices from OQC using
    Catalyst."""

    config = get_lib_path("oqc_runtime", "OQC_LIB_DIR") + "/backend" + "/oqc.toml"

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """

        # TODO: Replace with the oqc shared library
        return "oqc", get_lib_path("oqc_runtime", "OQC_LIB_DIR") + "/librtd_oqc.so"

    def __init__(self, wires, backend, shots=1024, **kwargs):
        self._backend = backend
        _check_backend(backend=backend)
        _check_envvar()
        super().__init__(wires=wires, shots=shots, **kwargs)

    @property
    def backend(self):
        """Backend property of the device."""
        return self._backend

    def preprocess(
        self,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """This function defines the device transform program to be applied and an
        updated device configuration."""
        transform_program = TransformProgram()
        # TODO: Add transforms (check wires, check shots, no sample, only commuting measurements,
        # measurement from counts)
        return transform_program, execution_config

    def execute(self, circuits, execution_config):
        """Non-implemented python execution."""
        # Check availability
        raise NotImplementedError("The OQC device only supports Catalyst.")


def _check_backend(backend):
    """Helper function to check the backend."""
    if backend not in BACKENDS:
        raise ValueError(f"The backend {backend} is not supported. Valid devices are {BACKENDS}")


def _check_envvar():
    """Helper function to check the environment variables are set for authentification."""
    url = os.getenv("OQC_URL")
    email = os.getenv("OQC_EMAIL")
    password = os.getenv("OQC_PASSWORD")
    if not all((url, email, password)):
        raise ValueError("You must set url, email and password as environment variables.")
