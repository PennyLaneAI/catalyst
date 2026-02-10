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
import platform
from typing import Optional

from pennylane import CompilePipeline
from pennylane.devices import Device, ExecutionConfig

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

    config_filepath = get_lib_path("oqc_runtime", "OQC_LIB_DIR") + "/backend" + "/oqc.toml"

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """

        system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
        lib_path = get_lib_path("oqc_runtime", "OQC_LIB_DIR") + "/librtd_oqc" + system_extension
        return "oqc", lib_path

    def __init__(self, wires, backend, **kwargs):
        self._backend = backend
        _check_backend(backend=backend)
        _check_envvar()
        super().__init__(wires=wires, **kwargs)

    @property
    def backend(self):
        """Backend property of the device."""
        return self._backend

    def preprocess(
        self,
        execution_config: Optional[ExecutionConfig] = None,
    ):
        """This function defines the device transform program to be applied and an
        updated device configuration."""
        if execution_config is None:
            execution_config = ExecutionConfig()

        compile_pipeline = CompilePipeline()
        # TODO: Add transforms (check wires, check shots, no sample, only commuting measurements,
        # measurement from counts)
        return compile_pipeline, execution_config

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
        raise ValueError(
            """
            OQC credentials not found in environment variables.
            Please set the environment variables `OQC_EMAIL`, `OQC_PASSWORD` and `OQC_URL`.
            """
        )
