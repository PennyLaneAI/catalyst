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
import platform

from pennylane.devices import Device, ExecutionConfig
from pennylane.transforms.core import TransformProgram
from catalyst.compiler import get_lib_path

BACKENDS = ["default"]


def OQDDevicePipeline(device, qubit, gate):
    """
    Generate the compilation pipeline for an OQD device.

    Args:
        device (str): the path to the device toml file specifications.
        qubit (str): the path to the qubit toml file specifications.
        gate (str): the path to the gate toml file specifications.

    Returns:
        A list of tuples, with each tuple being a stage in the compilation pipeline.
        When using ``keep_intermediate=True`` from :func:`~.qjit`, the kept stages
        correspond to the tuples.
    """
    return [
        (
            "device-agnostic-pipeline",
            [
                "enforce-runtime-invariants-pipeline",
                "hlo-lowering-pipeline",
                "quantum-compilation-pipeline",
                "bufferization-pipeline",
            ],
        ),
        (
            "oqd_pipeline",
            [
                "func.func(ions-decomposition)",
                "func.func(gates-to-pulses{"
                + "device-toml-loc="
                + device
                + " qubit-toml-loc="
                + qubit
                + " gate-to-pulse-toml-loc="
                + gate
                + "})",
                "convert-ion-to-llvm",
            ],
        ),
        (
            "llvm-dialect-lowering-pipeline",
            [
                "llvm-dialect-lowering-pipeline",
            ],
        ),
    ]


class OQDDevice(Device):
    """The OQD device allows access to the hardware devices from OQD using Catalyst."""

    config_filepath = get_lib_path("oqd_runtime", "OQD_LIB_DIR") + "/backend" + "/oqd.toml"

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """

        system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
        lib_path = (
            get_lib_path("oqd_runtime", "OQD_LIB_DIR") + "/librtd_oqd_device" + system_extension
        )

        return "oqd", lib_path

    def __init__(
        self, wires, shots, backend="default", openapl_file_name="__openapl__output.json", **kwargs
    ):
        self._backend = backend
        self._openapl_file_name = openapl_file_name
        _check_backend(backend=backend)
        super().__init__(wires=wires, shots=shots, **kwargs)
        self.device_kwargs = {
            "openapl_file_name": self._openapl_file_name,
        }

    @property
    def openapl_file_name(self):
        """The OpenAPL output file name."""
        return self._openapl_file_name

    @property
    def backend(self):
        """Backend property of the device."""
        return self._backend

    def preprocess(
        self,
        execution_config: Optional[ExecutionConfig] = None,
    ):
        """Device preprocessing function.

        This function defines the device transform program to be applied and an updated device
        configuration.

        TODO: This function is boilerplate only
        """
        if execution_config is None:
            execution_config = ExecutionConfig()

        transform_program = TransformProgram()

        return transform_program, execution_config

    def execute(self, circuits, execution_config):
        """Python execution is not supported."""
        raise NotImplementedError("The OQD device only supports Catalyst.")


def _check_backend(backend):
    """Helper function to check the backend."""
    if backend not in BACKENDS:
        raise ValueError(f"The backend {backend} is not supported. Valid devices are {BACKENDS}")
