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
import os
import platform

from pennylane import CompilePipeline
from pennylane.devices import Device, ExecutionConfig
from catalyst.compiler import get_lib_path

BACKENDS = ["default"]


def get_default_artiq_config():
    """Get default ARTIQ cross-compilation configuration"""
    # Check environment variable
    kernel_ld = os.environ.get("ARTIQ_KERNEL_LD")
    if kernel_ld and os.path.exists(kernel_ld):
        return {"kernel_ld": kernel_ld}

    return None


def OQDDevicePipeline(device, qubit, gate, device_db=None):
    """
    Generate the compilation pipeline for an OQD device.

    Args:
        device (str): the path to the device toml file specifications.
        qubit (str): the path to the qubit toml file specifications.
        gate (str): the path to the gate toml file specifications.
        device_db (str, optional): the path to the device_db.json file for ARTIQ.
            If provided, generates ARTIQ-compatible output.
            If None, uses convert-ion-to-llvm for legacy OQD pipeline.

    Returns:
        A list of tuples, with each tuple being a stage in the compilation pipeline.
        When using ``keep_intermediate=True`` from :func:`~.qjit`, the kept stages
        correspond to the tuples.
    """
    # Common gates-to-pulses pass
    gates_to_pulses_pass = (
        "func.func(gates-to-pulses{"
        + "device-toml-loc="
        + device
        + " qubit-toml-loc="
        + qubit
        + " gate-to-pulse-toml-loc="
        + gate
        + "})"
    )

    # Build OQD pipeline based on whether device_db is provided
    if device_db is not None:
        oqd_passes = [
            "func.func(ions-decomposition)",
            gates_to_pulses_pass,
            "convert-ion-to-rtio{" + "device_db=" + device_db + "}",
            "convert-rtio-event-to-artiq",
        ]
        llvm_lowering_passes = [
            "llvm-dialect-lowering-stage",
            "emit-artiq-runtime",
        ]
    else:
        # Standard LLVM lowering route (legacy OQD pipeline)
        oqd_passes = [
            "func.func(ions-decomposition)",
            gates_to_pulses_pass,
            "convert-ion-to-llvm",
        ]
        llvm_lowering_passes = [
            "llvm-dialect-lowering-stage",
        ]

    return [
        (
            "device-agnostic-pipeline",
            [
                "quantum-compilation-stage",
                "hlo-lowering-stage",
                "gradient-lowering-stage",
                "bufferization-stage",
            ],
        ),
        (
            "oqd_pipeline",
            oqd_passes,
        ),
        (
            "llvm-dialect-lowering-stage",
            llvm_lowering_passes,
        ),
    ]


class OQDDevice(Device):
    """The OQD device allows access to the hardware devices from OQD using Catalyst.

    Args:
        wires: The number of wires/qubits.
        backend: Backend name (default: "default").
        openapl_file_name: Output file name for OpenAPL.
        artiq_config: ARTIQ cross-compilation configuration dict with keys:
            - kernel_ld: Path to ARTIQ's kernel.ld linker script
            - llc_path: Path to llc
            - lld_path: Path to ld.lld
    """

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
        self,
        wires,
        backend="default",
        openapl_file_name="__openapl__output.json",
        artiq_config=None,
        **kwargs,
    ):
        self._backend = backend
        self._openapl_file_name = openapl_file_name
        _check_backend(backend=backend)
        super().__init__(wires=wires, **kwargs)
        self.device_kwargs = {
            "openapl_file_name": self._openapl_file_name,
        }

        if artiq_config is not None:
            self._artiq_config = artiq_config
        else:
            self._artiq_config = get_default_artiq_config()

    @property
    def artiq_config(self):
        """ARTIQ cross-compilation configuration."""
        return self._artiq_config

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

        return CompilePipeline(), execution_config

    def execute(self, circuits, execution_config):
        """Python execution is not supported."""
        raise NotImplementedError("The OQD device only supports Catalyst.")


def _check_backend(backend):
    """Helper function to check the backend."""
    if backend not in BACKENDS:
        raise ValueError(f"The backend {backend} is not supported. Valid devices are {BACKENDS}")
