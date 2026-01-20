# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for OQD device ARTIQ compilation."""

import os
import shutil
import subprocess

import numpy as np
import pennylane as qml
import pytest

from catalyst import qjit
from catalyst.third_party.oqd import OQDDevice, OQDDevicePipeline, compile_to_artiq

MODULE_TEST_PATH = os.path.dirname(__file__)


def _get_oqd_pipelines():
    """Get OQD pipelines with device_db if available."""
    device_db_path = os.path.join(MODULE_TEST_PATH, "device_db.json")

    if not os.path.exists(device_db_path):
        device_db_path = None

    return OQDDevicePipeline(
        os.path.join(MODULE_TEST_PATH, "calibration_data", "device.toml"),
        os.path.join(MODULE_TEST_PATH, "calibration_data", "qubit.toml"),
        os.path.join(MODULE_TEST_PATH, "calibration_data", "gate.toml"),
        device_db_path,
    )


class TestOQDARTIQCompilation:
    """Test OQD device ARTIQ compilation."""

    def test_rx_gate_artiq_compilation(self):
        """Test RX gate compilation to ARTIQ ELF binary."""
        artiq_config = {
            "kernel_ld": os.path.join(MODULE_TEST_PATH, "kernel.ld"),
        }

        oqd_dev = OQDDevice(
            backend="default",
            wires=1,
            artiq_config=artiq_config,
        )
        qml.capture.enable()

        # Compile to LLVM IR only
        oqd_pipelines = _get_oqd_pipelines()

        @qjit(pipelines=oqd_pipelines, target="llvmir", keep_intermediate="pass")
        @qml.set_shots(4)
        @qml.qnode(oqd_dev)
        def circuit():
            x = np.pi / 2
            qml.RX(x, wires=0)
            return qml.counts(all_outcomes=True)

        # Compile to ARTIQ ELF
        output_elf_path = compile_to_artiq(circuit, oqd_dev.artiq_config)

        # Verify the ELF file was created
        assert output_elf_path is not None
        assert isinstance(output_elf_path, str)
        assert os.path.exists(output_elf_path), f"ELF file not found at {output_elf_path}"
        assert output_elf_path.endswith(
            ".elf"
        ), f"Output file should have .elf extension: {output_elf_path}"

        # Verify required ARTIQ symbols are present
        objdump_path = shutil.which("objdump")
        if objdump_path is None:
            pytest.skip("objdump not found in PATH")

        symbol_result = subprocess.run(
            [objdump_path, "-t", output_elf_path],
            capture_output=True,
            text=True,
            check=False,
        )
        assert (
            symbol_result.returncode == 0
        ), f"objdump -t failed on ELF file {output_elf_path}: {symbol_result.stderr}"

        symbols = symbol_result.stdout
        for check_symbol in ["__modinit__", "__kernel__"]:
            assert (
                check_symbol in symbols
            ), f"Required ARTIQ symbol '{check_symbol}' not found in ELF file {output_elf_path}"


if __name__ == "__main__":
    pytest.main(["-x", __file__])
