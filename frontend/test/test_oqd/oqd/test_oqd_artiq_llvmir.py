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

"""Test that LLVM IR generated from OQD pipeline contains required ARTIQ symbols."""

import os

import numpy as np
import pennylane as qml
import pytest

from catalyst import CompileError, qjit
from catalyst.third_party.oqd import OQDDevice, OQDDevicePipeline

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


class TestOQDARTIQLLVMIR:
    """Test OQD device ARTIQ LLVM IR generation."""

    @pytest.mark.usefixtures("use_capture")
    def test_rx_gate_llvmir_contains_artiq_symbols(self):
        """Test that LLVM IR contains required ARTIQ symbols (__modinit__ and __kernel__)."""
        artiq_config = {"kernel_ld": None}

        oqd_dev = OQDDevice(
            backend="default",
            wires=1,
            artiq_config=artiq_config,
        )

        # Compile to LLVM IR only
        oqd_pipelines = _get_oqd_pipelines()

        @qjit(pipelines=oqd_pipelines, target="llvmir")
        @qml.set_shots(4)
        @qml.qnode(oqd_dev)
        def circuit():
            x = np.pi / 2
            qml.RX(x, wires=0)
            return qml.counts(all_outcomes=True)

        # Get the LLVM IR
        llvm_ir = circuit.llvmir

        # Verify required ARTIQ symbols and structure
        assert "define void @__modinit__" in llvm_ir, "Missing __modinit__ function definition"
        assert (
            "tail call void @__kernel__()" in llvm_ir or "call void @__kernel__()" in llvm_ir
        ), "Missing call to __kernel__ in __modinit__"

        assert "define void @__kernel__()" in llvm_ir, "Missing __kernel__ function definition"

        # Verify RTIO runtime functions are declared
        assert "declare i64 @now_mu()" in llvm_ir, "Missing now_mu declaration"
        assert "declare void @at_mu(i64)" in llvm_ir, "Missing at_mu declaration"
        assert "declare void @rtio_init()" in llvm_ir, "Missing rtio_init declaration"
        assert "declare void @delay_mu(i64)" in llvm_ir, "Missing delay_mu declaration"
        assert "declare void @rtio_output(i32, i32)" in llvm_ir, "Missing rtio_output declaration"

        # Verify __kernel__ contains RTIO calls
        assert (
            "call fastcc void @rtio_init()" in llvm_ir or "call void @rtio_init()" in llvm_ir
        ), "Missing rtio_init call in __kernel__"

        # Test that the llvm ir is the same after getting it again
        assert circuit.llvmir == llvm_ir, "LLVM IR should be the same after getting it again"

        # Test artiq_config
        assert oqd_dev.artiq_config == artiq_config, "Same artiq_config should be returned"

    @pytest.mark.usefixtures("use_capture")
    def test_no_compilation_error(self):
        """Test that no compilation error is raised."""
        artiq_config = {"kernel_ld": None}

        oqd_dev = OQDDevice(
            backend="default",
            wires=1,
            artiq_config=artiq_config,
        )

        oqd_pipelines = _get_oqd_pipelines()

        @qjit(pipelines=oqd_pipelines, target="llvmir")
        @qml.set_shots(4)
        @qml.qnode(oqd_dev)
        def circuit():
            x = np.pi / 2
            qml.RX(x, wires=0)
            return qml.counts(all_outcomes=True)

        with pytest.raises(
            CompileError,
            match="Functions compiled with target='llvmir' cannot be executed directly. "
            "Access the generated LLVM IR via the '\\.llvmir' property\\.",
        ):
            circuit()


if __name__ == "__main__":
    pytest.main(["-x", __file__])
