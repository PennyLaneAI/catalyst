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

"""Test module for the convert-qecl-to-qecp dialect-conversion transform."""

import pennylane as qp
import pytest

from catalyst.python_interface.transforms.qecl import (
    convert_quantum_to_qecl_pass,
    inject_noise_to_qecl_pass,
)
from catalyst.python_interface.transforms.qecp import (
    ConvertQecLogicalToQecPhysicalPass,
    convert_qecl_to_qecp_pass,
)
from catalyst.python_interface.transforms.qecp.qec_code_lib import QecCode
from catalyst.utils.exceptions import CompileError

# pylint: disable=line-too-long


pytestmark = pytest.mark.xdsl


class TestQECLNoiseLoweringPassIntegration:
    """Integration lit tests for the convert-qecl-noise-to-qecp-noise pass"""

    # pylint: disable=line-too-long
    @pytest.mark.usefixtures("use_capture")
    def test_convert_qecl_noise_to_qecp_noise_pass_integration(self, run_filecheck_qjit):
        """Test the convert-qecl-noise-to-qecp-noise pass on the simplest possible, non-trivial circuit."""
        dev = qp.device("null.qubit", wires=1)

        @qp.qjit(target="mlir", keep_intermediate=True)
        @convert_qecl_to_qecp_pass(qec_code=QecCode("Steane", 7, 1, 3), number_errors=1)
        @inject_noise_to_qecl_pass
        @convert_quantum_to_qecl_pass(k=1)
        @qp.qnode(dev, shots=1)
        def circuit():
            # CHECK: builtin.unrealized_conversion_cast [[codeblock:%.*]] : !qecl.codeblock<1> to !qecp.codeblock<1 x 7>
            # CHECK: arith.constant dense
            # CHECK: arith.constant dense
            # CHECK: func.call @noise_subroutine_code
            # CHECK: builtin.unrealized_conversion_cast [[codeblock:%.*]] : !qecp.codeblock<1 x 7> to !qecl.codeblock<1>
            # CHECK: qecl.qec
            # CHECK: qecl.hadamard
            qp.H(0)
            # CHECK: builtin.unrealized_conversion_cast [[codeblock:%.*]] : !qecl.codeblock<1> to !qecp.codeblock<1 x 7>
            # CHECK: arith.constant dense
            # CHECK: arith.constant dense
            # CHECK: func.call @noise_subroutine_code
            # CHECK: builtin.unrealized_conversion_cast [[codeblock:%.*]] : !qecp.codeblock<1 x 7> to !qecl.codeblock<1>
            # CHECK: qecl.qec
            m0 = qp.measure(0)
            return qp.sample([m0])

        run_filecheck_qjit(circuit)


@pytest.mark.xfail(
    reason="The type conversion patterns are not applied in the current implementation"
)
class TestTypeConversionPattern:
    """Unit tests for the type conversion patterns of the convert-qecl-to-qecp pass."""

    @pytest.mark.parametrize("n", [7, 42])
    @pytest.mark.parametrize(
        "k", [1, pytest.param(2, marks=pytest.mark.xfail(reason="Only k = 1 is supported"))]
    )
    def test_codeblock_conversion(self, run_filecheck, n, k):
        """Test the type conversion pattern from !qecl.codeblock -> !qecp.codeblock for a few values
        of n and k.
        """
        program = f"""
        builtin.module {{
        // CHECK-LABEL: test_program
        func.func @test_program() {{
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecp.codeblock<{k} x {n}>
            %0 = "test.op"() : () -> !qecl.codeblock<{k}>

            // CHECK: [[cb1:%.+]] = "test.op"([[cb0]]) : (!qecp.codeblock<{k} x {n}>) -> !qecp.codeblock<{k} x {n}>
            %1 = "test.op"(%0) : (!qecl.codeblock<{k}>) -> !qecl.codeblock<{k}>
            return
        }}
        }}
        """
        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode("", n, k, 3)),)
        run_filecheck(program, pipeline)

    @pytest.mark.parametrize("width", [1, 2, 3])
    @pytest.mark.parametrize("n", [7, 42])
    @pytest.mark.parametrize(
        "k", [1, pytest.param(2, marks=pytest.mark.xfail(reason="Only k = 1 is supported"))]
    )
    def test_hyperreg_conversion(self, run_filecheck, width, n, k):
        """Test the type conversion pattern from !qecl.codeblock -> !qecp.codeblock for a few values
        of n and k.
        """
        program = f"""
        builtin.module {{
        // CHECK-LABEL: test_program
        func.func @test_program() {{
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecp.hyperreg<{width} x {k} x {n}>
            %0 = "test.op"() : () -> !qecl.hyperreg<{width} x {k}>

            // CHECK: [[cb1:%.+]] = "test.op"([[cb0]]) : (!qecp.hyperreg<{width} x {k} x {n}>) -> !qecp.hyperreg<{width} x {k} x {n}>
            %1 = "test.op"(%0) : (!qecl.hyperreg<{width} x {k}>) -> !qecl.hyperreg<{width} x {k}>
            return
        }}
        }}
        """
        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode("", n, k, 3)),)
        run_filecheck(program, pipeline)

    def test_codeblock_conversion_with_k_mismatch(self, run_filecheck):
        """Test that attempting to convert a codeblock type with a value of k different than the
        value of k given in the QEC code raise a CompileError.
        """
        program = """
        builtin.module {
        // CHECK-LABEL: test_program
        func.func @test_program() {
            %0 = "test.op"() : () -> !qecl.codeblock<2>
            return
        }
        }
        """
        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode("", 7, 1, 3)),)

        with pytest.raises(CompileError, match="Failed to convert type"):
            run_filecheck(program, pipeline)

    def test_hyperreg_conversion_with_k_mismatch(self, run_filecheck):
        """Test that attempting to convert a hyper-register type with a value of k different than
        the value of k given in the QEC code raise a CompileError.
        """
        program = """
        builtin.module {
        // CHECK-LABEL: test_program
        func.func @test_program() {
            %0 = "test.op"() : () -> !qecl.hyperreg<3 x 2>
            return
        }
        }
        """
        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode("", 7, 1, 3)),)

        with pytest.raises(CompileError, match="Failed to convert type"):
            run_filecheck(program, pipeline)
