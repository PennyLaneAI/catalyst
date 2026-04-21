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

# pylint: disable=line-too-long

import pytest

from catalyst.python_interface.transforms.qecp import (
    ConvertQecLogicalToQecPhysicalPass,
)
from catalyst.python_interface.transforms.qecp.qec_code_lib import QecCode
from catalyst.utils.exceptions import CompileError

pytestmark = pytest.mark.xdsl


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
