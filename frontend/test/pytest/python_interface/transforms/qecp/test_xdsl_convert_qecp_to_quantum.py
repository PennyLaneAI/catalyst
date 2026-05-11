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

"""Tests for the convert-qecp-to-quantum xDSL dialect-conversion pass."""

import pytest

from catalyst.python_interface.transforms.qecp.convert_qecp_to_quantum import (
    ConvertQecPhysicalToQuantumPass,
    convert_qecp_to_quantum_pass,
)

# pylint: disable=line-too-long

pytestmark = pytest.mark.xdsl


class TestConvertQecPhysicalToQuantumPass:
    """Unit tests for ConvertQecPhysicalToQuantumPass."""

    def test_pass_name(self):
        """The module pass reports a stable pipeline name."""
        assert ConvertQecPhysicalToQuantumPass.name == "convert-qecp-to-quantum"

    def test_compiler_transform_wrapper(self):
        """The compiler_transform wrapper is defined and callable."""
        assert callable(convert_qecp_to_quantum_pass)


class TestPhysicalCodeblockTypeConversion:
    """Type conversion from !qecp.codeblock<k x n> to !quantum.reg."""

    @pytest.mark.parametrize("k", [1])
    @pytest.mark.parametrize("n", [1, 3, 7])
    def test_codeblock_lowers_to_single_quantum_reg(self, run_filecheck, k, n):
        """Every physical codeblock type becomes one !quantum.reg (k, n are not encoded in the type)."""
        program = f"""
        builtin.module {{
        // CHECK-LABEL: test_program
        func.func @test_program() {{
            // CHECK: [[V:%.+]] = "test.op"() : () -> !quantum.reg
            %0 = "test.op"() : () -> !qecp.codeblock<{k} x {n}>
            // CHECK-NOT: !qecp.codeblock<
            return
        }}
        }}
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))

    def test_codeblock_use_chain(self, run_filecheck):
        """A converted codeblock SSA value can flow into a second op with matching lowered types."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_program
        func.func @test_program() {
            // CHECK: [[CB0:%.+]] = "test.op"() : () -> !quantum.reg
            %0 = "test.op"() : () -> !qecp.codeblock<1 x 2>
            // CHECK: [[CB1:%.+]] = "test.op"([[CB0:%.+]]) : (!quantum.reg) -> !quantum.reg
            %1 = "test.op"(%0) : (!qecp.codeblock<1 x 2>) -> !qecp.codeblock<1 x 2>
            return
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))

    def test_codeblock_as_func_argument_and_return(self, run_filecheck):
        """Codeblock types in argument and return positions both become !quantum.reg."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_sig
        func.func @test_sig(%arg0: !qecp.codeblock<1 x 2>) -> !qecp.codeblock<1 x 2> {
            // CHECK: [[R:%.+]] = "test.op"({{.+}}) : (!quantum.reg) -> !quantum.reg
            %0 = "test.op"(%arg0) : (!qecp.codeblock<1 x 2>) -> !qecp.codeblock<1 x 2>
            return %0 : !qecp.codeblock<1 x 2>
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))
