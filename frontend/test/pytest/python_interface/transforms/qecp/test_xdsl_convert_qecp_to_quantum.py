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

from catalyst.python_interface.dialects import qecp
from catalyst.python_interface.dialects.quantum.attributes import QubitType
from catalyst.python_interface.transforms.qecp.convert_qecp_to_quantum import (
    ConvertQecPhysicalToQuantumPass,
    QecPhysicalQubitTypeConversion,
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


class TestQecPhysicalQubitTypeConversion:
    """Type conversion from !qecp.qubit<role> to !quantum.bit."""

    @pytest.mark.parametrize("role", ["data", "aux"])
    def test_physical_qubit_lowers_to_quantum_bit(self, run_filecheck, role):
        """Physical qubit types (data or aux) become !quantum.bit."""
        program = f"""
        builtin.module {{
        // CHECK-LABEL: test_qubit_{role}
        func.func @test_qubit_{role}() {{
            // CHECK: [[Q:%.+]] = "test.op"() : () -> !quantum.bit
            %0 = "test.op"() : () -> !qecp.qubit<{role}>
            // CHECK-NOT: !qecp.qubit<
            return
        }}
        }}
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))

    def test_qubit_use_chain(self, run_filecheck):
        """A converted qubit SSA value can flow into a second op with matching lowered types."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_qubit_chain
        func.func @test_qubit_chain() {
            // CHECK: [[Q0:%.+]] = "test.op"() : () -> !quantum.bit
            %0 = "test.op"() : () -> !qecp.qubit<data>
            // CHECK: [[Q1:%.+]] = "test.op"([[Q0:%.+]]) : (!quantum.bit) -> !quantum.bit
            %1 = "test.op"(%0) : (!qecp.qubit<data>) -> !qecp.qubit<data>
            return
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))

    def test_qubit_as_func_argument_and_return(self, run_filecheck):
        """Qubit types in argument and return positions both become !quantum.bit."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_qubit_sig
        func.func @test_qubit_sig(%arg0: !qecp.qubit<data>) -> !qecp.qubit<data> {
            // CHECK: [[R:%.+]] = "test.op"({{.+}}) : (!quantum.bit) -> !quantum.bit
            %0 = "test.op"(%arg0) : (!qecp.qubit<data>) -> !qecp.qubit<data>
            return %0 : !qecp.qubit<data>
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))


class TestQecPhysicalQubitTypeConversionPatternUnit:
    """Direct unit tests on `QecPhysicalQubitTypeConversion.convert_type`."""

    @pytest.mark.parametrize("role", ["data", "aux"])
    def test_convert_type_returns_qubit_type(self, role):
        """`convert_type` maps any `QecPhysicalQubitType` to a bare `QubitType`."""
        pattern = QecPhysicalQubitTypeConversion()
        out = pattern.convert_type(qecp.QecPhysicalQubitType(role))
        assert isinstance(out, QubitType)
        assert out == QubitType()

    def test_convert_type_ignores_role_in_result_type(self):
        """Lowering is lossy: `!quantum.bit` does not encode data vs aux."""
        p = QecPhysicalQubitTypeConversion()
        data = p.convert_type(qecp.QecPhysicalQubitType("data"))
        aux = p.convert_type(qecp.QecPhysicalQubitType("aux"))
        assert data == aux
