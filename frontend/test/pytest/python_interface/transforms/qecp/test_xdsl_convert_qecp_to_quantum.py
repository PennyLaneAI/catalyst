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

    @pytest.mark.parametrize("n", [1, 3, 7])
    def test_codeblock_to_n_qubits_k1(self, run_filecheck, n):
        """For k = 1, each physical codeblock type becomes n quantum bit types (n from PhysicalCodeblockType)."""
        k = 1
        program = f"""
        builtin.module {{
        // CHECK-LABEL: test_program
        func.func @test_program() {{
            // CHECK: !quantum.reg
            %0 = "test.op"() : () -> !qecp.codeblock<{k} x {n}>
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


class TestAllocationConversion:
    """Lowering of qecp.alloc (+ qecp.extract_block) to quantum.alloc."""

    @pytest.mark.parametrize("n", [3, 7])
    def test_alloc_width_one_with_extract(self, run_filecheck, n):
        """A width-1 hyper-register becomes one quantum.alloc; extract_block is removed."""
        k = 1
        program = f"""
        builtin.module {{
        // CHECK-LABEL: test_alloc_one
        func.func @test_alloc_one() {{
            // CHECK-NOT: qecp.alloc
            // CHECK-NOT: qecp.extract_block
            // CHECK: [[QREG:%.+]] = quantum.alloc
            %h = qecp.alloc() : !qecp.hyperreg<1 x {k} x {n}>
            %cb = qecp.extract_block %h[0] : !qecp.hyperreg<1 x {k} x {n}> -> !qecp.codeblock<{k} x {n}>
            // CHECK: [[q0:%.+]] = qecp.extract [[QREG:%.+]][0] : !quantum.reg -> !quantum.bit
            %qubit = qecp.extract %cb[0] : !qecp.codeblock<{k} x {n}> -> !qecp.qubit<data>
            // CHECK: [[q1:%.+]] = qecp.hadamard [[q0:%.+]] : !quantum.bit
            %0 = qecp.hadamard %qubit : !qecp.qubit<data>
            return
        }}
        }}
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))

    @pytest.mark.parametrize("n", [3])
    def test_alloc_width_two_two_extracts(self, run_filecheck, n):
        """Width 2 produces two quantum.alloc ops; each extract index maps to its register."""
        k = 1
        program = f"""
        builtin.module {{
        // CHECK-LABEL: test_alloc_two
        func.func @test_alloc_two() {{
            // CHECK-NOT: qecp.extract_block
            // CHECK: [[QREG0:%.+]] = quantum.alloc
            // CHECK-NEXT: [[QREG1:%.+]] = quantum.alloc
            %h = qecp.alloc() : !qecp.hyperreg<2 x {k} x {n}>
            %cb = qecp.extract_block %h[0] : !qecp.hyperreg<2 x {k} x {n}> -> !qecp.codeblock<{k} x {n}>
            // CHECK: [[q0:%.+]] = qecp.extract [[QREG0:%.+]][0] : !quantum.reg -> !quantum.bit
            %q0 = qecp.extract %cb[0] : !qecp.codeblock<{k} x {n}> -> !qecp.qubit<data>
            // CHECK: [[q1:%.+]] = qecp.extract [[QREG0:%.+]][1] : !quantum.reg -> !quantum.bit
            %q1 = qecp.extract %cb[1] : !qecp.codeblock<{k} x {n}> -> !qecp.qubit<data>
            // CHECK: [[q2:%.+]], [[q3:%.+]] = qecp.cnot [[q0:%.+]], [[q1:%.+]] : !quantum.bit, !quantum.bit
            %0, %1 = qecp.cnot %q0, %q1 : !qecp.qubit<data>, !qecp.qubit<data>
            return
        }}
        }}
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))

    def test_alloc_unused_hyper_reg(self, run_filecheck):
        """Alloc with no extract uses is erased after inserting quantum.alloc ops."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_alloc_unused
        func.func @test_alloc_unused() {
            // CHECK: quantum.alloc
            // CHECK-NOT: qecp.alloc
            %h = qecp.alloc() : !qecp.hyperreg<1 x 1 x 2>
            return
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))
