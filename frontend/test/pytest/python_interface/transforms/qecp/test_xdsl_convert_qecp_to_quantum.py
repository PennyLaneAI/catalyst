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


class TestAuxAllocDeallocConversion:
    """Lowering of qecp.alloc_aux / qecp.dealloc_aux to quantum.alloc_qb / quantum.dealloc_qb."""

    def test_aux_alloc_dealloc(self, run_filecheck):
        """Auxiliary qubit allocation and deallocation map to quantum alloc_qb/dealloc_qb ops."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_aux_alloc_dealloc
        func.func @test_aux_alloc_dealloc() {
            // CHECK: [[q0:%.+]] = quantum.alloc_qb
            %0 = qecp.alloc_aux : !qecp.qubit<aux>
            // CHECK: quantum.dealloc_qb [[q0]]
            qecp.dealloc_aux %0 : !qecp.qubit<aux>
            // CHECK-NOT: qecp.alloc_aux
            // CHECK-NOT: qecp.dealloc_aux
            return
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))

    def test_convert_aux_operands(self, run_filecheck):
        """Multiple qcp.qubit<aux> types as operands are converted to quantum.bit."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_two_aux
        func.func @test_two_aux() {
            // CHECK-COUNT-2: quantum.alloc_qb
            // CHECK-NOT: qecp.alloc_aux
            %a = qecp.alloc_aux : !qecp.qubit<aux>
            %b = qecp.alloc_aux : !qecp.qubit<aux>
            // CHECK-NOT: qecp.qubit<aux>
            // CHECK: "test.op"({{%.+}}, {{%.+}}) : (!quantum.bit, !quantum.bit) -> ()
            "test.op"(%a, %b) : (!qecp.qubit<aux>, !qecp.qubit<aux>) -> ()
            return
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))


class TestExtractInsertQubitConversion:
    """Lowering of qecp.extract / qecp.insert to quantum.extract / quantum.insert."""

    def test_extract_lowering(self, run_filecheck):
        """A qecp.extract lowers to quantum.extract."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_extract
        func.func @test_extract() {
            // CHECK: [[REG:%.+]] = "test.op"() : () -> !quantum.reg
            %cb = "test.op"() : () -> !qecp.codeblock<1 x 4>
            // CHECK: [[q0:%.+]] = quantum.extract{{.*}}[0] : !quantum.reg -> !quantum.bit
            %q = qecp.extract %cb[0] : !qecp.codeblock<1 x 4> -> !qecp.qubit<data>
            // CHECK-NOT: qecp.extract
            // CHECK: [[q1:%.+]] = quantum.custom "PauliX"() [[q0:%.+]] : !quantum.bit
            %q1 = qecp.x %q : !qecp.qubit<data>
            // CHECK: [[mres:%.+]], [[q2:%.+]] = quantum.measure [[q1:%.+]] : i1, !quantum.bit
            %mres, %q2 = qecp.measure %q1 : i1, !qecp.qubit<data>
            return
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))

    def test_insert_lowering(self, run_filecheck):
        """A qecp.insert lowers to quantum.insert."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_insert_lowering
        func.func @test_insert_lowering() {
            %cb = "test.op"() : () -> !qecp.codeblock<1 x 2>
            %q0 = qecp.extract %cb[0] : !qecp.codeblock<1 x 2> -> !qecp.qubit<data>
            // CHECK: [[q1:%.+]] = quantum.extract {{%.+}}[1]
            %q1 = qecp.extract %cb[1] : !qecp.codeblock<1 x 2> -> !qecp.qubit<data>
            // CHECK: [[q2:%.+]] = quantum.custom "Hadamard"() [[q1:%.+]] : !quantum.bit
            %q2 = qecp.hadamard %q1 : !qecp.qubit<data>
            // CHECK: [[mres:%.+]], [[q3:%.+]] = quantum.measure [[q2:%.+]] : i1, !quantum.bit
            %mres, %q3 = qecp.measure %q2 : i1, !qecp.qubit<data>
            // CHECK: [[CB2:%.+]] = quantum.insert {{%.+}}[0], [[q3:%.+]] : !quantum.reg, !quantum.bit
            %cb2 = qecp.insert %cb[0], %q3 : !qecp.codeblock<1 x 2>, !qecp.qubit<data>
            return %cb2 : !qecp.codeblock<1 x 2>
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))


class TestGateMeasureConversion:
    """Lowering of gate and measurement operations in qecp to quantum."""

    def test_hadamard_lowering(self, run_filecheck):
        """qecp.hadamard lowers to quantum.custom "Hadamard"."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_hadamard
        func.func @test_hadamard() {
            %cb = "test.op"() : () -> !qecp.codeblock<1 x 1>
            %q0 = qecp.extract %cb[0] : !qecp.codeblock<1 x 1> -> !qecp.qubit<data>
            // CHECK: [[q1:%.+]] = quantum.custom "Hadamard"() [[q0:%.+]] : !quantum.bit
            %q1 = qecp.hadamard %q0 : !qecp.qubit<data>
            return %q1 : !qecp.qubit<data>
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))

    def test_paulix_lowering(self, run_filecheck):
        """qecp.x lowers to quantum.custom "PauliX"."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_pauli_x
        func.func @test_pauli_x() {
            %cb = "test.op"() : () -> !qecp.codeblock<1 x 1>
            %q0 = qecp.extract %cb[0] : !qecp.codeblock<1 x 1> -> !qecp.qubit<data>
            // CHECK: [[q1:%.+]] = quantum.custom "PauliX"() [[q0:%.+]] : !quantum.bit
            %q1 = qecp.x %q0 : !qecp.qubit<data>
            return %q1 : !qecp.qubit<data>
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))

    def test_pauliy_lowering(self, run_filecheck):
        """qecp.y lowers to quantum.custom "PauliY"."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_pauli_y
        func.func @test_pauli_y() {
            %cb = "test.op"() : () -> !qecp.codeblock<1 x 1>
            %q0 = qecp.extract %cb[0] : !qecp.codeblock<1 x 1> -> !qecp.qubit<data>
            // CHECK: [[q1:%.+]] = quantum.custom "PauliY"() [[q0:%.+]] : !quantum.bit
            %q1 = qecp.y %q0 : !qecp.qubit<data>
            return %q1 : !qecp.qubit<data>
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))

    def test_pauliz_lowering(self, run_filecheck):
        """qecp.z lowers to quantum.custom "PauliZ"."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_pauli_z
        func.func @test_pauli_z() {
            %cb = "test.op"() : () -> !qecp.codeblock<1 x 1>
            %q0 = qecp.extract %cb[0] : !qecp.codeblock<1 x 1> -> !qecp.qubit<data>
            // CHECK: [[q1:%.+]] = quantum.custom "PauliZ"() [[q0:%.+]] : !quantum.bit
            %q1 = qecp.z %q0 : !qecp.qubit<data>
            return %q1 : !qecp.qubit<data>
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))

    def test_identity_lowering(self, run_filecheck):
        """qecp.i lowers to quantum.custom "Identity"."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_identity
        func.func @test_identity() {
            %cb = "test.op"() : () -> !qecp.codeblock<1 x 1>
            %q0 = qecp.extract %cb[0] : !qecp.codeblock<1 x 1> -> !qecp.qubit<data>
            // CHECK: [[q1:%.+]] = quantum.custom "Identity"() [[q0:%.+]] : !quantum.bit
            %q1 = qecp.identity %q0 : !qecp.qubit<data>
            return %q1 : !qecp.qubit<data>
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))

    def test_s_lowering(self, run_filecheck):
        """qecp.s lowers to quantum.custom "S"."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_s
        func.func @test_s() {
            %cb = "test.op"() : () -> !qecp.codeblock<1 x 1>
            %q0 = qecp.extract %cb[0] : !qecp.codeblock<1 x 1> -> !qecp.qubit<data>
            // CHECK: [[q1:%.+]] = quantum.custom "S"() [[q0:%.+]] : !quantum.bit
            %q1 = qecp.s %q0 : !qecp.qubit<data>
            return %q1 : !qecp.qubit<data>
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))

    def test_cnot_lowering(self, run_filecheck):
        """qecp.cnot lowers to quantum.custom "CNOT"."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_cnot
        func.func @test_cnot() {
            %cb = "test.op"() : () -> !qecp.codeblock<1 x 2>
            %q0 = qecp.extract %cb[0] : !qecp.codeblock<1 x 2> -> !qecp.qubit<data>
            %q1 = qecp.extract %cb[1] : !qecp.codeblock<1 x 2> -> !qecp.qubit<data>
            // CHECK: [[q2:%.+]], [[q3:%.+]] = quantum.custom "CNOT"() [[q0:%.+]], [[q1:%.+]] : !quantum.bit, !quantum.bit
            %q2, %q3 = qecp.cnot %q0, %q1 : !qecp.qubit<data>, !qecp.qubit<data>
            // CHECK-NEXT: return [[q2:%.+]] : !quantum.bit
            return %q2 : !qecp.qubit<data>
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))

    def test_rot_conversion(self, run_filecheck):
        """qecp.rot lowers to quantum.custom "Rot"."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_noise_rot
        func.func @test_noise_rot() {
            // CHECK: [[reg:%.+]] = "test.op"() : () -> !quantum.reg
            %cb = "test.op"() : () -> !qecp.codeblock<1 x 1>
            %phi = "test.op"() : () -> f64
            %theta = "test.op"() : () -> f64
            %omega = "test.op"() : () -> f64
            // CHECK: [[q0:%.+]] = quantum.extract [[reg:%.+]][0] : !quantum.reg -> !quantum.bit
            %q0 = qecp.extract %cb[0] : !qecp.codeblock<1 x 1> -> !qecp.qubit<data>
            // CHECK-NEXT: [[q1:%.+]] = quantum.custom "Rot"({{.*}}, {{.*}}, {{.*}}) [[q0:%.+]] : !quantum.bit
            %q1 = qecp.rot(%phi, %theta, %omega) %q0 : !qecp.qubit<data>
            // CHECK-NEXT: return [[q1:%.+]] : !quantum.bit
            return %q1 : !qecp.qubit<data>
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))


class TestSubroutineConversion:
    """Lowering of subroutine funcOp and call ops with qecp types to quantum types."""

    def test_subroutine_qecp_codeblock_conversion(self, run_filecheck):
        """A subroutine with qecp.codeblock types in its signature lowers to a quantum.reg type."""
        program = """
        builtin.module {
        // CHECK-NOT: !qecp.codeblock
        // CHECK-NOT: !qecp.qubit
        // CHECK-LABEL: test_subroutine(%cb: !quantum.reg) -> !quantum.reg
        func.func @test_subroutine(%cb: !qecp.codeblock<1 x 1>) -> !qecp.codeblock<1 x 1> {
            // CHECK: [[q0:%.+]] = quantum.extract {{%.+}}[0] : !quantum.reg -> !quantum.bit
            %q0 = qecp.extract %cb[0] : !qecp.codeblock<1 x 1> -> !qecp.qubit<data>
            // CHECK-NEXT: [[q1:%.+]] = quantum.custom "Hadamard"() [[q0:%.+]] : !quantum.bit
            %q1 = qecp.hadamard %q0 : !qecp.qubit<data>
            // CHECK-NEXT: [[cb1:%.+]] = quantum.insert {{%.+}}[0], [[q1:%.+]] : !quantum.reg, !quantum.bit
            %cb1 = qecp.insert %cb[0], %q1 : !qecp.codeblock<1 x 1>, !qecp.qubit<data>
            // CHECK-NEXT: [[cb1:%.+]] : !quantum.reg
            return %cb1 : !qecp.codeblock<1 x 1>
        }

        // CHECK-LABEL: test_caller() -> !quantum.reg
        func.func @test_caller() -> !qecp.codeblock<1 x 1> {
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !quantum.reg
            %cb0 = "test.op"() : () -> !qecp.codeblock<1 x 1>
            // CHECK: [[cb1:%.+]] = func.call @test_subroutine([[cb0:%.+]]) : (!quantum.reg) -> !quantum.reg
            %cb1 = func.call @test_subroutine(%cb0) : (!qecp.codeblock<1 x 1>) -> !qecp.codeblock<1 x 1>
            return %cb1 : !qecp.codeblock<1 x 1>
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))

    def test_subroutine_qecp_qubit_conversion(self, run_filecheck):
        """A subroutine with qecp.qubit types in its signature lowers to quantum.bit types."""
        program = """
        builtin.module {
        // CHECK-NOT: !qecp.codeblock
        // CHECK-NOT: !qecp.qubit
        // CHECK-LABEL: test_subroutine(%q: !quantum.bit) -> !quantum.bit
        func.func @test_subroutine(%q: !qecp.qubit<data>) -> !qecp.qubit<data> {
            // CHECK: [[q1:%.+]] = quantum.custom "Hadamard"() [[q:%.+]] : !quantum.bit
            %q1 = qecp.hadamard %q : !qecp.qubit<data>
            // CHECK-NEXT: [[q1:%.+]] : !quantum.bit
            return %q1 : !qecp.qubit<data>
        }
        // CHECK-LABEL: test_caller() -> !quantum.bit
        func.func @test_caller() -> !qecp.qubit<data> {
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !quantum.reg
            %cb0 = "test.op"() : () -> !qecp.codeblock<1 x 1>
            // CHECK: [[q0:%.+]] = quantum.extract [[cb0:%.+]][0] : !quantum.reg -> !quantum.bit
            %q0 = qecp.extract %cb0[0] : !qecp.codeblock<1 x 1> -> !qecp.qubit<data>
            // CHECK: [[q1:%.+]] = func.call @test_subroutine([[q0:%.+]]) : (!quantum.bit) -> !quantum.bit
            %q1 = func.call @test_subroutine(%q0) : (!qecp.qubit<data>) -> !qecp.qubit<data>
            // CHECK-NEXT: [[q1:%.+]] : !quantum.bit
            return %q1 : !qecp.qubit<data>
        }
        }
        """
        run_filecheck(program, (ConvertQecPhysicalToQuantumPass(),))
