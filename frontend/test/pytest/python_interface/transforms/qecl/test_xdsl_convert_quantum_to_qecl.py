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

"""Test module for the convert-quantum-to-qecl dialect-conversion transform."""

import pennylane as qml
import pytest

from catalyst.python_interface.transforms.qecl import (
    ConvertQuantumToQecLogicalPass,
    convert_quantum_to_qecl_pass,
)
from catalyst.utils.exceptions import CompileError

pytestmark = pytest.mark.xdsl


@pytest.fixture(name="quantum_to_qecl_pipeline_k_1", scope="module")
def fixture_quantum_to_qecl_pipeline_k_1():
    return (ConvertQuantumToQecLogicalPass(k=1),)


@pytest.fixture(name="quantum_to_qecl_pipeline_k_2", scope="module")
def fixture_quantum_to_qecl_pipeline_k_2():
    return (ConvertQuantumToQecLogicalPass(k=2),)


# MARK: Test Op Conversion Patterns


@pytest.mark.filterwarnings("ignore:Unable to remove cast UnrealizedConversionCastOp")
class TestAllocPattern:
    """Unit tests for the `alloc` conversion pattern of the convert-quantum-to-qecl pass."""

    def test_alloc_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.alloc` ops are converted to their corresponding `qecl.alloc` ops for
        various initial numbers of qubits in the register.

        Perform the conversion for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: qecl.alloc() : !qecl.hyperreg<1 x 1>
            // CHECK-NOT: quantum.alloc
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: qecl.alloc() : !qecl.hyperreg<2 x 1>
            // CHECK-NOT: quantum.alloc
            %1 = quantum.alloc(2) : !quantum.reg

            // CHECK: qecl.alloc() : !qecl.hyperreg<3 x 1>
            // CHECK-NOT: quantum.alloc
            %2 = quantum.alloc(3) : !quantum.reg

            // CHECK: qecl.alloc() : !qecl.hyperreg<4 x 1>
            // CHECK-NOT: quantum.alloc
            %3 = quantum.alloc(4) : !quantum.reg

            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_alloc_with_use_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test the conversion of `quantum.alloc` ops includes the appropriate
        `unrealized_conversion_cast` ops when the quantum register has at least one use.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg:%.+]] = qecl.alloc() : !qecl.hyperreg<1 x 1>
            // CHECK: [[conv_cast_reg:%.+]] = builtin.unrealized_conversion_cast [[hreg]] : !qecl.hyperreg<1 x 1> to !quantum.reg
            // CHECK: "test.op"([[conv_cast_reg]]) : (!quantum.reg) -> !quantum.reg
            // CHECK-NOT: quantum.alloc
            %0 = quantum.alloc(1) : !quantum.reg
            %1 = "test.op"(%0) : (!quantum.reg) -> !quantum.reg

            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    @pytest.mark.xfail(reason="dynamic register size not supported", raises=NotImplementedError)
    def test_alloc_k_1_dyn(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.alloc` ops are converted to their corresponding `qecl.alloc` ops for a
        dynamic initial number of qubits in the register.

        Perform the conversion for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[n_cst:%.+]] = arith.constant 1 : i64
            %0 = arith.constant 1 : i64

            // CHECK: qecl.alloc([[n_cst]]) : !qecl.hyperreg<? x 1>
            // CHECK-NOT: quantum.alloc
            %1 = quantum.alloc(%0) : !quantum.reg

            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    @pytest.mark.xfail(reason="Only k = 1 is supported", raises=NotImplementedError)
    def test_alloc_k_2(self, run_filecheck, quantum_to_qecl_pipeline_k_2):
        """Test that `quantum.alloc` ops are converted to their corresponding `qecl.alloc` ops for
        various initial numbers of qubits in the register.

        Perform the conversion for k = 2.
        """
        program = """
        func.func @test_program() {
            // CHECK: qecl.alloc() : !qecl.hyperreg<1 x 2>
            // CHECK-NOT: quantum.alloc
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: qecl.alloc() : !qecl.hyperreg<1 x 2>
            // CHECK-NOT: quantum.alloc
            %1 = quantum.alloc(2) : !quantum.reg

            // CHECK: qecl.alloc() : !qecl.hyperreg<2 x 2>
            // CHECK-NOT: quantum.alloc
            %2 = quantum.alloc(3) : !quantum.reg

            // CHECK: qecl.alloc() : !qecl.hyperreg<2 x 2>
            // CHECK-NOT: quantum.alloc
            %3 = quantum.alloc(4) : !quantum.reg

            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_2)


@pytest.mark.filterwarnings("ignore:Unable to remove cast UnrealizedConversionCastOp")
class TestExtractPattern:
    """Unit tests for the `extract` conversion pattern of the convert-quantum-to-qecl pass."""

    def test_extract_width_1_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 1 and for k = 1. Also check that `qecl.encode[zero]` and
        `qecl.qec` ops are inserted after the `qecl.extract_block` op, since the codeblock is
        extracted immediately after a `qecl.alloc()` op.

        In this case, the extract index is static.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
            // CHECK: [[cb1:%.+]] = qecl.encode[zero] [[cb0]] : !qecl.codeblock<1>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb2]] : !qecl.codeblock<1> to !quantum.bit
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            %2 = "test.op"(%1) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_extract_width_1_dyn_idx_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 1 and for k = 1.

        In this case, the extract index is dynamic.
        """
        program = """
        func.func @test_program(%arg0 : i64) {
            // CHECK: [[hreg0:%.+]] = qecl.alloc
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: [[idx:%.+]] = arith.index_cast %arg0 : i64 to index
            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][[[idx]]] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
            // CHECK: [[cb1:%.+]] = qecl.encode[zero] [[cb0]] : !qecl.codeblock<1>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb2]] : !qecl.codeblock<1> to !quantum.bit
            %1 = quantum.extract %0[%arg0] : !quantum.reg -> !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            %2 = "test.op"(%1) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_extract_width_2_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 2 and for k = 1. Also check that `qecl.encode[zero]` and
        `qecl.qec` ops are inserted after each `qecl.extract_block` op.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = quantum.alloc(2) : !quantum.reg

            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0] : !qecl.hyperreg<2 x 1> -> !qecl.codeblock<1>
            // CHECK: [[cb1:%.+]] = qecl.encode[zero] [[cb0]] : !qecl.codeblock<1>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast1:%.+]] = builtin.unrealized_conversion_cast [[cb2]] : !qecl.codeblock<1> to !quantum.bit
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

            // CHECK: [[cb3:%.+]] = qecl.extract_block [[hreg0]][1] : !qecl.hyperreg<2 x 1> -> !qecl.codeblock<1>
            // CHECK: [[cb4:%.+]] = qecl.encode[zero] [[cb3]] : !qecl.codeblock<1>
            // CHECK: [[cb5:%.+]] = qecl.qec [[cb4]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast2:%.+]] = builtin.unrealized_conversion_cast [[cb5]] : !qecl.codeblock<1> to !quantum.bit
            %2 = quantum.extract %0[1] : !quantum.reg -> !quantum.bit

            // CHECK: "test.op"([[conv_cast1]]) : (!quantum.bit) -> !quantum.bit
            // CHECK: "test.op"([[conv_cast2]]) : (!quantum.bit) -> !quantum.bit
            %3 = "test.op"(%1) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            %4 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    @pytest.mark.xfail(reason="Not supported yet", raises=CompileError)
    def test_extract_width_1_multiple_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 1 and for k = 1, and that the `qecl.encode[zero]` and
        `qecl.qec` ops are only inserted after the initial alloc+extract_block op and not after
        subsequent extract_block ops.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
            // CHECK: [[cb1:%.+]] = qecl.encode[zero] [[cb0]] : !qecl.codeblock<1>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast1:%.+]] = builtin.unrealized_conversion_cast [[cb2]] : !qecl.codeblock<1> to !quantum.bit
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

            %2 = "test.op"(%0) : (!quantum.reg) -> !quantum.reg

            // CHECK: [[cb3:%.+]] = qecl.extract_block [[hreg1]][0] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
            // CHECK: [[conv_cast2:%.+]] = builtin.unrealized_conversion_cast [[cb3]] : !qecl.codeblock<1> to !quantum.bit
            // CHECK-NOT: qecl.encode
            // CHECK-NOT: qecl.qec
            %3 = quantum.extract %2[0] : !quantum.reg -> !quantum.bit

            // CHECK: "test.op"([[conv_cast1]]) : (!quantum.bit) -> !quantum.bit
            // CHECK: "test.op"([[conv_cast2]]) : (!quantum.bit) -> !quantum.bit
            %4 = "test.op"(%1) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            %5 = "test.op"(%3) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_extract_with_scf_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """TODO"""
        pass

    @pytest.mark.xfail(reason="Only k = 1 is supported", raises=NotImplementedError)
    def test_extract_width_1_k_2(self, run_filecheck, quantum_to_qecl_pipeline_k_2):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 1 and for k = 2.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0] : !qecl.hyperreg<1 x 2> -> !qecl.codeblock<2>
            // CHECK: [[cb1:%.+]] = qecl.encode[zero] [[cb0]] : !qecl.codeblock<2>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<2>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb2]] : !qecl.codeblock<1> to !quantum.bit
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            %2 = "test.op"(%1) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_2)

    @pytest.mark.xfail(reason="Only k = 1 is supported", raises=NotImplementedError)
    def test_extract_width_2_k_2(self, run_filecheck, quantum_to_qecl_pipeline_k_2):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 2 and for k = 2.

        Here, we extract two abstract qubits from the quantum register, but because k=2, we only
        extract a single codeblock from the allocated hyper-register at the QEC logical layer.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = quantum.alloc(2) : !quantum.reg

            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0] : !qecl.hyperreg<1 x 2> -> !qecl.codeblock<2>
            // CHECK: [[cb1:%.+]] = qecl.encode[zero] [[cb0]] : !qecl.codeblock<2>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<2>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb2]] : !qecl.codeblock<2> to !quantum.bit
            // CHECK-NOT: qecl.extract_block
            // CHECK-NOT: qecl.encode
            // CHECK-NOT: qecl.qec
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
            %2 = quantum.extract %0[1] : !quantum.reg -> !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            %3 = "test.op"(%1) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            %4 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_2)


@pytest.mark.filterwarnings("ignore:Unable to remove cast UnrealizedConversionCastOp")
class TestInsertPattern:
    """Unit tests for the `insert` conversion pattern of the convert-quantum-to-qecl pass."""

    def test_insert_width_1_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.insert` ops are converted to their corresponding `qecl.insert_block`
        ops for a registers with width = 1 and for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc
            %0 = quantum.alloc(1) : !quantum.reg
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

            // CHECK: [[hreg1:%.+]] = qecl.insert_block [[hreg0]][0], {{%.+}} : !qecl.hyperreg<1 x 1>, !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[hreg1]] : !qecl.hyperreg<1 x 1> to !quantum.reg
            %2 = quantum.insert %0[0], %1 : !quantum.reg, !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.reg) -> !quantum.reg
            %3 = "test.op"(%2) : (!quantum.reg) -> !quantum.reg  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)


class TestDeallocPattern:
    """Unit tests for the `dealloc` conversion pattern of the convert-quantum-to-qecl pass."""

    def test_dealloc_width_1_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.dealloc` ops are converted to their corresponding `qecl.dealloc` ops
        for a registers with width = 1 and for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: qecl.dealloc [[hreg0]]  : !qecl.hyperreg<1 x 1>
            quantum.dealloc %0 : !quantum.reg

            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_dealloc_width_2_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.dealloc` ops are converted to their corresponding `qecl.dealloc` ops
        for a registers with width = 2 and for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc() : !qecl.hyperreg<2 x 1>
            %0 = quantum.alloc(2) : !quantum.reg

            // CHECK: qecl.dealloc [[hreg0]]  : !qecl.hyperreg<2 x 1>
            quantum.dealloc %0 : !quantum.reg

            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    @pytest.mark.xfail(reason="Only k = 1 is supported", raises=NotImplementedError)
    def test_dealloc_width_1_k_2(self, run_filecheck, quantum_to_qecl_pipeline_k_2):
        """Test that `quantum.dealloc` ops are converted to their corresponding `qecl.dealloc` ops
        for a registers with width = 1 and for k = 2.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc() : !qecl.hyperreg<1 x 2>
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: qecl.dealloc [[hreg0]]  : !qecl.hyperreg<1 x 2>
            quantum.dealloc %0 : !quantum.reg

            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_2)


@pytest.mark.filterwarnings("ignore:Unable to remove cast UnrealizedConversionCastOp")
class TestGatePattern:
    """Unit tests for the `custom` (gate-op) conversion pattern of the convert-quantum-to-qecl
    pass.
    """

    def test_gate_hadamard_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that Hadamard gates (`quantum.custom "Hadamard"() ops) are converted to their
        corresponding `qecl.hadamard` ops for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: qecl.alloc
            // CHECK: qecl.extract_block
            // CHECK: qecl.encode[zero]
            // CHECK: [[cb0:%.+]] = qecl.qec
            %0 = quantum.alloc(1) : !quantum.reg
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

            // CHECK: [[cb1:%.+]] = qecl.hadamard [[cb0]][0] : !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb1]] : !qecl.codeblock<1> to !quantum.bit
            %2 = quantum.custom "Hadamard"() %1 : !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            %3 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_gate_s_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that S gates (`quantum.custom "S"() ops) are converted to their corresponding
        `qecl.s` ops for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: qecl.alloc
            // CHECK: qecl.extract_block
            // CHECK: qecl.encode[zero]
            // CHECK: [[cb0:%.+]] = qecl.qec
            %0 = quantum.alloc(1) : !quantum.reg
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

            // CHECK: [[cb1:%.+]] = qecl.s [[cb0]][0] : !qecl.codeblock<1>
            %2 = quantum.custom "S"() %1 : !quantum.bit

            // CHECK: [[cb2:%.+]] = qecl.s [[cb1]][0] adj : !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb2]] : !qecl.codeblock<1> to !quantum.bit
            %3 = quantum.custom "S"() %2 adj : !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            %4 = "test.op"(%3) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_gate_cnot_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that CNOT gates (`quantum.custom "CNOT"() ops) are converted to their corresponding
        `qecl.cnot` ops for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: qecl.alloc
            // CHECK: qecl.extract_block
            // CHECK: qecl.encode[zero]
            // CHECK: [[cb0:%.+]] = qecl.qec
            // CHECK: qecl.extract_block
            // CHECK: qecl.encode[zero]
            // CHECK: [[cb1:%.+]] = qecl.qec
            %0 = quantum.alloc(2) : !quantum.reg
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
            %2 = quantum.extract %0[1] : !quantum.reg -> !quantum.bit

            // CHECK: [[cb2:%.+]], [[cb3:%.+]] = qecl.cnot [[cb0]][0], [[cb1]][0] : !qecl.codeblock<1>, !qecl.codeblock<1>
            // CHECK: [[conv_cast1:%.+]] = builtin.unrealized_conversion_cast [[cb2]] : !qecl.codeblock<1> to !quantum.bit
            // CHECK: [[conv_cast2:%.+]] = builtin.unrealized_conversion_cast [[cb3]] : !qecl.codeblock<1> to !quantum.bit
            %3, %4 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit

            // CHECK: "test.op"([[conv_cast1]], [[conv_cast2]]) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
            %5, %6 = "test.op"(%3, %4) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)


@pytest.mark.filterwarnings("ignore:Unable to remove cast UnrealizedConversionCastOp")
class TestMeasurePattern:
    def test_measure_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that measurement ops (`quantum.measure`) are converted to their corresponding
        `qecl.measure` ops for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: qecl.alloc
            // CHECK: qecl.extract_block
            // CHECK: qecl.encode[zero]
            // CHECK: [[cb0:%.+]] = qecl.qec
            %0 = quantum.alloc(1) : !quantum.reg
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

            // CHECK: [[mres:%.+]], [[cb1:%.+]] = qecl.measure [[cb0]][0] : i1, !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb1]] : !qecl.codeblock<1> to !quantum.bit
            %mres, %2 = quantum.measure %1 : i1, !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            %3 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)


@pytest.mark.xfail(reason="Not supported yet")
class TestTodo:
    def test_conversion_with_scf_for_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """TODO"""
        program = """
        func.func @test_program() -> !quantum.reg {
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c0 = arith.constant 0 : index
            %0 = quantum.alloc( 1) : !quantum.reg
            %1 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %0) -> (!quantum.reg) {
                %4 = arith.constant 0.1 : f64
                %5 = quantum.extract %arg1[ 0] : !quantum.reg -> !quantum.bit
                %out_qubits = quantum.custom "RX"(%4) %5 : !quantum.bit
                %7 = quantum.insert %arg1[ 0], %out_qubits : !quantum.reg, !quantum.bit
                scf.yield %7 : !quantum.reg
            }
            return %1 : !quantum.reg
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)


# MARK: Integration Tests


class TestQuantumToQecLogicalPassIntegration:
    """Integration lit tests for the convert-quantum-to-qecl pass"""

    @pytest.mark.usefixtures("use_capture")
    def test_ghz_circuit(self, run_filecheck_qjit):
        dev = qml.device("null.qubit", wires=3)

        @qml.qjit(target="mlir")
        @convert_quantum_to_qecl_pass(k=1)
        @qml.qnode(dev, shots=1)
        def circuit():
            # CHECK: qecl.alloc() : !qecl.hyperreg<3 x 1>
            # CHECK: qecl.extract_block
            # CHECK: qecl.encode[zero]
            # CHECK: qecl.qec
            # CHECK: qecl.hadamard
            # CHECK: qecl.extract_block
            # CHECK: qecl.encode[zero]
            # CHECK: qecl.qec
            # CHECK: qecl.cnot
            # CHECK: qecl.extract_block
            # CHECK: qecl.encode[zero]
            # CHECK: qecl.qec
            # CHECK: qecl.cnot
            # CHECK: qecl.measure
            # CHECK: qecl.measure
            # CHECK: qecl.measure
            qml.H(0)
            qml.CNOT([0, 1])
            qml.CNOT([1, 2])
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            m2 = qml.measure(2)
            return qml.sample([m0, m1, m2])

        run_filecheck_qjit(circuit)
