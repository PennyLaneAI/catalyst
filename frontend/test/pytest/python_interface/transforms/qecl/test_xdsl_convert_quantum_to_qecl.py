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

import pennylane as qp
import pytest

from catalyst.python_interface.transforms.qecl import (
    ConvertQuantumToQecLogicalPass,
    convert_quantum_to_qecl_pass,
)
from catalyst.utils.exceptions import CompileError

pytestmark = pytest.mark.xdsl


@pytest.fixture(name="quantum_to_qecl_pipeline_k_1", scope="module")
def fixture_quantum_to_qecl_pipeline_k_1():
    """Fixture that returns the compilation pipeline containing the convert-quantum-to-qecl with
    k = 1.
    """
    return (ConvertQuantumToQecLogicalPass(k=1),)


@pytest.fixture(name="quantum_to_qecl_pipeline_k_2", scope="module")
def fixture_quantum_to_qecl_pipeline_k_2():
    """Fixture that returns the compilation pipeline containing the convert-quantum-to-qecl with
    k = 2.
    """
    return (ConvertQuantumToQecLogicalPass(k=2),)


# MARK: TestAllocPattern


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
            // CHECK: [[hreg10:%.+]] = qecl.alloc() : !qecl.hyperreg<1 x 1>
            // CHECK: [[cb10:%.+]] = qecl.extract_block [[hreg10]][0]
            // CHECK: [[cb11:%.+]] = qecl.encode[zero] [[cb10]]
            // CHECK: [[hreg11:%.+]] = qecl.insert_block [[hreg10]][0], [[cb11]]
            // CHECK-NOT: quantum.alloc
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: [[hreg20:%.+]] = qecl.alloc() : !qecl.hyperreg<2 x 1>
            // CHECK: [[lb:%.+]] = arith.constant 0 : index
            // CHECK: [[ub:%.+]] = arith.constant 2 : index
            // CHECK: [[step:%.+]] = arith.constant 1 : index
            // CHECK: [[hreg21:%.+]] = scf.for [[idx:%.+]] = [[lb]] to [[ub]] step [[step]] iter_args([[hreg2arg:%.+]] = [[hreg20]])
            // CHECK:     [[cb20:%.+]] = qecl.extract_block [[hreg2arg]][[[idx]]]
            // CHECK:     [[cb21:%.+]] = qecl.encode[zero] [[cb20]]
            // CHECK:     [[hreg22:%.+]] = qecl.insert_block [[hreg2arg]][[[idx]]], [[cb21]]
            // CHECK:     scf.yield [[hreg22]]
            // CHECK: }
            // CHECK-NOT: quantum.alloc
            %1 = quantum.alloc(2) : !quantum.reg

            // CHECK: [[hreg30:%.+]] = qecl.alloc() : !qecl.hyperreg<3 x 1>
            // CHECK: [[lb:%.+]] = arith.constant 0 : index
            // CHECK: [[ub:%.+]] = arith.constant 3 : index
            // CHECK: [[step:%.+]] = arith.constant 1 : index
            // CHECK: [[hreg31:%.+]] = scf.for [[idx:%.+]] = [[lb]] to [[ub]] step [[step]] iter_args([[hreg3arg:%.+]] = [[hreg30]])
            // CHECK:     [[cb30:%.+]] = qecl.extract_block [[hreg3arg]][[[idx]]]
            // CHECK:     [[cb31:%.+]] = qecl.encode[zero] [[cb30]]
            // CHECK:     [[hreg32:%.+]] = qecl.insert_block [[hreg3arg]][[[idx]]], [[cb31]]
            // CHECK:     scf.yield [[hreg32]]
            // CHECK: }
            // CHECK-NOT: quantum.alloc
            %2 = quantum.alloc(3) : !quantum.reg

            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_alloc_k_1_with_use(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test the conversion of `quantum.alloc` ops includes the appropriate
        `unrealized_conversion_cast` ops when the quantum register has at least one use.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc() : !qecl.hyperreg<1 x 1>
            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0]
            // CHECK: [[cb1:%.+]] = qecl.encode[zero] [[cb0]]
            // CHECK: [[hreg1:%.+]] = qecl.insert_block [[hreg0]][0], [[cb1]]
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[hreg1]] : !qecl.hyperreg<1 x 1> to !quantum.reg
            // CHECK: "test.op"([[conv_cast]]) : (!quantum.reg) -> !quantum.reg
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

            // CHECK-NOT: quantum.alloc
            // CHECK: qecl.alloc([[n_cst]]) : !qecl.hyperreg<? x 1>
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


# MARK: TestExtractPattern


@pytest.mark.filterwarnings("ignore:Unable to remove cast UnrealizedConversionCastOp")
class TestExtractPattern:
    """Unit tests for the `extract` conversion pattern of the convert-quantum-to-qecl pass."""

    def test_extract_k_1_width_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 1 and for k = 1. Also check that a `qecl.qec` op is inserted
        after the `qecl.extract_block` op.

        In this case, the extract index is static.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = "test.op"() : () -> !qecl.hyperreg<1 x 1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.hyperreg<1 x 1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.hyperreg<1 x 1> to !quantum.reg

            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
            // CHECK: [[cb1:%.+]] = qecl.qec [[cb0]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb1]] : !qecl.codeblock<1> to !quantum.bit
            %2 = quantum.extract %1[0] : !quantum.reg -> !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            %3 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_extract_k_1_width_1_dyn_idx(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 1 and for k = 1.

        In this case, the extract index is dynamic.
        """
        program = """
        func.func @test_program(%arg0 : i64) {
            // CHECK: [[hreg0:%.+]] = "test.op"() : () -> !qecl.hyperreg<1 x 1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.hyperreg<1 x 1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.hyperreg<1 x 1> to !quantum.reg

            // CHECK: [[idx:%.+]] = arith.index_cast %arg0 : i64 to index
            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][[[idx]]] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
            // CHECK: [[cb1:%.+]] = qecl.qec [[cb0]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb1]] : !qecl.codeblock<1> to !quantum.bit
            %2 = quantum.extract %1[%arg0] : !quantum.reg -> !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            %3 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_extract_k_1_width_2(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 2 and for k = 1. Also check that a `qecl.qec` op is inserted
        after each `qecl.extract_block` op.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = "test.op"() : () -> !qecl.hyperreg<2 x 1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.hyperreg<2 x 1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.hyperreg<2 x 1> to !quantum.reg

            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0] : !qecl.hyperreg<2 x 1> -> !qecl.codeblock<1>
            // CHECK: [[cb1:%.+]] = qecl.qec [[cb0]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast1:%.+]] = builtin.unrealized_conversion_cast [[cb1]] : !qecl.codeblock<1> to !quantum.bit
            %2 = quantum.extract %1[0] : !quantum.reg -> !quantum.bit

            // CHECK: [[cb2:%.+]] = qecl.extract_block [[hreg0]][1] : !qecl.hyperreg<2 x 1> -> !qecl.codeblock<1>
            // CHECK: [[cb3:%.+]] = qecl.qec [[cb2]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast2:%.+]] = builtin.unrealized_conversion_cast [[cb3]] : !qecl.codeblock<1> to !quantum.bit
            %3 = quantum.extract %1[1] : !quantum.reg -> !quantum.bit

            // CHECK: "test.op"([[conv_cast1]]) : (!quantum.bit) -> !quantum.bit
            // CHECK: "test.op"([[conv_cast2]]) : (!quantum.bit) -> !quantum.bit
            %4 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            %5 = "test.op"(%3) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    @pytest.mark.xfail(reason="Not supported yet", raises=CompileError)
    def test_extract_k_1_width_1_multiple(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 1 and for k = 1. In this case there are multiple extract
        operations interleaved with other ops.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = "test.op"() : () -> !qecl.hyperreg<1 x 1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.hyperreg<1 x 1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.hyperreg<1 x 1> to !quantum.reg

            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
            // CHECK: [[cb1:%.+]] = qecl.qec [[cb0]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast1:%.+]] = builtin.unrealized_conversion_cast [[cb1]] : !qecl.codeblock<1> to !quantum.bit
            %2 = quantum.extract %1[0] : !quantum.reg -> !quantum.bit

            %3 = "test.op"(%1) : (!quantum.reg) -> !quantum.reg

            // CHECK: [[cb2:%.+]] = qecl.extract_block [[hreg1]][0] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
            // CHECK: [[conv_cast2:%.+]] = builtin.unrealized_conversion_cast [[cb2]] : !qecl.codeblock<1> to !quantum.bit
            // CHECK-NOT: qecl.encode
            // CHECK-NOT: qecl.qec
            %4 = quantum.extract %3[0] : !quantum.reg -> !quantum.bit

            // CHECK: "test.op"([[conv_cast1]]) : (!quantum.bit) -> !quantum.bit
            // CHECK: "test.op"([[conv_cast2]]) : (!quantum.bit) -> !quantum.bit
            %5 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            %6 = "test.op"(%4) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    @pytest.mark.xfail(reason="Only k = 1 is supported", raises=NotImplementedError)
    def test_extract_k_2_width_1(self, run_filecheck, quantum_to_qecl_pipeline_k_2):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 1 and for k = 2.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = "test.op"() : () -> !qecl.hyperreg<1 x 2>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.hyperreg<1 x 2>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.hyperreg<1 x 2> to !quantum.reg

            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0] : !qecl.hyperreg<1 x 2> -> !qecl.codeblock<2>
            // CHECK: [[cb1:%.+]] = qecl.qec [[cb0]] : !qecl.codeblock<2>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb1]] : !qecl.codeblock<1> to !quantum.bit
            %2 = quantum.extract %1[0] : !quantum.reg -> !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            %3 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_2)

    @pytest.mark.xfail(reason="Only k = 1 is supported", raises=NotImplementedError)
    def test_extract_k_2_width_2(self, run_filecheck, quantum_to_qecl_pipeline_k_2):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 2 and for k = 2.

        Here, we extract two abstract qubits from the quantum register, but because k=2, we only
        extract a single codeblock from the allocated hyper-register at the QEC logical layer.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = "test.op"() : () -> !qecl.hyperreg<2 x 2>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.hyperreg<2 x 2>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.hyperreg<2 x 2> to !quantum.reg

            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0] : !qecl.hyperreg<2 x 2> -> !qecl.codeblock<2>
            // CHECK: [[cb1:%.+]] = qecl.qec [[cb0]] : !qecl.codeblock<2>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb1]] : !qecl.codeblock<2> to !quantum.bit
            // CHECK-NOT: qecl.extract_block
            // CHECK-NOT: qecl.qec
            %2 = quantum.extract %1[0] : !quantum.reg -> !quantum.bit
            %3 = quantum.extract %1[1] : !quantum.reg -> !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            %4 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            %5 = "test.op"(%3) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_2)


# MARK: TestInsertPattern


@pytest.mark.filterwarnings("ignore:Unable to remove cast UnrealizedConversionCastOp")
class TestInsertPattern:
    """Unit tests for the `insert` conversion pattern of the convert-quantum-to-qecl pass."""

    def test_insert_k_1_width_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.insert` ops are converted to their corresponding `qecl.insert_block`
        ops for a registers with width = 1 and for k = 1.

        In this case, the insertion index is static.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = "test.op"() : () -> !qecl.hyperreg<1 x 1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.hyperreg<1 x 1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.hyperreg<1 x 1> to !quantum.reg

            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecl.codeblock<1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %2 = "test.op"() : () -> !qecl.codeblock<1>
            %3 = builtin.unrealized_conversion_cast %2 : !qecl.codeblock<1> to !quantum.bit

            // CHECK: [[hreg1:%.+]] = qecl.insert_block [[hreg0]][0], [[cb0]] : !qecl.hyperreg<1 x 1>, !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[hreg1]] : !qecl.hyperreg<1 x 1> to !quantum.reg
            %4 = quantum.insert %1[0], %3 : !quantum.reg, !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.reg) -> !quantum.reg
            %5 = "test.op"(%4) : (!quantum.reg) -> !quantum.reg  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_insert_k_1_width_1_dyn_idx(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.insert` ops are converted to their corresponding `qecl.insert_block`
        ops for a registers with width = 1 and for k = 1.

        In this case, the insertion index is dynamic.
        """
        program = """
        func.func @test_program(%arg0 : i64) {
            // CHECK: [[hreg0:%.+]] = "test.op"() : () -> !qecl.hyperreg<1 x 1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.hyperreg<1 x 1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.hyperreg<1 x 1> to !quantum.reg

            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecl.codeblock<1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %2 = "test.op"() : () -> !qecl.codeblock<1>
            %3 = builtin.unrealized_conversion_cast %2 : !qecl.codeblock<1> to !quantum.bit

            // CHECK: [[idx:%.+]] = arith.index_cast %arg0 : i64 to index
            // CHECK: [[hreg1:%.+]] = qecl.insert_block [[hreg0]][[[idx]]], [[cb0]] : !qecl.hyperreg<1 x 1>, !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[hreg1]] : !qecl.hyperreg<1 x 1> to !quantum.reg
            %4 = quantum.insert %1[%arg0], %3 : !quantum.reg, !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.reg) -> !quantum.reg
            %5 = "test.op"(%4) : (!quantum.reg) -> !quantum.reg  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)


# MARK: TestDeallocPattern


class TestDeallocPattern:
    """Unit tests for the `dealloc` conversion pattern of the convert-quantum-to-qecl pass."""

    def test_dealloc_k_1_width_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.dealloc` ops are converted to their corresponding `qecl.dealloc` ops
        for a registers with width = 1 and for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = "test.op"() : () -> !qecl.hyperreg<1 x 1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.hyperreg<1 x 1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.hyperreg<1 x 1> to !quantum.reg

            // CHECK: qecl.dealloc [[hreg0]]  : !qecl.hyperreg<1 x 1>
            quantum.dealloc %1 : !quantum.reg

            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_dealloc_k_1_width_2(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that `quantum.dealloc` ops are converted to their corresponding `qecl.dealloc` ops
        for a registers with width = 2 and for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = "test.op"() : () -> !qecl.hyperreg<2 x 1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.hyperreg<2 x 1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.hyperreg<2 x 1> to !quantum.reg

            // CHECK: qecl.dealloc [[hreg0]]  : !qecl.hyperreg<2 x 1>
            quantum.dealloc %1 : !quantum.reg

            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    @pytest.mark.xfail(reason="Only k = 1 is supported", raises=NotImplementedError)
    def test_dealloc_k_2_width_1(self, run_filecheck, quantum_to_qecl_pipeline_k_2):
        """Test that `quantum.dealloc` ops are converted to their corresponding `qecl.dealloc` ops
        for a registers with width = 1 and for k = 2.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = "test.op"() : () -> !qecl.hyperreg<1 x 2>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.hyperreg<1 x 2>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.hyperreg<1 x 2> to !quantum.reg

            // CHECK: qecl.dealloc [[hreg0]]  : !qecl.hyperreg<1 x 2>
            quantum.dealloc %1 : !quantum.reg

            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_2)


# MARK: TestGatePattern


@pytest.mark.filterwarnings("ignore:Unable to remove cast UnrealizedConversionCastOp")
class TestGatePattern:
    """Unit tests for the `custom` (gate-op) conversion pattern of the convert-quantum-to-qecl
    pass.
    """

    def test_gate_identity_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that Identity gates (`quantum.custom "Identity"() ops) are converted to their
        corresponding `qecl.identity` ops for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecl.codeblock<1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.codeblock<1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.codeblock<1> to !quantum.bit

            // CHECK: [[cb1:%.+]] = qecl.identity [[cb0]][0] : !qecl.codeblock<1>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb2]] : !qecl.codeblock<1> to !quantum.bit
            %2 = quantum.custom "Identity"() %1 : !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            %3 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_gate_pauli_x_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that PauliX gates (`quantum.custom "PauliX"() ops) are converted to their
        corresponding `qecl.x` ops for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecl.codeblock<1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.codeblock<1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.codeblock<1> to !quantum.bit

            // CHECK: [[cb1:%.+]] = qecl.x [[cb0]][0] : !qecl.codeblock<1>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb2]] : !qecl.codeblock<1> to !quantum.bit
            %2 = quantum.custom "PauliX"() %1 : !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            %3 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_gate_pauli_y_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that PauliX gates (`quantum.custom "PauliY"() ops) are converted to their
        corresponding `qecl.y` ops for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecl.codeblock<1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.codeblock<1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.codeblock<1> to !quantum.bit

            // CHECK: [[cb1:%.+]] = qecl.y [[cb0]][0] : !qecl.codeblock<1>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb2]] : !qecl.codeblock<1> to !quantum.bit
            %2 = quantum.custom "PauliY"() %1 : !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            %3 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_gate_pauli_z_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that PauliX gates (`quantum.custom "PauliZ"() ops) are converted to their
        corresponding `qecl.z` ops for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecl.codeblock<1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.codeblock<1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.codeblock<1> to !quantum.bit

            // CHECK: [[cb1:%.+]] = qecl.z [[cb0]][0] : !qecl.codeblock<1>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb2]] : !qecl.codeblock<1> to !quantum.bit
            %2 = quantum.custom "PauliZ"() %1 : !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            %3 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_gate_hadamard_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that Hadamard gates (`quantum.custom "Hadamard"() ops) are converted to their
        corresponding `qecl.hadamard` ops for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecl.codeblock<1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.codeblock<1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.codeblock<1> to !quantum.bit

            // CHECK: [[cb1:%.+]] = qecl.hadamard [[cb0]][0] : !qecl.codeblock<1>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb2]] : !qecl.codeblock<1> to !quantum.bit
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
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecl.codeblock<1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.codeblock<1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.codeblock<1> to !quantum.bit

            // CHECK: [[cb1:%.+]] = qecl.s [[cb0]][0] : !qecl.codeblock<1>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<1>
            %2 = quantum.custom "S"() %1 : !quantum.bit

            // CHECK: [[cb3:%.+]] = qecl.s [[cb2]][0] adj : !qecl.codeblock<1>
            // CHECK: [[cb4:%.+]] = qecl.qec [[cb3]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb4]] : !qecl.codeblock<1> to !quantum.bit
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
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecl.codeblock<1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.codeblock<1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.codeblock<1> to !quantum.bit

            // CHECK: [[cb1:%.+]] = "test.op"() : () -> !qecl.codeblock<1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %2 = "test.op"() : () -> !qecl.codeblock<1>
            %3 = builtin.unrealized_conversion_cast %2 : !qecl.codeblock<1> to !quantum.bit

            // CHECK: [[cb2:%.+]], [[cb3:%.+]] = qecl.cnot [[cb0]][0], [[cb1]][0] : !qecl.codeblock<1>, !qecl.codeblock<1>
            // CHECK: [[cb4:%.+]] = qecl.qec [[cb2]] : !qecl.codeblock<1>
            // CHECK: [[cb5:%.+]] = qecl.qec [[cb3]] : !qecl.codeblock<1>
            // CHECK: [[conv_cast1:%.+]] = builtin.unrealized_conversion_cast [[cb4]] : !qecl.codeblock<1> to !quantum.bit
            // CHECK: [[conv_cast2:%.+]] = builtin.unrealized_conversion_cast [[cb5]] : !qecl.codeblock<1> to !quantum.bit
            %4, %5 = quantum.custom "CNOT"() %1, %3 : !quantum.bit, !quantum.bit

            // CHECK: "test.op"([[conv_cast1]], [[conv_cast2]]) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
            %6, %7 = "test.op"(%4, %5) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_unsupported_gate_raises_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that attempting to convert an unknown/unsupported gate raise a CompileError."""
        program = """
        func.func @test_program() {
            %0 = "test.op"() : () -> !qecl.codeblock<1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.codeblock<1> to !quantum.bit
            %2 = quantum.custom "Unknown"() %1 : !quantum.bit
            %3 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit // To prevent DCE
            return
        }
        """
        with pytest.raises(
            CompileError, match="Conversion of op 'quantum.custom' only supports gates"
        ):
            run_filecheck(program, quantum_to_qecl_pipeline_k_1)


# MARK: TestMeasurePattern


@pytest.mark.filterwarnings("ignore:Unable to remove cast UnrealizedConversionCastOp")
class TestMeasurePattern:
    """Unit tests for the `measure` op conversion pattern of the convert-quantum-to-qecl pass."""

    def test_measure_k_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that measurement ops (`quantum.measure`) are converted to their corresponding
        `qecl.measure` ops for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecl.codeblock<1>
            // CHECK-NOT: builtin.unrealized_conversion_cast
            %0 = "test.op"() : () -> !qecl.codeblock<1>
            %1 = builtin.unrealized_conversion_cast %0 : !qecl.codeblock<1> to !quantum.bit

            // CHECK: [[mres:%.+]], [[cb1:%.+]] = qecl.measure [[cb0]][0] : i1, !qecl.codeblock<1>
            // CHECK: [[conv_cast:%.+]] = builtin.unrealized_conversion_cast [[cb1]] : !qecl.codeblock<1> to !quantum.bit
            %mres, %2 = quantum.measure %1 : i1, !quantum.bit

            // CHECK: "test.op"([[conv_cast]]) : (!quantum.bit) -> !quantum.bit
            %3 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        run_filecheck(program, quantum_to_qecl_pipeline_k_1)


# MARK: TestInvalidInputIR


class TestInvalidInputIR:
    """Unit tests for invalid input IR"""

    def test_unconvertible_extract(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that attempting to convert a `quantum.extract` op raises a CompileError when there
        is insufficient information to convert the `quantum` type(s) to `qecl` types(s).
        """
        program = """
        func.func @test_program() {
            %0 = "test.op"() : () -> !quantum.reg
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
            %2 = "test.op"(%1) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        with pytest.raises(CompileError, match="Failed to convert op"):
            run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_unconvertible_insert(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that attempting to convert a `quantum.insert` op raises a CompileError when there
        is insufficient information to convert the `quantum` type(s) to `qecl` types(s).
        """
        program = """
        func.func @test_program() {
            %0 = "test.op"() : () -> !quantum.reg
            %1 = "test.op"() : () -> !quantum.bit
            %2 = quantum.insert %0[0], %1 : !quantum.reg, !quantum.bit
            %3 = "test.op"(%2) : (!quantum.reg) -> !quantum.reg  // To prevent DCE
            return
        }
        """
        with pytest.raises(CompileError, match="Failed to convert op"):
            run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_unconvertible_dealloc(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that attempting to convert a `quantum.dealloc` op raises a CompileError when there
        is insufficient information to convert the `quantum` type(s) to `qecl` types(s).
        """
        program = """
        func.func @test_program() {
            %0 = "test.op"() : () -> !quantum.reg
            quantum.dealloc %0 : !quantum.reg
            return
        }
        """
        with pytest.raises(CompileError, match="Failed to convert op"):
            run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_unconvertible_gate_1(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that attempting to convert a `quantum.custom` op raises a CompileError when there
        is insufficient information to convert the `quantum` type(s) to `qecl` types(s).

        In this case the `quantum.custom` op is a single-qubit gate.
        """
        program = """
        func.func @test_program() {
            %0 = "test.op"() : () -> !quantum.bit
            %1 = quantum.custom "Hadamard"() %0 : !quantum.bit
            %2 = "test.op"(%1) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        with pytest.raises(CompileError, match="Failed to convert op"):
            run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_unconvertible_gate_2(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that attempting to convert a `quantum.custom` op raises a CompileError when there
        is insufficient information to convert the `quantum` type(s) to `qecl` types(s).

        In this case the `quantum.custom` op is a two-qubit gate.
        """
        program = """
        func.func @test_program() {
            %0 = "test.op"() : () -> !quantum.bit
            %1 = "test.op"() : () -> !quantum.bit
            %2, %3 = quantum.custom "CNOT"() %0, %1 : !quantum.bit, !quantum.bit
            %4 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            %5 = "test.op"(%3) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        with pytest.raises(CompileError, match="Failed to convert op"):
            run_filecheck(program, quantum_to_qecl_pipeline_k_1)

    def test_unconvertible_measure(self, run_filecheck, quantum_to_qecl_pipeline_k_1):
        """Test that attempting to convert a `quantum.measure` op raises a CompileError when there
        is insufficient information to convert the `quantum` type(s) to `qecl` types(s).
        """
        program = """
        func.func @test_program() {
            %0 = "test.op"() : () -> !quantum.bit
            %1, %2 = quantum.measure %0 : i1, !quantum.bit
            %3 = "test.op"(%2) : (!quantum.bit) -> !quantum.bit  // To prevent DCE
            return
        }
        """
        with pytest.raises(CompileError, match="Failed to convert op"):
            run_filecheck(program, quantum_to_qecl_pipeline_k_1)


# MARK: Integration Tests


class TestQuantumToQecLogicalPassIntegration:
    """Integration lit tests for the convert-quantum-to-qecl pass"""

    @pytest.mark.usefixtures("use_capture")
    def test_circuit_basic(self, run_filecheck_qjit):
        """Test the convert-quantum-to-qecl pass on the simplest possible, non-trivial circuit."""
        dev = qp.device("null.qubit", wires=1)

        @qp.qjit(target="mlir")
        @convert_quantum_to_qecl_pass(k=1)
        @qp.qnode(dev, shots=1)
        def circuit():
            # CHECK: qecl.alloc() : !qecl.hyperreg<1 x 1>
            # CHECK: qecl.extract_block {{%.+}}[0] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
            # CHECK: qecl.encode[zero]
            # CHECK: qecl.insert_block {{%.+}}[0], {{%.+}}
            # CHECK: qecl.extract_block
            # CHECK: qecl.qec
            # CHECK: qecl.hadamard {{%.+}}[0]
            # CHECK: qecl.qec
            # CHECK: qecl.measure {{%.+}}[0]
            # CHECK: quantum.mcmobs
            # CHECK: quantum.sample
            # CHECK: qecl.insert_block
            # CHECK: qecl.dealloc
            qp.H(0)
            m0 = qp.measure(0)
            return qp.sample([m0])

        run_filecheck_qjit(circuit)

    @pytest.mark.usefixtures("use_capture")
    def test_circuit_ghz(self, run_filecheck_qjit):
        """Test the convert-quantum-to-qecl pass on a GHZ circuit."""
        dev = qp.device("null.qubit", wires=3)

        @qp.qjit(target="mlir")
        @convert_quantum_to_qecl_pass(k=1)
        @qp.qnode(dev, shots=1)
        def circuit():
            # CHECK: qecl.alloc() : !qecl.hyperreg<3 x 1>
            # CHECK: scf.for {{.*}} {
            # CHECK:   qecl.extract_block
            # CHECK:   qecl.encode[zero]
            # CHECK:   qecl.insert_block
            # CHECK:   scf.yield
            # CHECK: }
            # CHECK: qecl.extract_block
            # CHECK: qecl.qec
            # CHECK: qecl.hadamard
            # CHECK: qecl.extract_block
            # CHECK: qecl.qec
            # CHECK: qecl.cnot
            # CHECK: qecl.extract_block
            # CHECK: qecl.qec
            # CHECK: qecl.cnot
            # CHECK: qecl.measure
            # CHECK: qecl.measure
            # CHECK: qecl.measure
            # CHECK: quantum.mcmobs
            # CHECK: quantum.sample
            # CHECK: qecl.insert_block
            # CHECK: qecl.dealloc
            qp.H(0)
            qp.CNOT([0, 1])
            qp.CNOT([1, 2])
            m0 = qp.measure(0)
            m1 = qp.measure(1)
            m2 = qp.measure(2)
            return qp.sample([m0, m1, m2])

        run_filecheck_qjit(circuit)
