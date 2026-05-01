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

import numpy as np
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


@pytest.fixture(name="qecl_to_qecp_steane_pipeline", scope="module")
def fixture_qecl_to_qecp_steane_pipeline():
    """Fixture that returns the compilation pipeline containing the convert-qecl-to-qecp pass that
    uses the Steane code for lowering.
    """
    return (ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode.get("Steane")),)


# MARK: TestTypeConversion


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
        pipeline = (
            ConvertQecLogicalToQecPhysicalPass(
                qec_code=QecCode(
                    "",
                    n,
                    k,
                    3,
                    np.eye(n),
                    np.eye(n),
                )
            ),
        )
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
        pipeline = (
            ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode("", n, k, 3, np.eye(n), np.eye(n))),
        )
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
        pipeline = (
            ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode("", 7, 1, 3, np.eye(7), np.eye(7))),
        )

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
        pipeline = (
            ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode("", 7, 1, 3, np.eye(7), np.eye(7))),
        )

        with pytest.raises(CompileError, match="Failed to convert type"):
            run_filecheck(program, pipeline)


# MARK: TestAllocAndDealloc


class TestAllocAndDeallocConversionPatterns:
    """Test that qecl.allocate and qecl.deallocate operations for allocating hyperregisters
    of codeblocks are lowered as expected"""

    @pytest.mark.parametrize("width", [1, 2, 3])
    @pytest.mark.parametrize("n", [7, 42])
    @pytest.mark.parametrize(
        "k", [1, pytest.param(2, marks=pytest.mark.xfail(reason="Only k = 1 is supported"))]
    )
    def test_allocate_is_lowered(self, width, n, k, run_filecheck):
        """Test that a qecl.allocate operation is lowered as expected"""

        program = f"""
        builtin.module {{
        // CHECK-LABEL: test_program
        func.func @test_program() {{
            // CHECK: [[hreg0:%.+]] = qecp.alloc() : !qecp.hyperreg<{width} x {k} x {n}>
            %0 = qecl.alloc() : !qecl.hyperreg<{width} x {k}>
            return
        }}
        }}
        """

        pipeline = (
            ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode("", n, k, 3, np.eye(n), np.eye(n))),
        )
        run_filecheck(program, pipeline)

    @pytest.mark.parametrize("width", [1, 2, 3])
    @pytest.mark.parametrize("n", [7, 42])
    @pytest.mark.parametrize(
        "k", [1, pytest.param(2, marks=pytest.mark.xfail(reason="Only k = 1 is supported"))]
    )
    def test_deallocate_is_lowered(self, width, n, k, run_filecheck):
        """Test that a qecl.deallocate operation is lowered as expected"""

        program = f"""
        builtin.module {{
        // CHECK-LABEL: test_program
        func.func @test_program() {{
            // CHECK: [[hreg:%.+]] = "test.op"() : () -> !qecp.hyperreg<{width} x {k} x {n}>
            // CHECK-NEXT: qecp.dealloc [[hreg]] : !qecp.hyperreg<{width} x {k} x {n}>
            %0 = "test.op"() : () -> !qecl.hyperreg<{width} x {k}>
            qecl.dealloc %0 : !qecl.hyperreg<{width} x {k}>
            return
        }}
        }}
        """

        pipeline = (
            ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode("", n, k, 3, np.eye(n), np.eye(n))),
        )
        run_filecheck(program, pipeline)


# MARK: TestLoweringEncode


class TestInsertExtractConversionPatterns:
    """Test that qecl.extract_block and qecl.insert_block operations acting on hyperregisters
    of codeblocks are lowered as expected"""

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    @pytest.mark.parametrize("width", [1, 2, 3])
    @pytest.mark.parametrize("n", [7, 42])
    @pytest.mark.parametrize(
        "k", [1, pytest.param(2, marks=pytest.mark.xfail(reason="Only k = 1 is supported"))]
    )
    @pytest.mark.parametrize("idx", [0, 3, 6])
    def test_extract_block_is_lowered(self, width, k, n, idx, run_filecheck):
        """Test that a qecl.extract_block operation is lowered as expected"""

        program = f"""
        builtin.module {{
        // CHECK-LABEL: test_program
        func.func @test_program() {{
            // CHECK: [[hreg0:%.+]] = "test.op"() : () -> !qecp.hyperreg<{width} x {k} x {n}>
            // CHECK: qecp.extract_block [[hreg0]][{idx}] : !qecp.hyperreg<{width} x {k} x {n}> -> !qecp.codeblock<{k} x {n}>
            %0 = "test.op"() : () -> !qecl.hyperreg<{width} x {k}>
            %1 = qecl.extract_block %0[{idx}] : !qecl.hyperreg<{width} x {k}> -> !qecl.codeblock<{k}>
            return
        }}
        }}
        """

        pipeline = (
            ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode("", n, k, 3, np.eye(n), np.eye(n))),
        )
        run_filecheck(program, pipeline)

    @pytest.mark.parametrize("width", [1, 2, 3])
    @pytest.mark.parametrize("n", [7, 42])
    @pytest.mark.parametrize(
        "k", [1, pytest.param(2, marks=pytest.mark.xfail(reason="Only k = 1 is supported"))]
    )
    @pytest.mark.parametrize("idx", [0, 3, 6])
    def test_insert_block_is_lowered(self, width, k, n, idx, run_filecheck):
        """Test that a qecl.insert_block operation is lowered as expected"""

        program = f"""
        builtin.module {{
        // CHECK-LABEL: test_program
        func.func @test_program() {{
            // CHECK: [[cb:%.+]] = "test.op"() : () -> !qecp.codeblock<{k} x {n}>
            // CHECK: [[hreg:%.+]] = "test.op"() : () -> !qecp.hyperreg<{width} x {k} x {n}>
            // CHECK: qecp.insert_block [[hreg]][{idx}], [[cb]] : !qecp.hyperreg<{width} x {k} x {n}>, !qecp.codeblock<{k} x {n}>
            %0 = "test.op"() : () -> !qecl.codeblock<{k}>
            %1 = "test.op"() : () -> !qecl.hyperreg<{width} x {k}>
            %2 = qecl.insert_block %1[{idx}], %0 : !qecl.hyperreg<{width} x {k}>, !qecl.codeblock<{k}>
            return
        }}
        }}
        """

        pipeline = (
            ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode("", n, k, 3, np.eye(n), np.eye(n))),
        )
        run_filecheck(program, pipeline)


class TestLoweringEncode:
    """Test lowering the qecl.EncodeOp to a subroutine of qecp gates"""

    @pytest.mark.parametrize("code_name", ["test_name", "abcd"])
    @pytest.mark.parametrize(
        "k", [1, pytest.param(2, marks=pytest.mark.xfail(reason="Only k = 1 is supported"))]
    )
    def test_with_fake_code(self, code_name, k, run_filecheck):
        """Test that a single qecl.encode operation is lowered to a call to the encoding
        subroutine using a generic 'code' that relies on two data qubits (set by n) and
        two auxiliary qubits (set by the number of rows in the z_tanner graph)"""

        n = 2
        qec_code = QecCode(code_name, n=n, k=k, d=1, x_tanner=np.eye(n), z_tanner=np.eye(n))

        program = f"""
        builtin.module @module_circuit {{
                func.func @test_func() attributes {{quantum.node}} {{
                    // CHECK: [[codeblock:%.+]] = "test.op"() : () -> !qecp.codeblock<{k} x {n}>
                    // CHECK-NEXT: [[codeblock2:%.+]] = func.call @encode_zero_{code_name}([[codeblock]]) : (!qecp.codeblock<{k} x {n}>) -> !qecp.codeblock<{k} x {n}>
                    %0 = "test.op"() : () -> !qecl.codeblock<{k}>
                    %1 = qecl.encode ["zero"] %0 : !qecl.codeblock<{k}>
                    return
                }}
                // CHECK: func.func private @encode_zero_{code_name}([[codeblock:%.+]]: !qecp.codeblock<{k} x {n}>)
                // CHECK-NEXT: [[aux0:%.+]] = qecp.alloc_aux
                // CHECK-NEXT: [[aux1:%.+]] = qecp.alloc_aux
                // CHECK-NEXT: [[aux0_1:%.+]] = qecp.hadamard [[aux0]]
                // CHECK-NEXT: [[aux1_1:%.+]] = qecp.hadamard [[aux1]]
                // CHECK-NEXT: [[data0:%.+]] = qecp.extract [[codeblock]][0]
                // CHECK-NEXT: [[data1:%.+]] = qecp.extract [[codeblock]][1]
                // CHECK-NEXT: [[aux0_2:%.+]], [[data0_1:%.+]] = qecp.cnot [[aux0_1]], [[data0]]
                // CHECK-NEXT: [[aux1_2:%.+]], [[data1_1:%.+]] = qecp.cnot [[aux1_1]], [[data1]]
                // CHECK-NEXT: [[codeblock_1:%.+]] = qecp.insert [[codeblock]][0], [[data0_1]]
                // CHECK-NEXT: [[codeblock_2:%.+]] = qecp.insert [[codeblock_1]][1], [[data1_1]]
                // CHECK-NEXT: [[aux0_3:%.+]] = qecp.hadamard [[aux0_2]]
                // CHECK-NEXT: [[aux1_3:%.+]] = qecp.hadamard [[aux1_2]]
                // CHECK-NEXT: [[meas_val0:%.+]], [[aux0_out:%.+]] = qecp.measure [[aux0_3]]
                // CHECK-NEXT: [[meas_val1:%.+]], [[aux1_out:%.+]] = qecp.measure [[aux1_3]]
                // CHECK-NEXT: qecp.dealloc_aux [[aux0_out]]
                // CHECK-NEXT: qecp.dealloc_aux [[aux1_out]]
                // CHECK-NEXT: func.return [[codeblock_2]]
            }}
            """

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=qec_code),)
        run_filecheck(program, pipeline)

    def test_single_encode_with_Steane(self, run_filecheck):
        """Test that a single qecl.encode operation is lowered to a call to the encoding
        subroutine using the Steane code"""

        program = """
            builtin.module @module_circuit {
                func.func @test_func() attributes {quantum.node} {
                    // CHECK: [[codeblock:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
                    // CHECK-NEXT: [[codeblock2:%.+]] = func.call @encode_zero_Steane([[codeblock]]) : (!qecp.codeblock<1 x 7>) -> !qecp.codeblock<1 x 7>
                    %0 = "test.op"() : () -> !qecl.codeblock<1>
                    %1 = qecl.encode ["zero"] %0 : !qecl.codeblock<1>
                    return
                }
                // CHECK: func.func private @encode_zero_Steane([[codeblock:%.+]]: !qecp.codeblock<1 x 7>)
                // CHECK: [[aux0:%.+]] = qecp.alloc_aux
                // CHECK-NEXT: [[aux1:%.+]] = qecp.alloc_aux
                // CHECK-NEXT: [[aux2:%.+]] = qecp.alloc_aux
                // CHECK-NEXT: [[aux0_1:%.+]] = qecp.hadamard [[aux0]]
                // CHECK-NEXT: [[aux1_1:%.+]] = qecp.hadamard [[aux1]]
                // CHECK-NEXT: [[aux2_1:%.+]] = qecp.hadamard [[aux2]]
                // CHECK-NEXT: [[data0:%.+]] = qecp.extract [[codeblock]][0]
                // CHECK-NEXT: [[data1:%.+]] = qecp.extract [[codeblock]][1]
                // CHECK-NEXT: [[data2:%.+]] = qecp.extract [[codeblock]][2]
                // CHECK-NEXT: [[data3:%.+]] = qecp.extract [[codeblock]][3]
                // CHECK-NEXT: [[data4:%.+]] = qecp.extract [[codeblock]][4]
                // CHECK-NEXT: [[data5:%.+]] = qecp.extract [[codeblock]][5]
                // CHECK-NEXT: [[data6:%.+]] = qecp.extract [[codeblock]][6]
                // CHECK-NEXT: [[aux0_1a:%.+]], [[data0_1:%.+]] = qecp.cnot [[aux0_1]], [[data0]]
                // CHECK-NEXT: [[aux0_1b:%.+]], [[data1_1:%.+]] = qecp.cnot [[aux0_1a]], [[data1]]
                // CHECK-NEXT: [[aux0_1c:%.+]], [[data2_1:%.+]] = qecp.cnot [[aux0_1b]], [[data2]]
                // CHECK-NEXT: [[aux0_1d:%.+]], [[data3_1:%.+]] = qecp.cnot [[aux0_1c]], [[data3]]
                // CHECK-NEXT: [[aux1_1a:%.+]], [[data1_2:%.+]] = qecp.cnot [[aux1_1]], [[data1_1]]
                // CHECK-NEXT: [[aux1_1b:%.+]], [[data2_2:%.+]] = qecp.cnot [[aux1_1a]], [[data2_1]]
                // CHECK-NEXT: [[aux1_1c:%.+]], [[data4_1:%.+]] = qecp.cnot [[aux1_1b]], [[data4]]
                // CHECK-NEXT: [[aux1_1d:%.+]], [[data5_1:%.+]] = qecp.cnot [[aux1_1c]], [[data5]]
                // CHECK-NEXT: [[aux2_1a:%.+]], [[data2_3:%.+]] = qecp.cnot [[aux2_1]], [[data2_2]]
                // CHECK-NEXT: [[aux2_1b:%.+]], [[data3_2:%.+]] = qecp.cnot [[aux2_1a]], [[data3_1]]
                // CHECK-NEXT: [[aux2_1c:%.+]], [[data5_2:%.+]] = qecp.cnot [[aux2_1b]], [[data5_1]]
                // CHECK-NEXT: [[aux2_1d:%.+]], [[data6_1:%.+]] = qecp.cnot [[aux2_1c]], [[data6]]
                // CHECK-NEXT: [[codeblock_1:%.+]] = qecp.insert [[codeblock]][0], [[data0_1]]
                // CHECK-NEXT: [[codeblock_2:%.+]] = qecp.insert [[codeblock_1]][1], [[data1_2]]
                // CHECK-NEXT: [[codeblock_3:%.+]] = qecp.insert [[codeblock_2]][2], [[data2_3]]
                // CHECK-NEXT: [[codeblock_4:%.+]] = qecp.insert [[codeblock_3]][3], [[data3_2]]
                // CHECK-NEXT: [[codeblock_5:%.+]] = qecp.insert [[codeblock_4]][4], [[data4_1]]
                // CHECK-NEXT: [[codeblock_6:%.+]] = qecp.insert [[codeblock_5]][5], [[data5_2]]
                // CHECK-NEXT: [[codeblock_7:%.+]] = qecp.insert [[codeblock_6]][6], [[data6_1]]
                // CHECK-NEXT: [[aux0_2:%.+]] = qecp.hadamard [[aux0_1d]]
                // CHECK-NEXT: [[aux1_2:%.+]] = qecp.hadamard [[aux1_1d]]
                // CHECK-NEXT: [[aux2_2:%.+]] = qecp.hadamard [[aux2_1d]]
                // CHECK-NEXT: [[meas_val0:%.+]], [[aux0_out:%.+]] = qecp.measure [[aux0_2]]
                // CHECK-NEXT: [[meas_val1:%.+]], [[aux1_out:%.+]] = qecp.measure [[aux1_2]]
                // CHECK-NEXT: [[meas_val2:%.+]], [[aux2_out:%.+]] = qecp.measure [[aux2_2]]
                // CHECK-NEXT: qecp.dealloc_aux [[aux0_out]]
                // CHECK-NEXT: qecp.dealloc_aux [[aux1_out]]
                // CHECK-NEXT: qecp.dealloc_aux [[aux2_out]]
                // CHECK-NEXT: func.return [[codeblock_7]]
            }
            """

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode.get("Steane")),)
        run_filecheck(program, pipeline)

    def test_multiple_encodes_with_Steane(self, run_filecheck):
        """Test that a qecl.encode operation raises an error if we are not encoding to zero"""

        program = """
            builtin.module @module_circuit {
                func.func @test_func() attributes {quantum.node} {
                    // CHECK: [[codeblock1:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
                    // CHECK-NEXT: [[codeblock2:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
                    // CHECK-NEXT: func.call @encode_zero_Steane
                    // CHECK-NEXT: func.call @encode_zero_Steane
                    %0 = "test.op"() : () -> !qecl.codeblock<1>
                    %1 = "test.op"() : () -> !qecl.codeblock<1>
                    %2 = qecl.encode ["zero"] %0 : !qecl.codeblock<1>
                    %3 = qecl.encode ["zero"] %1 : !qecl.codeblock<1>
                    return
                }
                // CHECK: func.func private @encode_zero_Steane
            }
            """

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode.get("Steane")),)
        run_filecheck(program, pipeline)


# MARK: TestLoweringMeasure


class TestLoweringMeasure:
    """Unit tests for the `measure` pattern of the convert-qecl-to-qecp pass."""

    def test_measure(self, run_filecheck, qecl_to_qecp_steane_pipeline):
        """Test the lowering pattern for `qecl.measure` ops."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_program
        func.func @test_program() {
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
            %0 = "test.op"() : () -> !qecl.codeblock<1>

            // CHECK: [[mresp:%.+]], [[cb1:%.+]] = func.call @measure_transversal_Steane([[cb0]]) : ({{.*}}) -> (tensor<7xi1>, !qecp.codeblock<1 x 7>)
            // CHECK: [[mresl:%.+]] = qecp.decode_physical_meas [[mresp]] : tensor<7xi1> -> tensor<1xi1>
            // CHECK: [[zero:%.+]] = arith.constant 0 : index
            // CHECK: [[mres0:%.+]] = tensor.extract [[mresl]][[[zero]]] : tensor<1xi1>
            %mres0, %1 = qecl.measure %0[0] : i1, !qecl.codeblock<1>

            // CHECK: [[mres1:%.+]] = "test.op"([[mres0]]) : (i1) -> i1
            %mres1 = "test.op"(%mres0) : (i1) -> i1  // To prevent DCE
            %2 = "test.op"(%1) : (!qecl.codeblock<1>) -> !qecl.codeblock<1>  // To prevent DCE
            return
        }
        // CHECK-LABEL: func.func private @measure_transversal_Steane
        // CHECK-SAME: ([[cb_in:%.+]]: !qecp.codeblock<1 x 7>) -> (tensor<7xi1>, !qecp.codeblock<1 x 7>) {
        // CHECK: [[mres_t:%.+]] = tensor.empty() : tensor<7xi1>
        // CHECK: [[c0:%.+]] = arith.constant 0 : index
        // CHECK: [[c7:%.+]] = arith.constant 7 : index
        // CHECK: [[c1:%.+]] = arith.constant 1 : index
        // CHECK: [[mres_t_out:%.+]], [[cb_out:%.+]] = scf.for [[idx:%.+]] = [[c0]] to [[c7]] step [[c1]]
        // CHECK-SAME: iter_args([[mres_t_arg:%.+]] = [[mres_t]], [[cb_arg:%.+]] = [[cb_in]])
        // CHECK-SAME: -> (tensor<7xi1>, !qecp.codeblock<1 x 7>) {
        // CHECK:   [[q0:%.+]] = qecp.extract [[cb_arg]][[[idx]]] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
        // CHECK:   [[mres0:%.+]], [[q1:%.+]] = qecp.measure [[q0]] : i1, !qecp.qubit<data>
        // CHECK:   [[cb2:%.+]] = qecp.insert [[cb_arg]][[[idx]]], [[q1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
        // CHECK:   [[mres_t_1:%.+]] = tensor.insert [[mres0]] into [[mres_t_arg]][[[idx]]] : tensor<7xi1>
        // CHECK:   scf.yield [[mres_t_1]], [[cb2]] : tensor<7xi1>, !qecp.codeblock<1 x 7>
        // CHECK: }
        // CHECK: func.return [[mres_t_out]], [[cb_out]] : tensor<7xi1>, !qecp.codeblock<1 x 7>
        // CHECK: }
        }
        """
        run_filecheck(program, qecl_to_qecp_steane_pipeline)


# MARK: TestQECLNoiseLoweringPassIntegration


# We can remove this xfail and warning filter once `convert_qecl_to_qecp_pass` is complete
@pytest.mark.xfail(reason="The `convert_qecl_to_qecp_pass` is incomplete")
@pytest.mark.filterwarnings("ignore:Unable to remove cast UnrealizedConversionCastOp")
class TestQECLNoiseLoweringPassIntegration:
    """Integration lit tests for the convert-qecl-noise-to-qecp-noise pass"""

    # pylint: disable=line-too-long
    def test_convert_qecl_noise_to_qecp_noise_pass_integration(self, run_filecheck_qjit):
        """Test the convert-qecl-noise-to-qecp-noise pass on the simplest possible, non-trivial circuit."""
        dev = qp.device("null.qubit", wires=1)

        @qp.qjit(target="mlir", keep_intermediate=True, capture=True)
        @convert_qecl_to_qecp_pass(qec_code=QecCode.get("Steane"), number_errors=1)
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
