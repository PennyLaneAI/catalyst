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

# pylint: disable=line-too-long,too-many-lines


pytestmark = pytest.mark.xdsl


@pytest.fixture(name="qecl_to_qecp_steane_pipeline", scope="module")
def fixture_qecl_to_qecp_steane_pipeline():
    """Fixture that returns the compilation pipeline containing the convert-qecl-to-qecp pass that
    uses the Steane code for lowering.
    """
    return (ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode.get("Steane")),)


# pylint: disable=too-many-arguments
@pytest.fixture(name="get_generic_qec_code", scope="module")
def fixture_get_generic_qec_code():
    """Fixture factory that returns a function to create `QecCode` objects for generic QEC codes."""

    def _make_qec_code(
        n: int,
        k: int,
        d: int,
        *,
        name: str = "TestCode",
        n_aux: int = 3,
        x_tanner=None,
        z_tanner=None,
        transversal_1q_gates: dict[str, tuple[str, ...]] | None = None,
        transversal_2q_gates: dict[str, str] | None = None,
        unitary_encoding: dict = {},
    ) -> QecCode:
        rng = np.random.default_rng(seed=42)

        if x_tanner is None:
            x_tanner = rng.integers(low=0, high=1, size=(n_aux, n))

        if z_tanner is None:
            z_tanner = rng.integers(low=0, high=1, size=(n_aux, n))

        if transversal_1q_gates is None:
            transversal_1q_gates = {
                "x": ("X",) * n,
                "y": ("Y",) * n,
                "z": ("Z",) * n,
                "hadamard": ("H",) * n,
                "s": ("Sa",) * n,
            }

        if transversal_2q_gates is None:
            transversal_2q_gates = {"cnot": "CNOT"}

        return QecCode(
            name=name,
            n=n,
            k=k,
            d=d,
            x_tanner=x_tanner,
            z_tanner=z_tanner,
            transversal_1q_gates=transversal_1q_gates,
            transversal_2q_gates=transversal_2q_gates,
            unitary_encoding=unitary_encoding,
        )

    return _make_qec_code


# MARK: TestTypeConversion


class TestTypeConversionPattern:
    """Unit tests for the type conversion patterns of the convert-qecl-to-qecp pass."""

    @pytest.mark.parametrize("n", [7, 42])
    @pytest.mark.parametrize(
        "k", [1, pytest.param(2, marks=pytest.mark.xfail(reason="Only k = 1 is supported"))]
    )
    def test_codeblock_conversion(self, n, k, run_filecheck, get_generic_qec_code):
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
        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=get_generic_qec_code(n, k, d=3)),)
        run_filecheck(program, pipeline)

    @pytest.mark.parametrize("width", [1, 2, 3])
    @pytest.mark.parametrize("n", [7, 42])
    @pytest.mark.parametrize(
        "k", [1, pytest.param(2, marks=pytest.mark.xfail(reason="Only k = 1 is supported"))]
    )
    # pylint: disable=too-many-positional-arguments, too-many-arguments
    def test_hyperreg_conversion(self, width, n, k, run_filecheck, get_generic_qec_code):
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
        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=get_generic_qec_code(n, k, d=3)),)
        run_filecheck(program, pipeline)

    def test_codeblock_conversion_with_k_mismatch(self, run_filecheck, get_generic_qec_code):
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
        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=get_generic_qec_code(7, 1, 3)),)

        with pytest.raises(CompileError, match="Failed to convert type"):
            run_filecheck(program, pipeline)

    def test_hyperreg_conversion_with_k_mismatch(self, run_filecheck, get_generic_qec_code):
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
        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=get_generic_qec_code(7, 1, 3)),)

        with pytest.raises(CompileError, match="Failed to convert type"):
            run_filecheck(program, pipeline)


# MARK: Alloc/Dealloc


class TestAllocAndDeallocConversionPatterns:
    """Test that qecl.allocate and qecl.deallocate operations for allocating hyperregisters
    of codeblocks are lowered as expected"""

    @pytest.mark.parametrize("width", [1, 2, 3])
    @pytest.mark.parametrize("n", [7, 42])
    @pytest.mark.parametrize(
        "k", [1, pytest.param(2, marks=pytest.mark.xfail(reason="Only k = 1 is supported"))]
    )
    # pylint: disable=too-many-positional-arguments, too-many-arguments
    def test_allocate_is_lowered(self, width, n, k, run_filecheck, get_generic_qec_code):
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

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=get_generic_qec_code(n, k, 3)),)
        run_filecheck(program, pipeline)

    @pytest.mark.parametrize("width", [1, 2, 3])
    @pytest.mark.parametrize("n", [7, 42])
    @pytest.mark.parametrize(
        "k", [1, pytest.param(2, marks=pytest.mark.xfail(reason="Only k = 1 is supported"))]
    )
    # pylint: disable=too-many-positional-arguments, too-many-arguments
    def test_deallocate_is_lowered(self, width, n, k, run_filecheck, get_generic_qec_code):
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

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=get_generic_qec_code(n, k, 3)),)
        run_filecheck(program, pipeline)


# MARK: Insert/Extract


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
    def test_extract_block_is_lowered(self, width, k, n, idx, run_filecheck, get_generic_qec_code):
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

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=get_generic_qec_code(n, k, 3)),)
        run_filecheck(program, pipeline)

    @pytest.mark.parametrize("width", [1, 2, 3])
    @pytest.mark.parametrize("n", [7, 42])
    @pytest.mark.parametrize(
        "k", [1, pytest.param(2, marks=pytest.mark.xfail(reason="Only k = 1 is supported"))]
    )
    @pytest.mark.parametrize("idx", [0, 3, 6])
    def test_insert_block_is_lowered(self, width, k, n, idx, run_filecheck, get_generic_qec_code):
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

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=get_generic_qec_code(n, k, 3)),)
        run_filecheck(program, pipeline)


# MARK: Encode


class TestLoweringEncode:
    """Test lowering the qecl.EncodeOp to a subroutine of qecp gates"""

    @pytest.mark.parametrize("code_name", ["test_name", "abcd"])
    @pytest.mark.parametrize(
        "k", [1, pytest.param(2, marks=pytest.mark.xfail(reason="Only k = 1 is supported"))]
    )
    def test_with_fake_code(self, code_name, k, run_filecheck, get_generic_qec_code):
        """Test that a single qecl.encode operation is lowered to a call to the encoding
        subroutine using a generic 'code' that relies on two data qubits (set by n) and
        two auxiliary qubits (set by the number of rows in the z_tanner graph)"""

        n = 2
        qec_code = get_generic_qec_code(
            n=n,
            k=k,
            d=1,
            name=code_name,
            x_tanner=np.eye(n),
            z_tanner=np.eye(n),
        )

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

    def test_single_encode_with_Steane(self, run_filecheck, qecl_to_qecp_steane_pipeline):
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

        run_filecheck(program, qecl_to_qecp_steane_pipeline)

    def test_multiple_encodes_with_Steane(self, run_filecheck, qecl_to_qecp_steane_pipeline):
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

        run_filecheck(program, qecl_to_qecp_steane_pipeline)


# MARK: Tanner graphs


class TestTannerGraphInsertion:
    """Unit tests for the insertion of Tanner graph ops."""

    def test_tanner_graph_insertion_steane(self, run_filecheck, qecl_to_qecp_steane_pipeline):
        """Test that Tanner graph ops for the Steane code are correctly inserted at the beginning of
        the module.
        """

        program = """
        // CHECK-LABEL: test_module
        builtin.module @test_module {
        func.func @test_program()  {
            return
        }
        // CHECK-LABEL: func.func private @qec_cycle_Steane
        //      CHECK: [[row_idx_x:%.+]] = arith.constant
        // CHECK-SAME:   dense<[7, 7, 8, 7, 8, 9, 7, 9, 8, 8, 9, 9, 0, 1, 2, 3, 1, 2, 4, 5, 2, 3, 5, 6]> : tensor<24xi32>
        //      CHECK: [[col_ptr_x:%.+]] = arith.constant dense<[0, 1, 3, 6, 8, 9, 11, 12, 16, 20, 24]> : tensor<11xi32>
        //      CHECK: [[tanner_x:%.+]] = qecp.assemble_tanner [[row_idx_x]], [[col_ptr_x]] :
        // CHECK-SAME:   tensor<24xi32>, tensor<11xi32> -> !qecp.tanner_graph<24, 11, i32>
        //      CHECK: [[row_idx_z:%.+]] = arith.constant
        // CHECK-SAME:   dense<[7, 7, 8, 7, 8, 9, 7, 9, 8, 8, 9, 9, 0, 1, 2, 3, 1, 2, 4, 5, 2, 3, 5, 6]> : tensor<24xi32>
        //      CHECK: [[col_ptr_z:%.+]] = arith.constant dense<[0, 1, 3, 6, 8, 9, 11, 12, 16, 20, 24]> : tensor<11xi32>
        //      CHECK: [[tanner_z:%.+]] = qecp.assemble_tanner [[row_idx_z]], [[col_ptr_z]] :
        // CHECK-SAME:   tensor<24xi32>, tensor<11xi32> -> !qecp.tanner_graph<24, 11, i32>
        }
        """
        run_filecheck(program, qecl_to_qecp_steane_pipeline)


# MARK: QEC Cycle


class TestQecCycleLowering:
    """Unit tests for the `qecl.qec` conversion pattern of the convert-qecl-to-qecp pass."""

    def test_single_qec_cycle_Steane(self, run_filecheck, qecl_to_qecp_steane_pipeline):
        """Test that a `qecl.qec` op is lowered to a call to the QEC-cycle subroutine for the Steane
        code.
        """
        program = """
        // CHECK-LABEL: test_module
        builtin.module @test_module {
        // CHECK-LABEL: test_program
        func.func @test_program()  {
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
            %0 = "test.op"() : () -> !qecl.codeblock<1>

            // CHECK: [[cb1:%.+]] = func.call @qec_cycle_Steane([[cb0]]) : (!qecp.codeblock<1 x 7>) -> !qecp.codeblock<1 x 7>
            %1 = qecl.qec %0 : !qecl.codeblock<1>
            return
        }
        // CHECK-LABEL: qec_cycle_Steane([[cb0:%.+]]: !qecp.codeblock<1 x 7>) -> !qecp.codeblock<1 x 7>
        // CHECK: [[tanner_x:%.+]] = qecp.assemble_tanner {{.+}}, {{.+}} : tensor<24xi32>, tensor<11xi32> -> !qecp.tanner_graph<24, 11, i32>
        // CHECK: [[tanner_z:%.+]] = qecp.assemble_tanner {{.+}}, {{.+}} : tensor<24xi32>, tensor<11xi32> -> !qecp.tanner_graph<24, 11, i32>

        // COM: The block below takes results of X checks and performs Z corrections
        // CHECK: qecp.alloc_aux : !qecp.qubit<aux>
        // CHECK: qecp.alloc_aux : !qecp.qubit<aux>
        // CHECK: qecp.alloc_aux : !qecp.qubit<aux>
        // CHECK: qecp.hadamard {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.hadamard {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.hadamard {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.extract {{.*}} : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
        // CHECK: qecp.cnot {{.*}} : !qecp.qubit<aux>, !qecp.qubit<data>
        // CHECK: qecp.insert {{.*}} : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
        // CHECK: [[cb0:%.+]] = qecp.insert {{.*}}[6], {{.*}} : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
        // CHECK: qecp.hadamard {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.hadamard {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.hadamard {{.*}} : !qecp.qubit<aux>
        // CHECK: [[m0:%.+]], {{.*}} = qecp.measure {{.*}} : i1, !qecp.qubit<aux>
        // CHECK: [[m1:%.+]], {{.*}} = qecp.measure {{.*}} : i1, !qecp.qubit<aux>
        // CHECK: [[m2:%.+]], {{.*}} = qecp.measure {{.*}} : i1, !qecp.qubit<aux>
        // CHECK: qecp.dealloc_aux {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.dealloc_aux {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.dealloc_aux {{.*}} : !qecp.qubit<aux>
        // CHECK: [[esm:%.+]] = tensor.from_elements [[m0]], [[m1]], [[m2]] : tensor<3xi1>
        // CHECK: [[idx_t:%.+]] = qecp.decode_esm_css([[tanner_x]] : !qecp.tanner_graph<24, 11, i32>) [[esm]] : tensor<3xi1> -> tensor<1xindex>
        // CHECK: [[lb:%.+]] = arith.constant 0 : index
        // CHECK: [[ub:%.+]] = arith.constant 1 : index
        // CHECK: [[st:%.+]] = arith.constant 1 : index
        // CHECK: [[cb_x_out:%.+]] = scf.for [[i:%.+]] = [[lb]] to [[ub]] step [[st]] iter_args([[cb_arg:%.+]] = {{%.+}})
        // CHECK:   [[err_idx:%.+]] = tensor.extract [[idx_t]][[[i]]] : tensor<1xindex>
        // CHECK:   [[err_i64:%.+]] = arith.index_cast [[err_idx]] : index to i64
        // CHECK:   [[minus1:%.+]] = arith.constant -1 : i64
        // CHECK:   [[cond:%.+]] = arith.cmpi ne, [[err_i64]], [[minus1]] : i64
        // CHECK:   [[cond_out_cb:%.+]] = scf.if [[cond]]
        // CHECK:     [[q0:%.+]] = qecp.extract [[cb_arg]][[[err_idx]]] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
        // CHECK:     [[q1:%.+]] = qecp.z [[q0]] : !qecp.qubit<data>
        // CHECK:     [[cb_arg_1:%.+]] = qecp.insert [[cb_arg]][[[err_idx]]], [[q1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
        // CHECK:     scf.yield [[cb_arg_1]] : !qecp.codeblock<1 x 7>
        // CHECK:   } else {
        // CHECK:     scf.yield [[cb_arg]] : !qecp.codeblock<1 x 7>
        // CHECK:   }
        // CHECK: scf.yield [[cond_out_cb]] : !qecp.codeblock<1 x 7>
        // CHECK: }

        // COM: The block below takes results of X checks and performs Z corrections
        // CHECK: qecp.alloc_aux : !qecp.qubit<aux>
        // CHECK: qecp.alloc_aux : !qecp.qubit<aux>
        // CHECK: qecp.alloc_aux : !qecp.qubit<aux>
        // CHECK-NOT: qecp.hadamard
        // CHECK: qecp.extract {{.*}} : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
        // CHECK: qecp.cnot {{.*}} : !qecp.qubit<data>, !qecp.qubit<aux>
        // CHECK: qecp.insert {{.*}} : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
        // CHECK: [[cb0:%.+]] = qecp.insert {{.*}}[6], {{.*}} : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
        // CHECK-NOT: qecp.hadamard
        // CHECK: [[m0:%.+]], {{.*}} = qecp.measure {{.*}} : i1, !qecp.qubit<aux>
        // CHECK: [[m1:%.+]], {{.*}} = qecp.measure {{.*}} : i1, !qecp.qubit<aux>
        // CHECK: [[m2:%.+]], {{.*}} = qecp.measure {{.*}} : i1, !qecp.qubit<aux>
        // CHECK: qecp.dealloc_aux {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.dealloc_aux {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.dealloc_aux {{.*}} : !qecp.qubit<aux>
        // CHECK: [[esm:%.+]] = tensor.from_elements [[m0]], [[m1]], [[m2]] : tensor<3xi1>
        // CHECK: [[idx_t:%.+]] = qecp.decode_esm_css([[tanner_z]] : !qecp.tanner_graph<24, 11, i32>) [[esm]] : tensor<3xi1>  -> tensor<1xindex>
        // CHECK: [[lb:%.+]] = arith.constant 0 : index
        // CHECK: [[ub:%.+]] = arith.constant 1 : index
        // CHECK: [[st:%.+]] = arith.constant 1 : index
        // CHECK: [[cb_x_out:%.+]] = scf.for [[i:%.+]] = [[lb]] to [[ub]] step [[st]] iter_args([[cb_arg:%.+]] = {{%.+}})
        // CHECK:   [[err_idx:%.+]] = tensor.extract [[idx_t]][[[i]]] : tensor<1xindex>
        // CHECK:   [[err_i64:%.+]] = arith.index_cast [[err_idx]] : index to i64
        // CHECK:   [[minus1:%.+]] = arith.constant -1 : i64
        // CHECK:   [[cond:%.+]] = arith.cmpi ne, [[err_i64]], [[minus1]] : i64
        // CHECK:   [[cond_out_cb:%.+]] = scf.if [[cond]]
        // CHECK:     [[q0:%.+]] = qecp.extract [[cb_arg]][[[err_idx]]] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
        // CHECK:     [[q1:%.+]] = qecp.x [[q0]] : !qecp.qubit<data>
        // CHECK:     [[cb_arg_1:%.+]] = qecp.insert [[cb_arg]][[[err_idx]]], [[q1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
        // CHECK:     scf.yield [[cb_arg_1]] : !qecp.codeblock<1 x 7>
        // CHECK:   } else {
        // CHECK:     scf.yield [[cb_arg]] : !qecp.codeblock<1 x 7>
        // CHECK:   }
        // CHECK: scf.yield [[cond_out_cb]] : !qecp.codeblock<1 x 7>
        // CHECK: }
        // CHECK: func.return [[cb_x_out]] : !qecp.codeblock<1 x 7>
        }
        """
        run_filecheck(program, qecl_to_qecp_steane_pipeline)


# MARK: Measure


class TestLoweringMeasure:
    """Unit tests for the `measure` pattern of the convert-qecl-to-qecp pass."""

    def test_measure_steane(self, run_filecheck, qecl_to_qecp_steane_pipeline):
        """Test the lowering pattern for `qecl.measure` ops with the Steane code."""
        program = """
        builtin.module {
        // CHECK-LABEL: test_program
        func.func @test_program() {
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
            %0 = "test.op"() : () -> !qecl.codeblock<1>

            //      CHECK: [[mresp:%.+]], [[cb1:%.+]] = func.call @measure_transversal_Steane([[cb0]]) :
            // CHECK-SAME:   (!qecp.codeblock<1 x 7>) -> (tensor<3xi1>, !qecp.codeblock<1 x 7>)
            //      CHECK: [[mresl:%.+]] = func.call @decode_physical_measurements_Steane([[mresp]]) :
            // CHECK-SAME:   (tensor<3xi1>) -> tensor<1xi1>
            //      CHECK: [[zero:%.+]] = arith.constant 0 : index
            //      CHECK: [[mres0:%.+]] = tensor.extract [[mresl]][[[zero]]] : tensor<1xi1>
            %mres0, %1 = qecl.measure %0[0] : i1, !qecl.codeblock<1>

            // CHECK: [[mres1:%.+]] = "test.op"([[mres0]]) : (i1) -> i1
            %mres1 = "test.op"(%mres0) : (i1) -> i1  // To prevent DCE
            %2 = "test.op"(%1) : (!qecl.codeblock<1>) -> !qecl.codeblock<1>  // To prevent DCE
            return
        }
        // CHECK-LABEL: func.func private @measure_transversal_Steane
        //  CHECK-SAME:     ([[cb_in:%.+]]: !qecp.codeblock<1 x 7>) -> (tensor<3xi1>, !qecp.codeblock<1 x 7>)
        //       CHECK:   [[q40:%.+]] = qecp.extract [[cb_in]][4]
        //       CHECK:   [[q50:%.+]] = qecp.extract [[cb_in]][5]
        //       CHECK:   [[q60:%.+]] = qecp.extract [[cb_in]][6]
        //       CHECK:   [[m4:%.+]], [[q41:%.+]] = qecp.measure [[q40]]
        //       CHECK:   [[m5:%.+]], [[q51:%.+]] = qecp.measure [[q50]]
        //       CHECK:   [[m6:%.+]], [[q61:%.+]] = qecp.measure [[q60]]
        //   CHECK-NOT:   qecp.measure
        //       CHECK:   [[cb1:%.+]] = qecp.insert [[cb_in]][4], [[q41]]
        //       CHECK:   [[cb2:%.+]] = qecp.insert [[cb1]][5], [[q51]]
        //       CHECK:   [[cb3:%.+]] = qecp.insert [[cb2]][6], [[q61]]
        //       CHECK:   [[m_3xi1:%.+]] = tensor.from_elements [[m4]], [[m5]], [[m6]] : tensor<3xi1>
        //       CHECK:   func.return [[m_3xi1]], [[cb3]] : tensor<3xi1>, !qecp.codeblock<1 x 7>

        // CHECK-LABEL: func.func private @decode_physical_measurements_Steane
        //  CHECK-SAME:     ([[in_mres_3xi1:%.+]]: tensor<3xi1>) -> tensor<1xi1>
        //       CHECK:   [[c0:%.+]] = arith.constant false
        //       CHECK:   [[empty_i1:%.+]] = tensor.empty() : tensor<i1>
        //       CHECK:   [[init_i1:%.+]] = linalg.fill
        //  CHECK-SAME:     ins([[c0]] : i1) outs([[empty_i1]] : tensor<i1>) -> tensor<i1>
        //       CHECK:   [[reduced_i1:%.+]] = linalg.reduce
        //  CHECK-SAME:     ins([[in_mres_3xi1]]:tensor<3xi1>) outs([[init_i1]]:tensor<i1>)
        //  CHECK-SAME:     dimensions = [0]
        //       CHECK:   ([[in:%.+]]: i1, [[out:%.+]]: i1) {
        //       CHECK:     [[xor:%.+]] = arith.xori [[in]], [[out]] : i1
        //       CHECK:     linalg.yield [[xor]] : i1
        //       CHECK:   [[expanded_1xi1:%.+]] = tensor.expand_shape [[reduced_i1]] []
        //  CHECK-SAME:     output_shape [1] : tensor<i1> into tensor<1xi1>
        //       CHECK:   func.return [[expanded_1xi1]] : tensor<1xi1>
        }
        """
        run_filecheck(program, qecl_to_qecp_steane_pipeline)

    def test_measure_toy_code(self, run_filecheck, get_generic_qec_code):
        """Test the lowering pattern for `qecl.measure` ops with a toy QEC code.

        Note that the transversal-measurement subroutine is essentially the same as the Steane code
        test above, so we don't test it again here. The main purpose of this test is to check that
        the physical-measurement decoding subroutine works for different values of n and Pauli Z
        observables.
        """
        program = """
        builtin.module {
        // CHECK-LABEL: test_program
        func.func @test_program() {
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 6>
            %0 = "test.op"() : () -> !qecl.codeblock<1>

            //      CHECK: [[mresp:%.+]], [[cb1:%.+]] = func.call @measure_transversal_TestCode([[cb0]]) :
            // CHECK-SAME:   (!qecp.codeblock<1 x 6>) -> (tensor<4xi1>, !qecp.codeblock<1 x 6>)
            //      CHECK: [[mresl:%.+]] = func.call @decode_physical_measurements_TestCode([[mresp]]) :
            // CHECK-SAME:   (tensor<4xi1>) -> tensor<1xi1>
            //      CHECK: [[zero:%.+]] = arith.constant 0 : index
            //      CHECK: [[mres0:%.+]] = tensor.extract [[mresl]][[[zero]]] : tensor<1xi1>
            %mres0, %1 = qecl.measure %0[0] : i1, !qecl.codeblock<1>

            // CHECK: [[mres1:%.+]] = "test.op"([[mres0]]) : (i1) -> i1
            %mres1 = "test.op"(%mres0) : (i1) -> i1  // To prevent DCE
            %2 = "test.op"(%1) : (!qecl.codeblock<1>) -> !qecl.codeblock<1>  // To prevent DCE
            return
        }
        // CHECK-LABEL: func.func private @measure_transversal_TestCode
        //  CHECK-SAME:     ([[cb_0:%.+]]: !qecp.codeblock<1 x 6>) -> (tensor<4xi1>, !qecp.codeblock<1 x 6>)
        //       CHECK:   [[q0_0:%.+]] = qecp.extract [[cb_0]][0]
        //       CHECK:   [[q2_0:%.+]] = qecp.extract [[cb_0]][2]
        //       CHECK:   [[q3_0:%.+]] = qecp.extract [[cb_0]][3]
        //       CHECK:   [[q4_0:%.+]] = qecp.extract [[cb_0]][4]
        //   CHECK-DAG:   [[m0:%.+]], [[q0_1:%.+]] = qecp.measure [[q0_0]]
        //   CHECK-DAG:   [[m2:%.+]], [[q2_1:%.+]] = qecp.measure [[q2_0]]
        //   CHECK-DAG:   [[q3_1:%.+]] = qecp.hadamard [[q3_0]]
        //   CHECK-DAG:   [[m3:%.+]], [[q3_2:%.+]] = qecp.measure [[q3_1]]
        //   CHECK-DAG:   [[q4_1:%.+]] = qecp.s [[q4_0]] adj
        //   CHECK-DAG:   [[q4_2:%.+]] = qecp.hadamard [[q4_1]]
        //   CHECK-DAG:   [[m4:%.+]], [[q4_3:%.+]] = qecp.measure [[q4_2]]
        //   CHECK-NOT:   qecp.measure
        //       CHECK:   [[cb_1:%.+]] = qecp.insert [[cb_0]][0], [[q0_1]]
        //       CHECK:   [[cb_2:%.+]] = qecp.insert [[cb_1]][2], [[q2_1]]
        //       CHECK:   [[cb_3:%.+]] = qecp.insert [[cb_2]][3], [[q3_2]]
        //       CHECK:   [[cb_4:%.+]] = qecp.insert [[cb_3]][4], [[q4_3]]
        //       CHECK:   [[m_4xi1:%.+]] = tensor.from_elements [[m0]], [[m2]], [[m3]], [[m4]] : tensor<4xi1>
        //       CHECK:   func.return [[m_4xi1]], [[cb_4]] : tensor<4xi1>, !qecp.codeblock<1 x 6>

        // CHECK-LABEL: func.func private @decode_physical_measurements_TestCode
        //  CHECK-SAME:     ([[in_mres_4xi1:%.+]]: tensor<4xi1>) -> tensor<1xi1>
        //       CHECK:   [[c0:%.+]] = arith.constant false
        //       CHECK:   [[empty_i1:%.+]] = tensor.empty() : tensor<i1>
        //       CHECK:   [[init_i1:%.+]] = linalg.fill
        //  CHECK-SAME:     ins([[c0]] : i1) outs([[empty_i1]] : tensor<i1>) -> tensor<i1>
        //       CHECK:   [[reduced_i1:%.+]] = linalg.reduce
        //  CHECK-SAME:     ins([[in_mres_4xi1]]:tensor<4xi1>) outs([[init_i1]]:tensor<i1>)
        //  CHECK-SAME:     dimensions = [0]
        //       CHECK:   ([[in:%.+]]: i1, [[out:%.+]]: i1) {
        //       CHECK:     [[xor:%.+]] = arith.xori [[in]], [[out]] : i1
        //       CHECK:     linalg.yield [[xor]] : i1
        //       CHECK:   [[expanded_1xi1:%.+]] = tensor.expand_shape [[reduced_i1]] []
        //  CHECK-SAME:     output_shape [1] : tensor<i1> into tensor<1xi1>
        //       CHECK:   func.return [[expanded_1xi1]] : tensor<1xi1>
        }
        """
        qec_code = get_generic_qec_code(
            n=6,
            k=1,
            d=3,
            transversal_1q_gates={"z": ("Z", "I", "Z", "X", "Y", "I")},
        )
        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=qec_code),)

        run_filecheck(program, pipeline)

    @pytest.mark.parametrize(
        "gate_data",
        [
            (),
        ],
    )
    def test_measure_with_missing_pauli_z_def_raise(
        self, run_filecheck, gate_data, get_generic_qec_code
    ):
        """Test that running the convert-qecl-to-qecp pass without specifying a logical Z observable
        raise an error when creating the transversal-measurement subroutine.
        """
        program = """
        builtin.module {
        // CHECK-LABEL: test_program
        func.func @test_program() {
            %0 = "test.op"() : () -> !qecl.codeblock<1>
            %mres0, %1 = qecl.measure %0[0] : i1, !qecl.codeblock<1>
            %mres1 = "test.op"(%mres0) : (i1) -> i1  // To prevent DCE
            %2 = "test.op"(%1) : (!qecl.codeblock<1>) -> !qecl.codeblock<1>  // To prevent DCE
            return
        }
        }
        """

        qec_code = get_generic_qec_code(
            n=7,
            k=1,
            d=3,
            transversal_1q_gates=dict(gate_data),
        )
        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=qec_code),)

        with pytest.raises(
            CompileError, match="Failed to create transversal-measurement subroutine"
        ):
            run_filecheck(program, pipeline)

    def test_no_subroutine_if_no_measure(self, get_generic_qec_code, run_filecheck):
        """Test that the measure subroutine isn't generated and added to the code if it isn't needed"""

        program = """
        builtin.module {
        // CHECK-LABEL: test_program
        // CHECK-NOT: func.call @measure_transversal_TestCode
        func.func @test_program() {
            %0 = "test.op"() : () -> !qecl.codeblock<1>
            %1 = qecl.x %0[0] : !qecl.codeblock<1>
            return
        }
        }
        """

        qec_code = get_generic_qec_code(
            n=7,
            k=1,
            d=3,
            transversal_1q_gates={"x": ("X",) * 7},
        )
        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=qec_code),)

        # no z operator defined, a program with measure would fail with a compilation error
        run_filecheck(program, pipeline)


# MARK: TransversalGates


class TestLoweringTransversalGates:
    """Unit tests for lowering transversal gates in the convert-qecl-to-qecp pass."""

    def test_single_qubit_op_lowering_generic(self, run_filecheck, get_generic_qec_code):
        """Test that a generic QEC code lowers ops as instructed. In this case (n=3,
        x is transversal and applied on indicies 0 and 2), we expect to extract 3
        qubits, apply the pattern XIX on them, and re-insert them."""

        n, k = (3, 1)

        qec_code = get_generic_qec_code(
            n=n,
            k=k,
            d=1,
            transversal_1q_gates={"x": ("X", "I", "X"), "z": ("Z", "I", "Z")},
        )

        program = f"""
        builtin.module @module_circuit {{
                func.func @test_func() attributes {{quantum.node}} {{
                    // CHECK: [[codeblock:%.+]] = "test.op"() : () -> !qecp.codeblock<{k} x {n}>
                    // CHECK-NEXT: [[codeblock2:%.+]] = func.call @x_TestCode([[codeblock]]) : (!qecp.codeblock<{k} x {n}>) -> !qecp.codeblock<{k} x {n}>
                    // CHECK-NOT: qecl.x
                    %0 = "test.op"() : () -> !qecl.codeblock<{k}>
                    %1 = qecl.x %0[0] : !qecl.codeblock<{k}>
                    return
                }}
                // CHECK: func.func private @x_TestCode([[codeblock_in:%.+]]: !qecp.codeblock<{k} x {n}>)
                // CHECK-NEXT: [[q0:%.+]] = qecp.extract [[codeblock_in]][0] : !qecp.codeblock<{k} x {n}> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q1:%.+]] = qecp.extract [[codeblock_in]][1] : !qecp.codeblock<{k} x {n}> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q2:%.+]] = qecp.extract [[codeblock_in]][2] : !qecp.codeblock<{k} x {n}> -> !qecp.qubit<data>
                // CHECK: [[q0_1:%.+]] = qecp.x [[q0]] : !qecp.qubit<data>
                // CHECK: [[q1_1:%.+]] = qecp.identity [[q1]] : !qecp.qubit<data>
                // CHECK: [[q2_1:%.+]] = qecp.x [[q2]] : !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in1:%.+]] = qecp.insert [[codeblock_in]][0], [[q0_1]] : !qecp.codeblock<{k} x {n}>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in2:%.+]] = qecp.insert [[codeblock_in1]][1], [[q1_1]] : !qecp.codeblock<{k} x {n}>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_out:%.+]] = qecp.insert [[codeblock_in2]][2], [[q2_1]] : !qecp.codeblock<{k} x {n}>, !qecp.qubit<data>
                // CHECK-NEXT: func.return [[codeblock_out]]
            }}
            """

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=qec_code),)
        run_filecheck(program, pipeline)

    def test_two_qubit_op_lowering_generic(self, run_filecheck, get_generic_qec_code):
        """Test that a generic QEC code lowers ops as instructed. In this case (n=3,
        x is transversal and applied on indicies 0 and 2), we expect to extract 3
        qubits, apply the pattern XIX on them, and re-insert them."""

        n, k = (3, 1)

        qec_code = get_generic_qec_code(
            n=n,
            k=k,
            d=1,
            transversal_1q_gates={"z": ("Z", "I", "I")},
            transversal_2q_gates={"cnot": "CNOT"},
        )

        program = f"""
        builtin.module @module_circuit {{
                func.func @test_func() attributes {{quantum.node}} {{
                    // CHECK: [[ctrl_codeblock:%.+]] = "test.op"() : () -> !qecp.codeblock<{k} x {n}>
                    // CHECK-NEXT: [[trgt_codeblock:%.+]] = "test.op"() : () -> !qecp.codeblock<{k} x {n}>
                    // CHECK-NEXT: [[ctrl_codeblock_out:%.+]], [[trgt_codeblock_out:%.+]] = func.call @cnot_TestCode([[ctrl_codeblock]], [[trgt_codeblock]])  : (!qecp.codeblock<{k} x {n}>, !qecp.codeblock<{k} x {n}>) -> (!qecp.codeblock<{k} x {n}>, !qecp.codeblock<{k} x {n}>)
                    // CHECK-NOT: qecl.cnot
                    %0 = "test.op"() : () -> !qecl.codeblock<{k}>
                    %1 = "test.op"() : () -> !qecl.codeblock<{k}>
                    %2, %3 = qecl.cnot %0[0], %1[0] : !qecl.codeblock<{k}>, !qecl.codeblock<{k}>
                    return
                }}
                // CHECK: func.func private @cnot_TestCode([[ctrl_cb_in:%.+]]: !qecp.codeblock<{k} x {n}>, [[trgt_cb_in:%.+]]: !qecp.codeblock<{k} x {n}>)
                // CHECK-NEXT: [[ctrl_q0:%.+]] = qecp.extract [[ctrl_cb_in]][0] : !qecp.codeblock<{k} x {n}> -> !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_q1:%.+]] = qecp.extract [[ctrl_cb_in]][1] : !qecp.codeblock<{k} x {n}> -> !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_q2:%.+]] = qecp.extract [[ctrl_cb_in]][2] : !qecp.codeblock<{k} x {n}> -> !qecp.qubit<data>
                // CHECK-NEXT: [[trgt_q0:%.+]] = qecp.extract [[trgt_cb_in]][0] : !qecp.codeblock<{k} x {n}> -> !qecp.qubit<data>
                // CHECK-NEXT: [[trgt_q1:%.+]] = qecp.extract [[trgt_cb_in]][1] : !qecp.codeblock<{k} x {n}> -> !qecp.qubit<data>
                // CHECK-NEXT: [[trgt_q2:%.+]] = qecp.extract [[trgt_cb_in]][2] : !qecp.codeblock<{k} x {n}> -> !qecp.qubit<data>
                // CHECK: [[ctrl_q0_1:%.+]], [[trgt_q0_1:%.+]] = qecp.cnot [[ctrl_q0]], [[trgt_q0]] : !qecp.qubit<data>
                // CHECK: [[ctrl_q1_1:%.+]], [[trgt_q1_1:%.+]] = qecp.cnot [[ctrl_q1]], [[trgt_q1]] : !qecp.qubit<data>
                // CHECK: [[ctrl_q2_1:%.+]], [[trgt_q2_1:%.+]] = qecp.cnot [[ctrl_q2]], [[trgt_q2]] : !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_cb_in1:%.+]] = qecp.insert [[ctrl_cb_in]][0], [[ctrl_q0_1]] : !qecp.codeblock<{k} x {n}>, !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_cb_in2:%.+]] = qecp.insert [[ctrl_cb_in1]][1], [[ctrl_q1_1]] : !qecp.codeblock<{k} x {n}>, !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_cb_out:%.+]] = qecp.insert [[ctrl_cb_in2]][2], [[ctrl_q2_1]] : !qecp.codeblock<{k} x {n}>, !qecp.qubit<data>
                // CHECK-NEXT: [[trgt_cb_in1:%.+]] = qecp.insert [[trgt_cb_in]][0], [[trgt_q0_1]] : !qecp.codeblock<{k} x {n}>, !qecp.qubit<data>
                // CHECK-NEXT: [[trgt_cb_in2:%.+]] = qecp.insert [[trgt_cb_in1]][1], [[trgt_q1_1]] : !qecp.codeblock<{k} x {n}>, !qecp.qubit<data>
                // CHECK-NEXT: [[trgt_cb_out:%.+]] = qecp.insert [[trgt_cb_in2]][2], [[trgt_q2_1]] : !qecp.codeblock<{k} x {n}>, !qecp.qubit<data>
                // CHECK-NEXT: func.return [[ctrl_cb_out]], [[trgt_cb_out]]
            }}
            """

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=qec_code),)
        run_filecheck(program, pipeline)

    @pytest.mark.parametrize("gate", ("x", "y", "z"))
    def test_pauli_lowering_Steane(self, gate, run_filecheck, qecl_to_qecp_steane_pipeline):
        """Test that using the Steane code lowers Pauli ops as expected. These ops are applied
        on the last 3 qubits in the codeblock, and identity is applied to the rest."""

        program = f"""
        builtin.module @module_circuit {{
                func.func @test_func() attributes {{quantum.node}} {{
                    // CHECK: [[codeblock:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
                    // CHECK-NEXT: [[codeblock2:%.+]] = func.call @{gate}_Steane([[codeblock]]) : (!qecp.codeblock<1 x 7>) -> !qecp.codeblock<1 x 7>
                    // CHECK-NOT: qecl.{gate}
                    %0 = "test.op"() : () -> !qecl.codeblock<1>
                    %1 = qecl.{gate} %0[0] : !qecl.codeblock<1>
                    return
                }}
                // CHECK: func.func private @{gate}_Steane([[codeblock_in:%.+]]: !qecp.codeblock<1 x 7>)
                // CHECK-NEXT: [[q0:%.+]] = qecp.extract [[codeblock_in]][0] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q1:%.+]] = qecp.extract [[codeblock_in]][1] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q2:%.+]] = qecp.extract [[codeblock_in]][2] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q3:%.+]] = qecp.extract [[codeblock_in]][3] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q4:%.+]] = qecp.extract [[codeblock_in]][4] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q5:%.+]] = qecp.extract [[codeblock_in]][5] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q6:%.+]] = qecp.extract [[codeblock_in]][6] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK: [[q0_1:%.+]] = qecp.identity [[q0]] : !qecp.qubit<data>
                // CHECK: [[q1_1:%.+]] = qecp.identity [[q1]] : !qecp.qubit<data>
                // CHECK: [[q2_1:%.+]] = qecp.identity [[q2]] : !qecp.qubit<data>
                // CHECK: [[q3_1:%.+]] = qecp.identity [[q3]] : !qecp.qubit<data>
                // CHECK: [[q4_1:%.+]] = qecp.{gate} [[q4]] : !qecp.qubit<data>
                // CHECK: [[q5_1:%.+]] = qecp.{gate} [[q5]] : !qecp.qubit<data>
                // CHECK: [[q6_1:%.+]] = qecp.{gate} [[q6]] : !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in1:%.+]] = qecp.insert [[codeblock_in]][0], [[q0_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in2:%.+]] = qecp.insert [[codeblock_in1]][1], [[q1_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in3:%.+]] = qecp.insert [[codeblock_in2]][2], [[q2_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in4:%.+]] = qecp.insert [[codeblock_in3]][3], [[q3_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in5:%.+]] = qecp.insert [[codeblock_in4]][4], [[q4_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in6:%.+]] = qecp.insert [[codeblock_in5]][5], [[q5_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_out:%.+]] = qecp.insert [[codeblock_in6]][6], [[q6_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: func.return [[codeblock_out]]
            }}
            """

        run_filecheck(program, qecl_to_qecp_steane_pipeline)

    @pytest.mark.parametrize("gate, adj", [("s", "adj"), ("hadamard", ""), ("identity", "")])
    def test_single_qubit_gate_lowering_Steane(
        self, gate, adj, run_filecheck, qecl_to_qecp_steane_pipeline
    ):
        """Test that using the Steane code lowers Hadamard, Identity and S ops as expected. These ops
        are applied on all qubits in the codeblock. For the S operator, the adjoint is applied."""

        program = f"""
        builtin.module @module_circuit {{
                func.func @test_func() attributes {{quantum.node}} {{
                    // CHECK: [[codeblock:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
                    // CHECK-NEXT: [[codeblock2:%.+]] = func.call @{gate}_Steane([[codeblock]]) : (!qecp.codeblock<1 x 7>) -> !qecp.codeblock<1 x 7>
                    // CHECK-NOT: qecl.{gate}
                    %0 = "test.op"() : () -> !qecl.codeblock<1>
                    %1 = qecl.{gate} %0[0] : !qecl.codeblock<1>
                    return
                }}
                // CHECK: func.func private @{gate}_Steane([[codeblock_in:%.+]]: !qecp.codeblock<1 x 7>)
                // CHECK-NEXT: [[q0:%.+]] = qecp.extract [[codeblock_in]][0] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q1:%.+]] = qecp.extract [[codeblock_in]][1] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q2:%.+]] = qecp.extract [[codeblock_in]][2] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q3:%.+]] = qecp.extract [[codeblock_in]][3] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q4:%.+]] = qecp.extract [[codeblock_in]][4] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q5:%.+]] = qecp.extract [[codeblock_in]][5] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q6:%.+]] = qecp.extract [[codeblock_in]][6] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK: [[q0_1:%.+]] = qecp.{gate} [[q0]] {adj} : !qecp.qubit<data>
                // CHECK: [[q1_1:%.+]] = qecp.{gate} [[q1]] {adj} : !qecp.qubit<data>
                // CHECK: [[q2_1:%.+]] = qecp.{gate} [[q2]] {adj} : !qecp.qubit<data>
                // CHECK: [[q3_1:%.+]] = qecp.{gate} [[q3]] {adj} : !qecp.qubit<data>
                // CHECK: [[q4_1:%.+]] = qecp.{gate} [[q4]] {adj} : !qecp.qubit<data>
                // CHECK: [[q5_1:%.+]] = qecp.{gate} [[q5]] {adj} : !qecp.qubit<data>
                // CHECK: [[q6_1:%.+]] = qecp.{gate} [[q6]] {adj} : !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in1:%.+]] = qecp.insert [[codeblock_in]][0], [[q0_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in2:%.+]] = qecp.insert [[codeblock_in1]][1], [[q1_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in3:%.+]] = qecp.insert [[codeblock_in2]][2], [[q2_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in4:%.+]] = qecp.insert [[codeblock_in3]][3], [[q3_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in5:%.+]] = qecp.insert [[codeblock_in4]][4], [[q4_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in6:%.+]] = qecp.insert [[codeblock_in5]][5], [[q5_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_out:%.+]] = qecp.insert [[codeblock_in6]][6], [[q6_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: func.return [[codeblock_out]]
            }}
            """

        run_filecheck(program, qecl_to_qecp_steane_pipeline)

    def test_adjoint_s_lowering_Steane(self, run_filecheck, qecl_to_qecp_steane_pipeline):
        """Test that using the Steane code lowers S adjoint as expected (the physical S operator
        is applied to all qubits in the codeblock)."""

        program = """
        builtin.module @module_circuit {
                func.func @test_func() attributes {quantum.node} {
                    // CHECK: [[codeblock:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
                    // CHECK-NEXT: [[codeblock2:%.+]] = func.call @s_adj_Steane([[codeblock]]) : (!qecp.codeblock<1 x 7>) -> !qecp.codeblock<1 x 7>
                    // CHECK-NOT: qecl.s
                    %0 = "test.op"() : () -> !qecl.codeblock<1>
                    %1 = qecl.s %0[0] adj : !qecl.codeblock<1>
                    return
                }
                // CHECK: func.func private @s_adj_Steane([[codeblock_in:%.+]]: !qecp.codeblock<1 x 7>)
                // CHECK-NEXT: [[q0:%.+]] = qecp.extract [[codeblock_in]][0] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q1:%.+]] = qecp.extract [[codeblock_in]][1] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q2:%.+]] = qecp.extract [[codeblock_in]][2] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q3:%.+]] = qecp.extract [[codeblock_in]][3] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q4:%.+]] = qecp.extract [[codeblock_in]][4] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q5:%.+]] = qecp.extract [[codeblock_in]][5] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[q6:%.+]] = qecp.extract [[codeblock_in]][6] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK: [[q0_1:%.+]] = qecp.s [[q0]] : !qecp.qubit<data>
                // CHECK: [[q1_1:%.+]] = qecp.s [[q1]] : !qecp.qubit<data>
                // CHECK: [[q2_1:%.+]] = qecp.s [[q2]] : !qecp.qubit<data>
                // CHECK: [[q3_1:%.+]] = qecp.s [[q3]] : !qecp.qubit<data>
                // CHECK: [[q4_1:%.+]] = qecp.s [[q4]] : !qecp.qubit<data>
                // CHECK: [[q5_1:%.+]] = qecp.s [[q5]] : !qecp.qubit<data>
                // CHECK: [[q6_1:%.+]] = qecp.s [[q6]] : !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in1:%.+]] = qecp.insert [[codeblock_in]][0], [[q0_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in2:%.+]] = qecp.insert [[codeblock_in1]][1], [[q1_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in3:%.+]] = qecp.insert [[codeblock_in2]][2], [[q2_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in4:%.+]] = qecp.insert [[codeblock_in3]][3], [[q3_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in5:%.+]] = qecp.insert [[codeblock_in4]][4], [[q4_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_in6:%.+]] = qecp.insert [[codeblock_in5]][5], [[q5_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[codeblock_out:%.+]] = qecp.insert [[codeblock_in6]][6], [[q6_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: func.return [[codeblock_out]]
            }
            """

        run_filecheck(program, qecl_to_qecp_steane_pipeline)

    def test_cnot_lowering_Steane(self, run_filecheck, qecl_to_qecp_steane_pipeline):
        """Test that using the Steane code lowers ops as expected"""

        program = """
        builtin.module @module_circuit {
                func.func @test_func() attributes {quantum.node} {
                    // CHECK: [[ctrl_codeblock:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
                    // CHECK: [[trgt_codeblock:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
                    // CHECK-NEXT: [[ctrl_codeblock_out:%.+]], [[trgt_codeblock_out:%.+]] = func.call @cnot_Steane([[ctrl_codeblock]], [[trgt_codeblock]]) : (!qecp.codeblock<1 x 7>, !qecp.codeblock<1 x 7>) -> (!qecp.codeblock<1 x 7>, !qecp.codeblock<1 x 7>)
                    // CHECK-NOT: qecl.cnot
                    %0 = "test.op"() : () -> !qecl.codeblock<1>
                    %1 = "test.op"() : () -> !qecl.codeblock<1>
                    %2, %3 = qecl.cnot %0[0], %1[0] : !qecl.codeblock<1>, !qecl.codeblock<1>
                    return
                }
                // CHECK: func.func private @cnot_Steane([[ctrl_cb_in:%.+]]: !qecp.codeblock<1 x 7>, [[tgt_cb_in:%.+]]: !qecp.codeblock<1 x 7>)
                // CHECK-NEXT: [[ctrl_q0:%.+]] = qecp.extract [[ctrl_cb_in]][0] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_q1:%.+]] = qecp.extract [[ctrl_cb_in]][1] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_q2:%.+]] = qecp.extract [[ctrl_cb_in]][2] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_q3:%.+]] = qecp.extract [[ctrl_cb_in]][3] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_q4:%.+]] = qecp.extract [[ctrl_cb_in]][4] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_q5:%.+]] = qecp.extract [[ctrl_cb_in]][5] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_q6:%.+]] = qecp.extract [[ctrl_cb_in]][6] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[tgt_q0:%.+]] = qecp.extract [[tgt_cb_in]][0] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[tgt_q1:%.+]] = qecp.extract [[tgt_cb_in]][1] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[tgt_q2:%.+]] = qecp.extract [[tgt_cb_in]][2] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[tgt_q3:%.+]] = qecp.extract [[tgt_cb_in]][3] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[tgt_q4:%.+]] = qecp.extract [[tgt_cb_in]][4] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[tgt_q5:%.+]] = qecp.extract [[tgt_cb_in]][5] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK-NEXT: [[tgt_q6:%.+]] = qecp.extract [[tgt_cb_in]][6] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
                // CHECK: [[ctrl_q0_1:%.+]], [[tgt_q0_1:%.+]] = qecp.cnot [[ctrl_q0]], [[tgt_q0]] : !qecp.qubit<data>, !qecp.qubit<data>
                // CHECK: [[ctrl_q1_1:%.+]], [[tgt_q1_1:%.+]] = qecp.cnot [[ctrl_q1]], [[tgt_q1]] : !qecp.qubit<data>, !qecp.qubit<data>
                // CHECK: [[ctrl_q2_1:%.+]], [[tgt_q2_1:%.+]] = qecp.cnot [[ctrl_q2]], [[tgt_q2]] : !qecp.qubit<data>, !qecp.qubit<data
                // CHECK: [[ctrl_q3_1:%.+]], [[tgt_q3_1:%.+]] = qecp.cnot [[ctrl_q3]], [[tgt_q3]] : !qecp.qubit<data>, !qecp.qubit<data
                // CHECK: [[ctrl_q4_1:%.+]], [[tgt_q4_1:%.+]] = qecp.cnot [[ctrl_q4]], [[tgt_q4]] : !qecp.qubit<data>, !qecp.qubit<data
                // CHECK: [[ctrl_q5_1:%.+]], [[tgt_q5_1:%.+]] = qecp.cnot [[ctrl_q5]], [[tgt_q5]] : !qecp.qubit<data>, !qecp.qubit<data
                // CHECK: [[ctrl_q6_1:%.+]], [[tgt_q6_1:%.+]] = qecp.cnot [[ctrl_q6]], [[tgt_q6]] : !qecp.qubit<data>, !qecp.qubit<data
                // CHECK-NEXT: [[ctrl_cb_in1:%.+]] = qecp.insert [[ctrl_cb_in]][0], [[ctrl_q0_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_cb_in2:%.+]] = qecp.insert [[ctrl_cb_in1]][1], [[ctrl_q1_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_cb_in3:%.+]] = qecp.insert [[ctrl_cb_in2]][2], [[ctrl_q2_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_cb_in4:%.+]] = qecp.insert [[ctrl_cb_in3]][3], [[ctrl_q3_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_cb_in5:%.+]] = qecp.insert [[ctrl_cb_in4]][4], [[ctrl_q4_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_cb_in6:%.+]] = qecp.insert [[ctrl_cb_in5]][5], [[ctrl_q5_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[ctrl_cb_out:%.+]] = qecp.insert [[ctrl_cb_in6]][6], [[ctrl_q6_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[tgt_cb_in1:%.+]] = qecp.insert [[tgt_cb_in]][0], [[tgt_q0_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[tgt_cb_in2:%.+]] = qecp.insert [[tgt_cb_in1]][1], [[tgt_q1_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[tgt_cb_in3:%.+]] = qecp.insert [[tgt_cb_in2]][2], [[tgt_q2_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[tgt_cb_in4:%.+]] = qecp.insert [[tgt_cb_in3]][3], [[tgt_q3_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[tgt_cb_in5:%.+]] = qecp.insert [[tgt_cb_in4]][4], [[tgt_q4_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[tgt_cb_in6:%.+]] = qecp.insert [[tgt_cb_in5]][5], [[tgt_q5_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: [[tgt_cb_out:%.+]] = qecp.insert [[tgt_cb_in6]][6], [[tgt_q6_1]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
                // CHECK-NEXT: func.return [[ctrl_cb_out]], [[tgt_cb_out]]
            }
            """

        run_filecheck(program, qecl_to_qecp_steane_pipeline)

    def test_nontransveral_ops_ignored(self, run_filecheck, get_generic_qec_code):
        """Test that a generic QEC code lowers ops as instructed"""

        n, k = (3, 1)

        qec_code = get_generic_qec_code(
            n=n,
            k=k,
            d=1,
            transversal_1q_gates={"x": ("X", "X", "I"), "z": ("Z", "Z", "I")},
        )

        program = f"""
        builtin.module @module_circuit {{
                func.func @test_func() attributes {{quantum.node}} {{
                    // CHECK: [[codeblock:%.+]] = "test.op"() : () -> !qecp.codeblock<{k} x {n}>
                    // CHECK-NEXT: [[codeblock2:%.+]] = func.call @x_TestCode([[codeblock]]) : (!qecp.codeblock<{k} x {n}>) -> !qecp.codeblock<{k} x {n}>
                    // CHECK-NOT: qecl.x
                    // CHECK: qecl.y
                    %0 = "test.op"() : () -> !qecl.codeblock<{k}>
                    %1 = qecl.x %0[0] : !qecl.codeblock<{k}>
                    %2 = qecl.y %1[0] : !qecl.codeblock<{k}>
                    return
                }}
                // CHECK: func.func private @x_TestCode([[codeblock_in:%.+]]: !qecp.codeblock<{k} x {n}>)
            }}
            """

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=qec_code),)
        run_filecheck(program, pipeline)

    def test_only_needed_op_subroutines(self, qecl_to_qecp_steane_pipeline, run_filecheck):
        """Test that only the needed gate subroutine are generated for the circuit"""

        program = """
        builtin.module @module_circuit {
                func.func @test_func() attributes {quantum.node} {
                    %0 = "test.op"() : () -> !qecl.codeblock<1>
                    %1 = qecl.hadamard %0[0] : !qecl.codeblock<1>
                    %2 = qecl.s %1[0] adj : !qecl.codeblock<1>
                    return
                }
                // CHECK: func.func private @hadamard_Steane
                // CHECK: func.func private @s_adj_Steane
                // CHECK-NOT: func.func private @x_Steane
                // CHECK-NOT: func.func private @y_Steane
                // CHECK-NOT: func.func private @z_Steane
                // CHECK-NOT: func.func private @s_Steane
            }
            """

        run_filecheck(program, qecl_to_qecp_steane_pipeline)


# Mark: FabricateOp


class TestLoweringFabricateOp:
    """Test lowering for the qecl.fabricate op"""

    @pytest.mark.parametrize("init_state, adj", [("magic", ""), ("magic_conj", " adj")])
    def test_lower_fabricate_toy_code(self, init_state, adj, run_filecheck, get_generic_qec_code):
        """Test that `qecl.fabricate [magic]` op lowers to a call to the magic-state
        fabrication subroutine with a toy code, and that the subroutine performs H-T
        state injection on the code's state_prep_index followed by the unitary encoding
        as defined by `hadamard_indices` and `cnot_indices`."""

        qec_code = get_generic_qec_code(
            n=3,
            k=1,
            d=1,
            unitary_encoding={
                "ops": [
                    ("H", [0]),
                    ("H", [2]),
                    ("CNOT", [0, 1]),
                    ("CNOT", [2, 0]),
                ],
                "state_prep_index": 1,
            },
        )

        program = f"""
        builtin.module @module_circuit {{
            func.func @test_func() attributes {{quantum.node}} {{
                // CHECK:   [[magic_cb:%.+]] = func.call @fabricate_{init_state}_TestCode() : () -> !qecp.codeblock<1 x 3>
                %0 = qecl.fabricate[{init_state}] : !qecl.codeblock<1>
                return
            }}
            // CHECK-LABEL: func.func private @fabricate_{init_state}_TestCode() -> !qecp.codeblock<1 x 3>
            //       CHECK:   [[cb:%.+]] = qecp.alloc_cb : !qecp.codeblock<1 x 3>
            // Extract qubits
            //       CHECK-DAG:   [[q0:%.+]] = qecp.extract [[cb]][0] : !qecp.codeblock<1 x 3> -> !qecp.qubit<data>
            //       CHECK-DAG:   [[q1:%.+]] = qecp.extract [[cb]][1] : !qecp.codeblock<1 x 3> -> !qecp.qubit<data>
            //       CHECK-DAG:   [[q2:%.+]] = qecp.extract [[cb]][2] : !qecp.codeblock<1 x 3> -> !qecp.qubit<data>
            // State injection on the state_prep_index (q1)
            //       CHECK:   [[q1_1:%.+]] = qecp.hadamard [[q1]] : !qecp.qubit<data>
            //       CHECK:   [[q1_2:%.+]] = qecp.t [[q1_1]]{adj} : !qecp.qubit<data>
            // Unitary encoding
            //       CHECK:   [[q0_1:%.+]] = qecp.hadamard [[q0]] : !qecp.qubit<data>
            //       CHECK:   [[q2_1:%.+]] = qecp.hadamard [[q2]] : !qecp.qubit<data>
            //       CHECK:   [[q0_2:%.+]], [[q1_out:%.+]] = qecp.cnot [[q0_1]], [[q1_2]] : !qecp.qubit<data>, !qecp.qubit<data>
            //       CHECK:   [[q2_out:%.+]], [[q0_out:%.+]] = qecp.cnot [[q2_1]], [[q0_2]] : !qecp.qubit<data>, !qecp.qubit<data>
            // Insert qubits and return
            //       CHECK:   [[cb_1:%.+]] = qecp.insert [[cb]][0], [[q0_out]]
            //       CHECK:   [[cb_2:%.+]] = qecp.insert [[cb_1]][1], [[q1_out]]
            //       CHECK:   [[cb_3:%.+]] = qecp.insert [[cb_2]][2], [[q2_out]]
            //       CHECK:   func.return [[cb_3:%.+]] : !qecp.codeblock<1 x 3>
        }}
        """
        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=qec_code),)
        run_filecheck(program, pipeline)

    @pytest.mark.parametrize("init_state, adj", [("magic", ""), ("magic_conj", " adj")])
    def test_fabricate_lowering_steane(
        self, init_state, adj, run_filecheck, qecl_to_qecp_steane_pipeline
    ):
        """Test that `qecl.fabricate [magic]` op lowers to a call to the magic-state
        fabrication subroutine when using the Steane code, and that the subroutine performs
        H-T state injection on the Steane code's state_prep_index (qubit 6) followed by the
        unitary encoding."""

        program = f"""
        builtin.module @module_circuit {{
            func.func @test_func() attributes {{quantum.node}} {{
                // CHECK:   [[magic_cb:%.+]] = func.call @fabricate_{init_state}_Steane() : () -> !qecp.codeblock<1 x 7>
                %0 = qecl.fabricate[{init_state}] : !qecl.codeblock<1>
                return
            }}
            // CHECK-LABEL: func.func private @fabricate_{init_state}_Steane() -> !qecp.codeblock<1 x 7>
            //       CHECK:   [[cb:%.+]] = qecp.alloc_cb : !qecp.codeblock<1 x 7>
            //       CHECK-DAG:   [[q0:%.+]] = qecp.extract [[cb]][0] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
            //       CHECK-DAG:   [[q1:%.+]] = qecp.extract [[cb]][1] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
            //       CHECK-DAG:   [[q2:%.+]] = qecp.extract [[cb]][2] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
            //       CHECK-DAG:   [[q3:%.+]] = qecp.extract [[cb]][3] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
            //       CHECK-DAG:   [[q4:%.+]] = qecp.extract [[cb]][4] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
            //       CHECK-DAG:   [[q5:%.+]] = qecp.extract [[cb]][5] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
            //       CHECK-DAG:   [[q6:%.+]] = qecp.extract [[cb]][6] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
            // State injection on the state_prep_index (qubit 6): H then T
            //       CHECK-NEXT:   [[h_inj:%.+]] = qecp.hadamard [[q6]] : !qecp.qubit<data>
            //       CHECK-NEXT:   [[t_inj:%.+]] = qecp.t [[h_inj]]{adj} : !qecp.qubit<data>
            // Unitary encoding: Hadamards on indices 1, 2, 3
            //       CHECK-NEXT:   [[h1:%.+]] = qecp.hadamard [[q1]] : !qecp.qubit<data>
            //       CHECK-NEXT:   [[h2:%.+]] = qecp.hadamard [[q2]] : !qecp.qubit<data>
            //       CHECK-NEXT:   [[h3:%.+]] = qecp.hadamard [[q3]] : !qecp.qubit<data>
            // First few CNOTs of the encoding circuit
            //       CHECK-NEXT:   qecp.cnot [[h1]], [[q0]] : !qecp.qubit<data>, !qecp.qubit<data>
            //       CHECK-NEXT:   qecp.cnot [[h2]], [[q4]] : !qecp.qubit<data>, !qecp.qubit<data>
            //       CHECK-NEXT:   qecp.cnot [[t_inj]], [[q5]] : !qecp.qubit<data>, !qecp.qubit<data>
        }}
        """
        run_filecheck(program, qecl_to_qecp_steane_pipeline)

    def test_apply_t_steane(self, run_filecheck, qecl_to_qecp_steane_pipeline):
        """Test that the call signature for the apply_T subroutine is updated as expected."""

        program = """
        builtin.module @module_circuit {
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
                %0 = "test.op"() : () -> !qecl.codeblock<1>

                // CHECK: [[cb1:%.+]] = func.call @apply_T([[cb0]]) : (!qecp.codeblock<1 x 7>) -> !qecp.codeblock<1 x 7>
                %2 = func.call @apply_T(%0) : (!qecl.codeblock<1>) -> !qecl.codeblock<1>
                return
            }
            //      CHECK-LABEL: func.func private @apply_T([[in_codeblock:%.+]]: !qecp.codeblock<1 x 7>)
            // CHECK: func.call @fabricate_magic_Steane() : () -> !qecp.codeblock<1 x 7>
            // CHECK: qecp.dealloc_cb
            func.func private @apply_T(%0: !qecl.codeblock<1>) -> !qecl.codeblock<1> {
                %1 = qecl.fabricate[magic] : !qecl.codeblock<1>
                qecl.dealloc_cb %0 : !qecl.codeblock<1>
                func.return %1 : !qecl.codeblock<1>
            }
            //      CHECK-LABEL: func.func private @fabricate_magic_Steane
            // CHECK: qecp.alloc_cb
            // CHECK: qecp.h
            // CHECK: qecp.t [[qb:%.+]]
            // CHECK-NOT: qecp.t [[qb:%.+]] adj
            // CHECK: qecp.h
            // CHECK: qecp.cnot
            // CHECK-NOT: func.func private @fabricate_magic_conj_Steane()
        }
        """
        run_filecheck(program, qecl_to_qecp_steane_pipeline)

    def test_apply_adj_t_steane(self, run_filecheck, qecl_to_qecp_steane_pipeline):
        """Test that the call signature for the apply_T_adj subroutine is updated as expected."""

        program = """
        builtin.module @module_circuit {
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
                %0 = "test.op"() : () -> !qecl.codeblock<1>

                // CHECK: [[cb1:%.+]] = func.call @apply_T_adj([[cb0]]) : (!qecp.codeblock<1 x 7>) -> !qecp.codeblock<1 x 7>
                %1 = func.call @apply_T_adj(%0) : (!qecl.codeblock<1>) -> !qecl.codeblock<1>
                return
            }
            //      CHECK-LABEL: func.func private @apply_T_adj([[in_codeblock:%.+]]: !qecp.codeblock<1 x 7>)
            // CHECK: func.call @fabricate_magic_conj_Steane() : () -> !qecp.codeblock<1 x 7>
            func.func private @apply_T_adj(%0: !qecl.codeblock<1>) -> !qecl.codeblock<1> {
                %1 = qecl.fabricate[magic_conj] : !qecl.codeblock<1>
                qecl.dealloc_cb %0 : !qecl.codeblock<1>
                func.return %1 : !qecl.codeblock<1>
            }
            // CHECK-NOT: func.func private @fabricate_magic_Steane
            //      CHECK-LABEL: func.func private @fabricate_magic_conj_Steane
            // CHECK: qecp.alloc_cb
            // CHECK: qecp.h
            // CHECK: qecp.t [[qb:%.+]] adj
            // CHECK-NOT: qecp.t [[qb:%.+]] :
            // CHECK: qecp.h
            // CHECK: qecp.cnot
        }
        """
        run_filecheck(program, qecl_to_qecp_steane_pipeline)


# MARK: Control Flow


class TestControlFlow:
    """Test suite for control-flow conversion patterns in the convert-qecl-to-qecp pass."""

    def test_scf_if_integration(self, run_filecheck_qjit):
        """Test the convert-quantum-to-qecl and convert-qecl-to-qecp passes together on a simple
        program with an if statement.
        """
        dev = qp.device("null.qubit", wires=1)

        @qp.qjit(capture=True, target="mlir")
        @convert_qecl_to_qecp_pass(qec_code="Steane")
        @qp.transform(pass_name="symbol-dce")
        @convert_quantum_to_qecl_pass(k=1)
        @qp.qnode(dev, shots=1)
        def circuit(x: float):
            # CHECK-LABEL: func.func public @circuit(
            #  CHECK-SAME:     [[cond_arg:%.+]]: tensor<f64>

            # CHECK: [[c0:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
            # CHECK: [[cond_t:%.+]] = stablehlo.compare GT, [[cond_arg]], [[c0]] {{.*}} -> tensor<i1>
            # CHECK: [[cond:%.+]] = tensor.extract [[cond_t]]
            # CHECK: [[cb_out:%.+]] = scf.if [[cond]] -> (!qecp.codeblock<1 x 7>)
            # CHECK:     func.call @x_Steane({{%.+}}) : {{.*}} -> !qecp.codeblock<1 x 7>
            # CHECK:     scf.yield {{%.+}} : !qecp.codeblock<1 x 7>
            # CHECK: else
            # CHECK:     func.call @z_Steane({{%.+}}) : {{.*}} -> !qecp.codeblock<1 x 7>
            # CHECK:     scf.yield {{%.+}} : !qecp.codeblock<1 x 7>
            # CHECK: func.call @measure_transversal_Steane([[cb_out]])
            def true_branch():
                qp.X(0)

            def false_branch():
                qp.Z(0)

            qp.cond(x > 0, true_branch, false_branch)()

            return qp.sample(wires=[0])

        run_filecheck_qjit(circuit)

    def test_scf_for_integration(self, run_filecheck_qjit):
        """Test the convert-quantum-to-qecl and convert-qecl-to-qecp passes together on a simple
        program with a for loop.
        """
        dev = qp.device("null.qubit", wires=1)

        @qp.qjit(capture=True, target="mlir")
        @convert_qecl_to_qecp_pass(qec_code="Steane")
        @qp.transform(pass_name="symbol-dce")
        @convert_quantum_to_qecl_pass(k=1)
        @qp.qnode(dev, shots=1)
        def circuit():
            # CHECK-LABEL: func.func public @circuit(

            #  CHECK-DAG: [[c1:%.+]] = arith.constant 1 : index
            #  CHECK-DAG: [[c4:%.+]] = arith.constant 4 : index
            #  CHECK-DAG: [[c0:%.+]] = arith.constant 0 : index
            #      CHECK: qecp.alloc() : !qecp.hyperreg<1 x 1 x 7>
            #      CHECK: [[cb_out:%.+]] = scf.for {{%.+}} = [[c0]] to [[c4]] step [[c1]]
            # CHECK-SAME:         iter_args([[cb_arg:%.+]] = {{%.+}}) -> (!qecp.codeblock<1 x 7>)
            #      CHECK:     func.call @x_Steane([[cb_arg]]) : (!qecp.codeblock<1 x 7>)
            # CHECK-SAME:         -> !qecp.codeblock<1 x 7>
            #        COM:     <qec
            #      CHECK:     scf.yield {{%.+}} : !qecp.codeblock<1 x 7>
            #      CHECK: func.call @measure_transversal_Steane([[cb_out]])
            @qp.for_loop(0, 4, 1)
            def loop_pauli_x(i):  # pylint: disable=unused-argument
                qp.PauliX(0)

            loop_pauli_x()
            return qp.sample(wires=[0])

        run_filecheck_qjit(circuit)

    def test_scf_while_integration(self, run_filecheck_qjit):
        """Test the convert-quantum-to-qecl and convert-qecl-to-qecp passes together on a simple
        program with a while loop.
        """
        dev = qp.device("null.qubit", wires=1)

        @qp.qjit(capture=True, target="mlir")
        @convert_qecl_to_qecp_pass(qec_code="Steane")
        @qp.transform(pass_name="symbol-dce")
        @convert_quantum_to_qecl_pass(k=1)
        @qp.qnode(dev, shots=1)
        def circuit(value: float):
            # CHECK-LABEL: func.func public @circuit(
            #  CHECK-SAME:     [[top_arg:%.+]]: tensor<f64>

            #      CHECK: [[c2:%.+]] = stablehlo.constant dense<2.000000e+00> : tensor<f64>
            #      CHECK: {{%.+}}, [[cb_out:%.+]] = scf.while
            # CHECK-SAME:         ([[while_arg_b:%.+]] = [[top_arg]], [[cb_arg_b:%.+]] = {{%.+}}) :
            # CHECK-SAME:         (tensor<f64>, !qecp.codeblock<1 x 7>) -> (tensor<f64>, !qecp.codeblock<1 x 7>)
            #      CHECK:    [[cond_t:%.+]] = stablehlo.compare LT, [[while_arg_b]], [[c2]]
            #      CHECK:    [[cond:%.+]] = tensor.extract [[cond_t]]
            #      CHECK:    scf.condition([[cond]]) [[while_arg_b]], [[cb_arg_b]] :
            # CHECK-SAME:        tensor<f64>, !qecp.codeblock<1 x 7>
            #      CHECK: do
            #      CHECK: ^bb0([[while_arg_a:%.+]]: tensor<f64>, [[cb_arg_a:%.+]]: !qecp.codeblock<1 x 7>):
            #      CHECK:     func.call @x_Steane([[cb_arg_a]]) : (!qecp.codeblock<1 x 7>)
            # CHECK-SAME:         -> !qecp.codeblock<1 x 7>
            #      CHECK:     stablehlo.add [[while_arg_a]], [[c2]] : tensor<f64>
            #      CHECK:     scf.yield {{%.+}}, {{%.+}} : tensor<f64>, !qecp.codeblock<1 x 7>
            #      CHECK: func.call @measure_transversal_Steane([[cb_out]])
            @qp.while_loop(lambda x: x < 2.0)
            def loop_rx(x):
                qp.PauliX(0)
                return x + 2

            # apply the while loop
            loop_rx(value)

            return qp.sample(wires=[0])

        run_filecheck_qjit(circuit)


# MARK: Integration


class TestQECPLoweringIntegration:
    """Integration lit tests for convert-qecl-to-qecp"""

    def test_circuit_ghz_to_qecp(self, run_filecheck_qjit):
        """Test the convert-quantum-to-qecl and convert-qecl-to-qecp pass together on a
        GHZ circuit."""
        dev = qp.device("null.qubit", wires=3)

        @qp.qjit(capture=True, target="mlir")
        @convert_qecl_to_qecp_pass(qec_code="Steane")
        @convert_quantum_to_qecl_pass(k=1)
        @qp.qnode(dev, shots=1)
        def circuit():
            # CHECK: qecp.alloc() : !qecp.hyperreg<3 x 1 x 7>
            # CHECK: scf.for {{.*}} {
            # CHECK:   qecp.extract_block
            # CHECK:   func.call @encode_zero_Steane
            # CHECK:   qecp.insert_block
            # CHECK:   scf.yield
            # CHECK: }
            # CHECK: qecp.extract_block
            # CHECK: func.call @qec_cycle_Steane
            # CHECK: func.call @hadamard_Steane
            # CHECK: func.call @qec_cycle_Steane
            # CHECK: qecp.extract_block
            # CHECK: func.call @qec_cycle_Steane
            # CHECK: func.call @cnot_Steane
            # CHECK: func.call @qec_cycle_Steane
            # CHECK: qecp.extract_block
            # CHECK: func.call @qec_cycle_Steane
            # CHECK: func.call @cnot_Steane
            # CHECK: func.call @qec_cycle_Steane
            # CHECK: func.call @measure_transversal_Steane
            # CHECK: func.call @measure_transversal_Steane
            # CHECK: func.call @measure_transversal_Steane
            # CHECK: quantum.mcmobs
            # CHECK: qecp.insert_block
            # CHECK: quantum.sample
            # CHECK: qecp.dealloc
            qp.H(0)
            qp.CNOT([0, 1])
            qp.CNOT([1, 2])
            return qp.sample(wires=[0, 1, 2])

        run_filecheck_qjit(circuit)

    def test_convert_qecl_noise_to_qecp_noise_pass_integration(self, run_filecheck_qjit):
        """Test integration of the convert-qecl-noise-to-qecp-noise pass on the simplest possible,
        non-trivial circuit."""
        dev = qp.device("null.qubit", wires=1)

        @qp.qjit(target="mlir", capture=True)
        @convert_qecl_to_qecp_pass(qec_code="Steane", number_errors=1)
        @inject_noise_to_qecl_pass
        @convert_quantum_to_qecl_pass(k=1)
        @qp.qnode(dev, shots=1)
        def circuit():
            # CHECK: func.call @encode_zero_Steane
            # CHECK: arith.constant dense
            # CHECK: arith.constant dense
            # CHECK: func.call @noise_subroutine_code
            # CHECK: func.call @qec_cycle_Steane
            # CHECK: func.call @hadamard_Steane
            qp.H(0)
            # CHECK: arith.constant dense
            # CHECK: arith.constant dense
            # CHECK: func.call @noise_subroutine_code
            # CHECK: func.call @qec_cycle_Steane
            # CHECK: func.call @measure_transversal_Steane
            return qp.sample(wires=[0])

        run_filecheck_qjit(circuit)


# MARK: Generality


class TestGenerality:
    """Test the generality for other k=1 CSS codes beyond the Steane code by testing compilation
    with the Shor-913 code. Note that this code does not support any transversal phase gates. These
    tests check lowering to the qecp dialect, rather than execution and validity of results."""

    def test_transversal_gates(self, run_filecheck_qjit):
        """Test that compilation for the code runs as expected without raising any errors
        from the frontend through to the qecp layer."""

        dev = qp.device("lightning.qubit", wires=2)
        pipe = [("pipe", ["quantum-compilation-stage"])]

        @qp.qjit(capture=True, pipelines=pipe, target="mlir")
        @convert_qecl_to_qecp_pass(qec_code="Shor913", number_errors=0)
        @convert_quantum_to_qecl_pass(k=1)
        @qp.set_shots(1000)
        @qp.qnode(dev, mcm_method="one-shot")
        def circ():
            # CHECK: func.call @qec_cycle_Shor913
            # CHECK: func.call @x_Shor913
            # CHECK: func.call @z_Shor913
            # CHECK: func.call @cnot_Shor913
            # CHECK: func.call @measure_transversal_Shor913
            qp.X(0)
            qp.Z(1)
            qp.CNOT([0, 1])
            return qp.sample(wires=[0, 1])

        run_filecheck_qjit(circ)

    def test_x_shor(self, run_filecheck):
        """Test that using the Shor913 code lowers a logical Pauli X gate as expected.

        The logical Pauli X gate in the Shor913 code is realized by applying the following Pauli
        word to the physical codeblock:

            "ZIIZIIZII"
        """
        program = """
        builtin.module @module_circuit {
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[codeblock:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 9>
                // CHECK: [[codeblock2:%.+]] = func.call @x_Shor913([[codeblock]]) : (!qecp.codeblock<1 x 9>) -> !qecp.codeblock<1 x 9>
                // CHECK-NOT: qecl.x
                %0 = "test.op"() : () -> !qecl.codeblock<1>
                %1 = qecl.x %0[0] : !qecl.codeblock<1>
                return
            }
            // CHECK: func.func private @x_Shor913([[cb_0:%.+]]: !qecp.codeblock<1 x 9>)
            // CHECK: [[q0:%.+]] = qecp.extract [[cb_0]][0] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q1:%.+]] = qecp.extract [[cb_0]][1] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q2:%.+]] = qecp.extract [[cb_0]][2] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q3:%.+]] = qecp.extract [[cb_0]][3] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q4:%.+]] = qecp.extract [[cb_0]][4] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q5:%.+]] = qecp.extract [[cb_0]][5] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q6:%.+]] = qecp.extract [[cb_0]][6] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q7:%.+]] = qecp.extract [[cb_0]][7] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q8:%.+]] = qecp.extract [[cb_0]][8] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q0_1:%.+]] = qecp.z [[q0]] : !qecp.qubit<data>
            // CHECK: [[q1_1:%.+]] = qecp.identity [[q1]] : !qecp.qubit<data>
            // CHECK: [[q2_1:%.+]] = qecp.identity [[q2]] : !qecp.qubit<data>
            // CHECK: [[q3_1:%.+]] = qecp.z [[q3]] : !qecp.qubit<data>
            // CHECK: [[q4_1:%.+]] = qecp.identity [[q4]] : !qecp.qubit<data>
            // CHECK: [[q5_1:%.+]] = qecp.identity [[q5]] : !qecp.qubit<data>
            // CHECK: [[q6_1:%.+]] = qecp.z [[q6]] : !qecp.qubit<data>
            // CHECK: [[q7_1:%.+]] = qecp.identity [[q7]] : !qecp.qubit<data>
            // CHECK: [[q8_1:%.+]] = qecp.identity [[q8]] : !qecp.qubit<data>
            // CHECK: [[cb_1:%.+]] = qecp.insert [[cb_0]][0], [[q0_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_2:%.+]] = qecp.insert [[cb_1]][1], [[q1_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_3:%.+]] = qecp.insert [[cb_2]][2], [[q2_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_4:%.+]] = qecp.insert [[cb_3]][3], [[q3_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_5:%.+]] = qecp.insert [[cb_4]][4], [[q4_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_6:%.+]] = qecp.insert [[cb_5]][5], [[q5_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_7:%.+]] = qecp.insert [[cb_6]][6], [[q6_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_8:%.+]] = qecp.insert [[cb_7]][7], [[q7_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_9:%.+]] = qecp.insert [[cb_8]][8], [[q8_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: return [[cb_9]] : !qecp.codeblock<1 x 9>
        }
        """

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode.get("Shor913")),)
        run_filecheck(program, pipeline)

    def test_y_shor(self, run_filecheck):
        """Test that using the Shor913 code lowers a logical Pauli Y gate as expected.

        The logical Pauli Y gate in the Shor913 code is realized by applying the following Pauli
        word to the physical codeblock:

            "YXXZIIZII"
        """
        program = """
        builtin.module @module_circuit {
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[codeblock:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 9>
                // CHECK: [[codeblock2:%.+]] = func.call @y_Shor913([[codeblock]]) : (!qecp.codeblock<1 x 9>) -> !qecp.codeblock<1 x 9>
                // CHECK-NOT: qecl.y
                %0 = "test.op"() : () -> !qecl.codeblock<1>
                %1 = qecl.y %0[0] : !qecl.codeblock<1>
                return
            }
            // CHECK: func.func private @y_Shor913([[cb_0:%.+]]: !qecp.codeblock<1 x 9>)
            // CHECK: [[q0:%.+]] = qecp.extract [[cb_0]][0] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q1:%.+]] = qecp.extract [[cb_0]][1] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q2:%.+]] = qecp.extract [[cb_0]][2] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q3:%.+]] = qecp.extract [[cb_0]][3] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q4:%.+]] = qecp.extract [[cb_0]][4] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q5:%.+]] = qecp.extract [[cb_0]][5] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q6:%.+]] = qecp.extract [[cb_0]][6] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q7:%.+]] = qecp.extract [[cb_0]][7] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q8:%.+]] = qecp.extract [[cb_0]][8] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q0_1:%.+]] = qecp.y [[q0]] : !qecp.qubit<data>
            // CHECK: [[q1_1:%.+]] = qecp.x [[q1]] : !qecp.qubit<data>
            // CHECK: [[q2_1:%.+]] = qecp.x [[q2]] : !qecp.qubit<data>
            // CHECK: [[q3_1:%.+]] = qecp.z [[q3]] : !qecp.qubit<data>
            // CHECK: [[q4_1:%.+]] = qecp.identity [[q4]] : !qecp.qubit<data>
            // CHECK: [[q5_1:%.+]] = qecp.identity [[q5]] : !qecp.qubit<data>
            // CHECK: [[q6_1:%.+]] = qecp.z [[q6]] : !qecp.qubit<data>
            // CHECK: [[q7_1:%.+]] = qecp.identity [[q7]] : !qecp.qubit<data>
            // CHECK: [[q8_1:%.+]] = qecp.identity [[q8]] : !qecp.qubit<data>
            // CHECK: [[cb_1:%.+]] = qecp.insert [[cb_0]][0], [[q0_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_2:%.+]] = qecp.insert [[cb_1]][1], [[q1_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_3:%.+]] = qecp.insert [[cb_2]][2], [[q2_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_4:%.+]] = qecp.insert [[cb_3]][3], [[q3_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_5:%.+]] = qecp.insert [[cb_4]][4], [[q4_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_6:%.+]] = qecp.insert [[cb_5]][5], [[q5_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_7:%.+]] = qecp.insert [[cb_6]][6], [[q6_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_8:%.+]] = qecp.insert [[cb_7]][7], [[q7_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_9:%.+]] = qecp.insert [[cb_8]][8], [[q8_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: return [[cb_9]] : !qecp.codeblock<1 x 9>
        }
        """

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode.get("Shor913")),)
        run_filecheck(program, pipeline)

    def test_z_shor(self, run_filecheck):
        """Test that using the Shor913 code lowers a logical Pauli Z gate as expected.

        The logical Pauli Z gate in the Shor913 code is realized by applying the following Pauli
        word to the physical codeblock:

            "XXXIIIIII"
        """
        program = """
        builtin.module @module_circuit {
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[codeblock:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 9>
                // CHECK: [[codeblock2:%.+]] = func.call @z_Shor913([[codeblock]]) : (!qecp.codeblock<1 x 9>) -> !qecp.codeblock<1 x 9>
                // CHECK-NOT: qecl.z
                %0 = "test.op"() : () -> !qecl.codeblock<1>
                %1 = qecl.z %0[0] : !qecl.codeblock<1>
                return
            }
            // CHECK: func.func private @z_Shor913([[cb_0:%.+]]: !qecp.codeblock<1 x 9>)
            // CHECK: [[q0:%.+]] = qecp.extract [[cb_0]][0] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q1:%.+]] = qecp.extract [[cb_0]][1] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q2:%.+]] = qecp.extract [[cb_0]][2] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q3:%.+]] = qecp.extract [[cb_0]][3] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q4:%.+]] = qecp.extract [[cb_0]][4] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q5:%.+]] = qecp.extract [[cb_0]][5] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q6:%.+]] = qecp.extract [[cb_0]][6] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q7:%.+]] = qecp.extract [[cb_0]][7] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q8:%.+]] = qecp.extract [[cb_0]][8] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q0_1:%.+]] = qecp.x [[q0]] : !qecp.qubit<data>
            // CHECK: [[q1_1:%.+]] = qecp.x [[q1]] : !qecp.qubit<data>
            // CHECK: [[q2_1:%.+]] = qecp.x [[q2]] : !qecp.qubit<data>
            // CHECK: [[q3_1:%.+]] = qecp.identity [[q3]] : !qecp.qubit<data>
            // CHECK: [[q4_1:%.+]] = qecp.identity [[q4]] : !qecp.qubit<data>
            // CHECK: [[q5_1:%.+]] = qecp.identity [[q5]] : !qecp.qubit<data>
            // CHECK: [[q6_1:%.+]] = qecp.identity [[q6]] : !qecp.qubit<data>
            // CHECK: [[q7_1:%.+]] = qecp.identity [[q7]] : !qecp.qubit<data>
            // CHECK: [[q8_1:%.+]] = qecp.identity [[q8]] : !qecp.qubit<data>
            // CHECK: [[cb_1:%.+]] = qecp.insert [[cb_0]][0], [[q0_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_2:%.+]] = qecp.insert [[cb_1]][1], [[q1_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_3:%.+]] = qecp.insert [[cb_2]][2], [[q2_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_4:%.+]] = qecp.insert [[cb_3]][3], [[q3_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_5:%.+]] = qecp.insert [[cb_4]][4], [[q4_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_6:%.+]] = qecp.insert [[cb_5]][5], [[q5_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_7:%.+]] = qecp.insert [[cb_6]][6], [[q6_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_8:%.+]] = qecp.insert [[cb_7]][7], [[q7_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb_9:%.+]] = qecp.insert [[cb_8]][8], [[q8_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: return [[cb_9]] : !qecp.codeblock<1 x 9>
        }
        """

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode.get("Shor913")),)
        run_filecheck(program, pipeline)

    def test_cnot_shor(self, run_filecheck):
        """Test that using the Shor913 code lowers a logical CNOT gate as expected.

        The logical CNOT gate in the Shor913 code is realized by transversally applying physical
        CNOT gates qubit-wise between two codeblocks.
        """
        program = """
        builtin.module @module_circuit {
            func.func @test_func() attributes {quantum.node} {
                // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 9>
                // CHECK: [[cb1:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 9>
                // CHECK: [[cb2:%.+]], [[cb3:%.+]] = func.call @cnot_Shor913([[cb0]], [[cb1]]) :
                // CHECK-SAME: (!qecp.codeblock<1 x 9>, !qecp.codeblock<1 x 9>) -> (!qecp.codeblock<1 x 9>, !qecp.codeblock<1 x 9>)
                // CHECK-NOT: qecl.cnot
                %0 = "test.op"() : () -> !qecl.codeblock<1>
                %1 = "test.op"() : () -> !qecl.codeblock<1>
                %2, %3 = qecl.cnot %0[0], %1[0] : !qecl.codeblock<1>, !qecl.codeblock<1>
                return
            }
            // CHECK: func.func private @cnot_Shor913([[cb0_0:%.+]]: !qecp.codeblock<1 x 9>, [[cb1_0:%.+]]: !qecp.codeblock<1 x 9>)
            // CHECK: [[q00_0:%.+]] = qecp.extract [[cb0]][0] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q01_0:%.+]] = qecp.extract [[cb0]][1] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q02_0:%.+]] = qecp.extract [[cb0]][2] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q03_0:%.+]] = qecp.extract [[cb0]][3] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q04_0:%.+]] = qecp.extract [[cb0]][4] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q05_0:%.+]] = qecp.extract [[cb0]][5] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q06_0:%.+]] = qecp.extract [[cb0]][6] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q07_0:%.+]] = qecp.extract [[cb0]][7] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q08_0:%.+]] = qecp.extract [[cb0]][8] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q10_0:%.+]] = qecp.extract [[cb1]][0] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q11_0:%.+]] = qecp.extract [[cb1]][1] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q12_0:%.+]] = qecp.extract [[cb1]][2] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q13_0:%.+]] = qecp.extract [[cb1]][3] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q14_0:%.+]] = qecp.extract [[cb1]][4] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q15_0:%.+]] = qecp.extract [[cb1]][5] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q16_0:%.+]] = qecp.extract [[cb1]][6] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q17_0:%.+]] = qecp.extract [[cb1]][7] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q18_0:%.+]] = qecp.extract [[cb1]][8] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // CHECK: [[q00_1:%.+]], [[q10_1:%.+]] = qecp.cnot [[q00_0]], [[q10_0]]
            // CHECK: [[q01_1:%.+]], [[q11_1:%.+]] = qecp.cnot [[q01_0]], [[q11_0]]
            // CHECK: [[q02_1:%.+]], [[q12_1:%.+]] = qecp.cnot [[q02_0]], [[q12_0]]
            // CHECK: [[q03_1:%.+]], [[q13_1:%.+]] = qecp.cnot [[q03_0]], [[q13_0]]
            // CHECK: [[q04_1:%.+]], [[q14_1:%.+]] = qecp.cnot [[q04_0]], [[q14_0]]
            // CHECK: [[q05_1:%.+]], [[q15_1:%.+]] = qecp.cnot [[q05_0]], [[q15_0]]
            // CHECK: [[q06_1:%.+]], [[q16_1:%.+]] = qecp.cnot [[q06_0]], [[q16_0]]
            // CHECK: [[q07_1:%.+]], [[q17_1:%.+]] = qecp.cnot [[q07_0]], [[q17_0]]
            // CHECK: [[q08_1:%.+]], [[q18_1:%.+]] = qecp.cnot [[q08_0]], [[q18_0]]
            // CHECK: [[cb0_1:%.+]] = qecp.insert [[cb0_0]][0], [[q00_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb0_2:%.+]] = qecp.insert [[cb0_1]][1], [[q01_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb0_3:%.+]] = qecp.insert [[cb0_2]][2], [[q02_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb0_4:%.+]] = qecp.insert [[cb0_3]][3], [[q03_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb0_5:%.+]] = qecp.insert [[cb0_4]][4], [[q04_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb0_6:%.+]] = qecp.insert [[cb0_5]][5], [[q05_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb0_7:%.+]] = qecp.insert [[cb0_6]][6], [[q06_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb0_8:%.+]] = qecp.insert [[cb0_7]][7], [[q07_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb0_9:%.+]] = qecp.insert [[cb0_8]][8], [[q08_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb1_1:%.+]] = qecp.insert [[cb1_0]][0], [[q10_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb1_2:%.+]] = qecp.insert [[cb1_1]][1], [[q11_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb1_3:%.+]] = qecp.insert [[cb1_2]][2], [[q12_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb1_4:%.+]] = qecp.insert [[cb1_3]][3], [[q13_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb1_5:%.+]] = qecp.insert [[cb1_4]][4], [[q14_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb1_6:%.+]] = qecp.insert [[cb1_5]][5], [[q15_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb1_7:%.+]] = qecp.insert [[cb1_6]][6], [[q16_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb1_8:%.+]] = qecp.insert [[cb1_7]][7], [[q17_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: [[cb1_9:%.+]] = qecp.insert [[cb1_8]][8], [[q18_1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            // CHECK: return [[cb0_9]], [[cb1_9]] : !qecp.codeblock<1 x 9>, !qecp.codeblock<1 x 9>
        }
        """

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode.get("Shor913")),)
        run_filecheck(program, pipeline)

    def test_qec_cycle_shor(self, run_filecheck):
        """Test that a `qecl.qec` op is lowered to a call to the QEC-cycle subroutine for the
        Shor913 code.
        """
        program = """
        // CHECK-LABEL: test_module
        builtin.module @test_module {
        // CHECK-LABEL: test_program
        func.func @test_program()  {
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 9>
            %0 = "test.op"() : () -> !qecl.codeblock<1>

            // CHECK: [[cb1:%.+]] = func.call @qec_cycle_Shor913([[cb0]]) : (!qecp.codeblock<1 x 9>) -> !qecp.codeblock<1 x 9>
            %1 = qecl.qec %0 : !qecl.codeblock<1>
            return
        }
        // CHECK-LABEL: qec_cycle_Shor913([[cb0:%.+]]: !qecp.codeblock<1 x 9>) -> !qecp.codeblock<1 x 9>
        // CHECK: [[tanner_x:%.+]] = qecp.assemble_tanner {{.+}}, {{.+}} : tensor<24xi32>, tensor<12xi32> -> !qecp.tanner_graph<24, 12, i32>
        // CHECK: [[tanner_z:%.+]] = qecp.assemble_tanner {{.+}}, {{.+}} : tensor<24xi32>, tensor<16xi32> -> !qecp.tanner_graph<24, 16, i32>

        // COM: The block below takes results of X checks and performs Z corrections
        // CHECK: qecp.alloc_aux : !qecp.qubit<aux>
        // CHECK: qecp.alloc_aux : !qecp.qubit<aux>
        // CHECK: qecp.hadamard {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.hadamard {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.extract {{.*}} : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
        // CHECK: qecp.cnot {{.*}} : !qecp.qubit<aux>, !qecp.qubit<data>
        // CHECK: qecp.insert {{.*}} : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
        // CHECK: [[cb0:%.+]] = qecp.insert {{.*}}[6], {{.*}} : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
        // CHECK: qecp.hadamard {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.hadamard {{.*}} : !qecp.qubit<aux>
        // CHECK: [[m0:%.+]], {{.*}} = qecp.measure {{.*}} : i1, !qecp.qubit<aux>
        // CHECK: [[m1:%.+]], {{.*}} = qecp.measure {{.*}} : i1, !qecp.qubit<aux>
        // CHECK: qecp.dealloc_aux {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.dealloc_aux {{.*}} : !qecp.qubit<aux>
        // CHECK: [[esm:%.+]] = tensor.from_elements [[m0]], [[m1]] : tensor<2xi1>
        // CHECK: [[idx_t:%.+]] = qecp.decode_esm_css([[tanner_x]] : !qecp.tanner_graph<24, 12, i32>) [[esm]] : tensor<2xi1> -> tensor<1xindex>
        // CHECK: [[lb:%.+]] = arith.constant 0 : index
        // CHECK: [[ub:%.+]] = arith.constant 1 : index
        // CHECK: [[st:%.+]] = arith.constant 1 : index
        // CHECK: [[cb_x_out:%.+]] = scf.for [[i:%.+]] = [[lb]] to [[ub]] step [[st]] iter_args([[cb_arg:%.+]] = {{%.+}})
        // CHECK:   [[err_idx:%.+]] = tensor.extract [[idx_t]][[[i]]] : tensor<1xindex>
        // CHECK:   [[err_i64:%.+]] = arith.index_cast [[err_idx]] : index to i64
        // CHECK:   [[minus1:%.+]] = arith.constant -1 : i64
        // CHECK:   [[cond:%.+]] = arith.cmpi ne, [[err_i64]], [[minus1]] : i64
        // CHECK:   [[cond_out_cb:%.+]] = scf.if [[cond]]
        // CHECK:     [[q0:%.+]] = qecp.extract [[cb_arg]][[[err_idx]]] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
        // CHECK:     [[q1:%.+]] = qecp.z [[q0]] : !qecp.qubit<data>
        // CHECK:     [[cb_arg_1:%.+]] = qecp.insert [[cb_arg]][[[err_idx]]], [[q1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
        // CHECK:     scf.yield [[cb_arg_1]] : !qecp.codeblock<1 x 9>
        // CHECK:   } else {
        // CHECK:     scf.yield [[cb_arg]] : !qecp.codeblock<1 x 9>
        // CHECK:   }
        // CHECK: scf.yield [[cond_out_cb]] : !qecp.codeblock<1 x 9>
        // CHECK: }

        // COM: The block below takes results of X checks and performs Z corrections
        // CHECK: qecp.alloc_aux : !qecp.qubit<aux>
        // CHECK: qecp.alloc_aux : !qecp.qubit<aux>
        // CHECK: qecp.alloc_aux : !qecp.qubit<aux>
        // CHECK: qecp.alloc_aux : !qecp.qubit<aux>
        // CHECK: qecp.alloc_aux : !qecp.qubit<aux>
        // CHECK: qecp.alloc_aux : !qecp.qubit<aux>
        // CHECK-NOT: qecp.hadamard
        // CHECK: qecp.extract {{.*}} : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
        // CHECK: qecp.cnot {{.*}} : !qecp.qubit<data>, !qecp.qubit<aux>
        // CHECK: qecp.insert {{.*}} : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
        // CHECK: [[cb0:%.+]] = qecp.insert {{.*}}[6], {{.*}} : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
        // CHECK-NOT: qecp.hadamard
        // CHECK: [[m0:%.+]], {{.*}} = qecp.measure {{.*}} : i1, !qecp.qubit<aux>
        // CHECK: [[m1:%.+]], {{.*}} = qecp.measure {{.*}} : i1, !qecp.qubit<aux>
        // CHECK: [[m2:%.+]], {{.*}} = qecp.measure {{.*}} : i1, !qecp.qubit<aux>
        // CHECK: [[m3:%.+]], {{.*}} = qecp.measure {{.*}} : i1, !qecp.qubit<aux>
        // CHECK: [[m4:%.+]], {{.*}} = qecp.measure {{.*}} : i1, !qecp.qubit<aux>
        // CHECK: [[m5:%.+]], {{.*}} = qecp.measure {{.*}} : i1, !qecp.qubit<aux>
        // CHECK: qecp.dealloc_aux {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.dealloc_aux {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.dealloc_aux {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.dealloc_aux {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.dealloc_aux {{.*}} : !qecp.qubit<aux>
        // CHECK: qecp.dealloc_aux {{.*}} : !qecp.qubit<aux>
        // CHECK: [[esm:%.+]] = tensor.from_elements [[m0]], [[m1]], [[m2]], [[m3]], [[m4]], [[m5]] : tensor<6xi1>
        // CHECK: [[idx_t:%.+]] = qecp.decode_esm_css([[tanner_z]] : !qecp.tanner_graph<24, 16, i32>) [[esm]] : tensor<6xi1>  -> tensor<1xindex>
        // CHECK: [[lb:%.+]] = arith.constant 0 : index
        // CHECK: [[ub:%.+]] = arith.constant 1 : index
        // CHECK: [[st:%.+]] = arith.constant 1 : index
        // CHECK: [[cb_x_out:%.+]] = scf.for [[i:%.+]] = [[lb]] to [[ub]] step [[st]] iter_args([[cb_arg:%.+]] = {{%.+}})
        // CHECK:   [[err_idx:%.+]] = tensor.extract [[idx_t]][[[i]]] : tensor<1xindex>
        // CHECK:   [[err_i64:%.+]] = arith.index_cast [[err_idx]] : index to i64
        // CHECK:   [[minus1:%.+]] = arith.constant -1 : i64
        // CHECK:   [[cond:%.+]] = arith.cmpi ne, [[err_i64]], [[minus1]] : i64
        // CHECK:   [[cond_out_cb:%.+]] = scf.if [[cond]]
        // CHECK:     [[q0:%.+]] = qecp.extract [[cb_arg]][[[err_idx]]] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
        // CHECK:     [[q1:%.+]] = qecp.x [[q0]] : !qecp.qubit<data>
        // CHECK:     [[cb_arg_1:%.+]] = qecp.insert [[cb_arg]][[[err_idx]]], [[q1]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
        // CHECK:     scf.yield [[cb_arg_1]] : !qecp.codeblock<1 x 9>
        // CHECK:   } else {
        // CHECK:     scf.yield [[cb_arg]] : !qecp.codeblock<1 x 9>
        // CHECK:   }
        // CHECK: scf.yield [[cond_out_cb]] : !qecp.codeblock<1 x 9>
        // CHECK: }
        // CHECK: func.return [[cb_x_out]] : !qecp.codeblock<1 x 9>
        }
        """

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode.get("Shor913")),)
        run_filecheck(program, pipeline)

    def test_fabricate_magic_state_shor(self, run_filecheck):
        """Test that the `fabricate` op for the magic state is generated as expected for the
        Shor913 code. Note that without transversal S, we can't lower the apply_T subroutine,
        so we can only test the generation of the `fabricate` subroutine.

        Since this is only used in applying T at the moment, this isn't reachable from any
        frontend code, but we can still check that it works."""

        program = """
        builtin.module @module_circuit {
            func.func @test_func() attributes {quantum.node} {
                // CHECK:   [[magic_cb:%.+]] = func.call @fabricate_magic_Shor913() : () -> !qecp.codeblock<1 x 9>
                %0 = qecl.fabricate[magic] : !qecl.codeblock<1>
                return
            }
            // CHECK-LABEL: func.func private @fabricate_magic_Shor913() -> !qecp.codeblock<1 x 9>
            //       CHECK:   [[cb:%.+]] = qecp.alloc_cb : !qecp.codeblock<1 x 9>
            //   CHECK-DAG:   [[q0:%.+]] = qecp.extract [[cb]][0] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            //   CHECK-DAG:   [[q1:%.+]] = qecp.extract [[cb]][1] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            //   CHECK-DAG:   [[q2:%.+]] = qecp.extract [[cb]][2] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            //   CHECK-DAG:   [[q3:%.+]] = qecp.extract [[cb]][3] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            //   CHECK-DAG:   [[q4:%.+]] = qecp.extract [[cb]][4] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            //   CHECK-DAG:   [[q5:%.+]] = qecp.extract [[cb]][5] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            //   CHECK-DAG:   [[q6:%.+]] = qecp.extract [[cb]][6] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            //   CHECK-DAG:   [[q7:%.+]] = qecp.extract [[cb]][7] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            //   CHECK-DAG:   [[q8:%.+]] = qecp.extract [[cb]][8] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            // COM: State injection on the state_prep_index (qubit 0): H then T
            //       CHECK:   [[h_inj:%.+]] = qecp.hadamard [[q0]] : !qecp.qubit<data>
            //       CHECK:   [[q0_1:%.+]] = qecp.t [[h_inj]] : !qecp.qubit<data>
            // COM: Unitary encoding: initial CNOTs
            //       CHECK:   [[q0_2:%.+]], [[q3_1:%.+]] = qecp.cnot [[q0_1]], [[q3]] : !qecp.qubit<data>, !qecp.qubit<data>
            //       CHECK:   [[q0_3:%.+]], [[q6_1:%.+]] = qecp.cnot [[q0_2]], [[q6]] : !qecp.qubit<data>, !qecp.qubit<data>
            // COM: Unitary encoding: Hadamards on indices 0, 3, 6
            //       CHECK:   [[q0_4:%.+]] = qecp.hadamard [[q0_3]] : !qecp.qubit<data>
            //       CHECK:   [[q3_2:%.+]] = qecp.hadamard [[q3_1]] : !qecp.qubit<data>
            //       CHECK:   [[q6_2:%.+]] = qecp.hadamard [[q6_1]] : !qecp.qubit<data>
            // COM: Unitary encoding: more CNOTs - [n, n+1] and [n, n+2] for n in [0, 3, 6]
            //       CHECK:   [[q0_5:%.+]], [[q1_1:%.+]] = qecp.cnot [[q0_4]], [[q1]] : !qecp.qubit<data>, !qecp.qubit<data>
            //       CHECK:   qecp.cnot [[q0_5]], [[q2]] : !qecp.qubit<data>, !qecp.qubit<data>
            //       CHECK:   [[q3_3:%.+]], [[q4_1:%.+]] = qecp.cnot [[q3_2]], [[q4]] : !qecp.qubit<data>, !qecp.qubit<data>
            //       CHECK:   qecp.cnot [[q3_3]], [[q5]] : !qecp.qubit<data>, !qecp.qubit<data>
            //       CHECK:   [[q6_3:%.+]], [[q7_1:%.+]] = qecp.cnot [[q6_2]], [[q7]] : !qecp.qubit<data>, !qecp.qubit<data>
            //       CHECK:   qecp.cnot [[q6_3]], [[q8]] : !qecp.qubit<data>, !qecp.qubit<data>

        }
        """
        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode.get("Shor913")),)
        run_filecheck(program, pipeline)

    def test_measure_shor(self, run_filecheck):
        """Test that using the Shor913 code lowers a logical measurement as expected.

        Recall that a logical computational-basis measurement amounts to measuring the logical Pauli
        Z observable, which in the Shor913 code is "XXXIIIIII". In order to perform a physical X
        measurement, diagonalizing gates are inserted before performing the computational basis
        measurement (the diagonalizing gate for X measurements is 'H').
        """
        program = """
        builtin.module @module_circuit {
            func.func @test_func() attributes {quantum.node} {
                //      CHECK: [[cb_0:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 9>
                //      CHECK: [[mres_3xi1:%.+]], [[cb_1:%.+]] = func.call @measure_transversal_Shor913([[cb_0]]) :
                // CHECK-SAME:   (!qecp.codeblock<1 x 9>) -> (tensor<3xi1>, !qecp.codeblock<1 x 9>)
                //      CHECK: [[mres_1xi1:%.+]] = func.call @decode_physical_measurements_Shor913([[mres_3xi1]]) :
                // CHECK-SAME:   (tensor<3xi1>) -> tensor<1xi1>
                //  CHECK-NOT: qecl.measure
                %0 = "test.op"() : () -> !qecl.codeblock<1>
                %mres, %1 = qecl.measure %0[0] : i1, !qecl.codeblock<1>
                return
            }
            // CHECK-LABEL: func.func private @measure_transversal_Shor913(
            //  CHECK-SAME:     [[cb_0]]: !qecp.codeblock<1 x 9>) -> (tensor<3xi1>, !qecp.codeblock<1 x 9>)
            //       CHECK:   [[q0_0:%.+]] = qecp.extract %0[0] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            //       CHECK:   [[q1_0:%.+]] = qecp.extract %0[1] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            //       CHECK:   [[q2_0:%.+]] = qecp.extract %0[2] : !qecp.codeblock<1 x 9> -> !qecp.qubit<data>
            //       CHECK:   [[q0_1:%.+]] = qecp.hadamard [[q0_0]] : !qecp.qubit<data>
            //       CHECK:   [[q1_1:%.+]] = qecp.hadamard [[q1_0]] : !qecp.qubit<data>
            //       CHECK:   [[q2_1:%.+]] = qecp.hadamard [[q2_0]] : !qecp.qubit<data>
            //       CHECK:   [[m0:%.+]], [[q0_2:%.+]] = qecp.measure [[q0_1]] : i1, !qecp.qubit<data>
            //       CHECK:   [[m1:%.+]], [[q1_2:%.+]] = qecp.measure [[q1_1]] : i1, !qecp.qubit<data>
            //       CHECK:   [[m2:%.+]], [[q2_2:%.+]] = qecp.measure [[q2_1]] : i1, !qecp.qubit<data>
            //       CHECK:   [[cb_1:%.+]] = qecp.insert [[cb_0]][0], [[q0_2]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            //       CHECK:   [[cb_2:%.+]] = qecp.insert [[cb_1]][1], [[q1_2]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            //       CHECK:   [[cb_3:%.+]] = qecp.insert [[cb_2]][2], [[q2_2]] : !qecp.codeblock<1 x 9>, !qecp.qubit<data>
            //       CHECK:   [[m_3xi1:%.+]] = tensor.from_elements [[m0]], [[m1]], [[m2]] : tensor<3xi1>
            //       CHECK:   return [[m_3xi1]], [[cb_3]] : tensor<3xi1>, !qecp.codeblock<1 x 9>

            // CHECK-LABEL: func.func private @decode_physical_measurements_Shor913
            //  CHECK-SAME:     ([[in_mres_3xi1:%.+]]: tensor<3xi1>) -> tensor<1xi1>
            //       CHECK:   [[c0:%.+]] = arith.constant false
            //       CHECK:   [[empty_i1:%.+]] = tensor.empty() : tensor<i1>
            //       CHECK:   [[init_i1:%.+]] = linalg.fill
            //  CHECK-SAME:     ins([[c0]] : i1) outs([[empty_i1]] : tensor<i1>) -> tensor<i1>
            //       CHECK:   [[reduced_i1:%.+]] = linalg.reduce
            //  CHECK-SAME:     ins([[in_mres_3xi1]]:tensor<3xi1>) outs([[init_i1]]:tensor<i1>)
            //  CHECK-SAME:     dimensions = [0]
            //       CHECK:   ([[in:%.+]]: i1, [[out:%.+]]: i1) {
            //       CHECK:     [[xor:%.+]] = arith.xori [[in]], [[out]] : i1
            //       CHECK:     linalg.yield [[xor]] : i1
            //       CHECK:   [[expanded_1xi1:%.+]] = tensor.expand_shape [[reduced_i1]] []
            //  CHECK-SAME:     output_shape [1] : tensor<i1> into tensor<1xi1>
            //       CHECK:   return [[expanded_1xi1]] : tensor<1xi1>
        }
        """

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code=QecCode.get("Shor913")),)
        run_filecheck(program, pipeline)
