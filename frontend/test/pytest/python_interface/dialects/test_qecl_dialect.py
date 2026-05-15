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

"""Unit tests for the xDSL QecLogical dialect."""

from typing import cast

import pytest
from xdsl.dialects import test
from xdsl.dialects.builtin import I64, IndexType, IntegerAttr, IntegerType, UnitAttr, i64
from xdsl.ir import AttributeCovT, Operation, OpResult, SSAValue

from catalyst.python_interface.dialects import qecl

pytestmark = pytest.mark.xdsl


@pytest.fixture(scope="module", name="assert_valid_idx_attr")
def fixture_assert_valid_idx_attr():
    """Fixture factory that returns a function to validate the `idx_attr` attribute of an xDSL
    operation.
    """

    def _validate_idx(op: Operation, idx: int | IntegerAttr | SSAValue):
        idx_attr = op.properties.get("idx_attr")
        if isinstance(idx, (int, IntegerAttr)):
            assert idx_attr is not None
            if isinstance(idx, int):
                assert idx_attr == IntegerAttr(idx, IndexType())
            elif isinstance(idx, IntegerAttr):
                assert idx_attr == IntegerAttr(idx.value.data, IndexType())
        else:
            assert idx_attr is None

    return _validate_idx


# Test function taken from xdsl/utils/test_value.py
def create_ssa_value(t: AttributeCovT) -> OpResult[AttributeCovT]:
    """Create a single SSA value with the given type for testing purposes."""
    op = test.TestOp(result_types=(t,))
    return cast(OpResult[AttributeCovT], op.results[0])


all_ops = list(qecl.QecLogical.operations)
all_attrs = list(qecl.QecLogical.attributes)

expected_ops_names = {
    "AllocOp": "qecl.alloc",
    "DeallocOp": "qecl.dealloc",
    "ExtractCodeblockOp": "qecl.extract_block",
    "InsertCodeblockOp": "qecl.insert_block",
    "EncodeOp": "qecl.encode",
    "NoiseOp": "qecl.noise",
    "QecCycleOp": "qecl.qec",
    "IdentityOp": "qecl.identity",
    "PauliXOp": "qecl.x",
    "PauliYOp": "qecl.y",
    "PauliZOp": "qecl.z",
    "HadamardOp": "qecl.hadamard",
    "SOp": "qecl.s",
    "CnotOp": "qecl.cnot",
    "MeasureOp": "qecl.measure",
}

expected_attrs_names = {
    "LogicalCodeblockInitStateAttr": "qecl.codeblock_init_state",
    "LogicalCodeblockType": "qecl.codeblock",
    "LogicalHyperRegisterType": "qecl.hyperreg",
}


def test_qecl_dialect_name():
    """Test that the QecLogical dialect name is correct."""
    assert qecl.QecLogical.name == "qecl"


@pytest.mark.parametrize("op", all_ops)
def test_all_operations_names(op):
    """Test that all operations have the expected name."""
    op_class_name = op.__name__
    expected_name = expected_ops_names.get(op_class_name)
    assert (
        expected_name is not None
    ), f"Unexpected operation {op_class_name} found in QecLogical dialect"
    assert op.name == expected_name


@pytest.mark.parametrize("attr", all_attrs)
def test_all_attributes_names(attr):
    """Test that all attributes have the expected name."""
    attr_class_name = attr.__name__
    expected_name = expected_attrs_names.get(attr_class_name)
    assert (
        expected_name is not None
    ), f"Unexpected attribute {attr_class_name} found in QecLogical dialect"
    assert attr.name == expected_name


class TestQecLogicalTypes:
    """Tests relating to the qecl types."""

    @pytest.mark.parametrize("k", [1, 2, IntegerAttr(1, 64)])
    def test_qecl_type_constructor_codeblock(self, k: int | IntegerAttr[I64]):
        """Test the constructor of qecl.LogicalCodeblockType."""
        codeblock = qecl.LogicalCodeblockType(k)

        expected_k = k if isinstance(k, IntegerAttr) else IntegerAttr(k, 64)
        assert codeblock.k == expected_k

    @pytest.mark.parametrize("width", [1, 3, IntegerAttr(3, 64)])
    @pytest.mark.parametrize("k", [1, 2, IntegerAttr(1, 64)])
    def test_qecl_type_constructor_hyper_reg(
        self, width: int | IntegerAttr[I64], k: int | IntegerAttr[I64]
    ):
        """Test the constructor of qecl.LogicalHyperRegisterType."""
        hyper_reg = qecl.LogicalHyperRegisterType(width, k)

        expected_width = width if isinstance(width, IntegerAttr) else IntegerAttr(width, 64)
        expected_k = k if isinstance(k, IntegerAttr) else IntegerAttr(k, 64)
        assert hyper_reg.width == expected_width
        assert hyper_reg.k == expected_k


class TestQecLogicalOps:
    """Tests relating to the qecl ops."""

    width = IntegerAttr(3, 64)
    k = IntegerAttr(1, 64)
    idx_attr = IntegerAttr(0, IndexType())

    def _get_hyper_reg_value(self):
        return create_ssa_value(qecl.LogicalHyperRegisterType(self.width, self.k))

    def _get_codeblock_value(self):
        return create_ssa_value(qecl.LogicalCodeblockType(self.k))

    def test_qecl_op_constructor_alloc(self):
        """Test the constructor of the qecl.alloc op."""
        alloc_op = qecl.AllocOp(qecl.LogicalHyperRegisterType(self.width, self.k))
        assert len(alloc_op.result_types) == 1
        assert isinstance(alloc_op.result_types[0], qecl.LogicalHyperRegisterType)
        assert alloc_op.result_types[0].width == self.width
        assert alloc_op.result_types[0].k == self.k

    def test_qecl_op_constructor_dealloc(self):
        """Test the constructor of the qecl.dealloc op."""
        dealloc_op = qecl.DeallocOp(self._get_hyper_reg_value())
        assert len(dealloc_op.result_types) == 0

    @pytest.mark.parametrize(
        "idx", [0, IntegerAttr(0, IndexType()), IntegerAttr(0, i64), create_ssa_value(IndexType())]
    )
    def test_qecl_op_constructor_extract_block(self, idx, assert_valid_idx_attr):
        """Test the constructor of the qecl.extract_block op.

        Also check that when the `idx` input is static that the `idx_attr` of the op always has type
        `index`.
        """
        extract_block_op = qecl.ExtractCodeblockOp(hyper_reg=self._get_hyper_reg_value(), idx=idx)
        assert len(extract_block_op.result_types) == 1
        assert isinstance(extract_block_op.result_types[0], qecl.LogicalCodeblockType)
        assert extract_block_op.result_types[0].k == self.k

        assert_valid_idx_attr(extract_block_op, idx)

    @pytest.mark.parametrize(
        "idx", [0, IntegerAttr(0, IndexType()), IntegerAttr(0, i64), create_ssa_value(IndexType())]
    )
    def test_qecl_op_constructor_insert_block(self, idx, assert_valid_idx_attr):
        """Test the constructor of the qecl.insert_block op.

        Also check that when the `idx` input is static that the `idx_attr` of the op always has type
        `index`.
        """
        insert_block_op = qecl.InsertCodeblockOp(
            in_hyper_reg=self._get_hyper_reg_value(), idx=idx, codeblock=self._get_codeblock_value()
        )
        assert len(insert_block_op.result_types) == 1
        assert isinstance(insert_block_op.result_types[0], qecl.LogicalHyperRegisterType)
        assert insert_block_op.result_types[0].width == self.width
        assert insert_block_op.result_types[0].k == self.k

        assert_valid_idx_attr(insert_block_op, idx)

    @pytest.mark.parametrize("init_state", ["zero", qecl.LogicalCodeblockInitStateAttr("zero")])
    def test_qecl_op_constructor_encode(self, init_state):
        """Test the constructor of the qecl.encode op."""
        encode_op = qecl.EncodeOp(in_codeblock=self._get_codeblock_value(), init_state=init_state)
        assert len(encode_op.result_types) == 1
        assert isinstance(encode_op.result_types[0], qecl.LogicalCodeblockType)
        assert encode_op.result_types[0].k == self.k

    def test_qecl_op_constructor_noise(self):
        """Test the constructor of the qecl.noise op."""
        noise_op = qecl.NoiseOp(in_codeblock=self._get_codeblock_value())
        assert len(noise_op.result_types) == 1
        assert isinstance(noise_op.result_types[0], qecl.LogicalCodeblockType)
        assert noise_op.result_types[0].k == self.k

    def test_qecl_op_constructor_qec(self):
        """Test the constructor of the qecl.qec op."""
        qec_op = qecl.QecCycleOp(in_codeblock=self._get_codeblock_value())
        assert len(qec_op.result_types) == 1
        assert isinstance(qec_op.result_types[0], qecl.LogicalCodeblockType)
        assert qec_op.result_types[0].k == self.k

    @pytest.mark.parametrize("op", [qecl.IdentityOp, qecl.PauliXOp, qecl.PauliYOp, qecl.PauliZOp])
    @pytest.mark.parametrize("idx", [0, IntegerAttr(0, IndexType()), create_ssa_value(IndexType())])
    def test_qecl_op_constructor_paulis(self, op, idx):
        """Test the constructors of the qecl Pauli gate ops."""
        pauli_op = op(in_codeblock=self._get_codeblock_value(), idx=idx)
        assert len(pauli_op.result_types) == 1
        assert isinstance(pauli_op.result_types[0], qecl.LogicalCodeblockType)
        assert pauli_op.result_types[0].k == self.k

    @pytest.mark.parametrize("idx", [0, IntegerAttr(0, IndexType()), create_ssa_value(IndexType())])
    def test_qecl_op_constructor_hadamard(self, idx):
        """Test the constructor of the qecl.hadamard op."""
        hadamard_op = qecl.HadamardOp(in_codeblock=self._get_codeblock_value(), idx=idx)
        assert len(hadamard_op.result_types) == 1
        assert isinstance(hadamard_op.result_types[0], qecl.LogicalCodeblockType)
        assert hadamard_op.result_types[0].k == self.k

    @pytest.mark.parametrize("idx", [0, IntegerAttr(0, IndexType()), create_ssa_value(IndexType())])
    @pytest.mark.parametrize("adj", [False, True, UnitAttr()])
    def test_qecl_op_constructor_s(self, idx, adj):
        """Test the constructor of the qecl.s op."""
        s_op = qecl.SOp(in_codeblock=self._get_codeblock_value(), idx=idx, adjoint=adj)
        assert len(s_op.result_types) == 1
        assert isinstance(s_op.result_types[0], qecl.LogicalCodeblockType)
        assert s_op.result_types[0].k == self.k

        if adj:
            assert s_op.properties.get("adjoint") == UnitAttr()
        else:
            assert s_op.properties.get("adjoint") is None

    @pytest.mark.parametrize(
        "idx_ctrl", [0, IntegerAttr(0, IndexType()), create_ssa_value(IndexType())]
    )
    @pytest.mark.parametrize(
        "idx_trgt", [0, IntegerAttr(0, IndexType()), create_ssa_value(IndexType())]
    )
    def test_qecl_op_constructor_cnot(self, idx_ctrl, idx_trgt):
        """Test the constructor of the qecl.cnot op."""
        cnot_op = qecl.CnotOp(
            in_ctrl_codeblock=self._get_codeblock_value(),
            idx_ctrl=idx_ctrl,
            in_trgt_codeblock=self._get_codeblock_value(),
            idx_trgt=idx_trgt,
        )
        assert len(cnot_op.result_types) == 2
        assert isinstance(cnot_op.result_types[0], qecl.LogicalCodeblockType)
        assert cnot_op.result_types[0].k == self.k
        assert isinstance(cnot_op.result_types[1], qecl.LogicalCodeblockType)
        assert cnot_op.result_types[1].k == self.k

    @pytest.mark.parametrize("idx", [0, IntegerAttr(0, IndexType()), create_ssa_value(IndexType())])
    def test_qecl_op_constructor_measure(self, idx):
        """Test the constructor of the qecl.measure op."""
        measure_op = qecl.MeasureOp(in_codeblock=self._get_codeblock_value(), idx=idx)
        assert len(measure_op.result_types) == 2
        assert measure_op.result_types[0] == IntegerType(1)
        assert isinstance(measure_op.result_types[1], qecl.LogicalCodeblockType)
        assert measure_op.result_types[1].k == self.k


@pytest.mark.parametrize(
    "pretty_print", [pytest.param(True, id="pretty_print"), pytest.param(False, id="generic_print")]
)
def test_assembly_format(run_filecheck, pretty_print):
    """Test the assembly format of the qecl ops."""
    program = r"""
    // CHECK: [[codeblock:%.+]] = "test.op"() : () -> !qecl.codeblock<1>
    %codeblock = "test.op"() : () -> !qecl.codeblock<1>

    // CHECK: [[hyperreg:%.+]] = "test.op"() : () -> !qecl.hyperreg<3 x 1>
    %hyperreg = "test.op"() : () -> !qecl.hyperreg<3 x 1>

    // CHECK: [[hreg0:%.+]] = qecl.alloc() : !qecl.hyperreg<3 x 1>
    %hreg0 = qecl.alloc() : !qecl.hyperreg<3 x 1>

    // CHECK: qecl.dealloc [[hreg0]] : !qecl.hyperreg<3 x 1>
    qecl.dealloc %hreg0 : !qecl.hyperreg<3 x 1>

    // CHECK: [[block0:%.+]] = qecl.extract_block [[hyperreg]][{{\s*}}0] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<1>
    %block0 = qecl.extract_block %hyperreg[ 0] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<1>

    // CHECK: qecl.insert_block [[hyperreg]][{{\s*}}0], [[block0]] : !qecl.hyperreg<3 x 1>, !qecl.codeblock<1>
    %hreg1 = qecl.insert_block %hyperreg[ 0], %block0 : !qecl.hyperreg<3 x 1>, !qecl.codeblock<1>

    // CHECK: [[block1:%.+]] = qecl.encode{{\s*}}[zero] [[block0]] : !qecl.codeblock<1>
    %block1 = qecl.encode [zero] %block0 : !qecl.codeblock<1>

    // CHECK: [[block1_1:%.+]] = qecl.noise [[block1]] : !qecl.codeblock<1>
    %block1_1 = qecl.noise %block1 : !qecl.codeblock<1>

    // CHECK: [[block2:%.+]] = qecl.qec [[block1_1]] : !qecl.codeblock<1>
    %block2 = qecl.qec %block1_1 : !qecl.codeblock<1>

    // CHECK: [[block3:%.+]] = qecl.identity [[block2]][{{\s*}}0] : !qecl.codeblock<1>
    %block3 = qecl.identity %block2[0] : !qecl.codeblock<1>

    // CHECK: [[block4:%.+]] = qecl.x [[block3]][{{\s*}}0] : !qecl.codeblock<1>
    %block4 = qecl.x %block3[0] : !qecl.codeblock<1>

    // CHECK: [[block5:%.+]] = qecl.x [[block4]][{{\s*}}0] : !qecl.codeblock<1>
    %block5 = qecl.x %block4[0] : !qecl.codeblock<1>

    // CHECK: [[block6:%.+]] = qecl.y [[block5]][{{\s*}}0] : !qecl.codeblock<1>
    %block6 = qecl.y %block5[0] : !qecl.codeblock<1>

    // CHECK: [[block7:%.+]] = qecl.z [[block6]][{{\s*}}0] : !qecl.codeblock<1>
    %block7 = qecl.z %block6[0] : !qecl.codeblock<1>

    // CHECK: [[block8:%.+]] = qecl.hadamard [[block7]][{{\s*}}0] : !qecl.codeblock<1>
    %block8 = qecl.hadamard %block7[0] : !qecl.codeblock<1>

    // CHECK: [[block9:%.+]] = qecl.s [[block8]][{{\s*}}0] : !qecl.codeblock<1>
    %block9 = qecl.s %block8[0] : !qecl.codeblock<1>

    // CHECK: [[block10:%.+]] = qecl.s [[block9]][{{\s*}}0] adj : !qecl.codeblock<1>
    %block10 = qecl.s %block9[0] adj : !qecl.codeblock<1>

    // CHECK: [[block_ctrl:%.+]] = "test.op"() : () -> !qecl.codeblock<1>
    // CHECK: [[block11:%.+]], [[block12:%.+]] = qecl.cnot [[block_ctrl]][{{\s*}}0], [[block10]][{{\s*}}0]
    %block_ctrl = "test.op"() : () -> !qecl.codeblock<1>
    %block11, %block12 = qecl.cnot %block_ctrl[0], %block10[0] : !qecl.codeblock<1>, !qecl.codeblock<1>

    // CHECK: [[block13:%.+]] = "test.op"() : () -> !qecl.codeblock<1>
    // CHECK: [[block14:%.+]] = qecl.noise [[block13:%.+]] : !qecl.codeblock<1>
    %block13 = "test.op"() : () -> !qecl.codeblock<1>
    %block14 = qecl.noise %block13 : !qecl.codeblock<1>

    """

    run_filecheck(program, roundtrip=True, verify=True, pretty_print=pretty_print)


class TestQecLogicalHelpers:
    """Tests for the QEC logical dialect helper functions"""

    @pytest.mark.parametrize(
        "in_hyper_reg_type",
        [
            qecl.LogicalHyperRegisterType(1, 1),
            qecl.LogicalHyperRegisterType(1, 3),
            qecl.LogicalHyperRegisterType(3, 1),
            qecl.LogicalHyperRegisterType(3, 3),
        ],
    )
    def test_get_logical_hyper_reg_type(self, in_hyper_reg_type):
        """Test that the qecl.get_logical_hyper_reg_type function returns the correct type when
        given an SSA value or an operation.
        """
        in_hyper_reg_ssa_val = create_ssa_value(in_hyper_reg_type)
        out_hyper_reg_type_from_ssa = qecl.get_logical_hyper_reg_type(in_hyper_reg_ssa_val)
        assert in_hyper_reg_type == out_hyper_reg_type_from_ssa

        in_hyper_reg_defining_op = in_hyper_reg_ssa_val.op
        out_hyper_reg_type_from_op = qecl.get_logical_hyper_reg_type(in_hyper_reg_defining_op)
        assert in_hyper_reg_type == out_hyper_reg_type_from_op
