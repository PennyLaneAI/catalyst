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

"""Unit tests for the xDSL QecPhysical dialect."""

from typing import cast

import pytest
from xdsl.dialects import test
from xdsl.dialects.builtin import IndexType, IntegerAttr, UnitAttr
from xdsl.ir import AttributeCovT, OpResult

from catalyst.python_interface.dialects import qecp

pytestmark = pytest.mark.xdsl


# Test function taken from xdsl/utils/test_value.py
def create_ssa_value(t: AttributeCovT) -> OpResult[AttributeCovT]:
    """Create a single SSA value with the given type for testing purposes."""
    op = test.TestOp(result_types=(t,))
    return cast(OpResult[AttributeCovT], op.results[0])


all_ops = list(qecp.QecPhysical.operations)
all_attrs = list(qecp.QecPhysical.attributes)

expected_ops_names = {
    "AllocOp": "qecp.alloc",
    "DeallocOp": "qecp.dealloc",
    "AllocAuxQubitOp": "qecp.alloc_aux",
    "DeallocAuxQubitOp": "qecp.dealloc_aux",
    "ExtractCodeblockOp": "qecp.extract_block",
    "InsertCodeblockOp": "qecp.insert_block",
    "ExtractQubitOp": "qecp.extract",
    "InsertQubitOp": "qecp.insert",
    "IdentityOp": "qecp.identity",
    "PauliXOp": "qecp.x",
    "PauliYOp": "qecp.y",
    "PauliZOp": "qecp.z",
    "HadamardOp": "qecp.hadamard",
    "SOp": "qecp.s",
    "CnotOp": "qecp.cnot",
}

expected_attrs_names = {
    "QecPhysicalQubitRoleAttr": "qecp.qubit_role",
    "QecPhysicalQubitType": "qecp.qubit",
    "PhysicalCodeblockType": "qecp.codeblock",
    "PhysicalHyperRegisterType": "qecp.hyperreg",
}


def test_qecp_dialect_name():
    """Test that the QecPhysical dialect name is correct."""
    assert qecp.QecPhysical.name == "qecp"


@pytest.mark.parametrize("op", all_ops)
def test_all_operations_names(op):
    """Test that all operations have the expected name."""
    op_class_name = op.__name__
    expected_name = expected_ops_names.get(op_class_name)
    assert (
        expected_name is not None
    ), f"Unexpected operation {op_class_name} found in QecPhysical dialect"
    assert op.name == expected_name


@pytest.mark.parametrize("attr", all_attrs)
def test_all_attributes_names(attr):
    """Test that all attributes have the expected name."""
    attr_class_name = attr.__name__
    expected_name = expected_attrs_names.get(attr_class_name)
    assert (
        expected_name is not None
    ), f"Unexpected attribute {attr_class_name} found in QecPhysical dialect"
    assert attr.name == expected_name


class TestQecPhysicalTypes:
    """Tests relating to the qecp types."""

    @pytest.mark.parametrize(
        "role", ["data", qecp.QecPhysicalQubitRole.Data, "aux", qecp.QecPhysicalQubitRole.Aux]
    )
    def test_qecp_type_constructor_qubit(self, role):
        """Test the constructor of qecp.QecPhysicalQubitType."""
        q_data = qecp.QecPhysicalQubitType(role)
        expected_role = qecp.QecPhysicalQubitRoleAttr(role)
        assert q_data.role == expected_role

    @pytest.mark.parametrize("k", [1, 2, IntegerAttr(1, 64)])
    @pytest.mark.parametrize("n", [1, 7, IntegerAttr(7, 64)])
    def test_qecp_type_constructor_codeblock(self, k, n):
        """Test the constructor of qecp.PhysicalCodeblockType."""
        codeblock = qecp.PhysicalCodeblockType(k, n)
        expected_k = k if isinstance(k, IntegerAttr) else IntegerAttr(k, 64)
        expected_n = n if isinstance(n, IntegerAttr) else IntegerAttr(n, 64)
        assert codeblock.k == expected_k
        assert codeblock.n == expected_n

    @pytest.mark.parametrize("width", [1, 3, IntegerAttr(3, 64)])
    @pytest.mark.parametrize("k", [1, 2, IntegerAttr(1, 64)])
    @pytest.mark.parametrize("n", [1, 7, IntegerAttr(7, 64)])
    def test_qecp_type_constructor_hyper_reg(self, width, k, n):
        """Test the constructor of qecp.PhysicalHyperRegisterType."""
        hyper_reg = qecp.PhysicalHyperRegisterType(width, k, n)
        expected_width = width if isinstance(width, IntegerAttr) else IntegerAttr(width, 64)
        expected_k = k if isinstance(k, IntegerAttr) else IntegerAttr(k, 64)
        expected_n = n if isinstance(n, IntegerAttr) else IntegerAttr(n, 64)
        assert hyper_reg.width == expected_width
        assert hyper_reg.k == expected_k
        assert hyper_reg.n == expected_n


class TestQecPhysicalOps:
    """Tests relating to the qecp ops."""

    width = IntegerAttr(3, 64)
    k = IntegerAttr(1, 64)
    n = IntegerAttr(7, 64)
    idx_attr = IntegerAttr(0, IndexType())

    def _get_hyper_reg_value(self):
        return create_ssa_value(qecp.PhysicalHyperRegisterType(self.width, self.k, self.n))

    def _get_codeblock_value(self):
        return create_ssa_value(qecp.PhysicalCodeblockType(self.k, self.n))

    def _get_qubit_data_value(self):
        return create_ssa_value(qecp.QecPhysicalQubitType("data"))

    def _get_qubit_aux_value(self):
        return create_ssa_value(qecp.QecPhysicalQubitType("aux"))

    def test_qecp_op_constructor_alloc(self):
        """Test the constructor of the qecp.alloc op."""
        alloc_op = qecp.AllocOp(qecp.PhysicalHyperRegisterType(self.width, self.k, self.n))
        assert len(alloc_op.result_types) == 1
        assert isinstance(alloc_op.result_types[0], qecp.PhysicalHyperRegisterType)
        assert alloc_op.result_types[0].width == self.width
        assert alloc_op.result_types[0].k == self.k
        assert alloc_op.result_types[0].n == self.n

    def test_qecp_op_constructor_dealloc(self):
        """Test the constructor of the qecp.dealloc op."""
        dealloc_op = qecp.DeallocOp(self._get_hyper_reg_value())
        assert len(dealloc_op.result_types) == 0

    def test_qecp_op_constructor_alloc_aux(self):
        """Test the constructor of the qecp.alloc_aux op."""
        alloc_aux_op = qecp.AllocAuxQubitOp()
        assert len(alloc_aux_op.result_types) == 1
        assert isinstance(alloc_aux_op.result_types[0], qecp.QecPhysicalQubitType)
        assert alloc_aux_op.result_types[0].role.data == qecp.QecPhysicalQubitRole.Aux

    def test_qecp_op_constructor_dealloc_aux(self):
        """Test the constructor of the qecp.dealloc_aux op."""
        dealloc_aux_op = qecp.DeallocAuxQubitOp(self._get_qubit_aux_value())
        assert len(dealloc_aux_op.result_types) == 0

    @pytest.mark.parametrize(
        "idx", [0, IntegerAttr.from_index_int_value(0), create_ssa_value(IndexType())]
    )
    def test_qecp_op_constructor_extract_block(self, idx):
        """Test the constructor of the qecp.extract_block op."""
        extract_block_op = qecp.ExtractCodeblockOp(hyper_reg=self._get_hyper_reg_value(), idx=idx)
        assert len(extract_block_op.result_types) == 1
        assert isinstance(extract_block_op.result_types[0], qecp.PhysicalCodeblockType)
        assert extract_block_op.result_types[0].k == self.k
        assert extract_block_op.result_types[0].n == self.n

    @pytest.mark.parametrize(
        "idx", [0, IntegerAttr.from_index_int_value(0), create_ssa_value(IndexType())]
    )
    def test_qecp_op_constructor_insert_block(self, idx):
        """Test the constructor of the qecp.insert_block op."""
        insert_block_op = qecp.InsertCodeblockOp(
            in_hyper_reg=self._get_hyper_reg_value(), idx=idx, codeblock=self._get_codeblock_value()
        )
        assert len(insert_block_op.result_types) == 1
        assert isinstance(insert_block_op.result_types[0], qecp.PhysicalHyperRegisterType)
        assert insert_block_op.result_types[0].width == self.width
        assert insert_block_op.result_types[0].k == self.k
        assert insert_block_op.result_types[0].n == self.n

    @pytest.mark.parametrize(
        "idx", [0, IntegerAttr.from_index_int_value(0), create_ssa_value(IndexType())]
    )
    def test_qecp_op_constructor_extract(self, idx):
        """Test the constructor of the qecp.extract op."""
        extract_op = qecp.ExtractQubitOp(codeblock=self._get_codeblock_value(), idx=idx)
        assert len(extract_op.result_types) == 1
        assert isinstance(extract_op.result_types[0], qecp.QecPhysicalQubitType)
        assert extract_op.result_types[0].role.data == qecp.QecPhysicalQubitRole.Data

    @pytest.mark.parametrize(
        "idx", [0, IntegerAttr.from_index_int_value(0), create_ssa_value(IndexType())]
    )
    def test_qecp_op_constructor_insert(self, idx):
        """Test the constructor of the qecp.insert op."""
        insert_op = qecp.InsertQubitOp(
            in_codeblock=self._get_codeblock_value(), idx=idx, qubit=self._get_qubit_data_value()
        )
        assert len(insert_op.result_types) == 1
        assert isinstance(insert_op.result_types[0], qecp.PhysicalCodeblockType)
        assert insert_op.result_types[0].k == self.k
        assert insert_op.result_types[0].n == self.n

    @pytest.mark.parametrize("op", [qecp.IdentityOp, qecp.PauliXOp, qecp.PauliYOp, qecp.PauliZOp])
    @pytest.mark.parametrize(
        "qubit",
        [
            create_ssa_value(qecp.QecPhysicalQubitType("data")),
            create_ssa_value(qecp.QecPhysicalQubitType("aux")),
        ],
    )
    def test_qecp_op_constructor_pauli(self, op, qubit):
        """Test the constructor of the qecp Pauli gate op."""
        pauli_op = op(qubit)
        assert len(pauli_op.operands) == 1
        assert pauli_op.operand_types[0] == qubit.type
        assert len(pauli_op.result_types) == 1
        assert pauli_op.result_types[0] == qubit.type

    @pytest.mark.parametrize(
        "qubit",
        [
            create_ssa_value(qecp.QecPhysicalQubitType("data")),
            create_ssa_value(qecp.QecPhysicalQubitType("aux")),
        ],
    )
    def test_qecp_op_constructor_hadamard(self, qubit):
        """Test the constructor of the qecp.hadamard op."""
        hadamard_op = qecp.HadamardOp(qubit)
        assert len(hadamard_op.operands) == 1
        assert hadamard_op.operand_types[0] == qubit.type
        assert len(hadamard_op.result_types) == 1
        assert hadamard_op.result_types[0] == qubit.type

    @pytest.mark.parametrize(
        "qubit",
        [
            create_ssa_value(qecp.QecPhysicalQubitType("data")),
            create_ssa_value(qecp.QecPhysicalQubitType("aux")),
        ],
    )
    @pytest.mark.parametrize("adj", [False, True, UnitAttr()])
    def test_qecp_op_constructor_s(self, qubit, adj):
        """Test the constructor of the qecp.s op."""
        s_op = qecp.SOp(qubit, adjoint=adj)
        assert len(s_op.operands) == 1
        assert s_op.operand_types[0] == qubit.type
        assert len(s_op.result_types) == 1
        assert s_op.result_types[0] == qubit.type

        if adj:
            assert s_op.properties.get("adjoint") == UnitAttr()
        else:
            assert s_op.properties.get("adjoint") is None

    @pytest.mark.parametrize(
        "qubit_ctrl",
        [
            create_ssa_value(qecp.QecPhysicalQubitType("data")),
            create_ssa_value(qecp.QecPhysicalQubitType("aux")),
        ],
    )
    @pytest.mark.parametrize(
        "qubit_trgt",
        [
            create_ssa_value(qecp.QecPhysicalQubitType("data")),
            create_ssa_value(qecp.QecPhysicalQubitType("aux")),
        ],
    )
    def test_qecp_op_constructor_cnot(self, qubit_ctrl, qubit_trgt):
        """Test the constructor of the qecp.cnot op."""
        cnot_op = qecp.CnotOp(qubit_ctrl, qubit_trgt)
        assert len(cnot_op.operands) == 2
        assert cnot_op.operand_types[0] == qubit_ctrl.type
        assert cnot_op.operand_types[1] == qubit_trgt.type
        assert len(cnot_op.result_types) == 2
        assert cnot_op.result_types[0] == qubit_ctrl.type
        assert cnot_op.result_types[1] == qubit_trgt.type


@pytest.mark.parametrize(
    "pretty_print", [pytest.param(True, id="pretty_print"), pytest.param(False, id="generic_print")]
)
def test_assembly_format(run_filecheck, pretty_print):
    """Test the assembly format of the qecp ops."""
    program = r"""
    // CHECK: [[q_data:%.+]] = "test.op"() : () -> !qecp.qubit<data>
    %q_data = "test.op"() : () -> !qecp.qubit<data>

    // CHECK: [[q_aux:%.+]] = "test.op"() : () -> !qecp.qubit<aux>
    %q_aux = "test.op"() : () -> !qecp.qubit<aux>

    // CHECK: [[codeblock:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
    %codeblock = "test.op"() : () -> !qecp.codeblock<1 x 7>

    // CHECK: [[hyperreg:%.+]] = "test.op"() : () -> !qecp.hyperreg<3 x 1 x 7>
    %hyperreg = "test.op"() : () -> !qecp.hyperreg<3 x 1 x 7>

    // CHECK: [[hreg0:%.+]] = qecp.alloc() : !qecp.hyperreg<3 x 1 x 7>
    %hreg0 = qecp.alloc() : !qecp.hyperreg<3 x 1 x 7>

    // CHECK: qecp.dealloc [[hreg0]] : !qecp.hyperreg<3 x 1 x 7>
    qecp.dealloc %hreg0 : !qecp.hyperreg<3 x 1 x 7>

    // CHECK: [[q_aux1:%.+]] = qecp.alloc_aux : !qecp.qubit<aux>
    %q_aux1 = qecp.alloc_aux : !qecp.qubit<aux>

    // CHECK: qecp.dealloc_aux [[q_aux1]] : !qecp.qubit<aux>
    qecp.dealloc_aux %q_aux1 : !qecp.qubit<aux>

    // CHECK: [[block0:%.+]] = qecp.extract_block [[hyperreg]][{{\s*}}0] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>
    %block0 = qecp.extract_block %hyperreg[ 0] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>

    // CHECK: qecp.insert_block [[hyperreg]][{{\s*}}0], [[block0]] : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 7>
    %hreg1 = qecp.insert_block %hyperreg[ 0], %block0 : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 7>

    // CHECK: [[q0:%.+]] = qecp.extract [[block0]][{{\s*}}0] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
    %q0 = qecp.extract %block0[ 0] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>

    // CHECK: [[block1:%.+]] = qecp.insert [[block0]][{{\s*}}0], [[q0]] : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
    %block1 = qecp.insert %block0[ 0], %q0 : !qecp.codeblock<1 x 7>, !qecp.qubit<data>

    // CHECK: [[qd0:%.+]] = "test.op"() : () -> !qecp.qubit<data>
    // CHECK: [[qa0:%.+]] = "test.op"() : () -> !qecp.qubit<aux>
    %qd0 = "test.op"() : () -> !qecp.qubit<data>
    %qa0 = "test.op"() : () -> !qecp.qubit<aux>

    // CHECK: [[qd1:%.+]] = qecp.identity [[qd0]] : !qecp.qubit<data>
    // CHECK: [[qa1:%.+]] = qecp.identity [[qa0]] : !qecp.qubit<aux>
    %qd1 = qecp.identity %qd0 : !qecp.qubit<data>
    %qa1 = qecp.identity %qa0 : !qecp.qubit<aux>

    // CHECK: [[qd2:%.+]] = qecp.x [[qd1]] : !qecp.qubit<data>
    // CHECK: [[qa2:%.+]] = qecp.x [[qa1]] : !qecp.qubit<aux>
    %qd2 = qecp.x %qd1 : !qecp.qubit<data>
    %qa2 = qecp.x %qa1 : !qecp.qubit<aux>

    // CHECK: [[qd3:%.+]] = qecp.y [[qd2]] : !qecp.qubit<data>
    // CHECK: [[qa3:%.+]] = qecp.y [[qa2]] : !qecp.qubit<aux>
    %qd3 = qecp.y %qd2 : !qecp.qubit<data>
    %qa3 = qecp.y %qa2 : !qecp.qubit<aux>

    // CHECK: [[qd4:%.+]] = qecp.z [[qd3]] : !qecp.qubit<data>
    // CHECK: [[qa4:%.+]] = qecp.z [[qa3]] : !qecp.qubit<aux>
    %qd4 = qecp.z %qd3 : !qecp.qubit<data>
    %qa4 = qecp.z %qa3 : !qecp.qubit<aux>

    // CHECK: [[qd5:%.+]] = qecp.hadamard [[qd4]] : !qecp.qubit<data>
    // CHECK: [[qa5:%.+]] = qecp.hadamard [[qa4]] : !qecp.qubit<aux>
    %qd5 = qecp.hadamard %qd4 : !qecp.qubit<data>
    %qa5 = qecp.hadamard %qa4 : !qecp.qubit<aux>

    // CHECK: [[qd6:%.+]] = qecp.s [[qd5]] : !qecp.qubit<data>
    // CHECK: [[qa6:%.+]] = qecp.s [[qa5]] : !qecp.qubit<aux>
    %qd6 = qecp.s %qd5 : !qecp.qubit<data>
    %qa6 = qecp.s %qa5 : !qecp.qubit<aux>

    // CHECK: [[qd7:%.+]] = qecp.s [[qd6]] adj : !qecp.qubit<data>
    // CHECK: [[qa7:%.+]] = qecp.s [[qa6]] adj : !qecp.qubit<aux>
    %qd7 = qecp.s %qd6 adj : !qecp.qubit<data>
    %qa7 = qecp.s %qa6 adj : !qecp.qubit<aux>

    // CHECK: [[qd10:%.+]] = "test.op"() : () -> !qecp.qubit<data>
    // CHECK: [[qd20:%.+]] = "test.op"() : () -> !qecp.qubit<data>
    // CHECK: [[qa10:%.+]] = "test.op"() : () -> !qecp.qubit<aux>
    // CHECK: [[qa20:%.+]] = "test.op"() : () -> !qecp.qubit<aux>
    %qd10 = "test.op"() : () -> !qecp.qubit<data>
    %qd20 = "test.op"() : () -> !qecp.qubit<data>
    %qa10 = "test.op"() : () -> !qecp.qubit<aux>
    %qa20 = "test.op"() : () -> !qecp.qubit<aux>

    // CHECK: [[qd8:%.+]], [[qd9:%.+]] = qecp.cnot [[qd10]], [[qd20]] : !qecp.qubit<data>, !qecp.qubit<data>
    %qd11, %qd21 = qecp.cnot %qd10, %qd20 : !qecp.qubit<data>, !qecp.qubit<data>

    // CHECK: [[qd12:%.+]], [[qa21:%.+]] = qecp.cnot [[qd11]], [[qa20]] : !qecp.qubit<data>, !qecp.qubit<aux>
    %qd12, %qa21 = qecp.cnot %qd11, %qa20 : !qecp.qubit<data>, !qecp.qubit<aux>

    // CHECK: [[qa11:%.+]], [[qd22:%.+]] = qecp.cnot [[qa10]], [[qd21]] : !qecp.qubit<aux>, !qecp.qubit<data>
    %qa11, %qd22 = qecp.cnot %qa10, %qd21 : !qecp.qubit<aux>, !qecp.qubit<data>

    // CHECK: [[qa13:%.+]], [[qa22:%.+]] = qecp.cnot [[qa11]], [[qa21]] : !qecp.qubit<aux>, !qecp.qubit<aux>
    %qa12, %qa22 = qecp.cnot %qa11, %qa21 : !qecp.qubit<aux>, !qecp.qubit<aux>
    """

    run_filecheck(program, roundtrip=True, verify=True, pretty_print=pretty_print)


class TestQecPhysicalHelpers:
    """Tests for the QEC physical dialect helper functions"""

    @pytest.mark.parametrize(
        "in_hyper_reg_type",
        [
            qecp.PhysicalHyperRegisterType(1, 1, 1),
            qecp.PhysicalHyperRegisterType(1, 1, 7),
            qecp.PhysicalHyperRegisterType(3, 1, 1),
            qecp.PhysicalHyperRegisterType(3, 1, 7),
            qecp.PhysicalHyperRegisterType(3, 2, 1),
            qecp.PhysicalHyperRegisterType(3, 2, 7),
        ],
    )
    def test_get_physical_hyper_reg_type(self, in_hyper_reg_type):
        """Test that the qecl.get_physical_hyper_reg_type function returns the correct type when
        given an SSA value or an operation.
        """
        in_hyper_reg_ssa_val = create_ssa_value(in_hyper_reg_type)
        out_hyper_reg_type_from_ssa = qecp.get_physical_hyper_reg_type(in_hyper_reg_ssa_val)
        assert in_hyper_reg_type == out_hyper_reg_type_from_ssa

        in_hyper_reg_defining_op = in_hyper_reg_ssa_val.op
        out_hyper_reg_type_from_op = qecp.get_physical_hyper_reg_type(in_hyper_reg_defining_op)
        assert in_hyper_reg_type == out_hyper_reg_type_from_op

    @pytest.mark.parametrize(
        "in_codeblock_type",
        [
            qecp.PhysicalCodeblockType(1, 1),
            qecp.PhysicalCodeblockType(1, 7),
            qecp.PhysicalCodeblockType(7, 1),
            qecp.PhysicalCodeblockType(7, 7),
        ],
    )
    def test_get_physical_codeblock_type(self, in_codeblock_type):
        """Test that the qecl.get_physical_codeblock_type function returns the correct type when
        given an SSA value or an operation.
        """
        in_codeblock_ssa_val = create_ssa_value(in_codeblock_type)
        out_codeblock_type_from_ssa = qecp.get_physical_codeblock_type(in_codeblock_ssa_val)
        assert in_codeblock_type == out_codeblock_type_from_ssa

        in_codeblock_defining_op = in_codeblock_ssa_val.op
        out_codeblock_type_from_op = qecp.get_physical_codeblock_type(in_codeblock_defining_op)
        assert in_codeblock_type == out_codeblock_type_from_op

    @pytest.mark.parametrize(
        "in_qubit_type",
        [
            qecp.QecPhysicalQubitType("data"),
            qecp.QecPhysicalQubitType("aux"),
        ],
    )
    def test_get_physical_qubit_type(self, in_qubit_type):
        """Test that the qecl.get_physical_qubit_type function returns the correct type when
        given an SSA value or an operation.
        """
        in_qubit_ssa_val = create_ssa_value(in_qubit_type)
        out_qubit_type_from_ssa = qecp.get_physical_qubit_type(in_qubit_ssa_val)
        assert in_qubit_type == out_qubit_type_from_ssa

        in_qubit_defining_op = in_qubit_ssa_val.op
        out_qubit_type_from_op = qecp.get_physical_qubit_type(in_qubit_defining_op)
        assert in_qubit_type == out_qubit_type_from_op
