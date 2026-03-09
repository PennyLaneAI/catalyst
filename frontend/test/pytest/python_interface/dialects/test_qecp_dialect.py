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

import pytest
from xdsl.dialects import test
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.ir import AttributeCovT, OpResult

from catalyst.python_interface.dialects import qecp

pytestmark = pytest.mark.xdsl


# Test function taken from xdsl/utils/test_value.py
def create_ssa_value(t: AttributeCovT) -> OpResult[AttributeCovT]:
    """Create a single SSA value with the given type for testing purposes."""
    op = test.TestOp(result_types=(t,))
    return op.results[0]


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


def test_type_constructors():
    """Test the constructors of each type defined in the qecp dialect work as expected."""

    q_data = qecp.QecPhysicalQubitType("data")
    assert isinstance(q_data.role, qecp.QecPhysicalQubitRoleAttr)
    assert q_data.role.data == str(qecp.QecPhysicalQubitRole.Data)

    q_aux = qecp.QecPhysicalQubitType("aux")
    assert isinstance(q_aux.role, qecp.QecPhysicalQubitRoleAttr)
    assert q_aux.role.data == str(qecp.QecPhysicalQubitRole.Aux)

    codeblock = qecp.PhysicalCodeblockType(1, 7)
    assert isinstance(codeblock.k, IntegerAttr)
    assert codeblock.k.value.data == 1
    assert codeblock.k.type == IntegerType(64)
    assert codeblock.n.value.data == 7
    assert codeblock.n.type == IntegerType(64)

    hyper_reg = qecp.PhysicalHyperRegisterType(3, 1, 7)
    assert isinstance(hyper_reg.width, IntegerAttr)
    assert hyper_reg.width.value.data == 3
    assert hyper_reg.width.type == IntegerType(64)
    assert isinstance(hyper_reg.k, IntegerAttr)
    assert hyper_reg.k.value.data == 1
    assert hyper_reg.k.type == IntegerType(64)
    assert isinstance(hyper_reg.n, IntegerAttr)
    assert hyper_reg.n.value.data == 7
    assert hyper_reg.n.type == IntegerType(64)


def test_op_constructors():
    """Test the constructors of each op defined in the qecp dialect work as expected."""
    width = 3
    k = 1
    n = 7

    hyper_reg = create_ssa_value(qecp.PhysicalHyperRegisterType(width, k, n))
    codeblock = create_ssa_value(qecp.PhysicalCodeblockType(k, n))
    q_aux = create_ssa_value(qecp.QecPhysicalQubitType("aux"))
    q_data = create_ssa_value(qecp.QecPhysicalQubitType("data"))

    # alloc
    alloc_op = qecp.AllocOp(qecp.PhysicalHyperRegisterType(width, k, n))
    assert len(alloc_op.result_types) == 1
    assert isinstance(alloc_op.result_types[0], qecp.PhysicalHyperRegisterType)
    assert alloc_op.result_types[0].width.value.data == width
    assert alloc_op.result_types[0].k.value.data == k
    assert alloc_op.result_types[0].n.value.data == n

    # dealloc
    dealloc_op = qecp.DeallocOp(hyper_reg)
    assert len(dealloc_op.result_types) == 0

    # alloc_aux
    alloc_aux_op = qecp.AllocAuxQubitOp()
    assert len(alloc_aux_op.result_types) == 1
    assert isinstance(alloc_aux_op.result_types[0], qecp.QecPhysicalQubitType)
    assert alloc_aux_op.result_types[0].role.data == qecp.QecPhysicalQubitRole.Aux

    # dealloc_aux
    dealloc_aux_op = qecp.DeallocAuxQubitOp(q_aux)
    assert len(dealloc_aux_op.result_types) == 0

    # extract_block
    extract_block_op = qecp.ExtractCodeblockOp(hyper_reg=hyper_reg, idx=0)
    assert len(extract_block_op.result_types) == 1
    assert isinstance(extract_block_op.result_types[0], qecp.PhysicalCodeblockType)
    assert extract_block_op.result_types[0].k.value.data == k
    assert extract_block_op.result_types[0].n.value.data == n

    # insert_block
    insert_block_op = qecp.InsertCodeblockOp(in_hyper_reg=hyper_reg, idx=0, codeblock=codeblock)
    assert len(insert_block_op.result_types) == 1
    assert isinstance(insert_block_op.result_types[0], qecp.PhysicalHyperRegisterType)
    assert insert_block_op.result_types[0].width.value.data == width
    assert insert_block_op.result_types[0].k.value.data == k
    assert insert_block_op.result_types[0].n.value.data == n

    # extract
    extract_op = qecp.ExtractQubitOp(codeblock=codeblock, idx=0)
    assert len(extract_op.result_types) == 1
    assert isinstance(extract_op.result_types[0], qecp.QecPhysicalQubitType)
    assert extract_op.result_types[0].role.data == qecp.QecPhysicalQubitRole.Data

    # insert
    insert_op = qecp.InsertQubitOp(in_codeblock=codeblock, idx=0, qubit=q_data)
    assert len(insert_op.result_types) == 1
    assert isinstance(insert_op.result_types[0], qecp.PhysicalCodeblockType)
    assert insert_op.result_types[0].k.value.data == k
    assert insert_op.result_types[0].n.value.data == n


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
    """

    run_filecheck(program, roundtrip=True, verify=True, pretty_print=pretty_print)
