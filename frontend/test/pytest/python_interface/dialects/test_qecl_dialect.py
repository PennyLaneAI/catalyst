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

import pytest
from xdsl.dialects import test
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.ir import AttributeCovT, OpResult

from catalyst.python_interface.dialects import qecl

pytestmark = pytest.mark.xdsl


# Test function taken from xdsl/utils/test_value.py
def create_ssa_value(t: AttributeCovT) -> OpResult[AttributeCovT]:
    """Create a single SSA value with the given type for testing purposes."""
    op = test.TestOp(result_types=(t,))
    return op.results[0]


all_ops = list(qecl.QecLogical.operations)
all_attrs = list(qecl.QecLogical.attributes)

expected_ops_names = {
    "AllocOp": "qecl.alloc",
    "DeallocOp": "qecl.dealloc",
    "ExtractCodeblockOp": "qecl.extract_block",
    "InsertCodeblockOp": "qecl.insert_block",
}

expected_attrs_names = {
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


def test_type_constructors():
    """Test the constructors of each type defined in the qecl dialect work as expected."""
    codeblock = qecl.LogicalCodeblockType(1)
    assert isinstance(codeblock.k, IntegerAttr)
    assert codeblock.k.value.data == 1
    assert codeblock.k.type == IntegerType(64)

    hyper_reg = qecl.LogicalHyperRegisterType(3, 1)
    assert isinstance(hyper_reg.width, IntegerAttr)
    assert hyper_reg.width.value.data == 3
    assert hyper_reg.width.type == IntegerType(64)
    assert isinstance(hyper_reg.k, IntegerAttr)
    assert hyper_reg.k.value.data == 1
    assert hyper_reg.k.type == IntegerType(64)


def test_op_constructors():
    """Test the constructors of each op defined in the qecl dialect work as expected."""
    hyper_reg = create_ssa_value(qecl.LogicalHyperRegisterType(3, 1))
    codeblock = create_ssa_value(qecl.LogicalCodeblockType(1))

    breakpoint()

    # alloc
    alloc_op = qecl.AllocOp(result_types=(qecl.LogicalHyperRegisterType(3, 1),))
    assert len(alloc_op.result_types) == 1
    assert isinstance(alloc_op.result_types[0], qecl.LogicalHyperRegisterType)

    # dealloc
    dealloc_op = qecl.DeallocOp(operands=(hyper_reg,))
    assert len(dealloc_op.result_types) == 0

    # extract_block
    extract_block_op = qecl.ExtractCodeblockOp(hyper_reg=hyper_reg, idx=0)
    assert len(extract_block_op.result_types) == 1
    assert isinstance(extract_block_op.result_types[0], qecl.LogicalCodeblockType)

    # insert_block
    insert_block_op = qecl.InsertCodeblockOp(in_hyper_reg=hyper_reg, idx=0, codeblock=codeblock)
    assert len(insert_block_op.result_types) == 1
    assert isinstance(insert_block_op.result_types[0], qecl.LogicalHyperRegisterType)


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
    """

    run_filecheck(program, roundtrip=True, verify=True, pretty_print=pretty_print)
