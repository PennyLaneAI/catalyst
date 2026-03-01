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
from xdsl.dialects.test import TestOp
from xdsl.ir import AttributeCovT, OpResult

from catalyst.python_interface.dialects import qecp

pytestmark = pytest.mark.xdsl


# Test function taken from xdsl/utils/test_value.py
def create_ssa_value(t: AttributeCovT) -> OpResult[AttributeCovT]:
    """Create a single SSA value with the given type for testing purposes."""
    op = TestOp(result_types=(t,))
    return op.results[0]


all_ops = list(qecp.QecPhysical.operations)
all_attrs = list(qecp.QecPhysical.attributes)

expected_ops_names = {}

expected_attrs_names = {
    "QecPhysicalQubitRoleAttr": "qecp.qubit_role",
    "QecPhysicalQubitType": "qecp.qubit",
    "PhysicalCodeblockType": "qecp.codeblock",
    "PhysicalHyperRegisterType": "qecp.hyperreg",
}


q_data = create_ssa_value(qecp.QecPhysicalQubitType("data"))
q_aux = create_ssa_value(qecp.QecPhysicalQubitType("aux"))
codeblock = create_ssa_value(qecp.PhysicalCodeblockType(1, 7))
hyperreg = create_ssa_value(qecp.PhysicalHyperRegisterType(3, 1, 7))


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


@pytest.mark.parametrize(
    "pretty_print", [pytest.param(True, id="pretty_print"), pytest.param(False, id="generic_print")]
)
def test_assembly_format(run_filecheck, pretty_print):
    """Test the assembly format of the mbqc ops."""
    program = r"""
    // CHECK: [[q_data:%.+]] = "test.op"() : () -> !qecp.qubit<data>
    %q_data = "test.op"() : () -> !qecp.qubit<data>

    // CHECK: [[q_aux:%.+]] = "test.op"() : () -> !qecp.qubit<aux>
    %q_aux = "test.op"() : () -> !qecp.qubit<aux>

    // CHECK: [[codeblock:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
    %codeblock = "test.op"() : () -> !qecp.codeblock<1 x 7>

    // CHECK: [[hyperreg:%.+]] = "test.op"() : () -> !qecp.hyperreg<3 x 1 x 7>
    %hyperreg = "test.op"() : () -> !qecp.hyperreg<3 x 1 x 7>
    """

    run_filecheck(program, roundtrip=True, verify=True, pretty_print=pretty_print)
