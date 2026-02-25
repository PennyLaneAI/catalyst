# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the xDSL PBC dialect."""

import pytest

from catalyst.python_interface.dialects import PBC

pytestmark = pytest.mark.xdsl

all_ops = list(PBC.operations)
all_attrs = list(PBC.attributes)

expected_ops_names = {
    "FabricateOp": "pbc.fabricate",
    "LayerOp": "pbc.layer",
    "PPMeasurementOp": "pbc.ppm",
    "PPRotationArbitraryOp": "pbc.ppr.arbitrary",
    "PPRotationOp": "pbc.ppr",
    "PrepareStateOp": "pbc.prepare",
    "SelectPPMeasurementOp": "pbc.select.ppm",
    "YieldOp": "pbc.yield",
}

expected_attrs_names = {
    "LogicalInit": "pbc.enum",
}


def test_pbc_dialect_name():
    """Test that the PBC dialect name is correct."""
    assert PBC.name == "pbc"


@pytest.mark.parametrize("op", all_ops)
def test_all_operations_names(op):
    """Test that all operations have the expected name."""
    op_class_name = op.__name__
    expected_name = expected_ops_names.get(op_class_name)
    assert expected_name is not None, f"Unexpected operation {op_class_name} found in PBC dialect"
    assert op.name == expected_name


@pytest.mark.parametrize("attr", all_attrs)
def test_all_attributes_names(attr):
    """Test that all attributes have the expected name."""
    attr_class_name = attr.__name__
    expected_name = expected_attrs_names.get(attr_class_name)
    assert expected_name is not None, f"Unexpected attribute {attr_class_name} found in PBC dialect"
    assert attr.name == expected_name


@pytest.mark.parametrize(
    "pretty_print", [pytest.param(True, id="pretty_print"), pytest.param(False, id="generic_print")]
)
def test_assembly_format(run_filecheck, pretty_print):
    """Test the assembly format of the pbc ops."""
    program = """
    // CHECK: [[Q0:%.+]], [[Q1:%.+]], [[Q2:%.+]] = "test.op"() : () -> (!quantum.bit
    %q0, %q1, %q2 = "test.op"() : () -> (!quantum.bit, !quantum.bit, !quantum.bit)

    // CHECK: [[PARAM:%.+]] = "test.op"() : () -> f64
    %param = "test.op"() : () -> f64

    // CHECK: [[COND:%.+]] = "test.op"() : () -> i1
    %cond = "test.op"() : () -> i1

    // CHECK: {{%.+}} = pbc.fabricate magic : !quantum.bit
    %fabricated = pbc.fabricate magic : !quantum.bit

    // CHECK: {{%.+}} = pbc.prepare zero [[Q0]] : !quantum.bit
    %prepared = pbc.prepare zero %q0 : !quantum.bit

    // CHECK: {{%.+}}, {{%.+}} = pbc.ppr.arbitrary ["X", "Y"]([[PARAM]]) [[Q0]], [[Q1]] : !quantum.bit,
    %arb0, %arb1 = pbc.ppr.arbitrary ["X", "Y"](%param) %q0, %q1 : !quantum.bit, !quantum.bit

    // CHECK: {{%.+}}, {{%.+}} = pbc.ppr ["X", "I"](4) [[Q0]], [[Q1]] : !quantum.bit,
    %r0, %r1 = pbc.ppr ["X", "I"](4) %q0, %q1 : !quantum.bit, !quantum.bit

    // CHECK: {{%.+}}, {{%.+}}, {{%.+}} = pbc.ppm ["X", "Z"] [[Q0]], [[Q1]] : i1, !quantum.bit,
    %measured, %m0, %m1 = pbc.ppm ["X", "Z"] %q0, %q1 : i1, !quantum.bit, !quantum.bit

    // CHECK: {{%.+}}, {{%.+}} = pbc.select.ppm([[COND]], ["X"], ["Z"]) [[Q0]] : i1, !quantum.bit
    %select_measured, %select_out = pbc.select.ppm (%cond, ["X"], ["Z"]) %q0 : i1, !quantum.bit

    """

    run_filecheck(program, roundtrip=True, verify=True, pretty_print=pretty_print)
