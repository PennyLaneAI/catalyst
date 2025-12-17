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
"""Unit tests for the Unified Compiler transform interpreter."""

from typing import Any

import pytest

# pylint: disable=wrong-import-position
pytest.mark.xdsl
xdsl = pytest.importorskip("xdsl")

from xdsl.dialects import builtin, test, transform
from xdsl.ir import Attribute, Block, Region, SSAValue

from catalyst.python_interface.pass_api.transform_interpreter import (
    TransformFunctionsExt,
    TransformInterpreterPass,
    _create_schedule,
)


def _get_xdsl_attr_from_pyval(val) -> Attribute:
    """Get an xDSL attribute corresponding to a Python value."""
    attr = None

    match val:
        case bool():
            attr = builtin.IntegerAttr(int(val), 1)
        case int():
            attr = builtin.IntegerAttr(val, 64)
        case float():
            attr = builtin.FloatAttr(val, 64)
        case str():
            attr = builtin.StringAttr(val)
        case list() | tuple():
            attr = builtin.ArrayAttr(tuple(_get_xdsl_attr_from_pyval(elt) for elt in val))
        case dict():
            attr = builtin.DictionaryAttr({k: _get_xdsl_attr_from_pyval(v) for k, v in val.items()})
        case _:
            raise ValueError(f"Invalid type {val}")

    return attr


def create_apply_registered_pass_op(
    pass_name, options: dict[str, Any], in_mod: SSAValue[transform.OperationType] | None = None
) -> transform.ApplyRegisteredPassOp:
    """Create an ApplyRegisteredPassOp using the provided pass name and
    pass options."""
    lowered_options = {}
    for arg, value in options.items():
        option_key = str(arg).replace("_", "-")
        lowered_options[option_key] = _get_xdsl_attr_from_pyval(value)

    in_mod = (
        in_mod or test.TestOp(result_types=(transform.OperationType("builtin.module"),)).results[0]
    )

    pass_op = transform.ApplyRegisteredPassOp.create(
        properties={
            "pass_name": builtin.StringAttr(pass_name),
            "options": builtin.DictionaryAttr(lowered_options),
        },
        operands=(in_mod,),
        result_types=(transform.OperationType(in_mod.operation),),
    )
    return pass_op


def create_named_sequence_op(
    pass_names: list[str],
    pass_options: list[dict[str, Any]],
) -> transform.NamedSequenceOp:
    """Create a NamedSequenceOp using the provided registered pass ops."""

    named_sequence_block = Block(arg_types=(transform.OperationType("builtin.module"),))
    in_mod = named_sequence_block.args[0]
    ops: list[transform.ApplyRegisteredPassOp] = []

    for pass_name, options in zip(pass_names, pass_options):
        pass_op = create_apply_registered_pass_op(pass_name, in_mod, options)
        ops.append(pass_op)
        in_mod = pass_op.results[0]
    ops.append(transform.YieldOp())

    named_sequence_block.add_ops(ops)
    named_sequence_region = Region(named_sequence_block)
    named_sequence_op = transform.NamedSequenceOp(
        sym_name=TransformInterpreterPass.entry_point,
        function_type=[transform.OperationType("builtin.module"), ()],
        body=named_sequence_region,
    )
    return named_sequence_op


def create_test_module(
    pass_names: list[str], pass_options: list[dict[str, Any]]
) -> builtin.ModuleOp:
    """Create a module containing a NamedSequenceOp with the provided passes."""
    named_sequence_op = create_named_sequence_op(pass_names, pass_options)
    module = builtin.ModuleOp([named_sequence_op])
    return module


class TestCreateSchedule:
    """Unit tests for creating a schedule containing CL args for passes with options."""

    def test_pass_no_options(self):
        """Test that passes with no options are parsed correctly."""

    def test_pass_basic_options(self):
        """Test that passes with basic options (int, float, bool, string) are parsed correctly."""

    def test_pass_array_options(self):
        """Test that passes with array options are parsed correctly."""

    def test_pass_dict_options(self):
        """Test that passes with dict options are parsed correctly."""

    def test_pass_nested_container_options(self):
        """Test that passes with options that are nested containers are parsed correctly."""

    def test_multiple_passes(self):
        """Test that scheduling multiple passes works correctly."""


class TestTransformFunctionsExt:
    """Unit tests for the TransformFunctionsExt interpreter."""

    def test_xdsl_pass(self):
        """Test that interpreting an xDSL pass works correctly."""

    def test_xdsl_pass_with_options(self):
        """Test that interpreting an xDSL pass works correctly."""

    def test_mlir_pass(self):
        """Test that interpreting an MLIR pass works correctly."""

    def test_mlir_pass_with_options(self):
        """Test that interpreting an MLIR pass works correctly."""


class TestTransformInterpreterPass:
    """Unit tests for the TransformInterpreterPass."""

    def test_interpret_named_sequence_xdsl_passes(self):
        """Test that a NamedSequenceOp can be interpreted correctly when it only
        contains xDSL passes."""

    def test_interpret_named_sequence_mlir_passes(self):
        """Test that a NamedSequenceOp can be interpreted correctly when it only
        contains MLIR passes."""

    def test_interpret_named_sequence_mixed_passes(self):
        """Test that a NamedSequenceOp can be interpreted correctly when it contains
        both xDSL and MLIR passes."""


class TestTransformInterpreterPassIntegration:
    """Integration tests for the TransformInterpreterPass."""

    def test_qjit_with_passes_interpreted_correctly(self):
        """Test that applying the TransformInterpreter to a qjitted qnode's
        module correctly transforms it."""
