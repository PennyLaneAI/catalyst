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

import subprocess
from typing import Any
from unittest.mock import MagicMock

import pytest
from xdsl.context import Context
from xdsl.dialects import builtin, test, transform
from xdsl.ir import Attribute, Block, Region, SSAValue
from xdsl.passes import ModulePass, PassPipeline

from catalyst.python_interface.pass_api.transform_interpreter import (
    TransformFunctionsExt,
    TransformInterpreterPass,
    _create_schedule,
)
from catalyst.python_interface.transforms import MergeRotationsPass

pytestmark = pytest.mark.xdsl


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
    pass_name,
    options: dict[str, Any] | None = None,
    in_mod: SSAValue[transform.OperationType] | None = None,
) -> transform.ApplyRegisteredPassOp:
    """Create an ApplyRegisteredPassOp using the provided pass name and
    pass options."""
    options = options or {}
    in_mod = (
        in_mod or test.TestOp(result_types=(transform.OperationType("builtin.module"),)).results[0]
    )

    lowered_options = {}
    for arg, value in options.items():
        option_key = str(arg).replace("_", "-")
        lowered_options[option_key] = _get_xdsl_attr_from_pyval(value)

    pass_op = transform.ApplyRegisteredPassOp.create(
        properties={
            "pass_name": builtin.StringAttr(pass_name),
            "options": builtin.DictionaryAttr(lowered_options),
        },
        operands=(in_mod,),
        result_types=(transform.OperationType(in_mod.type.operation),),
    )
    return pass_op


class TestCreateSchedule:
    """Unit tests for creating a schedule containing CL args for passes with options."""

    def test_pass_no_options(self):
        """Test that passes with no options are parsed correctly."""
        pass_op = create_apply_registered_pass_op("test-pass")
        schedule = _create_schedule([pass_op])
        assert len(schedule) == 1
        assert schedule[0] == "--test-pass"

    def test_pass_basic_options(self):
        """Test that passes with basic options (int, float, bool, string) are parsed correctly."""
        pass_op = create_apply_registered_pass_op(
            "test-pass",
            options={"int-opt": 1, "float-opt": 1.5, "bool-opt": False, "str-opt": "test_string"},
        )
        schedule = _create_schedule([pass_op])
        assert len(schedule) == 1
        assert (
            schedule[0] == "--test-pass=int-opt=1 float-opt=1.5 bool-opt=false str-opt=test_string"
        )

    def test_pass_array_options(self):
        """Test that passes with array options are parsed correctly."""
        pass_op = create_apply_registered_pass_op("test-pass", options={"list-opt": (1, 2, 3, 4)})
        schedule = _create_schedule([pass_op])
        assert len(schedule) == 1
        assert schedule[0] == "--test-pass=list-opt=1,2,3,4"

    def test_pass_dict_options(self):
        """Test that passes with dict options are parsed correctly."""
        pass_op = create_apply_registered_pass_op(
            "test-pass", options={"dict-opt": {"a": 1, "b": 2, "c": 3, "d": 4}}
        )
        schedule = _create_schedule([pass_op])
        assert len(schedule) == 1
        assert schedule[0] == "--test-pass=dict-opt={a=1 b=2 c=3 d=4}"

    def test_pass_nested_container_options(self):
        """Test that passes with options that are nested containers are parsed correctly."""
        pass_op = create_apply_registered_pass_op(
            "test-pass",
            options={
                "list-opt": ({"a": 1, "b": 2}, {"c": 1.5}, {"d": False, "e": True}),
                "dict-opt": {"f": (1, 2), "g": 1},
            },
        )
        schedule = _create_schedule([pass_op])
        assert len(schedule) == 1
        assert (
            schedule[0]
            == "--test-pass=list-opt={a=1 b=2},{c=1.5},{d=false e=true} dict-opt={f=1,2 g=1}"
        )

    def test_multiple_passes(self):
        """Test that scheduling multiple passes works correctly."""
        pass_op1 = create_apply_registered_pass_op("test-pass1")
        pass_op2 = create_apply_registered_pass_op("test-pass2", options={"int-opt": 1})
        pass_op3 = create_apply_registered_pass_op(
            "test-pass3", options={"list-opt": (False, True, False)}
        )
        schedule = _create_schedule([pass_op1, pass_op2, pass_op3])
        assert len(schedule) == 3
        assert schedule[0] == "--test-pass1"
        assert schedule[1] == "--test-pass2=int-opt=1"
        assert schedule[2] == "--test-pass3=list-opt=false,true,false"


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


class OptionsTestPass(ModulePass):
    """ModulePass for testing pass options."""

    name = "test-options-pass"

    def __init__(self, **options):
        self.options = options

    def apply(self, ctx, op):  # pylint: disable=unused-argument
        print(f"Applying test-options-pass with options {self.options}")
        return op


class TestTransformFunctionsExt:
    """Unit tests for the TransformFunctionsExt interpreter."""

    @pytest.mark.parametrize(
        "pass_options", [{}, {"a": 1, "b": 1.5, "c": (1, 2, 3, 4), "d": "string-input"}]
    )
    def test_xdsl_pass(self, pass_options, capsys):
        """Test that interpreting an xDSL pass works correctly."""
        ctx = Context()
        ctx.load_dialect(builtin.Builtin)
        ctx.load_dialect(transform.Transform)

        fns = TransformFunctionsExt(ctx, passes={"test-options-pass": lambda: OptionsTestPass})
        pass_op = create_apply_registered_pass_op(
            pass_name="test-options-pass", options=pass_options
        )

        mod = builtin.ModuleOp([])
        outs = fns.run_apply_registered_pass_op(None, pass_op, (mod,))
        assert outs.values[0] is mod
        captured = capsys.readouterr()
        assert captured.out.strip() == f"Applying test-options-pass with options {pass_options}"

    @pytest.mark.parametrize(
        "pass_options, cl_options",
        [
            ({}, ""),
            (
                {"a": 1, "b": 1.5, "c": (1, 2, 3, 4), "d": "string-input"},
                "=a=1 b=1.5 c=1,2,3,4 d=string-input",
            ),
        ],
    )
    def test_mlir_pass(self, pass_options, cl_options, mocker):
        """Test that interpreting an MLIR pass works correctly."""
        # The passes dict is empty, so when we're interpreting the pass it will be assumed to
        # be an MLIR pass
        ctx = Context()
        ctx.load_dialect(builtin.Builtin)
        ctx.load_dialect(transform.Transform)

        fns = TransformFunctionsExt(ctx, passes={})
        pass_op = create_apply_registered_pass_op(
            pass_name="test-options-pass", options=pass_options
        )
        captured_cmd = None

        def dummy_subprocess_run(cmd, input=None, **__):
            nonlocal captured_cmd
            captured_cmd = subprocess.list2cmdline(cmd)
            return MagicMock(args=cmd, stdout=input, returncode=0)

        mocker.patch("subprocess.run", side_effect=dummy_subprocess_run)

        mod = builtin.ModuleOp([])
        # This is just a silly step needed because the interpreter assumes that we're transforming
        # a nested module, so `mod` needs to have a parent op to work correctly
        _ = builtin.ModuleOp([mod])
        outs = fns.run_apply_registered_pass_op(None, pass_op, (mod,))

        assert captured_cmd is not None
        # args = list(filter(lambda arg: arg.startswith("--"), captured_cmd.split(" ")))
        # assert f"--test-options-pass{cl_options}" in args
        assert f"--test-options-pass{cl_options}" in captured_cmd


# TODO: Add tests for xdsl->pyval conversion
# Also, maybe move the conversion functions to utils folder


class TestTransformInterpreterPass:
    """Unit tests for the TransformInterpreterPass."""

    def test_interpret_named_sequence(self):
        """Test that a NamedSequenceOp can be interpreted correctly when it contains
        both xDSL and MLIR passes."""

    def test_qjit_with_passes_interpreted_correctly(self):
        """Test that applying the TransformInterpreter to a qjitted qnode's
        module correctly transforms it."""
