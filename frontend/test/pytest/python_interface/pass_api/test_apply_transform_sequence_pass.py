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

# pylint: disable=line-too-long

import subprocess
from typing import Any
from unittest.mock import MagicMock

import pennylane as qp
import pytest
from xdsl.context import Context
from xdsl.dialects import builtin, func, test, transform
from xdsl.ir import Attribute, SSAValue
from xdsl.passes import ModulePass

from catalyst import qjit
from catalyst.python_interface import QuantumParser
from catalyst.python_interface.conversion import parse_generic_to_xdsl_module, xdsl_from_qjit
from catalyst.python_interface.dialects import quantum
from catalyst.python_interface.pass_api.apply_transform_sequence import (
    ApplyTransformSequencePass,
    _create_mlir_cli_schedule,
)
from catalyst.python_interface.transforms import merge_rotations_pass

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
        result_types=(transform.OperationType("builtin.module"),),
    )
    return pass_op


class TestCreateMLIRSchedule:
    """Unit tests for creating a schedule containing CL args for passes with options."""

    def test_pass_no_options(self):
        """Test that passes with no options are parsed correctly."""
        pass_op = create_apply_registered_pass_op("test-pass")
        schedule = _create_mlir_cli_schedule(pass_ops=[pass_op])
        assert len(schedule) == 1
        assert schedule[0] == "--test-pass"

    def test_pass_basic_options(self):
        """Test that passes with basic options (int, float, bool, string) are parsed correctly."""
        pass_op = create_apply_registered_pass_op(
            "test-pass",
            options={
                "int-opt": 1,
                "float-opt": 1.5,
                "bool-opt": False,
                "str-opt": "test_string",
                "str-opt-with-spaces": "foo bar",
            },
        )
        schedule = _create_mlir_cli_schedule(pass_ops=[pass_op])
        assert len(schedule) == 1
        assert schedule[0] == (
            "--test-pass=int-opt=1 float-opt=1.5 bool-opt=false str-opt='test_string' "
            "str-opt-with-spaces='foo bar'"
        )

    def test_pass_array_options(self):
        """Test that passes with array options are parsed correctly."""
        pass_op = create_apply_registered_pass_op("test-pass", options={"list-opt": (1, 2, 3, 4)})
        schedule = _create_mlir_cli_schedule(pass_ops=[pass_op])
        assert len(schedule) == 1
        assert schedule[0] == "--test-pass=list-opt=1,2,3,4"

    def test_pass_dict_options(self):
        """Test that passes with dict options are parsed correctly."""
        pass_op = create_apply_registered_pass_op(
            "test-pass", options={"dict-opt": {"a": 1, "b": 2, "c": 3, "d": 4}}
        )
        schedule = _create_mlir_cli_schedule(pass_ops=[pass_op])
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
        schedule = _create_mlir_cli_schedule(pass_ops=[pass_op])
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
        schedule = _create_mlir_cli_schedule(pass_ops=[pass_op1, pass_op2, pass_op3])
        assert len(schedule) == 3
        assert schedule[0] == "--test-pass1"
        assert schedule[1] == "--test-pass2=int-opt=1"
        assert schedule[2] == "--test-pass3=list-opt=false,true,false"


class OptionsPass(ModulePass):
    """ModulePass for testing pass options."""

    name = "options-pass"

    def __init__(self, **options):
        self.options = options

    def apply(self, ctx, op):  # pylint: disable=unused-argument
        """Apply the pass."""
        print(f"Applying options-pass with options {self.options}")
        return op


class TestApplyTransformSequencePass:
    """Tests for the ApplyTransformSequencePass."""

    def test_multiple_named_sequences(self, run_filecheck):
        """Test that a transforms can be interpreted correctly when there
        are multiple NamedSequences."""

        program_2_sequences = """
            // CHECK-LABEL: workflow
            builtin.module @workflow {
                builtin.module {
                    // CHECK-NOT: module attributes {transform.with_named_sequence}
                    builtin.module attributes {transform.with_named_sequence} {
                        // CHECK-NOT: transform.named_sequence @__transform_0
                        // CHECK-NOT: {{%.+}} = transform.apply_registered_pass "xdsl-cancel-inverses" to {{%.+}}
                        // CHECK-NOT: transform.yield
                        transform.named_sequence @__transform_0(%t0_arg0 : !transform.op<"builtin.module">) {
                            %t0_0 = transform.apply_registered_pass "xdsl-cancel-inverses" to %t0_arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
                            transform.yield
                        }

                        // CHECK-NOT: transform.named_sequence @__transform_1
                        // CHECK-NOT: {{%.+}} = transform.apply_registered_pass "xdsl-merge-rotations" to {{%.+}}
                        // CHECK-NOT: transform.yield
                        transform.named_sequence @__transform_1(%t1_arg0 : !transform.op<"builtin.module">) {
                            %t1_0 = transform.apply_registered_pass "xdsl-merge-rotations" to %t1_arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
                            transform.yield
                        }
                    }

                    // CHECK-LABEL: f
                    func.func public @f(%arg0: tensor<f64>) -> !quantum.bit {
                        // CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
                        // CHECK-NOT: quantum.custom
                        %c_0 = arith.constant 0 : i64
                        %0 = quantum.alloc( 1) : !quantum.reg
                        %1 = quantum.extract %0[%c_0] : !quantum.reg -> !quantum.bit
                        %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
                        %out_qubits_1 = quantum.custom "Hadamard"() %out_qubits : !quantum.bit
                        %extracted_arg0 = tensor.extract %arg0[] : tensor<f64>
                        %out_qubits_2 = quantum.custom "RX"(%extracted_arg0) %out_qubits_1 : !quantum.bit
                        %out_qubits_3 = quantum.custom "RX"(%extracted_arg0) %out_qubits_2 : !quantum.bit
                        return %out_qubits_3 : !quantum.bit
                    }
                }
            }
            """
        pipeline = (ApplyTransformSequencePass(),)
        run_filecheck(program_2_sequences, pipeline)

    def test_interpret_named_sequence(self, mocker, capsys):
        """Test that a NamedSequenceOp can be interpreted correctly when it contains
        both xDSL and MLIR passes."""
        program = """
            builtin.module @workflow {
                builtin.module {
                    builtin.module attributes {transform.with_named_sequence} {
                        transform.named_sequence @__transform_0(%t0_arg0 : !transform.op<"builtin.module">) {
                            %t0_0 = transform.apply_registered_pass "options-pass" to %t0_arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
                            %t0_1 = transform.apply_registered_pass "mlir-pass1" with options = {a = 1 : i64, b = "foo"} to %t0_0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
                            %t0_2 = transform.apply_registered_pass "options-pass" with options = {c = [1 : i64, 2 : i64, 3 : i64], d = false} to %t0_1 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
                            %t0_3 = transform.apply_registered_pass "mlir-pass2" to %t0_2 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
                            transform.yield
                        }
                    }
                }
            }
        """

        pass_options = [{}, {"a": 1, "b": "foo"}, {"c": (1, 2, 3), "d": False}, {}]
        mod = parse_generic_to_xdsl_module(program)

        captured_cmds = []
        num_calls = 0

        def mock_subprocess_run(cmd, **kwargs):
            """Mock implementation of subprocess.run"""
            nonlocal captured_cmds
            nonlocal num_calls
            captured_cmds.append(subprocess.list2cmdline(cmd))
            num_calls += 1
            return MagicMock(args=cmd, stdout=kwargs.get("input", ""), returncode=0)

        mocker.patch("subprocess.run", side_effect=mock_subprocess_run)

        _pass = ApplyTransformSequencePass(
            passes={"options-pass": lambda: OptionsPass}, callback=None
        )
        ctx = Context()
        ctx.load_dialect(builtin.Builtin)
        ctx.load_dialect(transform.Transform)
        _pass.apply(ctx, mod)

        # Assert that xDSL passes were applied correctly
        captured = capsys.readouterr()
        assert captured.out.strip().split("\n") == [
            f"Applying options-pass with options {pass_options[0]}",
            f"Applying options-pass with options {pass_options[2]}",
        ]

        # Assert that MLIR passes were applied correctly
        assert len(captured_cmds) == 2
        assert "--mlir-pass1=a=1 b='foo'" in captured_cmds[0]
        # We check that there is a space after the pass name to check that no options were specified
        assert "--mlir-pass2 " in captured_cmds[1]

    def test_interpret_named_sequence_consecutive_mlir_passes(self, mocker, capsys):
        """Test that a NamedSequenceOp can be interpreted correctly when it contains
        both xDSL and MLIR passes."""
        program = """
            builtin.module @workflow {
                builtin.module {
                    builtin.module attributes {transform.with_named_sequence} {
                        transform.named_sequence @__transform_0(%t0_arg0 : !transform.op<"builtin.module">) {
                            %t0_0 = transform.apply_registered_pass "options-pass" to %t0_arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
                            %t0_1 = transform.apply_registered_pass "mlir-pass1" with options = {a = 1 : i64, b = "foo"} to %t0_0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
                            %t0_2 = transform.apply_registered_pass "mlir-pass2" to %t0_1 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
                            %t0_3 = transform.apply_registered_pass "mlir-pass2" with options = {c = [1 : i64, 2 : i64, 3 : i64], d = false} to %t0_2 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
                            %t0_4 = transform.apply_registered_pass "mlir-pass1" to %t0_3 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
                            transform.yield
                        }
                    }
                }
            }
        """

        pass_options = [{}, {"a": 1, "b": "foo"}, {}, {"c": (1, 2, 3), "d": False}, {}]
        mod = parse_generic_to_xdsl_module(program)

        captured_cmds = []
        num_calls = 0

        def mock_subprocess_run(cmd, **kwargs):
            """Mock implementation of subprocess.run"""
            nonlocal captured_cmds
            nonlocal num_calls
            captured_cmds.append(subprocess.list2cmdline(cmd))
            num_calls += 1
            return MagicMock(args=cmd, stdout=kwargs.get("input", ""), returncode=0)

        mocker.patch("subprocess.run", side_effect=mock_subprocess_run)

        _pass = ApplyTransformSequencePass(
            passes={"options-pass": lambda: OptionsPass}, callback=None
        )
        ctx = Context()
        ctx.load_dialect(builtin.Builtin)
        ctx.load_dialect(transform.Transform)
        _pass.apply(ctx, mod)

        # Assert that xDSL passes were applied correctly
        captured = capsys.readouterr()
        assert captured.out.strip().split("\n") == [
            f"Applying options-pass with options {pass_options[0]}"
        ]

        # Assert that MLIR passes were applied correctly
        assert len(captured_cmds) == 1
        assert (
            '"--mlir-pass1=a=1 b=\'foo\'" --mlir-pass2 "--mlir-pass2=c=1,2,3 d=false" --mlir-pass1'
            in captured_cmds[0]
        )

    def test_interpret_named_sequence_no_passes(self):
        """Test that a NamedSequenceOp can be interpreted correctly when there are no passes."""
        program = """
            builtin.module @workflow {
                builtin.module {
                    builtin.module attributes {transform.with_named_sequence} {
                        transform.named_sequence @__transform_0(%t0_arg0 : !transform.op<"builtin.module">) {
                            transform.yield
                        }
                    }
                }
            }
        """
        mod = parse_generic_to_xdsl_module(program)
        _pass = ApplyTransformSequencePass()

        ctx = Context()
        ctx.load_dialect(builtin.Builtin)
        ctx.load_dialect(transform.Transform)
        _pass.apply(ctx, mod)

        expected_program = """
            builtin.module @workflow {
                builtin.module {
                }
            }
        """
        expected_mod = parse_generic_to_xdsl_module(expected_program)
        assert mod.is_structurally_equivalent(expected_mod)

    def test_callback_count_with_passes(self):
        """Test that the apply function calls the callback the correct number of times."""

        num_calls = 0

        def callback(
            previous_pass, module, next_pass, pass_level=None
        ):  # pylint: disable=unused-argument
            """Mock implementation of the callback function"""
            nonlocal num_calls
            num_calls += 1

        program = """
            builtin.module @workflow {
                builtin.module {
                    builtin.module attributes {transform.with_named_sequence} {
                        transform.named_sequence @__transform_0(%t0_arg0 : !transform.op<"builtin.module">) {
                            %t0_0 = transform.apply_registered_pass "options-pass" to %t0_arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
                            transform.yield
                        }
                    }
                }
            }
        """
        mod = parse_generic_to_xdsl_module(program)
        _pass = ApplyTransformSequencePass(
            passes={"options-pass": lambda: OptionsPass}, callback=callback
        )

        ctx = Context()
        ctx.load_dialect(builtin.Builtin)
        ctx.load_dialect(transform.Transform)
        _pass.apply(ctx, mod)

        # Should be 2 calls, 1 for init pass, 1 for "options-pass"
        assert num_calls == 2

    def test_callback_count_no_passes(self):
        """Test that the apply function calls the callback the correct number of times."""

        num_calls = 0

        def callback(
            previous_pass, module, next_pass, pass_level=None
        ):  # pylint: disable=unused-argument
            """Mock implementation of the callback function"""
            nonlocal num_calls
            num_calls += 1

        program = """
            builtin.module @workflow {
                builtin.module {
                    builtin.module attributes {transform.with_named_sequence} {
                        transform.named_sequence @__transform_0(%t0_arg0 : !transform.op<"builtin.module">) {
                            transform.yield
                        }
                    }
                }
            }
        """
        mod = parse_generic_to_xdsl_module(program)
        _pass = ApplyTransformSequencePass(
            passes={"options-pass": lambda: OptionsPass}, callback=callback
        )

        ctx = Context()
        ctx.load_dialect(builtin.Builtin)
        ctx.load_dialect(transform.Transform)
        _pass.apply(ctx, mod)

        # Should be 1 call for init pass
        assert num_calls == 1

    @pytest.mark.usefixtures("use_both_frontend")
    def test_qjit_with_passes_interpreted_correctly(self):
        """Test that applying the ApplyTransformSequencePass to a qjitted qnode's
        module correctly transforms it."""

        dev = qp.device("null.qubit", wires=4)

        @xdsl_from_qjit
        @qjit
        # merge_rotations_pass dispatches to an xDSL pass
        @merge_rotations_pass
        # qp.transforms.cancel_inverses dispatches to an MLIR pass
        @qp.transforms.cancel_inverses
        @qp.qnode(dev)
        def circuit():
            qp.RX(1.5, 0)
            qp.RX(1.5, 0)
            qp.X(1)
            qp.X(1)
            return qp.state()

        mod = circuit()
        _pass = ApplyTransformSequencePass()
        ctx = Context(allow_unregistered=True)
        _ = QuantumParser(ctx, "")  # This loads necessary dialects into the context
        # The Xs will be cancelled, and the RXs will be merged, so there should only be
        # one RX remaining. We need to find the qnode module again because the original
        # can be replaced by the pass
        _pass.apply(ctx, mod)

        qnode_mod = [op for op in mod.body.ops if isinstance(op, builtin.ModuleOp)][0]
        qnode_fn = [op for op in qnode_mod.body.ops if isinstance(op, func.FuncOp)][0]
        custom_ops = [op for op in qnode_fn.body.ops if isinstance(op, quantum.CustomOp)]
        assert len(custom_ops) == 1
        assert custom_ops[0].gate_name.data == "RX"


if __name__ == "__main__":
    pytest.main(["-x", __file__])
