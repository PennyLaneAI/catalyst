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
from io import StringIO
from unittest.mock import MagicMock

import pennylane as qp
import pytest
from xdsl.context import Context
from xdsl.dialects import builtin, func, transform
from xdsl.passes import ModulePass
from xdsl.printer import Printer

from catalyst import qjit
from catalyst.python_interface import QuantumParser
from catalyst.python_interface.conversion import parse_generic_to_xdsl_module, xdsl_from_qjit
from catalyst.python_interface.dialects import quantum
from catalyst.python_interface.pass_api.apply_transform_sequence import (
    ApplyTransformSequencePass,
)
from catalyst.python_interface.transforms import merge_rotations_pass
from catalyst.python_interface.utils import get_pyval_from_xdsl_attr

pytestmark = pytest.mark.xdsl


def _mock_and_get_active_passes(input: str, active_mlir_passes: list) -> str:
    """Collect non-empty apply_registered_pass ops from ``input``.

    Mimics ``--apply-transform-sequence`` by detaching transform modules
    afterwards, and returns the remaining IR for use as mock stdout.
    """
    mod = parse_generic_to_xdsl_module(input)
    for op in mod.walk():
        if not isinstance(op, transform.ApplyRegisteredPassOp):
            continue
        if op.pass_name.data == "empty":
            continue
        options = get_pyval_from_xdsl_attr(op.options)
        active_mlir_passes.append((op.pass_name.data, options))

    # Find and remove all transform.with_named_sequence ops.
    to_detach = [
        op
        for op in mod.walk()
        if isinstance(op, builtin.ModuleOp)
        and op is not mod
        and op.get_attr_or_prop("transform.with_named_sequence") is not None
    ]
    for op in to_detach:
        op.detach()

    buffer = StringIO()
    Printer(stream=buffer, print_generic_format=True).print_op(mod)
    return buffer.getvalue()


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
        active_mlir_passes = []
        num_calls = 0

        def mock_subprocess_run(cmd, **kwargs):
            """Mock implementation of subprocess.run"""
            nonlocal captured_cmds
            nonlocal num_calls
            nonlocal active_mlir_passes
            captured_cmds.append(subprocess.list2cmdline(cmd))
            num_calls += 1
            transformed_ir = _mock_and_get_active_passes(
                kwargs.get("input", ""), active_mlir_passes
            )
            return MagicMock(args=cmd, stdout=transformed_ir, returncode=0)

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
        assert all("--apply-transform-sequence" in cmd for cmd in captured_cmds)
        assert active_mlir_passes == [
            ("mlir-pass1", {"a": 1, "b": "foo"}),
            ("mlir-pass2", {}),
        ]

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
        active_mlir_passes = []
        num_calls = 0

        def mock_subprocess_run(cmd, **kwargs):
            """Mock implementation of subprocess.run"""
            nonlocal captured_cmds
            nonlocal num_calls
            nonlocal active_mlir_passes
            captured_cmds.append(subprocess.list2cmdline(cmd))
            num_calls += 1
            transformed_ir = _mock_and_get_active_passes(
                kwargs.get("input", ""), active_mlir_passes
            )
            return MagicMock(args=cmd, stdout=transformed_ir, returncode=0)

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
        assert "--apply-transform-sequence" in captured_cmds[0]
        assert active_mlir_passes == [
            ("mlir-pass1", {"a": 1, "b": "foo"}),
            ("mlir-pass2", {}),
            ("mlir-pass2", {"c": (1, 2, 3), "d": False}),
            ("mlir-pass1", {}),
        ]

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

    def test_qjit_with_passes_interpreted_correctly(self, capture_mode):
        """Test that applying the ApplyTransformSequencePass to a qjitted qnode's
        module correctly transforms it."""

        dev = qp.device("null.qubit", wires=4)

        @xdsl_from_qjit
        @qjit(capture=capture_mode)
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

    def test_qjit_with_passes_callback_names(self, capture_mode):
        """Test that applying the ApplyTransformSequencePass to a qjitted qnode's
        module correctly transforms it through callback path

        Also ensure the callback is called as required.
        """

        seen_pass_names = []

        def callback(
            previous_pass, module, next_pass, pass_level=None
        ):  # pylint: disable=unused-argument
            """Record pass names reported to the callback."""
            pass_obj = previous_pass if previous_pass is not None else next_pass
            if pass_obj is None:
                return
            name = pass_obj.name if hasattr(pass_obj, "name") else pass_obj
            if name not in seen_pass_names:
                seen_pass_names.append(name)

        dev = qp.device("null.qubit", wires=4)

        @xdsl_from_qjit
        @qjit(capture=capture_mode)
        @merge_rotations_pass
        @qp.transforms.cancel_inverses
        @qp.qnode(dev)
        def circuit():
            qp.RX(1.5, 0)
            qp.RX(1.5, 0)
            qp.X(1)
            qp.X(1)
            return qp.state()

        mod = circuit()
        _pass = ApplyTransformSequencePass(callback=callback)
        ctx = Context(allow_unregistered=True)
        _ = QuantumParser(ctx, "")
        _pass.apply(ctx, mod)

        assert "cancel-inverses" in seen_pass_names
        assert "xdsl-merge-rotations" in seen_pass_names

        if capture_mode:
            assert "graph-decomposition" in seen_pass_names

        qnode_mod = [op for op in mod.body.ops if isinstance(op, builtin.ModuleOp)][0]
        qnode_fn = [op for op in qnode_mod.body.ops if isinstance(op, func.FuncOp)][0]
        custom_ops = [op for op in qnode_fn.body.ops if isinstance(op, quantum.CustomOp)]
        assert len(custom_ops) == 1
        assert custom_ops[0].gate_name.data == "RX"


if __name__ == "__main__":
    pytest.main(["-x", __file__])
