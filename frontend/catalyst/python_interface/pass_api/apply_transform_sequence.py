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
"""This file contains the pass that applies all passes present in the program representation."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from io import StringIO
from typing import Any, Callable

from xdsl.context import Context
from xdsl.dialects import builtin, transform
from xdsl.ir import Attribute
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PassPipeline
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern
from xdsl.printer import Printer

from catalyst.compiler import _quantum_opt
from catalyst.python_interface.utils import get_pyval_from_xdsl_attr

available_passes: dict[str, Callable[[], type[ModulePass]]] = {}


def register_pass(name, _callable):
    """Registers the passes available in the dictionary"""
    available_passes[name] = _callable  # pragma: no cover


def is_xdsl_pass(pass_name: str) -> bool:
    """Check if a pass name corresponds to an xDSL implemented pass.

    This function checks if the pass is registered in PennyLane's unified compiler
    pass registry, which dynamically tracks all available xDSL passes.

    Args:
        pass_name (str): Name of the pass to check

    Returns:
        bool: True if this is an xDSL compiler pass
    """
    # catalyst.python_interface.xdsl_universe collects all transforms in
    # catalyst.python_interface.transforms, so importing from that file
    # updates the global xDSL transforms registry.
    # pylint: disable=import-outside-toplevel
    from catalyst.python_interface.xdsl_universe import CATALYST_XDSL_UNIVERSE as _

    return pass_name in available_passes


@dataclass(frozen=True)
class ApplyTransformSequencePass(ModulePass):
    """
    Looks for nested modules. Nested modules in this context are guaranteed to correspond
    to qnodes. These modules are already annotated with which passes are to be executed.
    The pass ApplyTransformSequencePass will run passes annotated in the qnode modules.

    At the end, we delete the list of passes as they have already been applied.
    """

    name = "apply-transform-sequence"

    passes: dict[str, Callable[[], type[ModulePass]]] = field(
        default_factory=dict[str, Callable[[], type[ModulePass]]]
    )

    callback: (
        Callable[[ModulePass | None, builtin.ModuleOp, ModulePass | None, int], None] | None
    ) = None

    def __post_init__(self):
        """Update passes to include global pass registry."""
        self.passes.update(available_passes)

    @staticmethod
    def find_transform_entry_point(mod: builtin.ModuleOp) -> builtin.ModuleOp | None:
        """Find the transform entry point inside a module."""
        for op in mod.body.walk():
            if op.get_attr_or_prop("transform.with_named_sequence") is not None:
                return op

        return None

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """Applies the transformation"""
        nested_modules = []
        for _op in op.ops:
            if isinstance(_op, builtin.ModuleOp):
                nested_modules.append(_op)

        pattern_cls = (
            partial(ApplyTransformSequencePattern, callback=self.callback)
            if self.callback
            else ApplyTransformSequenceNoCallbackPattern
        )

        for mod in nested_modules:
            transformer = self.find_transform_entry_point(mod)
            if transformer is None:
                continue  # pragma: no cover

            # Need to create a new pattern for each nested module to properly handle callbacks
            pattern = pattern_cls(ctx=ctx, passes=self.passes)
            rewriter = PatternRewriter(transformer)
            pattern.match_and_rewrite(transformer, rewriter)


class ApplyTransformSequencePattern(RewritePattern):
    """RewritePattern for applying transform sequences when a callback is provided."""

    def __init__(
        self,
        ctx: Context,
        passes: dict[str, Callable[[], type[ModulePass]]],
        callback: Callable,
    ):
        self.ctx = ctx
        self.passes = passes
        self.callback = callback
        self.pass_level = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, transformer: builtin.ModuleOp, rewriter: PatternRewriter):
        """Rewrite modules containing transform.named_sequences."""
        payload: builtin.ModuleOp = transformer.parent_op()
        rewriter.erase_op(transformer)

        pass_ops = []
        for ns in transformer.ops:
            assert isinstance(ns, transform.NamedSequenceOp)
            pass_ops.extend(
                op for op in ns.body.ops if isinstance(op, transform.ApplyRegisteredPassOp)
            )

        if len(pass_ops) == 0:
            self._pre_pass_callback(None, payload)
            return

        for pass_op in pass_ops:
            payload = self.interpret_apply_registered_pass_op(pass_op, payload, rewriter)

    def _pre_pass_callback(self, compilation_pass, module):
        """Callback wrapper to run the callback function before a pass."""
        if self.pass_level == 0:
            # Since this is the first pass, there is no previous pass
            self.callback(None, module, compilation_pass, pass_level=0)

    def _post_pass_callback(self, compilation_pass, module):
        """Increment level and run callback if defined."""
        self.pass_level += 1
        self.callback(compilation_pass, module, None, pass_level=self.pass_level)

    def interpret_apply_registered_pass_op(
        self,
        op: transform.ApplyRegisteredPassOp,
        module: builtin.ModuleOp,
        rewriter: PatternRewriter,
    ):
        """Interpret the transform.apply_registered_pass."""

        pass_name = op.pass_name.data

        # ---- xDSL path ----
        if pass_name in self.passes:
            pass_class = self.passes[pass_name]()
            options = {
                k.replace("-", "_"): v for k, v in get_pyval_from_xdsl_attr(op.options).items()
            }
            pass_instance = pass_class(**options)
            self._pre_pass_callback(pass_instance, module)
            pass_instance.apply(self.ctx, module)
            self._post_pass_callback(pass_instance, module)
            return module

        # ---- Catalyst path ----
        buffer = StringIO()
        Printer(stream=buffer, print_generic_format=True).print_op(module)
        schedule = _create_mlir_cli_schedule([op])
        self._pre_pass_callback(pass_name, module)
        modified = _quantum_opt(*schedule, "-mlir-print-op-generic", stdin=buffer.getvalue())

        data = Parser(self.ctx, modified).parse_module()
        rewriter.replace_op(module, data)
        self._post_pass_callback(pass_name, data)
        return data


class ApplyTransformSequenceNoCallbackPattern(RewritePattern):
    """RewritePattern for applying transform sequences when no callback is provided.
    This is functionally the same as the ``ApplyTransformSequenceWithCallbackPattern``,
    but uses a more optimized pathway, which is enabled by the lack of a callback."""

    def __init__(
        self,
        ctx: Context,
        passes: dict[str, Callable[[], type[ModulePass]]],
    ):
        self.ctx = ctx
        self.passes = passes
        self.pass_level = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, transformer: builtin.ModuleOp, rewriter: PatternRewriter):
        """Rewrite modules containing transform.named_sequences."""
        payload: builtin.ModuleOp = transformer.parent_op()
        rewriter.erase_op(transformer)

        pass_ops = []
        for ns in transformer.ops:
            assert isinstance(ns, transform.NamedSequenceOp)
            pass_ops.extend(
                op for op in ns.body.ops if isinstance(op, transform.ApplyRegisteredPassOp)
            )

        if len(pass_ops) == 0:
            return

        schedule = self._cluster_passes(pass_ops)
        for subschedule in schedule:
            if subschedule[0].pass_name.data in self.passes:
                self._apply_xdsl_pipeline(payload, subschedule)
            else:
                payload = self._apply_mlir_pipeline(payload, subschedule, rewriter)

    def _cluster_passes(
        self, pass_ops: list[transform.ApplyRegisteredPassOp]
    ) -> list[list[transform.ApplyRegisteredPassOp]]:
        """Create a schedule that clusters consecutive xDSL/MLIR passes together."""
        if not pass_ops:  # pragma: no cover
            return []

        schedule = []
        cur_schedule = [pass_ops[0]]
        cur_state = pass_ops[0].pass_name.data in self.passes

        for op in pass_ops[1:]:
            new_state = op.pass_name.data in self.passes
            if new_state == cur_state:
                cur_schedule.append(op)
            else:
                schedule.append(cur_schedule)
                cur_schedule = [op]
                cur_state = new_state

        schedule.append(cur_schedule)
        return schedule

    def _apply_xdsl_pipeline(
        self, payload: builtin.ModuleOp, pass_ops: list[transform.ApplyRegisteredPassOp]
    ) -> None:
        r"""Interpret a sequence of ``ApplyRegisteredPassOp``\ s corresponding to xDSL passes."""
        pipeline = []
        for op in pass_ops:
            pass_class = self.passes[op.pass_name.data]()
            options = {
                k.replace("-", "_"): v for k, v in get_pyval_from_xdsl_attr(op.options).items()
            }
            pipeline.append(pass_class(**options))

        pipeline = PassPipeline(pipeline)
        pipeline.apply(self.ctx, payload)

    def _apply_mlir_pipeline(
        self,
        payload: builtin.ModuleOp,
        pass_ops: list[transform.ApplyRegisteredPassOp],
        rewriter: PatternRewriter,
    ) -> builtin.ModuleOp:
        r"""Interpret a sequence of ``ApplyRegisteredPassOp``\ s corresponding to MLIR passes."""
        buffer = StringIO()
        Printer(stream=buffer, print_generic_format=True).print_op(payload)

        pipeline = _create_mlir_cli_schedule(pass_ops)
        modified = _quantum_opt(*pipeline, "-mlir-print-op-generic", stdin=buffer.getvalue())

        data = Parser(self.ctx, modified).parse_module()
        rewriter.replace_op(payload, data)
        return data


def _create_mlir_cli_schedule(pass_ops: Sequence[transform.ApplyRegisteredPassOp]) -> list[str]:
    """Create a pass schedule for applying MLIR pass via CLI flags.

    For a pass with options, the corresponding CLI flag will be the following:

    .. code-block::

        "--my-pass=arg1=val1 arg2=val2 ..."

    Args:
        pass_ops (Sequence[xdsl.dialects.transform.ApplyRegisteredPassOp]): The
            passes to schedule

    Returns:
        list[str]: A list containing strings that correspond to the CLI flags
        for the specified passes
    """
    schedule: list[str] = []

    for op in pass_ops:
        pass_name = op.pass_name.data
        pass_options = op.options.data

        if not pass_options:
            schedule.append(f"--{pass_name}")
            continue

        cli_options = []
        for opt, val in pass_options.items():
            cli_options.append(f"{opt}={_get_cli_option_from_attr(val)}")
        cli_options = " ".join(cli_options)

        cli_pass = f"--{pass_name}={cli_options}"
        schedule.append(cli_pass)

    return schedule


def _get_cli_option_from_attr(attr: Attribute) -> Any:
    """Convert an xDSL attribute corresponding to a pass option value into a valid
    CLI option value."""
    cli_val = None

    match attr:
        case builtin.IntegerAttr():
            # Booleans are represented as integer attributes with a bitwidth of 1
            if isinstance(attr.type, builtin.IntegerType) and attr.type.width.data == 1:
                # Python boolean's repr is capitalized so we use strings
                cli_val = "true" if attr.value.data else "false"
            else:
                cli_val = attr.value.data

        case builtin.FloatAttr():
            cli_val = attr.value.data

        case builtin.StringAttr():
            cli_val = f"'{attr.data}'"

        case builtin.ArrayAttr():
            cli_val = ",".join([str(_get_cli_option_from_attr(attr)) for attr in attr.data])

        case builtin.DictionaryAttr():
            mapping = []
            for k, v in attr.data.items():
                mapping.append(f"{k}={_get_cli_option_from_attr(v)}")
            cli_val = f"{{{' '.join(mapping)}}}"

        case _:  # pragma: no cover
            raise ValueError(f"Unsupported option type {attr}.")

    return cli_val
