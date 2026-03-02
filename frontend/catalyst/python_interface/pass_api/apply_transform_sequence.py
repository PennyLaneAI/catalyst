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

import io
from dataclasses import dataclass

from pennylane.typing import Callable
from xdsl.context import Context
from xdsl.dialects import builtin, transform
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PassPipeline
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.printer import Printer
from xdsl.rewriter import InsertPoint

from catalyst.compiler import _quantum_opt
from catalyst.python_interface.utils import get_pyval_from_xdsl_attr

from .transform_interpreter import TransformInterpreterPass, _create_schedule

available_passes = {}


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
class ApplyTransformSequence(ModulePass):
    """
    Looks for nested modules. Nested modules in this context are guaranteed to correspond
    to qnodes. These modules are already annotated with which passes are to be executed.
    The pass ApplyTransformSequence will run passes annotated in the qnode modules.

    At the end, we delete the list of passes as they have already been applied.
    """

    name = "apply-transform-sequence"
    callback: Callable[[ModulePass, builtin.ModuleOp, ModulePass], None] | None = None

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """Applies the transformation"""
        nested_modules = []
        for _op in op.ops:
            if isinstance(_op, builtin.ModuleOp):
                nested_modules.append(_op)

        for mod in nested_modules:
            pattern = ApplyTransformSequencePattern(ctx, available_passes, self.callback)
            transformer = next(_op for _op in mod.ops if isinstance(_op, builtin.ModuleOp))
            rewriter = PatternRewriter(transformer)
            pattern.match_and_rewrite(transformer, rewriter)


class ApplyTransformSequencePattern(RewritePattern):
    """RewritePattern for applying transform sequences."""

    def __init__(
        self,
        ctx: Context,
        passes: dict[str, Callable[[], type[ModulePass]]],
        callback: Callable | None = None,
    ):
        self.ctx = ctx
        self.passes = passes
        self.callback = callback
        self.pass_level = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, transformer: builtin.ModuleOp, rewriter: PatternRewriter):
        """Rewrite modules containing transform.named_sequences."""
        if not transformer.get_attr_or_prop("transform.with_named_sequence"):
            return

        payload: builtin.ModuleOp = transformer.parent_op()
        cur_payload = payload
        rewriter.erase_op(transformer)

        for i, ns in enumerate(transformer.ops):
            if len(ns.body.ops) == 1:
                if i == 0:
                    self._pre_pass_callback(None, cur_payload)
                    self.pass_level += 1
                continue

            for pass_op in ns.body.walk():
                if isinstance(pass_op, transform.ApplyRegisteredPassOp):
                    next_payload = self.interpret_apply_registered_pass_op(
                        pass_op, cur_payload, rewriter
                    )

                    if next_payload != cur_payload:
                        rewriter.replace_op(cur_payload, next_payload)
                    cur_payload = next_payload

    def _pre_pass_callback(self, compilation_pass, module):
        """Callback wrapper to run the callback function before a pass."""
        if not self.callback:
            return
        if self.pass_level == 0:
            # Since this is the first pass, there is no previous pass
            self.callback(None, module, compilation_pass, pass_level=0)

    def _post_pass_callback(self, compilation_pass, module):
        """Increment level and run callback if defined."""
        if not self.callback:
            return
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
            pipeline = PassPipeline((pass_instance,))
            self._pre_pass_callback(pass_instance, module)
            pipeline.apply(self.ctx, module)
            self._post_pass_callback(pass_instance, module)
            return module

        # ---- Catalyst path ----
        buffer = io.StringIO()
        Printer(stream=buffer, print_generic_format=True).print_op(module)
        schedule = _create_schedule([op])
        self._pre_pass_callback(pass_name, module)
        modified = _quantum_opt(*schedule, "-mlir-print-op-generic", stdin=buffer.getvalue())

        data = Parser(self.ctx, modified).parse_module()
        self._post_pass_callback(pass_name, data)
        return data
