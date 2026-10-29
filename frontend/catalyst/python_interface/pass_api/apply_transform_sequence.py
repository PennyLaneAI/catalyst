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
from itertools import groupby
from typing import Callable

from xdsl.context import Context
from xdsl.dialects import builtin, transform
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PassPipeline
from xdsl.pattern_rewriter import (
    InsertPoint,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
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


class ApplyTransformSequenceBase(RewritePattern):
    """Shared rewrite driver for applying transform sequences.

    Manages running xDSL ``PassPipeline`` or MLIR ``--apply-transform-sequence``.
    Subclasses must implement grouping and how each group is applied (e.g. with or without
    callbacks).
    """

    def __init__(
        self,
        ctx: Context,
        passes: dict[str, Callable[[], type[ModulePass]]],
    ):
        self.ctx = ctx
        self.passes = passes

    def create_groups(
        self, pass_ops: Sequence[transform.ApplyRegisteredPassOp]
    ) -> Sequence[tuple[bool, tuple[transform.ApplyRegisteredPassOp, ...]]]:
        """Partition ``pass_ops`` into ``(is_xdsl, group)`` runs."""
        raise NotImplementedError  # pragma: no cover

    def on_empty(self, payload: builtin.ModuleOp) -> None:
        """Called when the transform sequence has no registered passes."""
        raise NotImplementedError  # pragma: no cover

    def apply_xdsl_pipeline(
        self,
        payload: builtin.ModuleOp,
        group: Sequence[transform.ApplyRegisteredPassOp],
    ) -> builtin.ModuleOp:
        """Apply an xDSL pass group to ``payload``; return the (possibly same) module."""
        raise NotImplementedError  # pragma: no cover

    def apply_mlir_pipeline(
        self,
        payload: builtin.ModuleOp,
        rewriter: PatternRewriter,
        transformer: builtin.ModuleOp,
        group: Sequence[transform.ApplyRegisteredPassOp],
        insertion_point: InsertPoint,
    ) -> tuple[builtin.ModuleOp, InsertPoint]:
        """Apply an MLIR pass group; return updated ``(payload, insertion_point)``."""
        raise NotImplementedError  # pragma: no cover

    @op_type_rewrite_pattern
    def match_and_rewrite(self, transformer: builtin.ModuleOp, rewriter: PatternRewriter):
        """Rewrite modules containing transform.named_sequences."""
        payload: builtin.ModuleOp = transformer.parent_op()
        # Detach (do not erase) so we can clone the transformer for MLIR groups.
        if transformer.next_op is not None:
            insertion_point = InsertPoint.before(transformer.next_op)
        else:
            insertion_point = InsertPoint.at_end(payload.body.block)

        transformer.detach()

        pass_ops = []
        for ns in transformer.ops:
            assert isinstance(ns, transform.NamedSequenceOp)
            pass_ops.extend(
                op for op in ns.body.ops if isinstance(op, transform.ApplyRegisteredPassOp)
            )

        if len(pass_ops) == 0:
            self.on_empty(payload)
            return

        for is_xdsl_group, group in self.create_groups(pass_ops):
            if is_xdsl_group:
                payload = self.apply_xdsl_pipeline(payload, group)
            else:
                payload, insertion_point = self.apply_mlir_pipeline(
                    payload, rewriter, transformer, group, insertion_point
                )

    def _is_xdsl_pass(self, op: transform.ApplyRegisteredPassOp) -> bool:
        """Check whether a registered pass op corresponds to an xDSL or MLIR pass."""
        # self.passes maps pass names to xDSL passes. If the pass name is in self.passes, we know
        # that it is an xDSL pass. Otherwise, we assume that the name corresponds to an MLIR pass.
        return op.pass_name.data in self.passes

    @staticmethod
    def _clone_transformer_for_mlir_group(
        transformer: builtin.ModuleOp,
        group: Sequence[transform.ApplyRegisteredPassOp],
    ) -> builtin.ModuleOp:
        """Clone ``transformer``, rewriting passes outside ``group`` to no-ops."""
        group_set = set(group)
        clone = transformer.clone()

        for orig_ns, clone_ns in zip(transformer.ops, clone.ops, strict=True):

            for orig_op, clone_op in zip(orig_ns.body.ops, clone_ns.body.ops, strict=True):
                if orig_op not in group_set and isinstance(
                    orig_op, transform.ApplyRegisteredPassOp
                ):
                    clone_op.pass_name = builtin.StringAttr("empty")
                    clone_op.options = builtin.DictionaryAttr({})

        return clone

    def _run_xdsl_pipeline(
        self, payload: builtin.ModuleOp, pass_ops: Sequence[transform.ApplyRegisteredPassOp]
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

    def _run_mlir_pipeline(
        self,
        payload: builtin.ModuleOp,
        rewriter: PatternRewriter,
    ) -> builtin.ModuleOp:
        r"""Apply MLIR passes embedded in ``transformer_clone`` via apply-transform-sequence."""
        buffer = StringIO()
        Printer(stream=buffer, print_generic_format=True).print_op(payload)

        modified = _quantum_opt(
            "--apply-transform-sequence", "-mlir-print-op-generic", stdin=buffer.getvalue()
        )

        data = Parser(self.ctx, modified).parse_module()
        rewriter.replace_op(payload, data)
        return data


class ApplyTransformSequenceNoCallbackPattern(ApplyTransformSequenceBase):
    """RewritePattern for applying transform sequences when no callback is provided.

    Batches consecutive xDSL / MLIR passes and applies them without per-pass hooks.
    """

    def create_groups(
        self, pass_ops: Sequence[transform.ApplyRegisteredPassOp]
    ) -> Sequence[tuple[bool, tuple[transform.ApplyRegisteredPassOp, ...]]]:
        """Group consecutive xDSL vs MLIR registered passes."""
        return [
            (is_xdsl, tuple(group)) for is_xdsl, group in groupby(pass_ops, key=self._is_xdsl_pass)
        ]

    def on_empty(self, payload: builtin.ModuleOp) -> None:
        """No-op when there are no registered passes."""
        return

    def apply_xdsl_pipeline(
        self,
        payload: builtin.ModuleOp,
        group: Sequence[transform.ApplyRegisteredPassOp],
    ) -> builtin.ModuleOp:
        """Apply ``group`` in-process via an xDSL ``PassPipeline``."""
        self._run_xdsl_pipeline(payload, group)
        return payload

    def apply_mlir_pipeline(
        self,
        payload: builtin.ModuleOp,
        rewriter: PatternRewriter,
        transformer: builtin.ModuleOp,
        group: Sequence[transform.ApplyRegisteredPassOp],
        insertion_point: InsertPoint,
    ) -> tuple[builtin.ModuleOp, InsertPoint]:
        """Clone the transform module for ``group`` and run ``--apply-transform-sequence``."""
        clone = self._clone_transformer_for_mlir_group(transformer, group)
        rewriter.insert_op(clone, insertion_point)
        payload = self._run_mlir_pipeline(payload, rewriter)
        insertion_point = InsertPoint.at_start(payload.body.block)
        return payload, insertion_point


class ApplyTransformSequencePattern(ApplyTransformSequenceBase):
    """RewritePattern for applying transform sequences when a callback is provided.

    Applies one registered pass at a time so the callback can observe the module
    before/after each pass. MLIR passes use the same detach/clone/
    ``--apply-transform-sequence`` path as the no-callback pattern.
    """

    def __init__(
        self,
        ctx: Context,
        passes: dict[str, Callable[[], type[ModulePass]]],
        callback: Callable,
    ):
        super().__init__(ctx=ctx, passes=passes)
        self.callback = callback
        self.pass_level = 0

    def create_groups(
        self, pass_ops: Sequence[transform.ApplyRegisteredPassOp]
    ) -> Sequence[tuple[bool, tuple[transform.ApplyRegisteredPassOp, ...]]]:
        """One registered pass per group so callbacks fire between each pass."""
        return [(self._is_xdsl_pass(op), (op,)) for op in pass_ops]

    def on_empty(self, payload: builtin.ModuleOp) -> None:
        """Fire the initial callback when there are no registered passes."""
        self._pre_pass_callback(None, payload)

    def apply_xdsl_pipeline(
        self,
        payload: builtin.ModuleOp,
        group: Sequence[transform.ApplyRegisteredPassOp],
    ) -> builtin.ModuleOp:
        """Apply ``group`` via ``_run_xdsl_pipeline`` with pre/post callbacks."""
        # Groups are size-1; pass name is enough for the callback (same as MLIR).
        pass_name = group[0].pass_name.data
        self._pre_pass_callback(pass_name, payload)
        self._run_xdsl_pipeline(payload, group)
        self._post_pass_callback(pass_name, payload)
        return payload

    def apply_mlir_pipeline(
        self,
        payload: builtin.ModuleOp,
        rewriter: PatternRewriter,
        transformer: builtin.ModuleOp,
        group: Sequence[transform.ApplyRegisteredPassOp],
        insertion_point: InsertPoint,
    ) -> tuple[builtin.ModuleOp, InsertPoint]:
        """Apply an MLIR pass group with pre/post callbacks around transform-sequence."""
        # Groups are size-1; use the pass name for the callback (no ModulePass instance).
        pass_name = group[0].pass_name.data
        self._pre_pass_callback(pass_name, payload)

        clone = self._clone_transformer_for_mlir_group(transformer, group)
        rewriter.insert_op(clone, insertion_point)
        payload = self._run_mlir_pipeline(payload, rewriter)
        insertion_point = InsertPoint.at_start(payload.body.block)

        self._post_pass_callback(pass_name, payload)
        return payload, insertion_point

    def _pre_pass_callback(self, compilation_pass, module):
        """Callback wrapper to run the callback function before a pass."""
        if self.pass_level == 0:
            # Since this is the first pass, there is no previous pass
            self.callback(None, module, compilation_pass, pass_level=0)

    def _post_pass_callback(self, compilation_pass, module):
        """Increment level and run callback if defined."""
        self.pass_level += 1
        self.callback(compilation_pass, module, None, pass_level=self.pass_level)
