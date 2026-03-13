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

"""Quantum to QEC Logical dialect conversion.

This module contains the implementation of the xDSL quantum-to-qecl dialect-conversion pass.
"""
import math
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir import BlockOps
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from catalyst.python_interface.dialects import qecl, quantum


@dataclass(frozen=True)
class AllocConversion(RewritePattern):
    """Converts `quantum.alloc` ops to equivalent `qecl.alloc` ops."""

    k: int

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: quantum.AllocOp, rewriter: PatternRewriter):
        """Rewrite pattern for `quantum.alloc` ops."""
        nqubits_attr = op.properties.get("nqubits_attr")
        if nqubits_attr is None:
            raise NotImplementedError(
                f"Failed to convert op '{op}': conversion pattern for '{op.name}' does not support "
                f"a dynamic number of qubits"
            )

        assert isinstance(
            nqubits_attr, IntegerAttr
        ), f"Expected 'nqubits_attr' to be an IntegerAttr, but got {type(nqubits_attr)}"

        nqubits = nqubits_attr.value.data

        hyper_reg_width = math.ceil(nqubits / self.k)
        rewriter.replace_op(
            op,
            qecl.AllocOp(qecl.LogicalHyperRegisterType(width=hyper_reg_width, k=self.k)),
        )

        BlockOps


@dataclass(frozen=True)
class ConvertQuantumToQecLogicalPass(ModulePass):
    """
    Convert quantum instructions to QEC logical instructions.
    """

    name = "convert-quantum-to-qecl"

    k: int

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(AllocConversion(self.k)).rewrite_module(op)
