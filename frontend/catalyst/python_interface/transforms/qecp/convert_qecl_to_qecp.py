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

"""QEC Logical to QEC Physical dialect conversion.

This module contains the implementation of the xDSL convert-qecl-to-qecp dialect-conversion pass.
To apply this pass, the QEC code must be know.
"""

from dataclasses import dataclass
from enum import StrEnum

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass

from catalyst.python_interface.pass_api.compiler_transform import compiler_transform


class QecCodeKey(StrEnum):
    """The set of supported QEC codes."""

    # List the supported QEC codes here, e.g.
    # STEANE_7_1_3 = "steane[[7,1,3]]"


# MARK: Conversion Pass


@dataclass(frozen=True)
class ConvertQecLogicalToQecPhysicalPass(ModulePass):
    """
    Convert QEC logical instructions to QEC physical instructions.
    """

    name = "convert-qecl-to-qecp"

    qec_code: QecCodeKey

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """Apply the convert-qecl-to-qecp pass."""

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    # Insert conversion patterns here
                ]
            )
        ).rewrite_module(op)

        # Certain patterns leave behind `builtin.unrealized_conversion_cast` ops;
        # this pass removes them
        ReconcileUnrealizedCastsPass().apply(ctx, op)


convert_qecl_to_qecp_pass = compiler_transform(ConvertQecLogicalToQecPhysicalPass)
