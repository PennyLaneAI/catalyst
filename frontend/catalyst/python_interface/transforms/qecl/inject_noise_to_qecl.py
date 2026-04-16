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

"""Inject noise into a QEC logical IR."""

from dataclasses import dataclass
from xdsl import passes, context
from xdsl.dialects import builtin
from xdsl import pattern_rewriter
from xdsl.rewriter import InsertPoint
from catalyst.python_interface.dialects import qecl
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform


class InjectNoiseToQECLPattern(pattern_rewriter.RewritePattern):
    """Pattern to inject noise into a QEC logical IR."""

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        qecop: qecl.QecCycleOp,
        rewriter: pattern_rewriter.PatternRewriter,
        /,
    ):
        """Inject a qecl.noise operation for each qecl.qec operation in the QEC logical IR."""

        codeblock = None
        for result in qecop.operands[0].owner.results:
            if result == qecop.in_codeblock:
                codeblock = result
                break
        noiseop = qecl.NoiseOp(codeblock)
        rewriter.insert_op(noiseop, InsertPoint.before(qecop))
        rewriter.replace_all_uses_with(codeblock, noiseop.results[0])


@dataclass(frozen=True)
class InjectNoiseToQECLPass(passes.ModulePass):
    """Pass to inject noise into a QEC logical IR."""

    name = "inject-noiseop-to-qecl"

    def apply(self, ctx: context.Context, op: builtin.ModuleOp) -> None:
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([InjectNoiseToQECLPattern()]),
            apply_recursively=False,
        ).rewrite_module(op)


inject_noise_to_qecl_pass = compiler_transform(InjectNoiseToQECLPass)
