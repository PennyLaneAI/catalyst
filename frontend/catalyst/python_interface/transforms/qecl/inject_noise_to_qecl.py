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

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin
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

        # 1. Create a qecl.noise operation and insert it before the qecl.qec operation.
        codeblock = qecop.in_codeblock
        noiseop = qecl.NoiseOp(codeblock)
        rewriter.insert_op(noiseop, InsertPoint.before(qecop))

        # 2. Replace all uses of the codeblock with the output of the noise operation,
        # except for the first use (the qecl.noise operation).
        codeblock.replace_uses_with_if(
            noiseop.out_codeblock, lambda use: use is not codeblock.first_use
        )

        # 3. Notify the rewriter that all operations using the codeblock have been modified,
        # except for the first use (the qecl.noise operation).
        for use in codeblock.uses:
            if use is not codeblock.first_use:
                rewriter.notify_op_modified(use.operation)


@dataclass(frozen=True)
class InjectNoiseToQECLPass(passes.ModulePass):
    """Pass to inject noise into a QEC logical IR."""

    name = "inject-noise-to-qecl"

    def apply(self, ctx: context.Context, op: builtin.ModuleOp) -> None:
        pattern_rewriter.PatternRewriteWalker(
            InjectNoiseToQECLPattern(),
            apply_recursively=False,
        ).rewrite_module(op)


inject_noise_to_qecl_pass = compiler_transform(InjectNoiseToQECLPass)
