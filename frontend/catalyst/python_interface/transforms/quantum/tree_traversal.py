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

"""This file contains the implementation of the tree_traversal transform,
written using xDSL."""

from dataclasses import dataclass

from xdsl import context, passes
from xdsl.dialects import builtin

from catalyst.python_interface.pass_api import compiler_transform


@dataclass(frozen=True)
class TreeTraversalPass(passes.ModulePass):
    """Pass for tree traversal (placeholder)."""

    name = "xdsl-tree-traversal"

    def apply(
        self, _ctx: context.Context, op: builtin.ModuleOp
    ) -> None:  # pylint: disable=unused-argument
        """Apply the tree traversal pass (no-op)."""


tree_traversal_pass = compiler_transform(TreeTraversalPass)
