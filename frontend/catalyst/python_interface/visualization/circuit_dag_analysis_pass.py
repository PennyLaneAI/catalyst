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

"""Contains the CircuitDAGAnalysisPass for generating a DAG from an xDSL module."""

from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any

import xdsl
from xdsl.dialects import builtin, func, scf
from xdsl.ir import Block, Region

from catalyst.python_interface.dialects import quantum
from catalyst.python_interface.visualization.dag_builder import DAGBuilder


class CircuitDAGAnalysisPass:
    def __init__(self, dag_builder: DAGBuilder) -> None:
        """Initialize the analysis pass."""
        self.dag_builder: DAGBuilder = dag_builder

    @singledispatchmethod
    def visit_op(self, op: Any) -> None:
        """Default handler for unknown operation types.

        This method is dispatched based on the type of 'op'.

        Args:
            op (Any): An xDSL operation.
        """

    @visit_op.register
    def visit_region(self, region: Region) -> None:
        """Visit an xDSL Region operation."""
        for block in region.blocks:
            self.visit_block(block)

    @visit_op.register
    def visit_block(self, block: Block) -> None:
        """Visit an xDSL Block operation."""
        for op in block.ops:
            self.visit_op(op)

    def run(self, module: builtin.ModuleOp) -> None:
        """Apply the analysis pass on the module."""

        for op in module.ops:
            self.visit_op(op)
