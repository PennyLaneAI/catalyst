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

"""Contains the ConstructCircuitDAG tool for constructing a DAG from an xDSL module."""

from functools import singledispatchmethod
from typing import Any

from xdsl.dialects import builtin
from xdsl.ir import Block, Operation, Region

from catalyst.python_interface.visualization.dag_builder import DAGBuilder


class ConstructCircuitDAG:
    """A tool that traverses an xDSL module and constructs a Directed Acyclic Graph (DAG)
    of it's quantum program using an injected DAGBuilder instance. This tool does not mutate the xDSL module.
    """

    def __init__(self, dag_builder: DAGBuilder) -> None:
        self.dag_builder: DAGBuilder = dag_builder

    def construct(self, module: builtin.ModuleOp) -> None:
        """Constructs the DAG from the module.

        Args:
            module (xdsl.builtin.ModuleOp): The module containing the quantum program to visualize.

        """
        for op in module.ops:
            self._visit(op)

    # =======================
    # 2. IR TRAVERSAL
    # =======================

    @singledispatchmethod
    def _visit(self, op: Any) -> None:
        """Central dispatch method (Visitor Pattern). Routes the operation 'op'
        to the specialized handler registered for its type."""

    @_visit.register
    def _operation(self, operation: Operation) -> None:
        """Visit an xDSL Operation."""
        for region in operation.regions:
            self._visit(region)

    @_visit.register
    def _region(self, region: Region) -> None:
        """Visit an xDSL Region operation."""
        for block in region.blocks:
            self._visit(block)

    @_visit.register
    def _block(self, block: Block) -> None:
        """Visit an xDSL Block operation, dispatching handling for each contained Operation."""
        for op in block.ops:
            self._visit(op)
