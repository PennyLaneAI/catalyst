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

from xdsl.dialects import builtin, func
from xdsl.ir import Block, Operation, Region

from catalyst.python_interface.dialects import quantum
from catalyst.python_interface.visualization.dag_builder import DAGBuilder


class ConstructCircuitDAG:
    """A tool that traverses an xDSL module and constructs a Directed Acyclic Graph (DAG)
    of it's quantum program using an injected DAGBuilder instance. This tool does not mutate the xDSL module.
    """

    def __init__(self, dag_builder: DAGBuilder) -> None:
        """Initialize the utility by injecting the DAG builder dependency.

        Args:
            dag_builder (DAGBuilder): The concrete builder instance used for graph construction.
        """
        self.dag_builder: DAGBuilder = dag_builder

        # Keep track of nesting clusters using a stack
        # NOTE: `None` corresponds to the base graph 'cluster'
        self._cluster_stack: list[str | None] = [None]

    def _reset(self) -> None:
        """Resets the instance."""
        self._cluster_stack: list[str | None] = [None]

    # =================================
    # 1. CORE DISPATCH AND ENTRY POINT
    # =================================

    @singledispatchmethod
    def _visit(self, op: Any) -> None:
        """Central dispatch method (Visitor Pattern). Routes the operation 'op'
        to the specialized handler registered for its type."""

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

    @_visit.register
    def _operation(self, operation: Operation) -> None:
        """Visit an xDSL Operation."""

        # Visualize FuncOp's as bounding boxes
        if isinstance(operation, func.FuncOp) and operation.sym_name.data not in {
            "setup",
            "teardown",
        }:
            cluster_id = f"cluster_{id(operation)}"
            self.dag_builder.add_cluster(
                cluster_id,
                node_label=operation.sym_name.data,
                cluster_id=self._cluster_stack[-1],
            )
            self._cluster_stack.append(cluster_id)

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

    @_visit.register
    def _device_init(self, op: quantum.DeviceInitOp) -> None:
        """Handles the initialization of a quantum device."""
        node_id = f"node_{id(op)}"
        self.dag_builder.add_node(
            node_id,
            label=op.device_name.data,
            cluster_id=self._cluster_stack[-1],
            fillcolor="white",
            shape="rectangle",
        )

    @_visit.register
    def _func_return(self, op: func.ReturnOp) -> None:
        """Handle func.return to exit FuncOp's cluster scope."""

        # NOTE: Skip first two because the first is the base graph, second is the jit_* workflow FuncOp
        if len(self._cluster_stack) > 2:
            # If we hit a func.return operation we know we are leaving
            # the FuncOp's scope and so we can pop the ID off the stack.
            self._cluster_stack.pop()
