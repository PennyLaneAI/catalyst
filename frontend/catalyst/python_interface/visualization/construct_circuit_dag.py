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

from xdsl.dialects import builtin, func, scf
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
        if isinstance(operation, func.FuncOp):
            cluster_id = f"cluster_{id(operation)}"
            self.dag_builder.add_cluster(
                cluster_id,
                node_label=operation.sym_name.data,
                cluster_id=self._cluster_stack[-1],
            )
            self._cluster_stack.append(cluster_id)

        if isinstance(operation, scf.IfOp):
            # Loop through each branch and visualize as a cluster
            for i, branch in enumerate(operation.regions):
                cluster_id = f"cluster_ifop_branch{i}_{id(operation)}"
                self.dag_builder.add_cluster(
                    cluster_id,
                    label=f"if ..." if i == 0 else "else",
                    cluster_id=self._cluster_stack[-1],
                )
                self._cluster_stack.append(cluster_id)

                # Go recursively into the branch to process internals
                self._visit(branch)

                # Pop branch cluster after processing to ensure
                # logical branches are treated as 'parallel'
                self._cluster_stack.pop()
        else:
            for region in operation.regions:
                self._visit(region)

        # Pop if the operation was a cluster creating operation
        # This ensures proper nesting
        ControlFlowOp = scf.ForOp | scf.WhileOp | scf.IfOp
        if isinstance(operation, ControlFlowOp):
            self._cluster_stack.pop()

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
            fillcolor="grey",
            color="black",
            penwidth=2,
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

    # =========================
    # 3. CONTROL FLOW HANDLERS
    # =========================

    @_visit.register
    def _for_op(self, op: scf.ForOp) -> None:
        """Handle an xDSL ForOp operation."""
        cluster_id = f"cluster_{id(op)}"
        self.dag_builder.add_cluster(
            cluster_id,
            node_label="for ...",
            cluster_id=self._cluster_stack[-1],
        )
        self._cluster_stack.append(cluster_id)

    @_visit.register
    def _while_op(self, op: scf.WhileOp) -> None:
        """Handle an xDSL WhileOp operation."""
        cluster_id = f"cluster_{id(op)}"
        self.dag_builder.add_cluster(
            cluster_id,
            node_label="while ...",
            cluster_id=self._cluster_stack[-1],
        )
        self._cluster_stack.append(cluster_id)

    @_visit.register
    def _if_op(self, op: scf.IfOp) -> None:
        """Handle an xDSL IfOp operation."""
        cluster_id = f"cluster_{id(op)}"
        self.dag_builder.add_cluster(
            cluster_id,
            node_label="if",
            cluster_id=self._cluster_stack[-1],
        )
        self._cluster_stack.append(cluster_id)
