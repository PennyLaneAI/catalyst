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

from xdsl.dialects import builtin, func
from xdsl.ir import Block, Operation, Region

from catalyst.python_interface.dialects import catalyst, quantum
from catalyst.python_interface.visualization.dag_builder import DAGBuilder


class ConstructCircuitDAG:
    """Utility tool following the director pattern to build a DAG representation of a compiled quantum program.

    This tool traverses an xDSL module and constructs a Directed Acyclic Graph (DAG)
    of it's quantum program using an injected DAGBuilder instance. This tool does not mutate the xDSL module.

    **Example**

    >>> builder = PyDotDAGBuilder()
    >>> director = ConstructCircuitDAG(builder)
    >>> director.construct(module)
    >>> director.dag_builder.to_string()
    ...
    """

    def __init__(self, dag_builder: DAGBuilder) -> None:
        self.dag_builder: DAGBuilder = dag_builder

        # Keep track of nesting clusters using a stack
        self._cluster_uid_stack: list[str] = []

        # Use counter internally for UID
        self._uid_counter: int = 0

    def _reset(self) -> None:
        """Resets the instance."""
        self._cluster_uid_stack: list[str] = []
        self._uid_counter: int = 0

    def construct(self, module: builtin.ModuleOp) -> None:
        """Constructs the DAG from the module.

        Args:
            module (xdsl.builtin.ModuleOp): The module containing the quantum program to visualize.

        """
        self._reset()
        for op in module.ops:
            self._visit_operation(op)

    # =============
    # IR TRAVERSAL
    # =============

    @singledispatchmethod
    def _visit_operation(self, operation: Operation) -> None:
        """Visit an xDSL Operation. Default to visiting each region contained in the operation."""
        for region in operation.regions:
            self._visit_region(region)

    def _visit_region(self, region: Region) -> None:
        """Visit an xDSL Region operation."""
        for block in region.blocks:
            self._visit_block(block)

    def _visit_block(self, block: Block) -> None:
        """Visit an xDSL Block operation, dispatching handling for each contained Operation."""
        for op in block.ops:
            self._visit_operation(op)

    # ============
    # DEVICE NODE
    # ============

    @_visit_operation.register
    def _device_init(self, operation: quantum.DeviceInitOp) -> None:
        """Handles the initialization of a quantum device."""
        node_id = f"device_node_{self._uid_counter}"
        self._uid_counter += 1
        self.dag_builder.add_node(
            node_id,
            label=operation.device_name.data,
            cluster_uid=self._cluster_uid_stack[-1],
            fillcolor="grey",
            color="black",
            penwidth=2,
            shape="rectangle",
        )

    # =======================
    # FuncOp NESTING UTILITY
    # =======================

    @_visit_operation.register
    def _func_op(self, operation: func.FuncOp) -> None:
        """Visit a FuncOp Operation."""

        label = operation.sym_name.data
        if "jit_" in operation.sym_name.data:
            label = "qjit"

        uid = f"funcop_cluster_{self._uid_counter}"
        self._uid_counter += 1
        parent_cluster_uid = None if self._cluster_uid_stack == [] else self._cluster_uid_stack[-1]
        self.dag_builder.add_cluster(
            uid,
            label=label,
            cluster_uid=parent_cluster_uid,
        )
        self._cluster_uid_stack.append(uid)

        self._visit_block(operation.regions[0].blocks[0])

    @_visit_operation.register
    def _func_return(self, operation: func.ReturnOp) -> None:
        """Handle func.return to exit FuncOp's cluster scope."""

        # NOTE: Skip first cluster as it is the "base" of the graph diagram.
        # In our case, it is the `qjit` bounding box.
        if len(self._cluster_uid_stack) > 1:
            # If we hit a func.return operation we know we are leaving
            # the FuncOp's scope and so we can pop the ID off the stack.
            self._cluster_uid_stack.pop()
