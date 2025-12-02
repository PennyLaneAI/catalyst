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

from xdsl.dialects import builtin, func, scf
from xdsl.ir import Block, Operation, Region

from catalyst.python_interface.dialects import catalyst, quantum
from catalyst.python_interface.inspection.xdsl_conversion import resolve_constant_params
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

    def _reset(self) -> None:
        """Resets the instance."""
        self._cluster_uid_stack: list[str] = []

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

    # =============
    # CONTROL FLOW
    # =============

    @_visit_operation.register
    def _for_op(self, operation: scf.ForOp) -> None:
        """Handle an xDSL ForOp operation."""
        lower_bound, upper_bound, step = (
            resolve_constant_params(operation.lb),
            resolve_constant_params(operation.ub),
            resolve_constant_params(operation.step),
        )

        index_var_name = operation.body.blocks[0].args[0].name_hint

        uid = f"cluster_{id(operation)}"
        self.dag_builder.add_cluster(
            uid,
            node_label=f"for {index_var_name} in range({lower_bound},{upper_bound},{step})",
            label="",
            cluster_uid=self._cluster_uid_stack[-1],
        )
        self._cluster_uid_stack.append(uid)

        for region in operation.regions:
            self._visit_region(region)

        self._cluster_uid_stack.pop()

    @_visit_operation.register
    def _while_op(self, operation: scf.WhileOp) -> None:
        """Handle an xDSL WhileOp operation."""
        uid = f"cluster_{id(operation)}"
        self.dag_builder.add_cluster(
            uid,
            node_label="while ...",
            label="",
            cluster_uid=self._cluster_uid_stack[-1],
        )
        self._cluster_uid_stack.append(uid)

        for region in operation.regions:
            self._visit_region(region)

        self._cluster_uid_stack.pop()

    @_visit_operation.register
    def _if_op(self, operation: scf.IfOp):
        """Handles the scf.IfOp operation."""
        uid = f"cluster_{id(operation)}"
        self.dag_builder.add_cluster(
            uid,
            node_label="",
            label="conditional",
            labeljust="l",
            cluster_uid=self._cluster_uid_stack[-1],
        )
        self._cluster_uid_stack.append(uid)

        # Loop through each branch and visualize as a cluster
        for i, branch in enumerate(operation.regions):
            uid = f"cluster_ifop_branch{i}_{id(operation)}"
            self.dag_builder.add_cluster(
                uid,
                node_label="if ..." if i == 0 else "else",
                label="",
                cluster_uid=self._cluster_uid_stack[-1],
            )
            self._cluster_uid_stack.append(uid)

            # Go recursively into the branch to process internals
            self._visit_region(branch)

            # Pop branch cluster after processing to ensure
            # logical branches are treated as 'parallel'
            self._cluster_uid_stack.pop()

        self._cluster_uid_stack.pop()

    # ============
    # DEVICE NODE
    # ============

    @_visit_operation.register
    def _device_init(self, operation: quantum.DeviceInitOp) -> None:
        """Handles the initialization of a quantum device."""
        node_id = f"node_{id(operation)}"
        self.dag_builder.add_node(
            node_id,
            label=operation.device_name.data,
            cluster_uid=self._cluster_uid_stack[-1],
            fillcolor="grey",
            color="black",
            penwidth=2,
            shape="rectangle",
        )

        for region in operation.regions:
            self._visit_region(region)

    # =======================
    # FuncOp NESTING UTILITY
    # =======================

    @_visit_operation.register
    def _func_op(self, operation: func.FuncOp) -> None:
        """Visit a FuncOp Operation."""

        # If this is the jit_* FuncOp, only draw if there's more than one qnode (launch kernel)
        # This avoids redundant nested clusters: jit_my_circuit -> my_circuit -> ...
        visualize = True
        label = operation.sym_name.data
        if "jit_" in operation.sym_name.data:
            num_qnodes = 0
            for op in operation.walk():
                if isinstance(op, catalyst.LaunchKernelOp):
                    num_qnodes += 1
            # Get everything after the jit_* prefix
            label = str(label).split("_", maxsplit=1)[-1]
            if num_qnodes == 1:
                visualize = False

        if visualize:
            uid = f"cluster_{id(operation)}"
            parent_cluster_uid = (
                None if self._cluster_uid_stack == [] else self._cluster_uid_stack[-1]
            )
            self.dag_builder.add_cluster(
                uid,
                label=label,
                cluster_uid=parent_cluster_uid,
            )
            self._cluster_uid_stack.append(uid)

        for region in operation.regions:
            self._visit_region(region)

    @_visit_operation.register
    def _func_return(self, operation: func.ReturnOp) -> None:
        """Handle func.return to exit FuncOp's cluster scope."""

        # NOTE: Skip first cluster as it is the "base" of the graph diagram.
        # If it is a multi-qnode workflow, it will represent the "workflow" function
        # If it is a single qnode, it will represent the quantum function.
        if len(self._cluster_uid_stack) > 1:
            # If we hit a func.return operation we know we are leaving
            # the FuncOp's scope and so we can pop the ID off the stack.
            self._cluster_uid_stack.pop()
