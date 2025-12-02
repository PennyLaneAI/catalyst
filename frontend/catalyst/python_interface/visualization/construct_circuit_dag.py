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

from collections import defaultdict
from functools import singledispatchmethod

from xdsl.dialects import builtin, func, scf
from xdsl.ir import Block, Operation, Region, SSAValue

from catalyst.python_interface.dialects import catalyst, quantum
from catalyst.python_interface.inspection.xdsl_conversion import (
    xdsl_to_qml_measurement,
    xdsl_to_qml_op,
)
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

        # Create a map of wire to node uid
        # Keys represent static (int) or dynamic wires (str)
        # Values represent the set of all node uids that are on that wire.
        self._wire_to_node_uid: dict[str | int, set[str]] = defaultdict(set)

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

    # ===================
    # QUANTUM OPERATIONS
    # ===================

    @_visit_operation.register
    def _unitary(
        self,
        op: quantum.CustomOp | quantum.GlobalPhaseOp | quantum.QubitUnitaryOp | quantum.MultiRZOp,
    ) -> None:
        """Generic handler for unitary gates."""

        # Create PennyLane instance
        qml_op = xdsl_to_qml_op(op)

        # Add node to current cluster
        node_uid = f"node_{id(op)}"
        self.dag_builder.add_node(
            uid=node_uid,
            label=str(qml_op),
            cluster_uid=self._cluster_uid_stack[-1],
        )

        # Search through previous ops found on current wires and connect
        prev_ops = set.union(*(self._wire_to_node_uid[wire] for wire in qml_op.wires))
        for prev_op in prev_ops:
            self.dag_builder.add_edge(prev_op, node_uid)

        # Update affected wires to source from this node UID
        for wire in qml_op.wires:
            self._wire_to_node_uid[wire] = {node_uid}

    # =====================
    # QUANTUM MEASUREMENTS
    # =====================

    @_visit_operation.register
    def _state_op(self, op: quantum.StateOp) -> None:
        """Handler for the terminal state measurement operation."""

        meas = xdsl_to_qml_measurement(op)
        node_uid = f"node_{id(op)}"
        # Build node on graph
        self.dag_builder.add_node(
            uid=node_uid,
            label=str(meas),
            cluster_uid=self._cluster_uid_stack[-1],
            fillcolor="lightpink",
            color="lightpink3",
        )

        for seen_wire, seen_nodes in self._wire_to_node_uid.items():
            for seen_node in seen_nodes:
                self.dag_builder.add_edge(seen_node, node_uid)

    @_visit_operation.register
    def _statistical_measurement_ops(
        self,
        op: quantum.ExpvalOp | quantum.VarianceOp | quantum.ProbsOp | quantum.SampleOp,
    ) -> None:
        """Handler for statistical measurement operations."""

        obs_op = op.obs.owner
        meas = xdsl_to_qml_measurement(op, xdsl_to_qml_measurement(obs_op))
        node_uid = f"node_{id(op)}"
        # Build node on graph
        self.dag_builder.add_node(
            uid=node_uid,
            label=str(meas),
            cluster_uid=self._cluster_uid_stack[-1],
            fillcolor="lightpink",
            color="lightpink3",
        )

        for wire in meas.wires:
            for seen_node in self._wire_to_node_uid[wire]:
                self.dag_builder.add_edge(seen_node, node_uid)

    @_visit_operation.register
    def _projective_measure_op(self, op: quantum.MeasureOp) -> None:
        """Handler for the single-qubit projective measurement operation."""

        meas = xdsl_to_qml_measurement(op)
        # Build node on graph
        self.dag_builder.add_node(
            uid=f"node_{id(op)}",
            label=str(meas),
            cluster_uid=self._cluster_uid_stack[-1],
        )

    # =============
    # CONTROL FLOW
    # =============

    @_visit_operation.register
    def _for_op(self, operation: scf.ForOp) -> None:
        """Handle an xDSL ForOp operation."""
        uid = f"cluster_{id(operation)}"
        self.dag_builder.add_cluster(
            uid,
            node_label=f"for ...",
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
        flattened_if_op: list[tuple[SSAValue | None, Region]] = _flatten_if_op(operation)

        uid = f"cluster_{id(operation)}"
        self.dag_builder.add_cluster(
            uid,
            node_label="",
            label="conditional",
            fontsize=10,
            labeljust="l",
            cluster_uid=self._cluster_uid_stack[-1],
        )
        self._cluster_uid_stack.append(uid)

        # Save wires state before all of the branches
        wire_map_before = self._wire_to_node_uid.copy()
        region_wire_maps: list[dict[int | str, set[str]]] = []

        # Loop through each branch and visualize as a cluster
        num_regions = len(flattened_if_op)
        for i, (condition_ssa, region) in enumerate(flattened_if_op):
            # Visualize with a cluster
            def _get_conditional_branch_label(i):
                if i == 0:
                    return "if ..."
                elif i == num_regions - 1:
                    return "else"
                else:
                    return "elif ..."

            uid = f"cluster_ifop_branch{i}_{id(operation)}"
            self.dag_builder.add_cluster(
                uid,
                node_label=_get_conditional_branch_label(i),
                label="",
                style="dashed",
                penwidth=1,
                cluster_uid=self._cluster_uid_stack[-1],
            )
            self._cluster_uid_stack.append(uid)

            # Make fresh wire map before going into region
            self._wire_to_node_uid = wire_map_before.copy()

            # Go recursively into the branch to process internals
            self._visit_region(region)

            # Update branch wire maps
            if self._wire_to_node_uid != wire_map_before:
                region_wire_maps.append(self._wire_to_node_uid)

            # Pop branch cluster after processing to ensure
            # logical branches are treated as 'parallel'
            self._cluster_uid_stack.pop()

        # Pop IfOp cluster before leaving this handler
        self._cluster_uid_stack.pop()

        # Check what wires were affected
        affected_wires = set(wire_map_before.keys())
        for region_wire_map in region_wire_maps:
            affected_wires.update(region_wire_map.keys())

        # Update state to be the union of all branch wire maps
        final_wire_map = defaultdict(set)
        for wire in affected_wires:
            all_nodes: set = set()
            for region_wire_map in region_wire_maps:
                if not wire in region_wire_map:
                    # Branch didn't touch this wire, so just use previous node
                    all_nodes.update(wire_map_before.get(wire, {}))
                else:
                    all_nodes.update(region_wire_map.get(wire, {}))

                final_wire_map[wire] = all_nodes
        self._wire_to_node_uid = final_wire_map

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
            for op in operation.body.ops:
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

        self._visit_block(operation.regions[0].blocks[0])

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


def _flatten_if_op(op: scf.IfOp) -> list[tuple[SSAValue | None, Region]]:
    """Recursively flattens a nested IfOp (if/elif/else chains)."""

    condition_ssa: SSAValue = op.operands[0]
    then_region, else_region = op.regions

    # Save condition SSA in case we want to visualize it eventually
    flattened_op: list[tuple[SSAValue | None, Region]] = [(condition_ssa, then_region)]

    # Peak into else region to see if there's another IfOp
    else_block: Block = else_region.block
    # Completely relies on the structure that the second last operation
    # will be an IfOp (seems to hold true)
    if isinstance(else_block.ops.last.prev_op, scf.IfOp):
        # Recursively flatten any IfOps found in said block
        nested_flattened_op = _flatten_if_op(else_block.ops.last.prev_op)
        flattened_op.extend(nested_flattened_op)
        return flattened_op

    # No more nested IfOps, therefore append final region
    # with no SSAValue
    flattened_op.extend([(None, else_region)])
    return flattened_op
