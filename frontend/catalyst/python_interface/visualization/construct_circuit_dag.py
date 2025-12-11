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
from functools import singledispatch, singledispatchmethod

from catalyst.python_interface.dialects import quantum
from catalyst.python_interface.inspection.xdsl_conversion import (
    xdsl_to_qml_measurement,
    xdsl_to_qml_op,
)
from catalyst.python_interface.visualization.dag_builder import DAGBuilder
from pennylane.measurements import (
    ExpectationMP,
    MeasurementProcess,
    ProbabilityMP,
    VarianceMP,
)
from pennylane.operation import Operator
from xdsl.dialects import builtin, func, scf
from xdsl.ir import Block, Operation, Region, SSAValue


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
        self._wire_to_node_uids: dict[str | int, set[str]] = defaultdict(set)

        # Use counter internally for UID
        self._node_uid_counter: int = 0
        self._cluster_uid_counter: int = 0

    def _reset(self) -> None:
        """Resets the instance."""
        self._cluster_uid_stack: list[str] = []
        self._node_uid_counter: int = 0
        self._cluster_uid_counter: int = 0
        self._wire_to_node_uids: dict[str | int, set[str]] = defaultdict(set)

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
    def _gate_op(
        self,
        op: quantum.CustomOp
        | quantum.GlobalPhaseOp
        | quantum.QubitUnitaryOp
        | quantum.MultiRZOp,
    ) -> None:
        """Generic handler for unitary gates."""

        # Create PennyLane instance
        qml_op = xdsl_to_qml_op(op)

        # Add node to current cluster
        node_uid = f"node{self._node_uid_counter}"
        self.dag_builder.add_node(
            uid=node_uid,
            label=get_label(qml_op),
            cluster_uid=self._cluster_uid_stack[-1],
            # NOTE: "record" allows us to use ports (https://graphviz.org/doc/info/shapes.html#record)
            shape="record",
        )
        self._node_uid_counter += 1

        # Search through previous ops found on current wires and connect
        prev_node_uids: set[str] = set.union(
            set(), *(self._wire_to_node_uids[wire] for wire in qml_op.wires)
        )
        for prev_node_uid in prev_node_uids:
            self.dag_builder.add_edge(prev_node_uid, node_uid)

        # Update affected wires to source from this node UID
        for wire in qml_op.wires:
            self._wire_to_node_uids[wire] = {node_uid}

    @_visit_operation.register
    def _projective_measure_op(self, op: quantum.MeasureOp) -> None:
        """Handler for the single-qubit projective measurement operation."""

        # Create PennyLane instance
        meas = xdsl_to_qml_measurement(op)

        # Add node to current cluster
        node_uid = f"node{self._node_uid_counter}"
        self.dag_builder.add_node(
            uid=node_uid,
            label=get_label(meas),
            cluster_uid=self._cluster_uid_stack[-1],
            # NOTE: "record" allows us to use ports (https://graphviz.org/doc/info/shapes.html#record)
            shape="record",
        )
        self._node_uid_counter += 1

        # Search through previous ops found on current wires and connect
        prev_node_uids: set[str] = set.union(
            set(), *(self._wire_to_node_uids[wire] for wire in meas.wires)
        )
        for prev_node_uid in prev_node_uids:
            self.dag_builder.add_edge(prev_node_uid, node_uid)

        # Update affected wires to source from this node UID
        for wire in meas.wires:
            self._wire_to_node_uids[wire] = {node_uid}

    # =====================
    # QUANTUM MEASUREMENTS
    # =====================

    @_visit_operation.register
    def _measurements(
        self,
        op: quantum.StateOp
        | quantum.ExpvalOp
        | quantum.VarianceOp
        | quantum.SampleOp
        | quantum.ProbsOp,
    ) -> None:
        """Handler for all quantum measurement operations."""

        prev_wires = []
        meas = None

        match op:
            case quantum.StateOp():
                meas = xdsl_to_qml_measurement(op)
                # NOTE: state can only handle all wires
                prev_wires = self._wire_to_node_uids.keys()

            case quantum.ExpvalOp() | quantum.VarianceOp():
                obs_op = op.obs.owner
                meas = xdsl_to_qml_measurement(op, xdsl_to_qml_measurement(obs_op))
                prev_wires = meas.wires.labels

            case quantum.SampleOp() | quantum.ProbsOp():
                obs_op = op.obs.owner

                # TODO: This doesn't logically make sense, but quantum.compbasis
                # is obs_op and function below just pulls out the static wires
                wires = xdsl_to_qml_measurement(obs_op)
                meas = xdsl_to_qml_measurement(op, wires=None if wires == [] else wires)

                if wires == []:
                    # If no wires specified, connect to all seen current wires
                    prev_wires = self._wire_to_node_uids.keys()
                else:
                    # Use the specific wires from the observable
                    prev_wires = wires

            case _:
                return

        node_uid = f"node{self._node_uid_counter}"
        self.dag_builder.add_node(
            uid=node_uid,
            label=get_label(meas),
            cluster_uid=self._cluster_uid_stack[-1],
            fillcolor="lightpink",
            color="lightpink3",
            # NOTE: "record" allows us to use ports (https://graphviz.org/doc/info/shapes.html#record)
            shape="record",
        )
        self._node_uid_counter += 1

        for wire in prev_wires:
            if wire in self._wire_to_node_uids:
                for seen_node in self._wire_to_node_uids[wire]:
                    self.dag_builder.add_edge(seen_node, node_uid, color="lightpink3")

    # =============
    # CONTROL FLOW
    # =============

    @_visit_operation.register
    def _for_op(self, operation: scf.ForOp) -> None:
        """Handle an xDSL ForOp operation."""

        uid = f"cluster{self._cluster_uid_counter}"
        self.dag_builder.add_cluster(
            uid,
            label="for loop",
            labeljust="l",
            cluster_uid=self._cluster_uid_stack[-1],
        )
        self._cluster_uid_stack.append(uid)
        self._cluster_uid_counter += 1

        self._visit_region(operation.regions[0])

        self._cluster_uid_stack.pop()

    @_visit_operation.register
    def _while_op(self, operation: scf.WhileOp) -> None:
        """Handle an xDSL WhileOp operation."""
        uid = f"cluster{self._cluster_uid_counter}"
        self.dag_builder.add_cluster(
            uid,
            label="while loop",
            labeljust="l",
            cluster_uid=self._cluster_uid_stack[-1],
        )
        self._cluster_uid_stack.append(uid)
        self._cluster_uid_counter += 1

        for region in operation.regions:
            self._visit_region(region)

        self._cluster_uid_stack.pop()

    @_visit_operation.register
    def _if_op(self, operation: scf.IfOp):
        """Handles the scf.IfOp operation."""
        uid = f"cluster{self._cluster_uid_counter}"
        self.dag_builder.add_cluster(
            uid,
            label="conditional",
            labeljust="l",
            cluster_uid=self._cluster_uid_stack[-1],
        )
        self._cluster_uid_stack.append(uid)
        self._cluster_uid_counter += 1

        # Save wires state before all of the branches
        wire_map_before = self._wire_to_node_uids.copy()
        region_wire_maps: list[dict[int | str, set[str]]] = []

        # Loop through each branch and visualize as a cluster
        flattened_if_op: list[Region] = _flatten_if_op(operation)
        num_regions = len(flattened_if_op)
        for i, region in enumerate(flattened_if_op):
            cluster_label = "elif"
            if i == 0:
                cluster_label = "if"
            elif i == num_regions - 1:
                cluster_label = "else"

            uid = f"cluster{self._cluster_uid_counter}"
            self.dag_builder.add_cluster(
                uid,
                label=cluster_label,
                labeljust="l",
                style="dashed",
                cluster_uid=self._cluster_uid_stack[-1],
            )
            self._cluster_uid_stack.append(uid)
            self._cluster_uid_counter += 1

            # Make fresh wire map before going into region
            self._wire_to_node_uids = wire_map_before.copy()

            # Go recursively into the branch to process internals
            self._visit_region(region)

            # Update branch wire maps
            if self._wire_to_node_uids != wire_map_before:
                region_wire_maps.append(self._wire_to_node_uids)

            # Pop branch cluster after processing to ensure
            # logical branches are treated as 'parallel'
            self._cluster_uid_stack.pop()

        # Pop IfOp cluster before leaving this handler
        self._cluster_uid_stack.pop()

        # Check what wires were affected
        affected_wires: set[str | int] = set(wire_map_before.keys())
        for region_wire_map in region_wire_maps:
            affected_wires.update(region_wire_map.keys())

        # Update state to be the union of all branch wire maps
        final_wire_map = defaultdict(set)
        for wire in affected_wires:
            all_nodes: set = set()
            for region_wire_map in region_wire_maps:
                if not wire in region_wire_map:
                    # IfOp region didn't apply anything on this wire
                    # so default to node before the IfOp
                    all_nodes.update(wire_map_before.get(wire, set()))
                else:
                    all_nodes.update(region_wire_map.get(wire, set()))
                final_wire_map[wire] = all_nodes
        self._wire_to_node_uids = final_wire_map

    # ============
    # DEVICE NODE
    # ============

    @_visit_operation.register
    def _device_init(self, operation: quantum.DeviceInitOp) -> None:
        """Handles the initialization of a quantum device."""
        node_id = f"node{self._node_uid_counter}"
        self.dag_builder.add_node(
            node_id,
            label=operation.device_name.data,
            cluster_uid=self._cluster_uid_stack[-1],
            fillcolor="grey",
            color="black",
            penwidth=2,
            shape="rectangle",
        )
        self._node_uid_counter += 1

    # =======================
    # FuncOp NESTING UTILITY
    # =======================

    @_visit_operation.register
    def _func_op(self, operation: func.FuncOp) -> None:
        """Visit a FuncOp Operation."""

        label = operation.sym_name.data
        if "jit_" in operation.sym_name.data:
            label = "qjit"

        uid = f"cluster{self._cluster_uid_counter}"
        parent_cluster_uid = (
            None if self._cluster_uid_stack == [] else self._cluster_uid_stack[-1]
        )
        self.dag_builder.add_cluster(
            uid,
            label=label,
            cluster_uid=parent_cluster_uid,
        )
        self._cluster_uid_counter += 1
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

        # Clear seen wires as we are exiting a FuncOp (qnode)
        self._wire_to_node_uids = defaultdict(set)


def _flatten_if_op(op: scf.IfOp) -> list[Region]:
    """Recursively flattens a nested IfOp (if/elif/else chains)."""

    then_region, else_region = op.regions

    flattened_op: list[Region] = [then_region]

    # Check to see if there are any nested quantum operations in the else block
    else_block: Block = else_region.block
    has_quantum_ops = False
    nested_if_op = None
    for op in else_block.ops:
        if isinstance(op, scf.IfOp):
            nested_if_op = op
            # No need to walk this op as this will be
            # recursively handled down below
            continue
        for internal_op in op.walk():
            if type(internal_op) in quantum.Quantum.operations:
                has_quantum_ops = True
                # No need to check anything else
                break

    if nested_if_op and not has_quantum_ops:
        # Recursively flatten any IfOps found in said block
        nested_flattened_op: list[Region] = _flatten_if_op(nested_if_op)
        flattened_op.extend(nested_flattened_op)
        return flattened_op

    # No more nested IfOps, therefore append final region
    flattened_op.append(else_region)
    return flattened_op


@singledispatch
def get_label(op: Operator | MeasurementProcess) -> str:
    """Gets the appropriate label for a PennyLane object."""
    return str(op)


@get_label.register
def _operator(op: Operator) -> str:
    """Returns the appropriate label for PennyLane Operator"""
    wires = list(op.wires.labels)
    if wires == []:
        wires_str = "all"
    else:
        wires_str = f"[{', '.join(map(str, wires))}]"
    # Using <...> lets us use ports (https://graphviz.org/doc/info/shapes.html#record)
    return f"<name> {op.name}|<wire> {wires_str}"


@get_label.register
def _meas(meas: MeasurementProcess) -> str:
    """Returns the appropriate label for a PennyLane MeasurementProcess using match/case."""

    wires_str = list(meas.wires.labels)
    if not wires_str:
        wires_str = "all"
    else:
        wires_str = f"[{', '.join(map(str, wires_str))}]"

    base_name = meas._shortname

    match meas:
        case ExpectationMP() | VarianceMP() | ProbabilityMP():
            if meas.obs is not None:
                obs_name = meas.obs.name
                base_name = f"{base_name}({obs_name})"

        case _:
            pass

    return f"<name> {base_name}|<wire> {wires_str}"
