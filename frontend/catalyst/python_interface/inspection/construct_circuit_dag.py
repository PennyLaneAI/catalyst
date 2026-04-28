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
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from functools import singledispatch, singledispatchmethod
from typing import Sequence

from pennylane.measurements import (
    ExpectationMP,
    MeasurementProcess,
    ProbabilityMP,
    VarianceMP,
)
from pennylane.operation import Operator
from pennylane.ops import GlobalPhase
from xdsl.dialects import builtin, func, scf
from xdsl.ir import Block, Operation, Region

from catalyst.python_interface.dialects import mbqc, pbc, quantum
from catalyst.python_interface.inspection.dag_builder import DAGBuilder
from catalyst.python_interface.inspection.xdsl_conversion import (
    ssa_to_qp_wires,
    xdsl_to_qp_measurement,
    xdsl_to_qp_op,
)

# Defines a set of operations from the quantum dialect
# that are not to be visualized (at the moment)
_SKIPPED_QUANTUM_OPS = (
    quantum.AllocOp,
    quantum.AllocQubitOp,
    quantum.ComputationalBasisOp,
    quantum.CountsOp,
    quantum.DeallocOp,
    quantum.DeallocQubitOp,
    quantum.DeviceReleaseOp,
    quantum.ExtractOp,
    quantum.FinalizeOp,
    quantum.HamiltonianOp,
    quantum.HermitianOp,
    quantum.InitializeOp,
    quantum.InsertOp,
    quantum.NamedObsOp,
    quantum.NumQubitsOp,
    quantum.PCPhaseOp,
    quantum.TensorOp,
    quantum.YieldOp,
)
# Handlers for PPRs and PPMs are defined
# but not sure how to visualize these yet
_SKIPPED_PBC_OPS = (
    pbc.FabricateOp,
    pbc.LayerOp,
    pbc.PrepareStateOp,
    pbc.SelectPPMeasurementOp,
    pbc.YieldOp,
)
# Any MBQC operation encountered will raise a
# VisualizationError
_SKIPPED_MBQC_OPS = ()

_SKIPPED_OPS = (*_SKIPPED_QUANTUM_OPS, *_SKIPPED_PBC_OPS, *_SKIPPED_MBQC_OPS)
_SUPPORTED_DIALECTS = {quantum.Quantum.name, pbc.PBC.name, mbqc.MBQC.name}


class _WireKind(Enum):
    """Wire type for ordering dependencies."""

    DEVICE = auto()
    DYNAMIC = auto()


class _ClusterKind(Enum):
    """Defines the structural role for a cluster of operations."""

    FUNC = auto()
    FOR_LOOP = auto()
    WHILE_LOOP = auto()
    CONDITIONAL = auto()
    BRANCH = auto()
    ADJOINT = auto()


@dataclass(frozen=True)
class ClusterEntry:
    """Unique descriptor for a cluster of operations."""

    uid: str
    kind: _ClusterKind


class VisualizationError(Exception):
    """Custom visualization error."""


class ConstructCircuitDAG:
    """Build a DAG representation of a compiled quantum program using the director pattern.

    This tool traverses an xDSL module and constructs a Directed Acyclic Graph (DAG)
    of it's quantum program using an injected DAGBuilder instance. This tool does not
    mutate the xDSL module.

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
        self._cluster_stack: list[ClusterEntry] = []

        # Create a map of wire to node uid
        # Keys represent static (int) or dynamic wires (_WireKind)
        # Values represent the set of all node uids that are on that wire.
        self._wire_to_node_uids: dict[_WireKind | int, set[str]] = defaultdict(set)

        # Track which node UIDs are dynamic
        self._dynamic_node_uids: set[str] = set()

        # Record last seen cluster type
        # as context for how to connect certain nodes
        self._last_cluster_entry: ClusterEntry | None = None

    def _reset(self) -> None:
        """Resets the instance."""
        self._wire_to_node_uids: dict[_WireKind | int, set[str]] = defaultdict(set)
        self._dynamic_node_uids: set[str] = set()
        self._cluster_stack: list[ClusterEntry] = []
        self._last_cluster_entry: ClusterEntry | None = None

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
        self._visualize_operation(operation)

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

    @singledispatchmethod
    def _visualize_operation(self, op: Operation) -> None:
        # NOTE: Currently only visualizing "quantum" operations
        if op.dialect_name() not in _SUPPORTED_DIALECTS:
            return
        if not isinstance(op, _SKIPPED_OPS):
            _ERROR_MSG = f"Visualization for operation '{op.name}' is currently not supported."
            raise VisualizationError(_ERROR_MSG)

    @_visualize_operation.register
    def _gate_op(self, op: quantum.GateOp) -> None:
        """Generic handler for unitary gates."""
        # Create PennyLane instance
        qp_op: Operator = xdsl_to_qp_op(op)

        # Add node to current cluster
        node_uid = self.dag_builder.add_node(
            label=get_label(qp_op),
            cluster_uid=self._cluster_stack[-1].uid,
            # NOTE: "record" allows us to use ports
            # (https://graphviz.org/doc/info/shapes.html#record)
            shape="record",
        )

        # Unlike standard gates, GlobalPhase does not have data dependencies
        # (it does not act on specific wires). Consequently, it is rendered as
        # a disjoint or 'floating' node within the current cluster to reflect
        # that it has no strict ordering requirements relative to other
        # quantum operations.
        if len(qp_op.wires) != 0:
            self._connect(qp_op.wires, node_uid)

    @_visualize_operation.register
    def _projective_measure_op(self, op: quantum.MeasureOp) -> None:
        """Handler for the single-qubit projective measurement operation."""

        # Create PennyLane instance
        meas: Operator = xdsl_to_qp_measurement(op)

        # Add node to current cluster
        node_uid = self.dag_builder.add_node(
            label=get_label(meas),
            cluster_uid=self._cluster_stack[-1].uid,
            # NOTE: "record" allows us to use ports
            # (https://graphviz.org/doc/info/shapes.html#record)
            shape="record",
        )

        self._connect(meas.wires, node_uid)

    @_visit_operation.register
    def _ppr(self, op: pbc.PPRotationOp | pbc.PPRotationArbitraryOp) -> None:
        """Handler for the PPR operation."""

        # Create label
        wires = ssa_to_qp_wires(op)
        wires_str = f"[{', '.join(map(str, wires))}]"
        pw = []
        for str_attr in op.pauli_product.data:
            pw.append(str(str_attr).replace('"', ""))
        pw = "".join(pw)

        attrs = {}
        if hasattr(op, "rotation_kind"):
            denominator = op.rotation_kind.value.data
            sign_str = "-" if denominator < 0 else ""
            angle = f"{sign_str}π/{abs(denominator)}"
            match abs(denominator):
                case 2:
                    attrs["fillcolor"] = "#D9D9D9"
                case 4:
                    attrs["fillcolor"] = "#F5BD70"
                case 8:
                    attrs["fillcolor"] = "#E3FFA1"
        else:
            angle = "φ"
            attrs["fillcolor"] = "#E3FFA1"
        label = f"<name> PPR-{pw} ({angle})|<wire> {wires_str}"

        # Add node to current cluster
        node_uid = self.dag_builder.add_node(
            label=label,
            cluster_uid=self._cluster_stack[-1].uid,
            # NOTE: "record" allows us to use ports
            # (https://graphviz.org/doc/info/shapes.html#record)
            shape="record",
            **attrs,
        )

        self._connect(wires, node_uid)

    @_visit_operation.register
    def _ppm(self, op: pbc.PPMeasurementOp) -> None:
        """Handler for the PPM operation."""

        wires = ssa_to_qp_wires(op)
        if wires == []:
            wires_str = "all"
        else:
            wires_str = f"[{', '.join(map(str, wires))}]"
        pw = []
        for str_attr in op.pauli_product.data:
            pw.append(str(str_attr).replace('"', ""))
        pw = "".join(pw)

        # Add node to current cluster
        node_uid = self.dag_builder.add_node(
            label=f"<name> PPM-{pw}|<wire> {wires_str}",
            cluster_uid=self._cluster_stack[-1].uid,
            # NOTE: "record" allows us to use ports
            # (https://graphviz.org/doc/info/shapes.html#record)
            shape="record",
            fillcolor="#70B3F5",
        )

        self._connect(wires, node_uid)

    # =====================
    # QUANTUM MEASUREMENTS
    # =====================

    @_visualize_operation.register
    def _terminal_measurements(self, op: quantum.TerminalMeasurementOp) -> None:
        """Handler for all quantum measurement operations."""

        meas = None

        match op:
            case quantum.StateOp():
                meas = xdsl_to_qp_measurement(op)
                # NOTE: state can only handle all wires

            case quantum.ExpvalOp() | quantum.VarianceOp():
                obs_op = op.obs.owner
                meas = xdsl_to_qp_measurement(op, xdsl_to_qp_measurement(obs_op))

            case quantum.SampleOp() | quantum.ProbsOp():
                obs_op = op.obs.owner

                # TODO: This doesn't logically make sense, but quantum.compbasis
                # is obs_op and function below just pulls out the static wires
                wires = xdsl_to_qp_measurement(obs_op)
                meas = xdsl_to_qp_measurement(op, wires=None if wires == [] else wires)

            case _:
                return

        node_uid = self.dag_builder.add_node(
            label=get_label(meas),
            cluster_uid=self._cluster_stack[-1].uid,
            fillcolor="lightpink",
            color="lightpink3",
            # NOTE: "record" allows us to use ports
            # (https://graphviz.org/doc/info/shapes.html#record)
            shape="record",
        )

        if not meas.wires:
            # wires = [] means connect to all active wires
            if _WireKind.DEVICE in self._wire_to_node_uids and len(self._wire_to_node_uids) == 1:
                # Case 1: Only the device node in past history
                all_prev_uids = self._wire_to_node_uids[_WireKind.DEVICE]
            else:
                # Case 2: Wire map is _WireKind.DEVICE + other stuff
                device_node_uid: set[str] = self._wire_to_node_uids.get(_WireKind.DEVICE, set())
                all_active = set().union(*self._wire_to_node_uids.values()) - device_node_uid

                # If we just exited a branching cluster (and are not in a nested one currently)
                # We need to connect to everything seen so far as all branches are a possibility.
                if self._exited_branching_cluster:
                    all_prev_uids = all_active
                else:
                    # Otherwise, just connect to static nodes as they block dynamic
                    # node connections
                    static_nodes = all_active - self._dynamic_node_uids
                    all_prev_uids = static_nodes or all_active
            for p_uid in all_prev_uids:
                self.dag_builder.add_edge(p_uid, node_uid, style="dashed", color="lightpink3")
        else:
            self._connect(meas.wires, node_uid, is_terminal_measurement=True, color="lightpink3")

    # =============
    # ADJOINT
    # =============

    @_visit_operation.register
    def _adjoint(self, operation: quantum.AdjointOp) -> None:
        """Handle a PennyLane adjoint operation."""

        cluster_uid = self.dag_builder.add_cluster(
            label="adjoint",
            labeljust="l",
            cluster_uid=self._cluster_stack[-1].uid,
        )
        self._cluster_stack.append(ClusterEntry(uid=cluster_uid, kind=_ClusterKind.ADJOINT))

        self._visit_region(operation.regions[0])

        self._cluster_stack.pop()

    # =============
    # CONTROL FLOW
    # =============

    @_visit_operation.register
    def _for_op(self, operation: scf.ForOp) -> None:
        """Handle an xDSL ForOp operation."""

        cluster_uid = self.dag_builder.add_cluster(
            label="for loop",
            labeljust="l",
            cluster_uid=self._cluster_stack[-1].uid,
        )
        self._cluster_stack.append(ClusterEntry(uid=cluster_uid, kind=_ClusterKind.FOR_LOOP))

        self._visit_region(operation.regions[0])

        self._cluster_stack.pop()

    @_visit_operation.register
    def _while_op(self, operation: scf.WhileOp) -> None:
        """Handle an xDSL WhileOp operation."""
        cluster_uid = self.dag_builder.add_cluster(
            label="while loop",
            labeljust="l",
            cluster_uid=self._cluster_stack[-1].uid,
        )
        self._cluster_stack.append(ClusterEntry(uid=cluster_uid, kind=_ClusterKind.WHILE_LOOP))

        for region in operation.regions:
            self._visit_region(region)

        self._cluster_stack.pop()

    @_visit_operation.register
    def _if_op(self, operation: scf.IfOp):
        """Handles the scf.IfOp operation."""

        # Create cluster for IfOp
        cluster_uid = self.dag_builder.add_cluster(
            label="conditional",
            labeljust="l",
            cluster_uid=self._cluster_stack[-1].uid,
        )
        self._cluster_stack.append(ClusterEntry(uid=cluster_uid, kind=_ClusterKind.CONDITIONAL))

        # Save wires state before all of the branches
        wire_map_before = deepcopy(self._wire_to_node_uids)

        region_wire_maps: list[dict[_WireKind | int, set[str]]] = []

        # Loop through each branch and visualize as a cluster
        flattened_if_op: list[Region] = _flatten_if_op(operation)
        num_regions = len(flattened_if_op)
        for i, region in enumerate(flattened_if_op):
            cluster_label = "elif"
            if i == 0:
                cluster_label = "if"
            elif i == num_regions - 1:
                cluster_label = "else"

            cluster_uid = self.dag_builder.add_cluster(
                label=cluster_label,
                labeljust="l",
                style="dashed",
                cluster_uid=self._cluster_stack[-1].uid,
            )
            self._cluster_stack.append(ClusterEntry(uid=cluster_uid, kind=_ClusterKind.BRANCH))

            # Make fresh wire map before going into region
            self._wire_to_node_uids = deepcopy(wire_map_before)

            # Go recursively into the branch to process internals
            self._visit_region(region)

            # Update branch wire maps
            if self._wire_to_node_uids != wire_map_before:
                # If the dynamic wire seen in this branch is different from the
                # one seen before the conditional *and* we have other static wires
                # clear the dyn_wires as the conditional becomes a blocking cluster.
                #
                # For example,
                #
                #    qp.H(x)
                #    if x == 2:
                #        qp.Y(x)
                #        qp.X(0)
                #    qp.Z(x)
                #
                before_dyn_node_uid: set[str] = wire_map_before.get(_WireKind.DYNAMIC, set())
                current_dyn_node_uid: set[str] = self._wire_to_node_uids[_WireKind.DYNAMIC]
                if current_dyn_node_uid != before_dyn_node_uid and len(self._wire_to_node_uids) > 1:
                    self._wire_to_node_uids[_WireKind.DYNAMIC] = set()
                region_wire_maps.append(self._wire_to_node_uids)

            # Pop branch cluster after processing to ensure
            # logical branches are treated as 'parallel'
            self._cluster_stack.pop()

        # Update affected wires with specific wires seen during branches
        affected_wires: set[int | _WireKind] = set()
        for region_wire_map in region_wire_maps:
            affected_wires.update(region_wire_map.keys())

        # Update state to be the union of all branch wire maps
        final_wire_map: dict[int | _WireKind, set[str]] = defaultdict(set)
        for wire in affected_wires:
            all_nodes: set[str] = set()
            for region_wire_map in region_wire_maps:
                all_nodes.update(region_wire_map.get(wire, set()))
            final_wire_map[wire] = all_nodes

        # If new dynamic wires are encountered during the conditional
        # the old ones from before are useless
        before_dyn_node_uid: set[str] = wire_map_before.get(_WireKind.DYNAMIC, set())
        current_dyn_node_uid: set[str] = final_wire_map[_WireKind.DYNAMIC]
        if before_dyn_node_uid and before_dyn_node_uid < current_dyn_node_uid:
            final_wire_map[_WireKind.DYNAMIC] -= before_dyn_node_uid

        # If we went through a single conditional and no dynamic wires were
        # encountered, clear the dynamic wires from before as the cluster should be
        # blocking
        if (
            before_dyn_node_uid == current_dyn_node_uid
            and sum(1 for s in self._cluster_stack if self._is_branching_cluster(s)) == 1
        ):
            final_wire_map[_WireKind.DYNAMIC] = set()

        # Pop IfOp cluster before leaving this handler
        self._last_cluster_entry = self._cluster_stack.pop()

        self._wire_to_node_uids = final_wire_map

    # ============
    # DEVICE NODE
    # ============

    @_visualize_operation.register
    def _device_init(self, operation: quantum.DeviceInitOp) -> None:
        """Handles the initialization of a quantum device."""
        node_uid = self.dag_builder.add_node(
            label=operation.device_name.data,
            cluster_uid=self._cluster_stack[-1].uid,
            fillcolor="grey",
            color="black",
            penwidth=2,
        )
        self._wire_to_node_uids[_WireKind.DEVICE].add(node_uid)

    # =======================
    # FuncOp NESTING UTILITY
    # =======================

    @_visit_operation.register
    def _func_op(self, operation: func.FuncOp) -> None:
        """Visit a FuncOp Operation."""

        if not operation.regions[0].blocks:
            _ERROR_MSG = (
                "Calls to functions without a definition are not yet compatible with 'draw_graph'. "
                f"Found external function call to {operation.sym_name.data}."
            )
            raise VisualizationError(_ERROR_MSG)

        label: str = (
            "qjit" if operation.sym_name.data.startswith("jit_") else operation.sym_name.data
        )

        # Create cluster representing the func
        parent_cluster_uid = None if not self._cluster_stack else self._cluster_stack[-1].uid
        cluster_uid = self.dag_builder.add_cluster(
            label=label,
            cluster_uid=parent_cluster_uid,
        )
        self._cluster_stack.append(ClusterEntry(uid=cluster_uid, kind=_ClusterKind.FUNC))

        self._visit_block(operation.regions[0].blocks[0])

    # pylint: disable=unused-argument
    @_visit_operation.register
    def _func_return(self, operation: func.ReturnOp) -> None:
        """Handle func.return to exit FuncOp's cluster scope."""

        # NOTE: Skip first cluster as it is the "base" of the graph diagram.
        # In our case, it is the `qjit` bounding box.
        if len(self._cluster_stack) > 1:
            # If we hit a func.return operation we know we are leaving
            # the FuncOp's scope and so we can pop the ID off the stack.
            self._cluster_stack.pop()

        # Clear seen wires as we are exiting a FuncOp (qnode)
        self._wire_to_node_uids = defaultdict(set)

    # =======================
    # NODE CONNECTIVITY
    # =======================

    def _connect(
        self, wires: Sequence, node_uid: str, is_terminal_measurement: bool = False, **edge_attrs
    ):
        """
        Connects a new node to its previous nodes in the DAG and updates the wire mapping in place.
        """

        # Record if it's a dynamic node for easy look-up
        is_dynamic = any(not isinstance(wire, int) for wire in wires) or len(wires) == 0
        if is_dynamic:
            self._dynamic_node_uids.add(node_uid)

        # Get all predecessor nodes
        prev_uids: set[str] = self._get_previous_uids(wires, is_dynamic)

        # Connect to all predecessors
        style = "dashed" if is_dynamic else "solid"
        for p_uid in prev_uids:
            self.dag_builder.add_edge(p_uid, node_uid, style=style, **edge_attrs)

        # Update wire mappings for future nodes
        if is_terminal_measurement:
            # No need to update wire mappings for MPs as they are terminal
            return

        self._update_wire_mapping(wires, node_uid, is_dynamic)

    def _get_previous_uids(self, wires: Sequence, is_dynamic: bool) -> set[str]:
        """Helper function to get the set of previous node uids."""

        prev_uids: set[str] = set()

        #######################
        ## PROCESS DYNAMIC NODES
        #######################

        if is_dynamic:
            if _WireKind.DEVICE in self._wire_to_node_uids:
                # Pop device node as this dynamic node will always superceed it
                device_node_uid: set[str] = self._wire_to_node_uids.pop(_WireKind.DEVICE)
                if not self._wire_to_node_uids:
                    return device_node_uid

            all_active = set().union(*self._wire_to_node_uids.values())

            # If we just exited a branching cluster (and are not in a nested one currently)
            # We need to connect to everything seen so far as all branches are a possibility.
            if self._exited_branching_cluster:
                return all_active

            # Otherwise, just connect to static nodes as they block dynamic
            # node connections
            static_nodes = all_active - self._dynamic_node_uids
            return static_nodes if static_nodes else all_active

        #######################
        ## PROCESS STATIC NODES
        #######################

        # Get all nodes seen on these static wires
        for wire in wires:
            prev_uids.update(self._wire_to_node_uids.get(wire, set()))

        # First time seeing this static wire
        if not prev_uids:
            # First time seeing this wire and no device node,
            # connect to the last seen dynamic node
            if _WireKind.DYNAMIC in self._wire_to_node_uids:
                prev_uids.update(self._wire_to_node_uids[_WireKind.DYNAMIC])

            # If no dynamic wire has been seen yet, connect to the device
            elif _WireKind.DEVICE in self._wire_to_node_uids:
                prev_uids.update(self._wire_to_node_uids[_WireKind.DEVICE])

        if (
            _WireKind.DYNAMIC in self._wire_to_node_uids
            and self._exited_branching_cluster
            and not self._inside_branch
        ):

            # Wire map contains both a dynamic wire and nodes on the static wires.
            # Only connect to dynamic wire if we just came from a condition.
            #
            # For example,
            #
            # if x:
            #   qp.X(0)
            # else:
            #   qp.Y(dyn)
            # qp.Z(0)
            #
            # We should have both X and Y connecting to the Z.

            # Also required for situations like,
            #
            #    qp.H(x)
            #    qp.S(0)
            #    if x == 3:
            #        if x == 2:
            #            qp.H(0)
            #    else:
            #        qp.RX(0,0)
            #
            # We don't want the RX in the final else condition to connect to the H(x)

            prev_uids.update(self._wire_to_node_uids[_WireKind.DYNAMIC])

        return prev_uids

    def _update_wire_mapping(self, wires: Sequence, node_uid: str, is_dynamic: bool) -> None:
        """Updates the wire mapping accordingly."""

        if is_dynamic:
            # Update last seen dynamic wire
            self._wire_to_node_uids.clear()
            self._wire_to_node_uids[_WireKind.DYNAMIC] = {node_uid}
        else:
            # Standard update for static wires
            for wire in wires:
                self._wire_to_node_uids[wire] = {node_uid}

            # If we just exited a branching cluster, update to have
            # no dynamic wires as the branching cluster itself acts as
            # a dynamic barrier
            if self._is_branching_cluster(self._last_cluster_entry):
                self._wire_to_node_uids[_WireKind.DYNAMIC] = set()

    def _is_branching_cluster(self, cluster: ClusterEntry | None) -> bool:
        """
        Whether or not the cluster is a cluster that results in
        many branches (e.g. conditionals -> if/elif/else)
        """
        if cluster is None:
            return False
        return cluster.kind == _ClusterKind.CONDITIONAL

    @property
    def _exited_branching_cluster(self) -> bool:
        """
        Check if we just exited a branching cluster
        and are not currently in a nested cluster.
        """
        inside_branching_cluster = any(self._is_branching_cluster(s) for s in self._cluster_stack)
        return self._is_branching_cluster(self._last_cluster_entry) and not inside_branching_cluster

    @property
    def _inside_branch(self) -> bool:
        """
        Check to see if we're inside of a branch of a branching cluster.
        """
        return self._cluster_stack[-1].kind == _ClusterKind.BRANCH if self._cluster_stack else False


def _flatten_if_op(operation: scf.IfOp) -> list[Region]:
    """Recursively flattens a nested IfOp (if/elif/else chains)."""

    then_region, else_region = operation.regions

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
    base_op = getattr(op, "base", op)

    # If Adjoint(GlobalPhase) or GlobalPhase, wires will be []
    # and so we don't need a port node. Controlled(GlobalPhase)
    # will contain control wires that need to be visualized on the
    # "wires" port.
    if isinstance(base_op, GlobalPhase) and len(op.wires) == 0:
        return str(op.name)

    wires = list(op.wires.labels)
    if not wires:
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

    # pylint: disable=protected-access
    base_name = meas._shortname

    match meas:
        case ExpectationMP() | VarianceMP() | ProbabilityMP():
            if meas.obs is not None:
                obs_name = meas.obs.name
                base_name = f"{base_name}({obs_name})"

        case _:
            pass

    return f"<name> {base_name}|<wire> {wires_str}"
