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

from catalyst.python_interface.dialects import qec, quantum
from catalyst.python_interface.inspection.dag_builder import DAGBuilder
from catalyst.python_interface.inspection.xdsl_conversion import (
    ssa_to_qml_wires,
    xdsl_to_qml_measurement,
    xdsl_to_qml_op,
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
_SKIPPED_QEC_OPS = (
    qec.FabricateOp,
    qec.LayerOp,
    qec.PrepareStateOp,
    qec.SelectPPMeasurementOp,
    qec.YieldOp,
)
# Any MBQC operation encountered will raise a
# VisualizationError
_SKIPPED_MBQC_OPS = ()


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
        self._cluster_uid_stack: list[str] = []

        # Create a map of wire to node uid
        # Keys represent static (int) or dynamic wires (str)
        # Values represent the set of all node uids that are on that wire.
        self._wire_to_node_uids: dict[str | int, set[str]] = defaultdict(set)

        # Track which node UIDs are dynamic
        self._dynamic_node_uids: set[str] = set()

        # Use counter internally for UID
        self._node_uid_counter: int = 0
        self._cluster_uid_counter: int = 0

        # Record last seen cluster UID
        # as context for how to connect certain nodes
        self._last_cluster_uid: str = ""

    def _reset(self) -> None:
        """Resets the instance."""
        self._cluster_uid_stack: list[str] = []
        self._node_uid_counter: int = 0
        self._cluster_uid_counter: int = 0
        self._wire_to_node_uids: dict[str | int, set[str]] = defaultdict(set)
        self._dynamic_node_uids: set[str] = set()
        self._last_cluster_uid: str = ""

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
        if op.dialect_name() not in {"quantum", "qec", "mbqc"}:
            return
        _SKIPPED_OPS = (*_SKIPPED_QUANTUM_OPS, *_SKIPPED_QEC_OPS, *_SKIPPED_MBQC_OPS)
        if not isinstance(op, _SKIPPED_OPS):
            raise VisualizationError(
                f"Visualization for operation '{op.name}' is currently not supported."
            )

    @_visualize_operation.register
    def _gate_op(
        self,
        op: (
            quantum.CustomOp
            | quantum.GlobalPhaseOp
            | quantum.QubitUnitaryOp
            | quantum.MultiRZOp
            | quantum.SetBasisStateOp
            | quantum.SetStateOp
            | quantum.PauliRotOp
        ),
    ) -> None:
        """Generic handler for unitary gates."""
        # Create PennyLane instance
        qml_op: Operator = xdsl_to_qml_op(op)

        # Add node to current cluster
        node_uid = f"node{self._node_uid_counter}"
        self.dag_builder.add_node(
            uid=node_uid,
            label=get_label(qml_op),
            cluster_uid=self._cluster_uid_stack[-1],
            # NOTE: "record" allows us to use ports
            # (https://graphviz.org/doc/info/shapes.html#record)
            shape="record",
        )
        self._node_uid_counter += 1

        # Unlike standard gates, GlobalPhase does not have data dependencies
        # (it does not act on specific wires). Consequently, it is rendered as
        # a disjoint or 'floating' node within the current cluster to reflect
        # that it has no strict ordering requirements relative to other
        # quantum operations.
        if len(qml_op.wires) != 0:
            self._connect(qml_op.wires, node_uid)

    @_visualize_operation.register
    def _projective_measure_op(self, op: quantum.MeasureOp) -> None:
        """Handler for the single-qubit projective measurement operation."""

        # Create PennyLane instance
        meas: Operator = xdsl_to_qml_measurement(op)

        # Add node to current cluster
        node_uid = f"node{self._node_uid_counter}"
        self.dag_builder.add_node(
            uid=node_uid,
            label=get_label(meas),
            cluster_uid=self._cluster_uid_stack[-1],
            # NOTE: "record" allows us to use ports
            # (https://graphviz.org/doc/info/shapes.html#record)
            shape="record",
        )
        self._node_uid_counter += 1

        self._connect(meas.wires, node_uid)

    @_visit_operation.register
    def _ppr(self, op: qec.PPRotationOp | qec.PPRotationArbitraryOp) -> None:
        """Handler for the PPR operation."""

        # Create label
        wires = ssa_to_qml_wires(op)
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
        node_uid = f"node{self._node_uid_counter}"
        self.dag_builder.add_node(
            uid=node_uid,
            label=label,
            cluster_uid=self._cluster_uid_stack[-1],
            # NOTE: "record" allows us to use ports
            # (https://graphviz.org/doc/info/shapes.html#record)
            shape="record",
            **attrs,
        )
        self._node_uid_counter += 1

        self._connect(wires, node_uid)

    @_visit_operation.register
    def _ppm(self, op: qec.PPMeasurementOp) -> None:
        """Handler for the PPM operation."""

        wires = ssa_to_qml_wires(op)
        if wires == []:
            wires_str = "all"
        else:
            wires_str = f"[{', '.join(map(str, wires))}]"
        pw = []
        for str_attr in op.pauli_product.data:
            pw.append(str(str_attr).replace('"', ""))
        pw = "".join(pw)

        # Add node to current cluster
        node_uid = f"node{self._node_uid_counter}"
        self.dag_builder.add_node(
            uid=node_uid,
            label=f"<name> PPM-{pw}|<wire> {wires_str}",
            cluster_uid=self._cluster_uid_stack[-1],
            # NOTE: "record" allows us to use ports
            # (https://graphviz.org/doc/info/shapes.html#record)
            shape="record",
            fillcolor="#70B3F5",
        )
        self._node_uid_counter += 1

        self._connect(wires, node_uid)

    # =====================
    # QUANTUM MEASUREMENTS
    # =====================

    @_visualize_operation.register
    def _measurements(
        self,
        op: (
            quantum.StateOp
            | quantum.ExpvalOp
            | quantum.VarianceOp
            | quantum.SampleOp
            | quantum.ProbsOp
        ),
    ) -> None:
        """Handler for all quantum measurement operations."""

        meas = None

        match op:
            case quantum.StateOp():
                meas = xdsl_to_qml_measurement(op)
                # NOTE: state can only handle all wires

            case quantum.ExpvalOp() | quantum.VarianceOp():
                obs_op = op.obs.owner
                meas = xdsl_to_qml_measurement(op, xdsl_to_qml_measurement(obs_op))

            case quantum.SampleOp() | quantum.ProbsOp():
                obs_op = op.obs.owner

                # TODO: This doesn't logically make sense, but quantum.compbasis
                # is obs_op and function below just pulls out the static wires
                wires = xdsl_to_qml_measurement(obs_op)
                meas = xdsl_to_qml_measurement(op, wires=None if wires == [] else wires)

            case _:
                return

        node_uid = f"node{self._node_uid_counter}"
        self.dag_builder.add_node(
            uid=node_uid,
            label=get_label(meas),
            cluster_uid=self._cluster_uid_stack[-1],
            fillcolor="lightpink",
            color="lightpink3",
            # NOTE: "record" allows us to use ports
            # (https://graphviz.org/doc/info/shapes.html#record)
            shape="record",
        )
        self._node_uid_counter += 1

        if not meas.wires:
            # wires = [] means connect to all active wires
            if "device" in self._wire_to_node_uids and len(self._wire_to_node_uids) == 1:
                # Case 1: Only the device node in past history
                all_prev_uids = self._wire_to_node_uids["device"]
            else:
                # Case 2: Wire map is "device" + other stuff
                device_node_uid: set[str] = self._wire_to_node_uids.get("device", set())
                all_active = set().union(*self._wire_to_node_uids.values()) - device_node_uid

                # If we just exited a conditional (and are not in a nested one currently)
                # We need to connect to everything seen so far as all branches are a possibility.
                exited_conditional_cluster = "conditional" in self._last_cluster_uid and not any(
                    "conditional" in s for s in self._cluster_uid_stack
                )
                if exited_conditional_cluster:
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

        uid = f"cluster{self._cluster_uid_counter}"
        self.dag_builder.add_cluster(
            uid,
            label="adjoint",
            labeljust="l",
            cluster_uid=self._cluster_uid_stack[-1],
        )
        self._cluster_uid_stack.append(uid)
        self._cluster_uid_counter += 1

        self._visit_region(operation.regions[0])

        self._cluster_uid_stack.pop()

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

        # Create cluster for IfOp
        uid = f"conditional_cluster{self._cluster_uid_counter}"
        self.dag_builder.add_cluster(
            uid,
            label="conditional",
            labeljust="l",
            cluster_uid=self._cluster_uid_stack[-1],
        )
        self._cluster_uid_stack.append(uid)
        self._cluster_uid_counter += 1

        # Save wires state before all of the branches
        wire_map_before = deepcopy(self._wire_to_node_uids)

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
                #    qml.H(x)
                #    if x == 2:
                #        qml.Y(x)
                #        qml.X(0)
                #    qml.Z(x)
                #
                before_dyn_node_uid: set[str] = wire_map_before.get("dyn_wire", set())
                current_dyn_node_uid: set[str] = self._wire_to_node_uids["dyn_wire"]
                if current_dyn_node_uid != before_dyn_node_uid and len(self._wire_to_node_uids) > 1:
                    self._wire_to_node_uids["dyn_wire"] = set()
                region_wire_maps.append(self._wire_to_node_uids)

            # Pop branch cluster after processing to ensure
            # logical branches are treated as 'parallel'
            self._cluster_uid_stack.pop()

        # Update affected wires with specific wires seen during branches
        affected_wires: set[str | int] = set()
        for region_wire_map in region_wire_maps:
            affected_wires.update(region_wire_map.keys())

        # Update state to be the union of all branch wire maps
        final_wire_map: dict[str | int, set[str]] = defaultdict(set)
        for wire in affected_wires:
            all_nodes: set[str] = set()
            for region_wire_map in region_wire_maps:
                all_nodes.update(region_wire_map.get(wire, set()))
            final_wire_map[wire] = all_nodes

        # If new dynamic wires are encountered during the conditional
        # the old ones from before are useless
        before_dyn_node_uid: set[str] = wire_map_before.get("dyn_wire", set())
        current_dyn_node_uid: set[str] = final_wire_map["dyn_wire"]
        if before_dyn_node_uid and before_dyn_node_uid < current_dyn_node_uid:
            final_wire_map["dyn_wire"] -= before_dyn_node_uid

        # If we went through a single conditional and no dynamic wires were
        # encountered, clear the dynamic wires from before as the cluster should be
        # blocking
        if (
            before_dyn_node_uid == current_dyn_node_uid
            and sum(1 for s in self._cluster_uid_stack if "conditional" in s) == 1
        ):
            final_wire_map["dyn_wire"] = set()

        # Pop IfOp cluster before leaving this handler
        self._last_cluster_uid = self._cluster_uid_stack.pop()

        self._wire_to_node_uids = final_wire_map

    # ============
    # DEVICE NODE
    # ============

    @_visualize_operation.register
    def _device_init(self, operation: quantum.DeviceInitOp) -> None:
        """Handles the initialization of a quantum device."""
        node_uid = f"node{self._node_uid_counter}"
        self.dag_builder.add_node(
            node_uid,
            label=operation.device_name.data,
            cluster_uid=self._cluster_uid_stack[-1],
            fillcolor="grey",
            color="black",
            penwidth=2,
        )
        self._node_uid_counter += 1
        self._wire_to_node_uids["device"].add(node_uid)

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
        parent_cluster_uid = None if not self._cluster_uid_stack else self._cluster_uid_stack[-1]
        self.dag_builder.add_cluster(
            uid,
            label=label,
            cluster_uid=parent_cluster_uid,
        )
        self._cluster_uid_counter += 1
        self._cluster_uid_stack.append(uid)

        self._visit_block(operation.regions[0].blocks[0])

    # pylint: disable=unused-argument
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
        is_dynamic = any(not isinstance(wire, int) for wire in wires)
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
            if "device" in self._wire_to_node_uids:
                # Pop device node as this dynamic node will always superceed it
                device_node_uid: set[str] = self._wire_to_node_uids.pop("device")
                if not self._wire_to_node_uids:
                    return device_node_uid

            all_active = set().union(*self._wire_to_node_uids.values())

            # If we just exited a conditional (and are not in a nested one currently)
            # We need to connect to everything seen so far as all branches are a possibility.
            exited_conditional_cluster = "conditional" in self._last_cluster_uid and not any(
                "conditional" in s for s in self._cluster_uid_stack
            )
            if exited_conditional_cluster:
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
            if "dyn_wire" in self._wire_to_node_uids:
                prev_uids.update(self._wire_to_node_uids["dyn_wire"])

            # If no dynamic wire has been seen yet, connect to the device
            elif "device" in self._wire_to_node_uids:
                prev_uids.update(self._wire_to_node_uids["device"])

        # Wire map contains both a dynamic wire and nodes on the static wires.
        # Only connect to dynamic wire if we just came from a condition.
        #
        # For example,
        #
        # if x:
        #   qml.X(0)
        # else:
        #   qml.Y(dyn)
        # qml.Z(0)
        #
        # We should have both X and Y connecting to the Z.

        # To do this carefully, we need to check if we're in a cluster's final
        # else condition by looking two steps behind in the stack,
        # _cluster_uid_stack = [..., "conditional_cluster*", "cluster*"]
        # This is required if we have situations like,
        #
        #    qml.H(x)
        #    qml.S(0)
        #    if x == 3:
        #        if x == 2:
        #            qml.H(0)
        #    else:
        #        qml.RX(0,0)
        #
        # We don't want the RX in the final else condition to connect to the H(x)

        after_conditional_cluster = "conditional" in self._last_cluster_uid
        inside_final_else_condition = False
        if len(self._cluster_uid_stack) > 2:
            inside_final_else_condition = "conditional" in self._cluster_uid_stack[-2]
        if (
            "dyn_wire" in self._wire_to_node_uids
            and after_conditional_cluster
            and not inside_final_else_condition
        ):
            prev_uids.update(self._wire_to_node_uids["dyn_wire"])

        return prev_uids

    def _update_wire_mapping(self, wires: Sequence, node_uid: str, is_dynamic: bool) -> None:
        """Updates the wire mapping accordingly."""

        if is_dynamic:
            # Update last seen dynamic wire
            self._wire_to_node_uids.clear()
            self._wire_to_node_uids["dyn_wire"] = {node_uid}
        else:
            # Standard update for static wires
            for wire in wires:
                self._wire_to_node_uids[wire] = {node_uid}

            # If we just exited a conditional, update to have
            # no dynamic wires as the conditional cluster itself acts as
            # a dynamic barrier
            if "conditional" in self._last_cluster_uid:
                self._wire_to_node_uids["dyn_wire"] = set()


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
