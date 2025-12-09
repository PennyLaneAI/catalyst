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

from functools import singledispatch, singledispatchmethod

from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from xdsl.dialects import builtin, func, scf
from xdsl.ir import Block, Operation, Region, SSAValue

from catalyst.python_interface.dialects import quantum
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

        # Use counter internally for UID
        self._node_uid_counter: int = 0
        self._cluster_uid_counter: int = 0

    def _reset(self) -> None:
        """Resets the instance."""
        self._cluster_uid_stack: list[str] = []
        self._node_uid_counter: int = 0
        self._cluster_uid_counter: int = 0

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
        node_uid = f"node{self._node_uid_counter}"
        self.dag_builder.add_node(
            uid=node_uid,
            label=get_label(qml_op),
            cluster_uid=self._cluster_uid_stack[-1],
            # NOTE: "record" allows us to use ports (https://graphviz.org/doc/info/shapes.html#record)
            shape="record",
        )
        self._node_uid_counter += 1

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

    # =====================
    # QUANTUM MEASUREMENTS
    # =====================

    @_visit_operation.register
    def _state_op(self, op: quantum.StateOp) -> None:
        """Handler for the terminal state measurement operation."""

        # Create PennyLane instance
        meas = xdsl_to_qml_measurement(op)

        # Add node to current cluster
        node_uid = f"node{self._node_uid_counter}"
        self.dag_builder.add_node(
            uid=node_uid,
            label=get_label(meas),
            cluster_uid=self._cluster_uid_stack[-1],
            fillcolor="lightpink",
            color="lightpink3",
        )
        self._node_uid_counter += 1

    @_visit_operation.register
    def _statistical_measurement_ops(
        self,
        op: quantum.ExpvalOp | quantum.VarianceOp,
    ) -> None:
        """Handler for statistical measurement operations."""

        # Create PennyLane instance
        obs_op = op.obs.owner
        meas = xdsl_to_qml_measurement(op, xdsl_to_qml_measurement(obs_op))

        # Add node to current cluster
        node_uid = f"node{self._node_uid_counter}"
        self.dag_builder.add_node(
            uid=node_uid,
            label=get_label(meas),
            cluster_uid=self._cluster_uid_stack[-1],
            fillcolor="lightpink",
            color="lightpink3",
        )
        self._node_uid_counter += 1

    @_visit_operation.register
    def _visit_sample_and_probs_ops(
        self,
        op: quantum.SampleOp | quantum.ProbsOp,
    ) -> None:
        """Handler for sample operations."""

        # Create PennyLane instance
        obs_op = op.obs.owner

        # TODO: This doesn't logically make sense, but quantum.compbasis
        # is obs_op and function below just pulls out the static wires
        wires = xdsl_to_qml_measurement(obs_op)
        meas = xdsl_to_qml_measurement(op, wires=None if wires == [] else wires)

        # Add node to current cluster
        node_uid = f"node{self._node_uid_counter}"
        self.dag_builder.add_node(
            uid=node_uid,
            label=get_label(meas),
            cluster_uid=self._cluster_uid_stack[-1],
            fillcolor="lightpink",
            color="lightpink3",
        )
        self._node_uid_counter += 1

    # =============
    # CONTROL FLOW
    # =============

    @_visit_operation.register
    def _for_op(self, operation: scf.ForOp) -> None:
        """Handle an xDSL ForOp operation."""

        uid = f"cluster{self._cluster_uid_counter}"
        self.dag_builder.add_cluster(
            uid,
            node_label="for loop",
            label="",
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
            node_label="while loop",
            label="",
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
        flattened_if_op: list[tuple[SSAValue | None, Region]] = _flatten_if_op(operation)

        uid = f"cluster{self._cluster_uid_counter}"
        self.dag_builder.add_cluster(
            uid,
            node_label="",
            label="conditional",
            labeljust="l",
            cluster_uid=self._cluster_uid_stack[-1],
        )
        self._cluster_uid_stack.append(uid)
        self._cluster_uid_counter += 1

        # Loop through each branch and visualize as a cluster
        num_regions = len(flattened_if_op)
        for i, (condition_ssa, region) in enumerate(flattened_if_op):
            node_label = "elif"
            if i == 0:
                node_label = "if"
            elif i == num_regions - 1:
                node_label = "else"

            uid = f"cluster{self._cluster_uid_counter}"
            self.dag_builder.add_cluster(
                uid,
                node_label=node_label,
                label="",
                style="dashed",
                penwidth=1,
                cluster_uid=self._cluster_uid_stack[-1],
            )
            self._cluster_uid_stack.append(uid)
            self._cluster_uid_counter += 1

            # Go recursively into the branch to process internals
            self._visit_region(region)

            # Pop branch cluster after processing to ensure
            # logical branches are treated as 'parallel'
            self._cluster_uid_stack.pop()

        # Pop IfOp cluster before leaving this handler
        self._cluster_uid_stack.pop()

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
        parent_cluster_uid = None if self._cluster_uid_stack == [] else self._cluster_uid_stack[-1]
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


@singledispatch
def get_label(op: Operator | MeasurementProcess) -> str:
    """Gets the appropriate label for a PennyLane object."""
    return str(op)


@get_label.register
def _operator(op: Operator) -> str:
    """Returns the appropriate label for an xDSL operation."""
    wires = list(op.wires.labels)
    if wires == []:
        wires_str = "all"
    else:
        wires_str = f"[{', '.join(map(str, wires))}]"
    return f"<name> {op.name}|<wire> {wires_str}"


@get_label.register
def _mp(mp: MeasurementProcess) -> str:
    """Returns the appropriate label for an xDSL operation."""
    return str(mp)
