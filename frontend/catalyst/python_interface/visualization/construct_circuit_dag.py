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
from typing import Any

from xdsl.dialects import builtin, scf
from xdsl.ir import Block, Operation, Region

from catalyst.python_interface.dialects import quantum
from catalyst.python_interface.visualization.dag_builder import DAGBuilder
from catalyst.python_interface.visualization.xdsl_conversion import (
    xdsl_to_qml_measurement,
    xdsl_to_qml_op,
)


class ConstructCircuitDAG:
    """A tool that analyzes an xDSL module and constructs a Directed Acyclic Graph (DAG)
    using an injected DAGBuilder instance. This tool does not mutate the xDSL module."""

    def __init__(self, dag_builder: DAGBuilder) -> None:
        """Initialize the analysis pass by injecting the DAG builder dependency.

        Args:
            dag_builder (DAGBuilder): The concrete builder instance used for graph construction.
        """
        self.dag_builder: DAGBuilder = dag_builder

        # Record clusters seen as a stack
        # beginning with the base graph (None)
        self._cluster_stack: list[str | None] = [None]

    def _reset(self) -> None:
        """Resets the instance."""
        self._cluster_stack: list[str | None] = [None]

    # =================================
    # 1. CORE DISPATCH AND ENTRY POINT
    # =================================

    @singledispatchmethod
    def visit(self, op: Any) -> None:
        """Central dispatch method (Visitor Pattern). Routes the operation 'op'
        to the specialized handler registered for its type."""
        pass

    def construct(self, module: builtin.ModuleOp) -> None:
        """Constructs the DAG from the module."""
        self._reset()

        for op in module.ops:
            self.visit(op)

    # =======================
    # 2. HIERARCHY TRAVERSAL
    # =======================
    # These methods navigate the recursive IR hierarchy (Op -> Region -> Block -> Op).

    @visit.register
    def _operation(self, operation: Operation) -> None:
        """Visit an xDSL Operation."""
        for region in operation.regions:
            self.visit(region)

    @visit.register
    def _region(self, region: Region) -> None:
        """Visit an xDSL Region operation."""
        for block in region.blocks:
            self.visit(block)

    @visit.register
    def _block(self, block: Block) -> None:
        """Visit an xDSL Block operation, dispatching handling for each contained Operation."""
        for op in block.ops:
            self.visit(op)

    # ======================================
    # 3. QUANTUM GATE & STATE PREP HANDLERS
    # ======================================
    # Handlers for operations that apply unitary transformations or set-up the quantum state.

    @visit.register
    def _unitary_and_state_prep(
        self,
        op: quantum.CustomOp
        | quantum.GlobalPhaseOp
        | quantum.QubitUnitaryOp
        | quantum.SetStateOp
        | quantum.MultiRZOp
        | quantum.SetBasisStateOp,
    ) -> None:
        """Generic handler for unitary gates and quantum state preparation operations."""

        qml_op = xdsl_to_qml_op(op)
        # Build node on graph
        self.dag_builder.add_node(
            node_id=f"node_{id(op)}",
            node_label=str(qml_op),
            parent_graph_id=self._cluster_stack[-1],
        )

    # =============================================
    # 4. QUANTUM MEASUREMENT HANDLERS
    # =============================================

    @visit.register
    def _state_op(self, op: quantum.StateOp) -> None:
        """Handler for the terminal state measurement operation."""

        meas = xdsl_to_qml_measurement(op)
        # Build node on graph
        self.dag_builder.add_node(
            node_id=f"node_{id(op)}",
            node_label=str(meas),
            parent_graph_id=self._cluster_stack[-1],
        )

    @visit.register
    def _statistical_measurement_ops(
        self,
        op: quantum.ExpvalOp | quantum.VarianceOp | quantum.ProbsOp | quantum.SampleOp,
    ) -> None:
        """Handler for statistical measurement operations."""

        obs_op = op.obs.owner
        meas = xdsl_to_qml_measurement(op, xdsl_to_qml_measurement(obs_op))
        # Build node on graph
        self.dag_builder.add_node(
            node_id=f"node_{id(op)}",
            node_label=str(meas),
            parent_graph_id=self._cluster_stack[-1],
        )

    @visit.register
    def _projective_measure_op(self, op: quantum.MeasureOp) -> None:
        """Handler for the single-qubit projective measurement operation."""

        meas = xdsl_to_qml_measurement(op)
        # Build node on graph
        self.dag_builder.add_node(
            node_id=f"node_{id(op)}",
            node_label=str(meas),
            parent_graph_id=self._cluster_stack[-1],
        )

    # =========================
    # 5. CONTROL FLOW HANDLERS
    # =========================

    @visit.register
    def _for_op(self, op: scf.ForOp) -> None:
        """Handle an xDSL ForOp operation."""
        pass

    @visit.register
    def _while_op(self, op: scf.WhileOp) -> None:
        """Handle an xDSL WhileOp operation."""
        pass

    @visit.register
    def _if_op(self, op: scf.IfOp) -> None:
        """Handle an xDSL IfOp operation."""
        pass
