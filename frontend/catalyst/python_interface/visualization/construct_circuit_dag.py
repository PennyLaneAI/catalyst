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
from xdsl.ir import Block, Region

from catalyst.python_interface.dialects import quantum
from catalyst.python_interface.visualization.dag_builder import DAGBuilder


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
        self._cluster_stack: list[str | None] = []

    # =================================
    # 1. CORE DISPATCH AND ENTRY POINT
    # =================================

    @singledispatchmethod
    def visit_op(self, op: Any) -> None:
        """Central dispatch method (Visitor Pattern). Routes the operation 'op'
        to the specialized handler registered for its type."""
        raise NotImplementedError(
            f"Dispatch not registered for operator of type {type(op)}"
        )

    def construct(self, module: builtin.ModuleOp) -> None:
        """Constructs the DAG from the module."""
        self._reset()

        for op in module.ops:
            self.visit_op(op)

    # =======================
    # 2. HIERARCHY TRAVERSAL
    # =======================
    # These methods navigate the recursive IR hierarchy (Op -> Region -> Block -> Op).

    @visit_op.register
    def visit_region(self, region: Region) -> None:
        """Visit an xDSL Region operation."""
        for block in region.blocks:
            self.visit_block(block)

    @visit_op.register
    def visit_block(self, block: Block) -> None:
        """Visit an xDSL Block operation, dispatching handling for each contained Operation."""
        for op in block.ops:
            self.visit_op(op)

    # ======================================
    # 3. QUANTUM GATE & STATE PREP HANDLERS
    # ======================================
    # Handlers for operations that apply unitary transformations or set-up the quantum state.

    @visit_op.register
    def _visit_unitary_and_state_prep(
        self,
        op: quantum.CustomOp,
    ) -> None:
        """Generic handler for unitary gates and quantum state preparation operations."""

        # Build node on graph
        self.dag_builder.add_node(
            node_id=f"node_{id(op)}",
            node_label=get_label(op),
        )

    # =============================================
    # 4. QUANTUM MEASUREMENT HANDLERS
    # =============================================

    @visit_op.register
    def _visit_state_op(self, op: quantum.StateOp) -> None:
        """Handler for the terminal state measurement operation."""

        # Build node on graph
        self.dag_builder.add_node(
            node_id=f"node_{id(op)}",
            node_label=get_label(op),
        )

    @visit_op.register
    def _visit_statistical_measurement_ops(
        self,
        op: quantum.ExpvalOp | quantum.VarianceOp | quantum.ProbsOp | quantum.SampleOp,
    ) -> None:
        """Handler for statistical measurement operations."""

        # Build node on graph
        self.dag_builder.add_node(
            node_id=f"node_{id(op)}",
            node_label=get_label(op),
        )

    @visit_op.register
    def _visit_projective_measure_op(self, op: quantum.MeasureOp) -> None:
        """Handler for the single-qubit projective measurement operation."""

        # Build node on graph
        self.dag_builder.add_node(
            node_id=f"node_{id(op)}",
            node_label=get_label(op),
        )

    # =========================
    # 5. CONTROL FLOW HANDLERS
    # =========================

    @visit_op.register
    def _visit_for_op(self, op: scf.ForOp) -> None:
        """Handle an xDSL ForOp operation."""
        pass

    @visit_op.register
    def _visit_while_op(self, op: scf.WhileOp) -> None:
        """Handle an xDSL WhileOp operation."""
        pass

    @visit_op.register
    def _visit_if_op(self, op: scf.IfOp) -> None:
        """Handle an xDSL IfOp operation."""
        pass


@singledispatch
def get_label(op: Any) -> str:
    """Gets a human readable label for a given xDSL operation.

    Returns:
        label (str): The appropriate label for a given xDSL operation. Defaults
            to the class name.
    """
    return type(op).__name__.replace("Op", "")


@get_label.register
def _get_custom_op_label(op: quantum.CustomOp) -> str:
    return op.gate_name.data


@get_label.register
def _get_state_op_label(op: quantum.StateOp) -> str:
    return op.name


@get_label.register
def _get_statistical_measurement_op_label(
    op: quantum.ExpvalOp | quantum.VarianceOp | quantum.ProbsOp | quantum.SampleOp,
) -> str:
    # e.g. op -> expval(Z(0))
    obs = op.obs
    mp = op.name.split(".")[-1]  # quantum.expval -> expval
    return f"{mp}({obs})"


@get_label.register
def _get_projective_measurement_op_label(op: quantum.MeasureOp) -> str:
    return op.name.split(".")[-1]  # quantum.measure -> measure
