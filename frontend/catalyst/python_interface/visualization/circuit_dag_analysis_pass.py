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

"""Contains the CircuitDAGAnalysisPass for generating a DAG from an xDSL module."""

from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any

import xdsl
from xdsl.dialects import builtin, func, scf
from xdsl.ir import Block, Region

from catalyst.python_interface.dialects import quantum
from catalyst.python_interface.visualization.dag_builder import DAGBuilder


class CircuitDAGAnalysisPass:
    """A Pass that analyzes an xDSL module and constructs a Directed Acyclic Graph (DAG)
    using an injected DAGBuilder instance. This is a non-mutating Analysis Pass."""

    def __init__(self, dag_builder: DAGBuilder) -> None:
        """Initialize the analysis pass by injecting the DAG builder dependency.

        Args:
            dag_builder (DAGBuilder): The concrete builder instance used for graph construction.
        """
        self.dag_builder: DAGBuilder = dag_builder

    # =================================
    # 1. CORE DISPATCH AND ENTRY POINT
    # =================================

    @singledispatchmethod
    def visit_op(self, op: Any) -> None:
        """Central dispatch method (Visitor Pattern). Routes the operation 'op'
        to the specialized handler registered for its type."""
        pass

    def run(self, module: builtin.ModuleOp) -> None:
        """Applies the analysis pass on the module."""
        for op in module.ops:
            self.visit_op(op)

    # =======================
    # 2. HIERARCHY TRAVERSAL
    # =======================
    # These methods navigate the recursive IR hierarchy (Op -> Region -> Block -> Op).

    @visit_op.register
    def visit_region(self, region: Region) -> None:
        """Visit an xDSL Region operation, delegating traversal to its Blocks."""
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
        op: (
            quantum.CustomOp
            | quantum.GlobalPhaseOp
            | quantum.QubitUnitaryOp
            | quantum.MultiRZOp
            | quantum.SetStateOp
            | quantum.SetBasisStateOp
        ),
    ) -> None:
        """Generic handler for unitary gates and quantum state preparation operations."""
        pass

    # =============================================
    # 4. QUANTUM MEASUREMENT & OBSERVABLE HANDLERS
    # =============================================

    @visit_op.register
    def _visit_state_op(self, op: quantum.StateOp) -> None:
        """Handler for the terminal StateOp."""
        pass

    @visit_op.register
    def _visit_statistical_measurement_ops(
        self,
        op: quantum.ExpvalOp | quantum.VarianceOp | quantum.ProbsOp | quantum.SampleOp,
    ) -> None:
        """Handler for statistical measurement operations."""
        pass

    @visit_op.register
    def _visit_projective_measure_op(self, op: quantum.MeasureOp) -> None:
        """Handler for the single-qubit projective MeasureOp."""
        pass

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
