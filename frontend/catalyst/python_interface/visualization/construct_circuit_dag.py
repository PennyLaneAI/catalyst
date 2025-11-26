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
from typing import Any

from xdsl.dialects import builtin
from xdsl.ir import Block, Operation, Region

from catalyst.python_interface.dialects import quantum
from catalyst.python_interface.visualization.dag_builder import DAGBuilder
from catalyst.python_interface.visualization.xdsl_conversion import (
    xdsl_to_qml_measurement,
    xdsl_to_qml_op,
)


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

        # Record clusters seen as a stack
        # beginning with the base graph (None)
        self._cluster_stack: list[str | None] = [None]

    def _reset(self) -> None:
        """Resets the instance."""
        self._cluster_stack: list[str | None] = [None]

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

        qml_op = xdsl_to_qml_op(op)
        # Build node on graph
        self.dag_builder.add_node(
            id=f"node_{id(op)}",
            label=str(qml_op),
            cluster_id=self._cluster_stack[-1],
        )

    # =====================
    # QUANTUM MEASUREMENTS
    # =====================

    @_visit_operation.register
    def _state_op(self, op: quantum.StateOp) -> None:
        """Handler for the terminal state measurement operation."""

        meas = xdsl_to_qml_measurement(op)
        # Build node on graph
        self.dag_builder.add_node(
            id=f"node_{id(op)}",
            label=str(meas),
            cluster_id=self._cluster_stack[-1],
        )

    @_visit_operation.register
    def _statistical_measurement_ops(
        self,
        op: quantum.ExpvalOp | quantum.VarianceOp | quantum.ProbsOp | quantum.SampleOp,
    ) -> None:
        """Handler for statistical measurement operations."""

        obs_op = op.obs.owner
        meas = xdsl_to_qml_measurement(op, xdsl_to_qml_measurement(obs_op))
        # Build node on graph
        self.dag_builder.add_node(
            id=f"node_{id(op)}",
            label=str(meas),
            cluster_id=self._cluster_stack[-1],
        )

    @_visit_operation.register
    def _projective_measure_op(self, op: quantum.MeasureOp) -> None:
        """Handler for the single-qubit projective measurement operation."""

        meas = xdsl_to_qml_measurement(op)
        # Build node on graph
        self.dag_builder.add_node(
            id=f"node_{id(op)}",
            label=str(meas),
            cluster_id=self._cluster_stack[-1],
        )
