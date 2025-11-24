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
"""Unit tests for the ConstructCircuitDAG utility."""

from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.usefixtures("requires_xdsl")

# pylint: disable=wrong-import-position
# This import needs to be after pytest in order to prevent ImportErrors
import pennylane as qml
from catalyst.python_interface.conversion import xdsl_from_qjit
from catalyst.python_interface.dialects.quantum import (
    CustomOp,
    QubitType,
)
from catalyst.python_interface.visualization.construct_circuit_dag import (
    ConstructCircuitDAG,
)
from catalyst.python_interface.visualization.dag_builder import DAGBuilder
from xdsl.dialects import test
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir.core import Block, Region


class FakeDAGBuilder(DAGBuilder):
    """
    A concrete implementation of DAGBuilder used ONLY for testing.
    It stores all graph manipulation calls in simple Python dictionaries
    for easy assertion of the final graph state.
    """

    def __init__(self):
        self._nodes = {}
        self._edges = []
        self._clusters = {}

    def add_node(self, id, label, cluster_id=None, **attrs) -> None:
        self._nodes[id] = {
            "id": id,
            "label": label,
            "parent_id": cluster_id,
            "attrs": attrs,
        }

    def add_edge(self, from_id: str, to_id: str, **attrs) -> None:
        self._edges.append(
            {
                "from": from_id,
                "to": to_id,
                "attrs": attrs,
            }
        )

    def add_cluster(
        self,
        id,
        node_label=None,
        cluster_id=None,
        **attrs,
    ) -> None:
        self._clusters[id] = {
            "id": id,
            "label": node_label,
            "parent_id": cluster_id,
            "attrs": attrs,
        }

    def get_nodes(self):
        return self._nodes.copy()

    def get_edges(self):
        return self._edges.copy()

    def get_clusters(self):
        return self._clusters.copy()

    def to_file(self, output_filename):
        pass

    def to_string(self) -> str:
        return "graph"


class TestInitialization:
    """Tests that the state is correctly initialized."""

    def test_dependency_injection(self):
        """Tests that relevant dependencies are injected."""

        mock_dag_builder = Mock(DAGBuilder)
        utility = ConstructCircuitDAG(mock_dag_builder)
        assert utility.dag_builder is mock_dag_builder


def test_does_not_mutate_module():
    """Test that the module is not mutated."""

    # Create block containing some ops
    op = test.TestOp()
    block = Block(ops=[op])
    # Create region containing some blocks
    region = Region(blocks=[block])
    # Create op containing the regions
    container_op = test.TestOp(regions=[region])
    # Create module op to house it all
    module_op = ModuleOp(ops=[container_op])

    module_op_str_before = str(module_op)

    mock_dag_builder = Mock(DAGBuilder)
    utility = ConstructCircuitDAG(mock_dag_builder)
    utility.construct(module_op)

    assert str(module_op) == module_op_str_before


class TestCreateOperatorNodes:
    """Tests that operators can be created and visualized as nodes."""

    @pytest.mark.unit
    @pytest.mark.parametrize("op", [qml.H(0), qml.X(0)])
    def test_custom_op(self, op):
        """Tests that the CustomOp operation node can be created and visualized."""

        # Build module with only a CustomOp
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            qml.apply(op)

        module = my_circuit()

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        # sanity check
        edges = utility.dag_builder.get_edges()
        assert edges == []
        clusters = utility.dag_builder.get_clusters()
        assert clusters == {}

        # Ensure DAG only has one node
        nodes = utility.dag_builder.get_nodes()
        assert len(nodes) == 1
        assert next(iter(nodes.values()))["label"] == str(op)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "op",
        [
            qml.GlobalPhase(0.5),
            qml.GlobalPhase(0.5, wires=0),
            qml.GlobalPhase(0.5, wires=[0, 1]),
        ],
    )
    def test_global_phase_op(self, op):
        # Build module with only a CustomOp
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            qml.apply(op)

        module = my_circuit()

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        # sanity check
        edges = utility.dag_builder.get_edges()
        assert edges == []
        clusters = utility.dag_builder.get_clusters()
        assert clusters == {}

        # Ensure DAG only has one node
        nodes = utility.dag_builder.get_nodes()
        assert len(nodes) == 1
        assert next(iter(nodes.values()))["label"] == str(op)

    def test_qubit_unitary_op(self):
        pass

    def test_set_state_op(self):
        pass

    def test_multi_rz_op(self):
        pass

    def test_set_basis_state_op(self):
        pass


class TestCreateMeasurementNodes:
    """Tests that measurements can be created and visualized as nodes."""

    def test_state_op(self):
        pass

    def test_statistical_measurement_op(self):
        pass

    def test_projective_measurement_op(self):
        pass
