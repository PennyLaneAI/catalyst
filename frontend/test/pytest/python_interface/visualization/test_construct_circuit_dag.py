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
from catalyst import measure
from catalyst.python_interface.conversion import xdsl_from_qjit
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
        cluster_id = "__base__" if cluster_id is None else cluster_id
        self._nodes[id] = {
            "id": id,
            "label": label,
            "cluster_id": cluster_id,
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
        cluster_id = "__base__" if cluster_id is None else cluster_id
        self._clusters[id] = {
            "id": id,
            "label": node_label,
            "cluster_id": cluster_id,
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


@pytest.mark.unit
def test_dependency_injection():
    """Tests that relevant dependencies are injected."""

    dag_builder = FakeDAGBuilder()
    utility = ConstructCircuitDAG(dag_builder)
    assert utility.dag_builder is dag_builder


@pytest.mark.unit
def test_does_not_mutate_module():
    """Test that the module is not mutated."""

    # Create module
    op = test.TestOp()
    block = Block(ops=[op])
    region = Region(blocks=[block])
    container_op = test.TestOp(regions=[region])
    module_op = ModuleOp(ops=[container_op])

    # Save state before
    module_op_str_before = str(module_op)

    # Process module
    mock_dag_builder = Mock(DAGBuilder)
    utility = ConstructCircuitDAG(mock_dag_builder)
    utility.construct(module_op)

    # Ensure not mutated
    assert str(module_op) == module_op_str_before


class TestCreateStaticOperatorNodes:
    """Tests that operators with static parameters can be created and visualized as nodes."""

    @pytest.mark.unit
    @pytest.mark.parametrize("op", [qml.H(0), qml.X(0), qml.SWAP([0, 1])])
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
        assert next(iter(nodes.values()))["cluster_id"] == "__base__"

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
        """Test that GlobalPhase can be handled."""

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
        assert next(iter(nodes.values()))["cluster_id"] == "__base__"

    @pytest.mark.unit
    def test_qubit_unitary_op(self):
        """Test that QubitUnitary operations can be handled."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            qml.QubitUnitary([[0, 1], [1, 0]], wires=0)

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
        assert next(iter(nodes.values()))["label"] == str(
            qml.QubitUnitary([[0, 1], [1, 0]], wires=0)
        )
        assert next(iter(nodes.values()))["cluster_id"] == "__base__"

    @pytest.mark.unit
    def test_multi_rz_op(self):
        """Test that MultiRZ operations can be handled."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            qml.MultiRZ(0.5, wires=[0])

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
        assert next(iter(nodes.values()))["label"] == str(qml.MultiRZ(0.5, wires=[0]))
        assert next(iter(nodes.values()))["cluster_id"] == "__base__"


class TestCreateStaticMeasurementNodes:
    """Tests that measurements with static parameters can be created and visualized as nodes."""

    @pytest.mark.unit
    def test_state_op(self):
        """Test that qml.state can be handled."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            return qml.state()

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
        assert next(iter(nodes.values()))["label"] == str(qml.state())
        assert next(iter(nodes.values()))["cluster_id"] == "__base__"

    @pytest.mark.unit
    @pytest.mark.parametrize("meas_fn", [qml.expval, qml.var])
    def test_expval_var_measurement_op(self, meas_fn):
        """Test that statistical measurement operators can be captured as nodes."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            return meas_fn(qml.Z(0))

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
        assert next(iter(nodes.values()))["label"] == str(meas_fn(qml.Z(0)))
        assert next(iter(nodes.values()))["cluster_id"] == "__base__"

    @pytest.mark.unit
    def test_probs_measurement_op(self):
        """Tests that the probs measurement function can be captured as a node."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            return qml.probs()

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
        assert next(iter(nodes.values()))["label"] == str(qml.probs())
        assert next(iter(nodes.values()))["cluster_id"] == "__base__"

    @pytest.mark.unit
    def test_sample_measurement_op(self):
        """Tests that the sample measurement function can be captured as a node."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.set_shots(10)
        @qml.qnode(dev)
        def my_circuit():
            return qml.sample()

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
        assert next(iter(nodes.values()))["label"] == str(qml.sample())
        assert next(iter(nodes.values()))["cluster_id"] == "__base__"

    @pytest.mark.unit
    def test_projective_measurement_op(self):
        """Test that projective measurements can be captured as nodes."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            measure(0)

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
        assert "MidMeasure" in next(iter(nodes.values()))["label"]
        assert next(iter(nodes.values()))["cluster_id"] == "__base__"
