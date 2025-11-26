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
from xdsl.dialects import test
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir.core import Block, Region

from catalyst.python_interface.conversion import xdsl_from_qjit
from catalyst.python_interface.visualization.construct_circuit_dag import (
    ConstructCircuitDAG,
)
from catalyst.python_interface.visualization.dag_builder import DAGBuilder


class FakeDAGBuilder(DAGBuilder):
    """
    A concrete implementation of DAGBuilder used ONLY for testing.
    It stores all graph manipulation calls in data structures
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
            "cluster_id": cluster_id,
            "attrs": attrs,
        }

    def add_edge(self, from_id: str, to_id: str, **attrs) -> None:
        self._edges.append(
            {
                "from_id": from_id,
                "to_id": to_id,
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
            "node_label": node_label,
            "cluster_label": attrs.get("label"),
            "cluster_id": cluster_id,
            "attrs": attrs,
        }

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    @property
    def clusters(self):
        return self._clusters

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


@pytest.mark.unit
class TestFuncOpVisualization:
    """Tests the visualization of FuncOps with bounding boxes"""

    def test_standard_qnode(self):
        """Tests that a standard QJIT'd QNode is visualized correctly"""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            qml.H(0)

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        graph_clusters = utility.dag_builder.clusters

        # Check labels we expected are there
        expected_cluster_labels = [
            "jit_my_workflow",
            "my_workflow",
            "setup",
            "teardown",
        ]
        generated_cluster_labels = {info["cluster_label"] for info in graph_clusters.values()}
        for cluster_label in expected_cluster_labels:
            assert cluster_label in generated_cluster_labels

        # Check nesting is correct
        # graph
        # └── jit_my_workflow
        #     └── my_workflow

        # Get the parent labels for each cluster and ensure they are what we expect.
        parent_labels = (
            graph_clusters[child_cluster["cluster_id"]]["cluster_label"]
            for child_cluster in graph_clusters.values()
            if child_cluster.get("cluster_id") is not None
        )
        cluster_label_to_parent_label: dict[str, str] = dict(
            zip(tuple(generated_cluster_labels), parent_labels)
        )
        assert cluster_label_to_parent_label["jit_my_workflow"] is None
        assert cluster_label_to_parent_label["my_qnode1"] == "jit_my_workflow"

    def test_nested_qnodes(self):
        """Tests that nested QJIT'd QNodes are visualized correctly"""

        dev = qml.device("null.qubit", wires=1)

        @qml.qnode(dev)
        def my_qnode2():
            qml.X(0)

        @qml.qnode(dev)
        def my_qnode1():
            qml.H(0)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        def my_workflow():
            my_qnode1()
            my_qnode2()

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        graph_clusters = utility.dag_builder.clusters

        # Check labels we expected are there as clusters
        expected_cluster_labels = [
            "jit_my_workflow",
            "my_qnode1",
            "my_qnode2",
            "setup",
            "teardown",
        ]
        assert len(graph_clusters) == len(expected_cluster_labels)
        cluster_labels = {info["label"] for info in graph_clusters.values()}
        for expected_name in expected_cluster_labels:
            assert expected_name in cluster_labels
        # Check nesting is correct
        # graph
        # └── jit_my_workflow
        #     ├── my_qnode1
        #     └── my_qnode2

        # Get the parent labels for each cluster and ensure they are what we expect.
        parent_labels = (
            graph_clusters[child_cluster["cluster_id"]]["cluster_label"]
            for child_cluster in graph_clusters.values()
            if child_cluster.get("cluster_id") is not None
        )
        cluster_label_to_parent_label: dict[str, str] = dict(
            zip(tuple(cluster_labels), parent_labels)
        )
        assert cluster_label_to_parent_label["jit_my_workflow"] is None
        assert cluster_label_to_parent_label["my_qnode1"] == "jit_my_workflow"
        assert cluster_label_to_parent_label["my_qnode2"] == "jit_my_workflow"


class TestControlFlowVisualization:
    """Tests that the control flow operations are visualized correctly as clusters."""

    @pytest.mark.unit
    def test_for_loop(self):
        """Test that the for loop is visualized correctly."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            for i in range(3):
                qml.H(0)

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        clusters = utility.dag_builder.clusters
        cluster_labels = {info["label"] for info in clusters.values()}
        assert "for ..." in cluster_labels

        # Ensure proper nesting of clusters

    @pytest.mark.unit
    def test_while_loop(self):
        """Test that the while loop is visualized correctly."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            counter = 0
            while counter < 5:
                qml.H(0)
                counter += 1

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        clusters = utility.dag_builder.clusters
        cluster_labels = {info["label"] for info in clusters.values()}
        assert "while ..." in cluster_labels

        # Ensure proper nesting of clusters

    @pytest.mark.unit
    def test_if_else_conditional(self):
        """Test that the conditional operation is visualized correctly."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            flag = 1
            if flag == 1:
                qml.X(0)
            else:
                qml.Y(0)

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        clusters = utility.dag_builder.clusters
        cluster_labels = {info["label"] for info in clusters.values()}
        assert "if ..." in cluster_labels
        assert "else" in cluster_labels

        # Ensure proper nesting of clusters

    @pytest.mark.unit
    def test_if_elif_else_conditional(self):
        """Test that the conditional operation is visualized correctly."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            flag = 1
            if flag == 1:
                qml.X(0)
            elif flag == 2:
                qml.Y(0)
            else:
                qml.Z(0)

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        clusters = utility.dag_builder.clusters
        cluster_labels = [info["label"] for info in clusters.values()]
        assert "if ..." in cluster_labels
        assert cluster_labels.count("if ...") == 2
        assert "else" in cluster_labels
        assert cluster_labels.count("else") == 2

        # Ensure proper nesting


class TestDeviceNode:
    """Tests that the device node is correctly visualized."""

    def test_standard_qnode(self):
        """Tests that a standard setup works."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            qml.H(0)

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        graph_nodes = utility.dag_builder.nodes

        # Basic check for node
        node_labels = {info["label"] for info in graph_nodes.values()}
        assert "NullQubit" in node_labels

        # Ensure nesting is correct
        graph_clusters = utility.dag_builder.clusters
        parent_labels = (
            graph_clusters[child_cluster["cluster_id"]]["cluster_label"]
            for child_cluster in graph_clusters.values()
            if child_cluster.get("cluster_id") is not None
        )
        cluster_label_to_parent_label: dict[str, str] = dict(zip(tuple(node_labels), parent_labels))
        assert cluster_label_to_parent_label["NullQubit"] == "my_workflow"

    def test_nested_qnodes(self):
        """Tests that nested QJIT'd QNodes are visualized correctly"""

        dev1 = qml.device("null.qubit", wires=1)
        dev2 = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev2)
        def my_qnode2():
            qml.X(0)

        @qml.qnode(dev1)
        def my_qnode1():
            qml.H(0)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        def my_workflow():
            my_qnode1()
            my_qnode2()

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        graph_clusters = utility.dag_builder.clusters
        graph_nodes = utility.dag_builder.nodes

        # Check labels we expected are there as clusters
        expected_cluster_labels = [
            "jit_my_workflow",
            "my_qnode1",
            "my_qnode2",
            "setup",
            "teardown",
        ]
        assert len(graph_clusters) == len(expected_cluster_labels)
        cluster_labels = {info["label"] for info in graph_clusters.values()}
        for expected_name in expected_cluster_labels:
            assert expected_name in cluster_labels

        # Check nesting is correct
        # graph
        # └── jit_my_workflow
        #     ├── my_qnode1
        #     │   └── node: NullQubit
        #     └── my_qnode2
        #         └── node: LightningQubit

        # Get the parent labels for each cluster and ensure they are what we expect.
        parent_labels = (
            graph_clusters[child_cluster["cluster_id"]]["cluster_label"]
            for child_cluster in graph_clusters.values()
            if child_cluster.get("cluster_id") is not None
        )
        cluster_label_to_parent_label: dict[str, str] = dict(
            zip(tuple(cluster_labels), parent_labels)
        )
        assert cluster_label_to_parent_label["jit_my_workflow"] is None
        assert cluster_label_to_parent_label["my_qnode1"] == "jit_my_workflow"
        assert cluster_label_to_parent_label["my_qnode2"] == "jit_my_workflow"

        # Check nodes are in the correct clusters
        parent_labels = (
            graph_clusters[child_node["cluster_id"]]["label"]
            for child_node in graph_nodes.values()
            if child_node.get("cluster_id") is not None
        )
        node_labels = {info["label"] for info in graph_nodes.values()}
        node_label_to_parent_label: dict[str, str] = dict(zip(tuple(node_labels), parent_labels))
        assert node_label_to_parent_label["NullQubit"] == "my_qnode1"
        assert node_label_to_parent_label["LightningSimulator"] == "my_qnode2"
