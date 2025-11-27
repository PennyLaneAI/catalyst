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
            "parent_cluster_id": "base" if cluster_id is None else cluster_id,
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
            "parent_cluster_id": "base" if cluster_id is None else cluster_id,
            "attrs": attrs,
        }

    def get_nodes_in_cluster(self, cluster_label: str) -> list[str]:
        """
        Returns a list of node labels that are direct children of the given cluster.
        """
        node_ids = []
        cluster_id = self.get_cluster_id_by_label(cluster_label)
        for node_data in self._nodes.values():
            if node_data["parent_cluster_id"] == cluster_id:
                node_ids.append(node_data["label"])
        return node_ids

    def get_child_clusters(self, parent_cluster_label: str) -> list[str]:
        """
        Returns a list of cluster labels that are direct children of the given parent cluster.
        """
        parent_cluster_id = self.get_cluster_id_by_label(parent_cluster_label)
        cluster_labels = []
        for cluster_data in self._clusters.values():
            if cluster_data["parent_cluster_id"] == parent_cluster_id:
                cluster_label = cluster_data["cluster_label"] or cluster_data["node_label"]
                cluster_labels.append(cluster_label)
        return cluster_labels

    def get_node_id_by_label(self, label: str) -> str | None:
        """
        Finds the ID of a node given its label.
        Assumes labels are unique for testing purposes.
        """
        for id, node_data in self._nodes.items():
            if node_data["label"] == label:
                return id
        return None

    def get_cluster_id_by_label(self, label: str) -> str | None:
        """
        Finds the ID of a cluster given its label.
        Assumes cluster labels are unique for testing purposes.
        """
        for id, cluster_data in self._clusters.items():
            cluster_label = cluster_data["cluster_label"] or cluster_data["node_label"]
            if cluster_label == label:
                return id
        return None

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

        # Check labels we expected are there
        graph_clusters = utility.dag_builder.clusters
        all_cluster_labels = {info["cluster_label"] for info in graph_clusters.values()}
        assert "jit_my_workflow" in all_cluster_labels
        assert "my_workflow" in all_cluster_labels

        # Check nesting is correct
        # graph
        # └── jit_my_workflow
        #     └── my_workflow

        # Check my_workflow is nested under jit_my_workflow
        assert "my_workflow" in utility.dag_builder.get_child_clusters("jit_my_workflow")
        # Check that jit_my_workflow is the first cluster on top of the graph
        jit_my_workflow_id = utility.dag_builder.get_cluster_id_by_label("jit_my_workflow")
        assert graph_clusters[jit_my_workflow_id]["parent_cluster_id"] == "base"

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
        graph_clusters = utility.dag_builder.clusters
        all_cluster_labels = {info["cluster_label"] for info in graph_clusters.values()}
        assert "jit_my_workflow" in all_cluster_labels
        assert "my_qnode1" in all_cluster_labels
        assert "my_qnode2" in all_cluster_labels

        # Check nesting is correct
        # graph
        # └── jit_my_workflow
        #     ├── my_qnode1
        #     └── my_qnode2

        # Check jit_my_workflow is under graph
        jit_my_workflow_id = utility.dag_builder.get_cluster_id_by_label("jit_my_workflow")
        assert graph_clusters[jit_my_workflow_id]["parent_cluster_id"] == "base"

        # Check both qnodes are under jit_my_workflow
        assert "my_qnode1" in utility.dag_builder.get_child_clusters("jit_my_workflow")
        assert "my_qnode2" in utility.dag_builder.get_child_clusters("jit_my_workflow")


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

        # Check that device node is within the my_workflow cluster
        nodes_in_my_workflow = utility.dag_builder.get_nodes_in_cluster("my_workflow")
        assert "NullQubit" in nodes_in_my_workflow

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

        # Check that device node is within the my_workflow cluster
        nodes_in_my_workflow = utility.dag_builder.get_nodes_in_cluster("my_qnode1")
        assert "NullQubit" in nodes_in_my_workflow
        nodes_in_my_workflow = utility.dag_builder.get_nodes_in_cluster("my_qnode2")
        assert "LightningSimulator" in nodes_in_my_workflow
