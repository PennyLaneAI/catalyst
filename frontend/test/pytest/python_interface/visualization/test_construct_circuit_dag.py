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

import re
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

    def add_node(self, uid, label, cluster_uid=None, **attrs) -> None:
        self._nodes[uid] = {
            "uid": uid,
            "label": label,
            "parent_cluster_uid": "base" if cluster_uid is None else cluster_uid,
            "attrs": attrs,
        }

    def add_edge(self, from_uid: str, to_uid: str, **attrs) -> None:
        self._edges.append(
            {
                "from_uid": from_uid,
                "to_uid": to_uid,
                "attrs": attrs,
            }
        )

    def add_cluster(
        self,
        uid,
        node_label=None,
        cluster_uid=None,
        **attrs,
    ) -> None:
        self._clusters[uid] = {
            "uid": uid,
            "node_label": node_label,
            "cluster_label": attrs.get("label"),
            "parent_cluster_uid": "base" if cluster_uid is None else cluster_uid,
            "attrs": attrs,
        }

    def get_nodes_in_cluster(self, cluster_label: str) -> list[str]:
        """
        Returns a list of node labels that are direct children of the given cluster.
        """
        node_uids = []
        cluster_uid = self.get_cluster_uid_by_label(cluster_label)
        for node_data in self._nodes.values():
            if node_data["parent_cluster_uid"] == cluster_uid:
                node_uids.append(node_data["label"])
        return node_uids

    def get_child_clusters(self, parent_cluster_label: str) -> list[str]:
        """
        Returns a list of cluster labels that are direct children of the given parent cluster.
        """
        parent_cluster_uid = self.get_cluster_uid_by_label(parent_cluster_label)
        cluster_labels = []
        for cluster_data in self._clusters.values():
            if cluster_data["parent_cluster_uid"] == parent_cluster_uid:
                cluster_label = cluster_data["cluster_label"] or cluster_data["node_label"]
                cluster_labels.append(cluster_label)
        return cluster_labels

    def get_node_uid_by_label(self, label: str) -> str | None:
        """
        Finds the ID of a node given its label.
        Assumes labels are unique for testing purposes.
        """
        for id, node_data in self._nodes.items():
            if node_data["label"] == label:
                return id
        return None

    def get_cluster_uid_by_label(self, label: str) -> str | None:
        """
        Finds the ID of a cluster given its label.
        Assumes cluster labels are unique for testing purposes.
        """
        # Work around for base graph
        if label == "base":
            return "base"
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


class TestFakeDAGBuilder:
    """Test the FakeDAGBuilder to ensure helper functions work as intended."""

    @pytest.fixture
    def builder_with_data(self):
        """Sets up an instance with a complex graph already built."""

        builder = FakeDAGBuilder()

        # Cluster set-up
        builder.add_cluster("c0", label="Company", cluster_uid=None)  # Add to base graph
        builder.add_cluster("c1", label="Marketing", cluster_uid="c0")
        builder.add_cluster("c2", label="Finance", cluster_uid="c0")

        # Node set-up
        builder.add_node("n0", "CEO", cluster_uid="c0")
        builder.add_node("n1", "Marketing Manager", cluster_uid="c1")
        builder.add_node("n2", "Finance Manager", cluster_uid="c2")

        return builder

    # Test ID look up

    def test_get_node_uid_by_label_success(self, builder_with_data):
        assert builder_with_data.get_node_uid_by_label("Finance Manager") == "n2"
        assert builder_with_data.get_node_uid_by_label("Marketing Manager") == "n1"
        assert builder_with_data.get_node_uid_by_label("CEO") == "n0"

    def test_get_node_uid_by_label_failure(self, builder_with_data):
        assert builder_with_data.get_node_uid_by_label("Software Manager") is None

    def test_get_cluster_uid_by_label_success(self, builder_with_data):
        assert builder_with_data.get_cluster_uid_by_label("Finance") == "c2"
        assert builder_with_data.get_cluster_uid_by_label("Marketing") == "c1"
        assert builder_with_data.get_cluster_uid_by_label("Company") == "c0"

    def test_get_cluster_uid_by_label_failure(self, builder_with_data):
        assert builder_with_data.get_cluster_uid_by_label("Software") is None

    # Test relationship probing

    def test_node_heirarchy(self, builder_with_data):
        finance_nodes = builder_with_data.get_nodes_in_cluster("Finance")
        assert finance_nodes == ["Finance Manager"]

        marketing_nodes = builder_with_data.get_nodes_in_cluster("Marketing")
        assert marketing_nodes == ["Marketing Manager"]

        company_nodes = builder_with_data.get_nodes_in_cluster("Company")
        assert company_nodes == ["CEO"]

    def test_cluster_heirarchy(self, builder_with_data):
        clusters_in_finance = builder_with_data.get_child_clusters("Finance")
        assert not clusters_in_finance

        clusters_in_marketing = builder_with_data.get_child_clusters("Marketing")
        assert not clusters_in_marketing

        clusters_in_company = builder_with_data.get_child_clusters("Company")
        assert {"Finance", "Marketing"} == set(clusters_in_company)

        clusters_in_base = builder_with_data.get_child_clusters("base")
        assert clusters_in_base == ["Company"]


@pytest.mark.unit
def test_dependency_injection():
    """Tests that relevant dependencies are injected."""

    mock_dag_builder = Mock(DAGBuilder)
    utility = ConstructCircuitDAG(mock_dag_builder)
    assert utility.dag_builder is mock_dag_builder


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
        assert "my_workflow" in all_cluster_labels

        # Check nesting is correct
        # graph
        # └── my_workflow

        # Check my_workflow is nested under my_workflow
        my_workflow_id = utility.dag_builder.get_cluster_uid_by_label("my_workflow")
        assert graph_clusters[my_workflow_id]["parent_cluster_uid"] == "base"

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
        assert "my_workflow" in all_cluster_labels
        assert "my_qnode1" in all_cluster_labels
        assert "my_qnode2" in all_cluster_labels

        # Check nesting is correct
        # graph
        # └── my_workflow
        #     ├── my_qnode1
        #     └── my_qnode2

        # Check my_workflow is under graph
        my_workflow_id = utility.dag_builder.get_cluster_uid_by_label("my_workflow")
        assert graph_clusters[my_workflow_id]["parent_cluster_uid"] == "base"

        # Check both qnodes are under my_workflow
        assert "my_qnode1" in utility.dag_builder.get_child_clusters("my_workflow")
        assert "my_qnode2" in utility.dag_builder.get_child_clusters("my_workflow")


class TestForOp:
    """Tests that the for loop control flow can be visualized correctly."""

    @pytest.mark.unit
    def test_basic_example(self):
        """Tests that the for loop cluster can be visualized correctly."""

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

        assert re.search(
            r"for arg\d in range\(0,3,1\)",
            utility.dag_builder.get_child_clusters("my_workflow"),
        )

    @pytest.mark.unit
    def test_nested_loop(self):
        """Tests that nested for loops are visualized correctly."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            for i in range(0, 5, 2):
                for j in range(1, 6, 2):
                    qml.H(0)

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        # Check first for loop
        child_clusters = utility.dag_builder.get_child_clusters("my_workflow")
        assert len(child_clusters) == 1
        assert re.search(
            r"for arg\d in range\(0,5,2\)",
            child_clusters,
        )

        # Check second for loop
        for_loop_label = utility.dag_builder.get_child_clusters(child_clusters[0])
        child_clusters = utility.dag_builder.get_child_clusters(for_loop_label)
        assert len(child_clusters) == 1
        assert re.search(
            r"for arg\d in range\(1,6,2\)",
            child_clusters,
        )

    def test_dynamic_start_stop_step(self):
        """Tests that dynamic start, stop, step variables can be displayed."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def step_dynamic(x):
            for i in range(0, 3, x):
                qml.H(0)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def stop_dynamic(x):
            for i in range(0, x, 1):
                qml.H(0)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def start_dynamic(x):
            for i in range(x, 3, 1):
                qml.H(0)

        args = (1,)
        utility = ConstructCircuitDAG(FakeDAGBuilder())

        start_dynamic_module = start_dynamic(*args)
        utility.construct(start_dynamic_module)
        assert re.search(
            r"for arg\d in range\(arg\d,3,1\)",
            utility.dag_builder.get_child_clusters("my_workflow"),
        )

        stop_dynamic_module = stop_dynamic(*args)
        utility.construct(stop_dynamic_module)
        assert re.search(
            r"for arg\d in range\(0,arg\d,1\)",
            utility.dag_builder.get_child_clusters("my_workflow"),
        )

        step_dynamic_module = step_dynamic(*args)
        utility.construct(step_dynamic_module)
        assert re.search(
            r"for arg\d in range\(0,3,arg\d\)",
            utility.dag_builder.get_child_clusters("my_workflow"),
        )


class TestWhileOp:
    """Tests that the while loop control flow can be visualized correctly."""

    @pytest.mark.unit
    def test_basic_example(self):
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

        assert "while ..." in utility.dag_builder.get_child_clusters("my_workflow")


class TestIfOp:
    """Tests that the conditional control flow can be visualized correctly."""

    @pytest.mark.unit
    def test_basic_example(self):
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

        assert "conditional" in utility.dag_builder.get_child_clusters("my_workflow")
        assert "if ..." in utility.dag_builder.get_child_clusters("conditional")
        assert "else" in utility.dag_builder.get_child_clusters("conditional")

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
