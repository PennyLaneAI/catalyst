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
from jax import util

pytestmark = pytest.mark.usefixtures("requires_xdsl")

# pylint: disable=wrong-import-position
# This import needs to be after pytest in order to prevent ImportErrors
import pennylane as qml
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
            "parent_cluster_uid": cluster_uid,
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
            "parent_cluster_uid": cluster_uid,
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

        graph_clusters = utility.dag_builder.clusters

        # Check nesting is correct
        # graph
        # └── qjit
        #     └── my_workflow

        # Check qjit is nested under graph
        assert graph_clusters["cluster0"]["cluster_label"] == "qjit"
        assert graph_clusters["cluster0"]["parent_cluster_uid"] is None

        # Check that my_workflow is under qjit
        assert graph_clusters["cluster1"]["cluster_label"] == "my_workflow"
        assert graph_clusters["cluster1"]["parent_cluster_uid"] == "cluster0"

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

        # Check nesting is correct
        # graph
        # └── qjit
        #     ├── my_qnode1
        #     └── my_qnode2

        # Check qjit is under graph
        assert graph_clusters["cluster0"]["cluster_label"] == "qjit"
        assert graph_clusters["cluster0"]["parent_cluster_uid"] is None

        # Check both qnodes are under my_workflow
        assert graph_clusters["cluster1"]["cluster_label"] == "my_qnode1"
        assert graph_clusters["cluster1"]["parent_cluster_uid"] == "cluster0"

        assert graph_clusters["cluster2"]["cluster_label"] == "my_qnode2"
        assert graph_clusters["cluster2"]["parent_cluster_uid"] == "cluster0"


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
        graph_clusters = utility.dag_builder.clusters

        # Check nesting is correct
        # graph
        # └── qjit
        #     └── my_workflow: NullQubit

        # Assert device node is inside my_workflow cluster
        assert graph_clusters["cluster1"]["cluster_label"] == "my_workflow"
        assert graph_nodes["node0"]["parent_cluster_uid"] == "cluster1"

        # Assert label is as expected
        assert graph_nodes["node0"]["label"] == "NullQubit"

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
        graph_nodes = utility.dag_builder.nodes
        graph_clusters = utility.dag_builder.clusters

        # Check nesting is correct
        # graph
        # └── qjit
        #     ├── my_qnode1: NullQubit
        #     └── my_qnode2: LightningSimulator

        # Assert lightning.qubit device node is inside my_qnode1 cluster
        assert graph_clusters["cluster1"]["cluster_label"] == "my_qnode1"
        assert graph_nodes["node0"]["parent_cluster_uid"] == "cluster1"

        # Assert label is as expected
        assert graph_nodes["node0"]["label"] == "NullQubit"

        # Assert null qubit device node is inside my_qnode2 cluster
        assert graph_clusters["cluster2"]["cluster_label"] == "my_qnode2"
        assert graph_nodes["node1"]["parent_cluster_uid"] == "cluster2"

        # Assert label is as expected
        assert graph_nodes["node1"]["label"] == "LightningSimulator"


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

        clusters = utility.dag_builder.clusters

        # cluster0 -> qjit
        # cluster1 -> my_workflow
        assert clusters["cluster2"]["node_label"] == "for ... in range(..., ..., ...)"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"

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

        clusters = utility.dag_builder.clusters

        # cluster0 -> qjit
        # cluster1 -> my_workflow
        assert clusters["cluster2"]["node_label"] == "for ... in range(..., ..., ...)"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"
        assert clusters["cluster3"]["node_label"] == "for ... in range(..., ..., ...)"
        assert clusters["cluster3"]["parent_cluster_uid"] == "cluster2"


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

        clusters = utility.dag_builder.clusters

        # cluster0 -> qjit
        # cluster1 -> my_workflow
        assert clusters["cluster2"]["node_label"] == "while ..."
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"

    @pytest.mark.unit
    def test_nested_loop(self):
        """Tests that nested while loops are visualized correctly."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            outer_counter = 0
            inner_counter = 0
            while outer_counter < 5:
                while inner_counter < 6:
                    qml.H(0)
                    inner_counter += 1
                outer_counter += 1

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        clusters = utility.dag_builder.clusters

        # cluster0 -> qjit
        # cluster1 -> my_workflow
        assert clusters["cluster2"]["node_label"] == "while ..."
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"
        assert clusters["cluster3"]["node_label"] == "while ..."
        assert clusters["cluster3"]["parent_cluster_uid"] == "cluster2"


class TestIfOp:
    """Tests that the conditional control flow can be visualized correctly."""

    @pytest.mark.unit
    def test_basic_example(self):
        """Test that the conditional operation is visualized correctly."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow(x):
            if x == 2:
                qml.X(0)
            else:
                qml.Y(0)

        args = (1,)
        module = my_workflow(*args)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        clusters = utility.dag_builder.clusters

        # cluster0 -> qjit
        # cluster1 -> my_workflow
        # Check conditional is a cluster within cluster1 (my_workflow)
        assert clusters["cluster2"]["cluster_label"] == "conditional"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"

        # Check three clusters live within cluster2 (conditional)
        assert clusters["cluster3"]["node_label"] == "if ..."
        assert clusters["cluster3"]["parent_cluster_uid"] == "cluster2"
        assert clusters["cluster4"]["node_label"] == "else"
        assert clusters["cluster4"]["parent_cluster_uid"] == "cluster2"

    @pytest.mark.unit
    def test_if_elif_else_conditional(self):
        """Test that the conditional operation is visualized correctly."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow(x):
            if x == 1:
                qml.X(0)
            elif x == 2:
                qml.Y(0)
            else:
                qml.Z(0)

        args = (1,)
        module = my_workflow(*args)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        clusters = utility.dag_builder.clusters

        # cluster0 -> qjit
        # cluster1 -> my_workflow
        # Check conditional is a cluster within my_workflow
        assert clusters["cluster2"]["cluster_label"] == "conditional"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"

        # Check three clusters live within conditional
        assert clusters["cluster3"]["node_label"] == "if ..."
        assert clusters["cluster3"]["parent_cluster_uid"] == "cluster2"
        assert clusters["cluster4"]["node_label"] == "elif ..."
        assert clusters["cluster4"]["parent_cluster_uid"] == "cluster2"
        assert clusters["cluster5"]["node_label"] == "else"
        assert clusters["cluster5"]["parent_cluster_uid"] == "cluster2"

    @pytest.mark.unit
    def test_nested_conditionals(self):
        """Tests that nested conditionals are visualized correctly."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow(x,y):
            if x == 1:
                if y == 2:
                    qml.H(0)
                else:
                    qml.Z(0)
                qml.X(0)
            else:
                qml.Z(0)

        args = (1,2)
        module = my_workflow(*args)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        clusters = utility.dag_builder.clusters
        clusters = utility.dag_builder.clusters

        # cluster0 -> qjit
        # cluster1 -> my_workflow

        # Check first conditional is a cluster within my_workflow
        assert clusters["cluster2"]["cluster_label"] == "conditional"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"

        # Check two clusters live within first conditional
        assert clusters["cluster3"]["node_label"] == "if ..."
        assert clusters["cluster3"]["parent_cluster_uid"] == "cluster2"
        # first conditional's else
        assert clusters["cluster6"]["node_label"] == "else"
        assert clusters["cluster6"]["parent_cluster_uid"] == "cluster2"

        # Check nested if / else is within the first if cluster
        assert clusters["cluster4"]["node_label"] == "if ..."
        assert clusters["cluster4"]["parent_cluster_uid"] == "cluster3"
        assert clusters["cluster5"]["node_label"] == "if ..."
        assert clusters["cluster5"]["parent_cluster_uid"] == "cluster3"
