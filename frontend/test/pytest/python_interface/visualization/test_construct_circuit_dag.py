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

pytestmark = pytest.mark.xdsl
xdsl = pytest.importorskip("xdsl")

# pylint: disable=wrong-import-position
# This import needs to be after pytest in order to prevent ImportErrors
import pennylane as qml
from xdsl.dialects import test
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir.core import Block, Region

from catalyst import measure
from catalyst.python_interface.conversion import xdsl_from_qjit
from catalyst.python_interface.visualization.construct_circuit_dag import (
    ConstructCircuitDAG,
    get_label,
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
        self._edges = {}
        self._clusters = {}

    def add_node(self, uid, label, cluster_uid=None, **attrs) -> None:
        self._nodes[uid] = {
            "uid": uid,
            "label": label,
            "parent_cluster_uid": cluster_uid,
            "attrs": attrs,
        }

    def add_edge(self, from_uid: str, to_uid: str, **attrs) -> None:
        # O(1) look up
        edge_key = (from_uid, to_uid)
        self._edges[edge_key] = {
            "attrs": attrs,
        }

    def add_cluster(
        self,
        uid,
        label=None,
        cluster_uid=None,
        **attrs,
    ) -> None:
        self._clusters[uid] = {
            "uid": uid,
            "label": label,
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


def test_dependency_injection():
    """Tests that relevant dependencies are injected."""

    mock_dag_builder = Mock(DAGBuilder)
    utility = ConstructCircuitDAG(mock_dag_builder)
    assert utility.dag_builder is mock_dag_builder


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
        assert graph_clusters["cluster0"]["label"] == "qjit"
        assert graph_clusters["cluster0"]["parent_cluster_uid"] is None

        # Check that my_workflow is under qjit
        assert graph_clusters["cluster1"]["label"] == "my_workflow"
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
        assert graph_clusters["cluster0"]["label"] == "qjit"
        assert graph_clusters["cluster0"]["parent_cluster_uid"] is None

        # Check both qnodes are under my_workflow
        assert graph_clusters["cluster1"]["label"] == "my_qnode1"
        assert graph_clusters["cluster1"]["parent_cluster_uid"] == "cluster0"

        assert graph_clusters["cluster2"]["label"] == "my_qnode2"
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
        assert graph_clusters["cluster1"]["label"] == "my_workflow"
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
        assert graph_clusters["cluster1"]["label"] == "my_qnode1"
        assert graph_nodes["node0"]["parent_cluster_uid"] == "cluster1"

        # Assert label is as expected
        assert graph_nodes["node0"]["label"] == "NullQubit"

        # Assert null qubit device node is inside my_qnode2 cluster
        assert graph_clusters["cluster2"]["label"] == "my_qnode2"
        # NOTE: node1 is the qml.H(0) in my_qnode1
        assert graph_nodes["node2"]["parent_cluster_uid"] == "cluster2"

        # Assert label is as expected
        assert graph_nodes["node2"]["label"] == "LightningSimulator"


class TestForOp:
    """Tests that the for loop control flow can be visualized correctly."""

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
        assert clusters["cluster2"]["label"] == "for loop"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"

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
        assert clusters["cluster2"]["label"] == "for loop"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"
        assert clusters["cluster3"]["label"] == "for loop"
        assert clusters["cluster3"]["parent_cluster_uid"] == "cluster2"


class TestWhileOp:
    """Tests that the while loop control flow can be visualized correctly."""

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
        assert clusters["cluster2"]["label"] == "while loop"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"

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
        assert clusters["cluster2"]["label"] == "while loop"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"
        assert clusters["cluster3"]["label"] == "while loop"
        assert clusters["cluster3"]["parent_cluster_uid"] == "cluster2"


class TestIfOp:
    """Tests that the conditional control flow can be visualized correctly."""

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
        assert clusters["cluster2"]["label"] == "conditional"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"

        # Check three clusters live within cluster2 (conditional)
        assert clusters["cluster3"]["label"] == "if"
        assert clusters["cluster3"]["parent_cluster_uid"] == "cluster2"
        assert clusters["cluster4"]["label"] == "else"
        assert clusters["cluster4"]["parent_cluster_uid"] == "cluster2"

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
        assert clusters["cluster2"]["label"] == "conditional"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"

        # Check three clusters live within conditional
        assert clusters["cluster3"]["label"] == "if"
        assert clusters["cluster3"]["parent_cluster_uid"] == "cluster2"
        assert clusters["cluster4"]["label"] == "elif"
        assert clusters["cluster4"]["parent_cluster_uid"] == "cluster2"
        assert clusters["cluster5"]["label"] == "else"
        assert clusters["cluster5"]["parent_cluster_uid"] == "cluster2"

    def test_nested_conditionals(self):
        """Tests that nested conditionals are visualized correctly."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow(x, y):
            if x == 1:
                if y == 2:
                    qml.H(0)
                else:
                    qml.Z(0)
                qml.X(0)
            else:
                qml.Z(0)

        args = (1, 2)
        module = my_workflow(*args)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        clusters = utility.dag_builder.clusters

        # cluster0 -> qjit
        # cluster1 -> my_workflow
        # cluster2 -> conditional (1)
        #   cluster3 -> if
        #       cluster4 -> conditional ()
        #           cluster5 -> if
        #           cluster6 -> else
        #   cluster7 -> else

        # Check first conditional is a cluster within my_workflow
        assert clusters["cluster2"]["label"] == "conditional"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"

        # Check 'if' cluster of first conditional has another conditional
        assert clusters["cluster3"]["label"] == "if"
        assert clusters["cluster3"]["parent_cluster_uid"] == "cluster2"

        # Second conditional
        assert clusters["cluster4"]["label"] == "conditional"
        assert clusters["cluster4"]["parent_cluster_uid"] == "cluster3"
        # Check 'if' and 'else' in second conditional
        assert clusters["cluster5"]["label"] == "if"
        assert clusters["cluster5"]["parent_cluster_uid"] == "cluster4"
        assert clusters["cluster6"]["label"] == "else"
        assert clusters["cluster6"]["parent_cluster_uid"] == "cluster4"

        # Check nested if / else is within the first if cluster
        assert clusters["cluster7"]["label"] == "else"
        assert clusters["cluster7"]["parent_cluster_uid"] == "cluster2"

    def test_nested_conditionals_with_quantum_ops(self):
        """Tests that nested conditionals are unflattend if quantum operations
        are present"""

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
                if x == 3:
                    qml.RX(0, 0)
                elif x == 4:
                    qml.RY(0, 0)
                else:
                    qml.RZ(0, 0)

        args = (1,)
        module = my_workflow(*args)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        clusters = utility.dag_builder.clusters

        # cluster0 -> qjit
        # cluster1 -> my_workflow
        #     node0 -> NullQubit
        #     cluster2 -> conditional (1)
        #       cluster3 -> if
        #           node1 -> X(0)
        #       cluster4 -> elif
        #           node2 -> Y(0)
        #       cluster5 -> else
        #           node3 -> Z(0)
        #           cluster6 -> conditional (2)
        #               cluster7 -> if
        #                    node4 -> RX(0,0)
        #               cluster8 -> elif
        #                    node5 -> RY(0,0)
        #               cluster9 -> else
        #                    node6 -> RZ(0,0)

        # check outer conditional (1)
        assert clusters["cluster2"]["label"] == "conditional"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"
        assert clusters["cluster3"]["label"] == "if"
        assert clusters["cluster3"]["parent_cluster_uid"] == "cluster2"
        assert clusters["cluster4"]["label"] == "elif"
        assert clusters["cluster4"]["parent_cluster_uid"] == "cluster2"
        assert clusters["cluster5"]["label"] == "else"
        assert clusters["cluster5"]["parent_cluster_uid"] == "cluster2"

        # Nested conditional (2) inside conditional (1)
        assert clusters["cluster6"]["label"] == "conditional"
        assert clusters["cluster6"]["parent_cluster_uid"] == "cluster5"
        assert clusters["cluster7"]["label"] == "if"
        assert clusters["cluster7"]["parent_cluster_uid"] == "cluster6"
        assert clusters["cluster8"]["label"] == "elif"
        assert clusters["cluster8"]["parent_cluster_uid"] == "cluster6"
        assert clusters["cluster9"]["label"] == "else"
        assert clusters["cluster9"]["parent_cluster_uid"] == "cluster6"

    def test_nested_conditionals_with_nested_quantum_ops(self):
        """Tests that nested conditionals are unflattend if quantum operations
        are present but nested in other operations"""

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
                for i in range(3):
                    qml.Z(0)
                if x == 3:
                    qml.RX(0, 0)
                elif x == 4:
                    qml.RY(0, 0)
                else:
                    qml.RZ(0, 0)

        args = (1,)
        module = my_workflow(*args)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        clusters = utility.dag_builder.clusters

        # cluster0 -> qjit
        # cluster1 -> my_workflow
        #     node0 -> NullQubit
        #     cluster2 -> conditional (1)
        #       cluster3 -> if
        #           node1 -> X(0)
        #       cluster4 -> elif
        #           node2 -> Y(0)
        #       cluster5 -> else
        #           cluster6 -> for loop
        #               node3 -> Z(0)
        #           cluster7 -> conditional (2)
        #               cluster8 -> if
        #                    node4 -> RX(0,0)
        #               cluster9 -> elif
        #                    node5 -> RY(0,0)
        #               cluster10 -> else
        #                    node6 -> RZ(0,0)

        # check outer conditional (1)
        assert clusters["cluster2"]["label"] == "conditional"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"
        assert clusters["cluster3"]["label"] == "if"
        assert clusters["cluster3"]["parent_cluster_uid"] == "cluster2"
        assert clusters["cluster4"]["label"] == "elif"
        assert clusters["cluster4"]["parent_cluster_uid"] == "cluster2"
        assert clusters["cluster5"]["label"] == "else"
        assert clusters["cluster5"]["parent_cluster_uid"] == "cluster2"

        # Nested conditional (2) inside conditional (1)
        assert clusters["cluster6"]["label"] == "for loop"
        assert clusters["cluster6"]["parent_cluster_uid"] == "cluster5"

        assert clusters["cluster7"]["label"] == "conditional"
        assert clusters["cluster7"]["parent_cluster_uid"] == "cluster5"
        assert clusters["cluster8"]["label"] == "if"
        assert clusters["cluster8"]["parent_cluster_uid"] == "cluster7"
        assert clusters["cluster9"]["label"] == "elif"
        assert clusters["cluster9"]["parent_cluster_uid"] == "cluster7"
        assert clusters["cluster10"]["label"] == "else"
        assert clusters["cluster10"]["parent_cluster_uid"] == "cluster7"


class TestGetLabel:
    """Tests the get_label utility."""

    @pytest.mark.parametrize(
        "op, label",
        [
            (qml.H(0), "<name> H|<wire> [0]"),
            (
                qml.QubitUnitary([[0, 1], [1, 0]], 0),
                "<name> QubitUnitary|<wire> [0]",
            ),
            (qml.SWAP([0, 1]), "<name> SWAP|<wire> [0, 1]"),
        ],
    )
    def test_standard_operator(self, op, label):
        """Tests against an operator instance."""
        assert get_label(op) == label

    def test_global_phase_operator(self):
        """Tests against a GlobalPhase operator instance."""
        assert get_label(qml.GlobalPhase(0.5)) == f"<name> GlobalPhase|<wire> all"

    @pytest.mark.parametrize(
        "meas, label",
        [
            (qml.state(), "<name> state|<wire> all"),
            (qml.expval(qml.Z(0)), "<name> expval(Z)|<wire> [0]"),
            (qml.var(qml.Z(0)), "<name> var(Z)|<wire> [0]"),
            (qml.probs(), "<name> probs|<wire> all"),
            (qml.probs(wires=0), "<name> probs|<wire> [0]"),
            (qml.probs(wires=[0, 1]), "<name> probs|<wire> [0, 1]"),
            (qml.sample(), "<name> sample|<wire> all"),
            (qml.sample(wires=0), "<name> sample|<wire> [0]"),
            (qml.sample(wires=[0, 1]), "<name> sample|<wire> [0, 1]"),
        ],
    )
    def test_standard_measurement(self, meas, label):
        """Tests against an operator instance."""
        assert get_label(meas) == label


class TestCreateStaticOperatorNodes:
    """Tests that operators with static parameters can be created and visualized as nodes."""

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

        # Ensure DAG only has one node
        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + operator

        # Make sure label has relevant info
        assert nodes["node1"]["label"] == get_label(op)

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

        # Ensure DAG only has one node
        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + operator

        # Compiler throws out the wires and they get converted to wires=[] no matter what
        assert nodes["node1"]["label"] == get_label(qml.GlobalPhase(0.5))

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

        # Ensure DAG only has one node
        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + operator

        assert nodes["node1"]["label"] == get_label(qml.QubitUnitary([[0, 1], [1, 0]], wires=0))

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

        # Ensure DAG only has one node
        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + operator

        assert nodes["node1"]["label"] == get_label(qml.MultiRZ(0.5, wires=[0]))

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

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + operator

        assert nodes["node1"]["label"] == f"<name> MidMeasure|<wire> [0]"


class TestCreateDynamicOperatorNodes:
    """Tests that operator nodes with dynamic parameters or wires can be created and visualized."""

    def test_static_dynamic_mix(self):
        """Tests that static and dynamic wires can both be used."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit(x):
            qml.SWAP([0, x])

        args = (1,)
        module = my_circuit(*args)

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + SWAP

        assert nodes["node1"]["label"] == f"<name> SWAP|<wire> [0, arg0]"

    def test_qnode_argument(self):
        """Tests that qnode arguments can be used as wires."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit(x, y):
            qml.H(x)
            qml.X(y)

        args = (1, 2)
        module = my_circuit(*args)

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 3  # Device node + H + X

        assert nodes["node1"]["label"] == f"<name> H|<wire> [arg0]"
        assert nodes["node2"]["label"] == f"<name> X|<wire> [arg1]"

    def test_for_loop_variable(self):
        """Tests that for loop iteration variables can be used as wires."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            for i in range(3):
                qml.H(i)

        module = my_circuit()

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + H

        assert nodes["node1"]["label"] == f"<name> H|<wire> [arg0]"

    def test_while_loop_variable(self):
        """Tests that while loop variables can be used as wires."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            counter = 0
            while counter < 5:
                qml.H(counter)
                counter += 1

        module = my_circuit()

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + H

        assert nodes["node1"]["label"] == f"<name> H|<wire> [arg0]"

    def test_conditional_variable(self):
        """Tests that conditional variables can be used."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit(x, y):
            if x == y:
                qml.H(x)

        args = (1, 2)
        module = my_circuit(*args)

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + H

        assert nodes["node1"]["label"] == f"<name> H|<wire> [arg0]"

    def test_through_clusters(self):
        """Tests that dynamic wire labels can be accessed through clusters."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit(x):
            for i in range(3):
                qml.H(x)
                qml.X(i)

        args = (1,)
        module = my_circuit(*args)

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 3  # Device node + H + X

        assert nodes["node1"]["label"] == f"<name> H|<wire> [arg0]"
        assert nodes["node2"]["label"] == f"<name> X|<wire> [arg1]"


class TestCreateStaticMeasurementNodes:
    """Tests that measurements with static parameters can be created and visualized as nodes."""

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

        # Ensure DAG only has one node
        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + operator

        assert nodes["node1"]["label"] == get_label(qml.state())

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

        # Ensure DAG only has one node
        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + measurement

        assert nodes["node1"]["label"] == get_label(meas_fn(qml.Z(0)))

    @pytest.mark.parametrize(
        "op",
        [
            qml.probs(),
            qml.probs(wires=0),
            qml.probs(wires=[0, 1]),
        ],
    )
    def test_probs_measurement_op(self, op):
        """Tests that the probs measurement function can be captured as a node."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            return op

        module = my_circuit()

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + probs

        assert nodes["node1"]["label"] == get_label(op)

    @pytest.mark.parametrize(
        "op",
        [
            qml.sample(),
            qml.sample(wires=0),
            qml.sample(wires=[0, 1]),
        ],
    )
    def test_valid_sample_measurement_op(self, op):
        """Tests that the sample measurement function can be captured as a node."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.set_shots(10)
        @qml.qnode(dev)
        def my_circuit():
            return op

        module = my_circuit()

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + sample

        assert nodes["node1"]["label"] == get_label(op)


class TestCreateDynamicMeasurementNodes:
    """Tests that measurements on dynamic wires render correctly."""

    def test_static_dynamic_mix(self):
        """Tests that static and dynamic wires can both be used."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit(x):
            return qml.probs(wires=[0, x])

        args = (1,)
        module = my_circuit(*args)

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + probs

        assert nodes["node1"]["label"] == f"<name> probs|<wire> [0, arg0]"

    def test_qnode_argument(self):
        """Tests that qnode arguments can be used as wires."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.set_shots(10)
        @qml.qnode(dev)
        def my_circuit(x, y):
            return qml.probs(wires=x), qml.expval(qml.Z(x)), qml.var(qml.X(y)), qml.sample(wires=x)

        args = (1, 2)
        module = my_circuit(*args)

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 5  # Device node + probs + expval + var + sample

        assert nodes["node1"]["label"] == f"<name> probs|<wire> [arg0]"
        assert nodes["node2"]["label"] == f"<name> expval(Z)|<wire> [arg0]"
        assert nodes["node3"]["label"] == f"<name> var(X)|<wire> [arg1]"
        assert nodes["node4"]["label"] == f"<name> sample|<wire> [arg0]"


class TestOperatorConnectivity:
    """Tests that operators are properly connected."""

    def test_static_connection_within_cluster(self):
        """Tests that connections can be made within the same cluster."""

        dev = qml.device("null.qubit", wires=3)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            qml.X(0)
            qml.Z(1)
            qml.Y(0)
            qml.H(1)
            qml.S(1)
            qml.T(2)

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        # node0 -> NullQubit

        # Check all nodes
        assert "X" in nodes["node1"]["label"]
        assert "Y" in nodes["node2"]["label"]
        assert "Z" in nodes["node3"]["label"]
        assert "H" in nodes["node4"]["label"]
        assert "S" in nodes["node5"]["label"]
        assert "T" in nodes["node6"]["label"]

        # Check edges
        #           -> X -> Y
        # NullQubit -> Z -> H -> S
        #           -> T
        assert len(edges) == 6
        assert ("node0", "node1") in edges
        assert ("node0", "node3") in edges
        assert ("node0", "node6") in edges
        assert ("node1", "node2") in edges
        assert ("node3", "node4") in edges
        assert ("node4", "node5") in edges

    def test_static_connection_through_for_loop(self):
        """Tests that connections can be made through a for loop cluster."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            qml.X(0)
            for i in range(3):
                qml.Y(0)

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes
        # node0 -> NullQubit

        # Check all nodes
        assert "X" in nodes["node1"]["label"]
        assert "Y" in nodes["node2"]["label"]

        # Check edges
        #    for loop
        # NullQubit -> X ----------> Y
        assert len(edges) == 2
        assert ("node0", "node1") in edges
        assert ("node1", "node2") in edges

    def test_static_connection_through_while_loop(self):
        """Tests that connections can be made through a while loop cluster."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            counter = 0
            qml.X(0)
            while counter < 5:
                qml.Y(0)
                counter += 1

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes
        # node0 -> NullQubit

        # Check all nodes
        assert "X" in nodes["node1"]["label"]
        assert "Y" in nodes["node2"]["label"]

        # Check edges
        #    while loop
        # NullQubit -> X ----------> Y
        assert len(edges) == 2
        assert ("node0", "node1") in edges
        assert ("node1", "node2") in edges

    def test_static_connection_through_conditional(self):
        """Tests that connections through conditionals make sense."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow(x):
            qml.X(0)
            qml.T(1)
            if x == 1:
                qml.RX(0, 0)
                qml.S(1)
            elif x == 2:
                qml.RY(0, 0)
            else:
                qml.RZ(0, 0)
            qml.H(0)

        args = (1,)
        module = my_workflow(*args)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        # node0 -> NullQubit

        # Check all nodes
        # NOTE: depth first traversal hence T first then X
        assert "T" in nodes["node1"]["label"]
        assert "X" in nodes["node2"]["label"]
        assert "RX" in nodes["node3"]["label"]
        assert "S" in nodes["node4"]["label"]
        assert "RY" in nodes["node5"]["label"]
        assert "RZ" in nodes["node6"]["label"]
        assert "H" in nodes["node7"]["label"]

        # Check all edges
        assert len(edges) == 9
        assert ("node0", "node1") in edges
        assert ("node0", "node2") in edges
        assert ("node1", "node4") in edges
        assert ("node2", "node3") in edges
        assert ("node2", "node5") in edges
        assert ("node2", "node6") in edges
        assert ("node3", "node7") in edges
        assert ("node5", "node7") in edges
        assert ("node6", "node7") in edges

    def test_static_connection_through_nested_conditional(self):
        """Tests that connections through nested conditionals make sense."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow(x):
            qml.X(0)
            qml.T(1)
            if x == 1:
                if x == 3:
                    qml.Y(1)
                else:
                    qml.Z(0)
            else:
                qml.RZ(0, 0)
            qml.H(0)

        args = (1,)
        module = my_workflow(*args)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        # node0 -> NullQubit

        # Check all nodes
        # NOTE: depth first traversal hence T first then PauliX
        assert "T" in nodes["node1"]["label"]
        assert "X" in nodes["node2"]["label"]
        assert "Y" in nodes["node3"]["label"]
        assert "Z" in nodes["node4"]["label"]
        assert "RZ" in nodes["node5"]["label"]
        assert "H" in nodes["node6"]["label"]

        # Check all edges
        assert len(edges) == 8
        assert ("node0", "node1")
        assert ("node0", "node2")
        assert ("node2", "node4") in edges  # X -> Z
        assert ("node2", "node5") in edges  # X -> RZ
        assert ("node2", "node6") in edges  # X -> H
        assert ("node5", "node6") in edges  # RZ -> H
        assert ("node4", "node6") in edges  # Z -> H
        assert ("node1", "node3") in edges  # T -> Y

    def test_multi_wire_connectivity(self):
        """Ensures that multi wire connectivity holds."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            qml.RX(0.1, 0)
            qml.RY(0.2, 1)
            qml.RZ(0.3, 2)
            qml.CNOT(wires=[0, 1])
            qml.Toffoli(wires=[1, 2, 0])

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        # node0 -> NullQubit

        # Check all nodes
        assert "RZ" in nodes["node1"]["label"]
        assert "RX" in nodes["node2"]["label"]
        assert "RY" in nodes["node3"]["label"]
        assert "CNOT" in nodes["node4"]["label"]
        assert "Toffoli" in nodes["node5"]["label"]

        # Check all edges
        assert len(edges) == 7
        assert ("node0", "node1") in edges  # Device -> RZ
        assert ("node0", "node2") in edges  # Device -> RX
        assert ("node0", "node3") in edges  # Device -> RY
        assert ("node3", "node4") in edges  # RX -> CNOT
        assert ("node2", "node4") in edges  # RY -> CNOT
        assert ("node1", "node5") in edges  # RZ -> Toffoli
        assert ("node4", "node5") in edges  # CNOT -> Toffoli

    def test_dynamic_wire_connectivity(self):
        """Tests standard scenario of interweaving static and dynamic operators."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow(x, y):
            qml.X(0)
            qml.Y(1)
            qml.Z(2)
            qml.H(x)
            qml.S(0)
            qml.T(2)
            qml.H(x)

        args = (1, 2)
        module = my_workflow(*args)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        # node0 -> NullQubit

        # Check all nodes
        assert "Z" in nodes["node1"]["label"]
        assert "Y" in nodes["node2"]["label"]
        assert "X" in nodes["node3"]["label"]
        assert "H" in nodes["node4"]["label"]
        assert "T" in nodes["node5"]["label"]
        assert "S" in nodes["node6"]["label"]
        assert "H" in nodes["node7"]["label"]

        # Check all edges
        assert len(edges) == 10

        # All static wires collapse into the dynamic hadamard (dashed edges)
        assert ("node0", "node1") in edges
        assert ("node0", "node2") in edges
        assert ("node0", "node3") in edges
        assert ("node1", "node4") in edges
        assert edges[("node1", "node4")]["attrs"]["style"] == "dashed"
        assert ("node2", "node4") in edges
        assert edges[("node2", "node4")]["attrs"]["style"] == "dashed"
        assert ("node3", "node4") in edges
        assert edges[("node3", "node4")]["attrs"]["style"] == "dashed"

        # H then fans out to the static S and T
        assert ("node4", "node5") in edges
        assert ("node4", "node6") in edges

        # Collapse again to the dynamic H
        assert ("node5", "node7") in edges
        assert edges[("node5", "node7")]["attrs"]["style"] == "dashed"
        assert ("node6", "node7") in edges
        assert edges[("node6", "node7")]["attrs"]["style"] == "dashed"

    def test_first_operator_is_dynamic(self):
        """Tests when the first operator is dynamic"""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow(x, y):
            qml.H(x)

        args = (1, 2)
        module = my_workflow(*args)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        # node0 -> NullQubit

        # Check all nodes
        assert "H" in nodes["node1"]["label"]

        # Check all edges
        assert len(edges) == 1
        assert ("node0", "node1") in edges

    def test_double_choke(self):
        """Tests when two dynamic operators are back to back"""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow(x, y):
            qml.X(0)
            qml.Y(x)
            qml.Z(y)

        args = (1, 2)
        module = my_workflow(*args)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        # node0 -> NullQubit

        # Check all nodes
        assert "X" in nodes["node1"]["label"]
        assert "Y" in nodes["node2"]["label"]
        assert "Z" in nodes["node3"]["label"]

        # Check all edges
        assert len(edges) == 3
        assert ("node0", "node1") in edges
        assert ("node1", "node2") in edges
        assert ("node2", "node3") in edges
        assert edges[("node2", "node3")]["attrs"]["style"] == "dashed"


class TestTerminalMeasurementConnectivity:
    """Test that terminal measurements connect properly."""

    @pytest.mark.parametrize("meas_fn", [qml.probs, qml.state])
    def test_connect_all_wires(self, meas_fn):
        """Tests connection to terminal measurements that operate on all wires."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            qml.X(0)
            qml.T(1)
            return meas_fn()

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        # node0 -> NullQubit

        # Check all nodes
        assert "X" in nodes["node1"]["label"]
        assert "T" in nodes["node2"]["label"]
        assert meas_fn.__name__ in nodes["node3"]["label"]

        # Check all edges
        assert len(edges) == 4
        assert ("node0", "node1") in edges
        assert ("node0", "node2") in edges
        assert ("node1", "node3") in edges
        assert ("node2", "node3") in edges

    def test_connect_specific_wires(self):
        """Tests connection to terminal measurements that operate on specific wires."""

        dev = qml.device("null.qubit", wires=5)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.set_shots(10)
        @qml.qnode(dev)
        def my_workflow():
            qml.X(0)
            qml.Y(1)
            qml.Z(2)
            qml.H(3)
            return (
                qml.expval(qml.Z(0)),
                qml.var(qml.Z(1)),
                qml.probs(wires=[2]),
                qml.sample(wires=[3]),
            )

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        # node0 -> NullQubit

        # Check all nodes
        assert "X" in nodes["node1"]["label"]
        assert "Y" in nodes["node2"]["label"]
        assert "Z" in nodes["node3"]["label"]
        assert "H" in nodes["node4"]["label"]
        assert "expval" in nodes["node5"]["label"]
        assert "var" in nodes["node6"]["label"]
        assert "probs" in nodes["node7"]["label"]
        assert "sample" in nodes["node8"]["label"]

        # Check all edges
        assert len(edges) == 8
        assert ("node0", "node1") in edges
        assert ("node0", "node2") in edges
        assert ("node0", "node3") in edges
        assert ("node0", "node4") in edges
        assert ("node1", "node5") in edges
        assert ("node2", "node6") in edges
        assert ("node3", "node7") in edges
        assert ("node4", "node8") in edges

    def test_multi_wire_connectivity(self):
        """Ensures that multi wire connectivity holds."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            qml.X(0)
            qml.Y(1)
            return qml.probs(wires=[0, 1])

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        # node0 -> NullQubit

        # Check all nodes
        assert "X" in nodes["node1"]["label"]
        assert "Y" in nodes["node2"]["label"]
        assert "probs" in nodes["node3"]["label"]

        # Check all edges
        assert len(edges) == 4
        assert ("node0", "node1") in edges
        assert ("node0", "node2") in edges
        assert ("node1", "node3") in edges
        assert ("node2", "node3") in edges

    def test_no_quantum_ops_before_measurement(self):
        """Tests a workflow with no quantum operations."""

        dev = qml.device("null.qubit", wires=2)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_empty_workflow():
            return qml.expval(qml.Z(0))

        module = my_empty_workflow()
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges

        assert len(edges) == 1
        # Node0 = NullQubit
        assert ("node0", "node1") in edges

    def test_terminal_measurement_after_static_dyn_op_mix(self):
        """Tests that a terminal measurement on a mix of dynamic and static wires connects properly."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow(x, y):
            qml.X(0)
            qml.Y(x)
            qml.Z(y)
            return qml.probs(wires=[0, x])

        args = (1, 2)
        module = my_workflow(*args)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        # node0 -> NullQubit

        # Check all nodes
        assert "X" in nodes["node1"]["label"]
        assert "Y" in nodes["node2"]["label"]
        assert "Z" in nodes["node3"]["label"]
        assert "probs" in nodes["node4"]["label"]

        # Check all edges
        assert len(edges) == 4
        assert ("node0", "node1") in edges
        assert ("node1", "node2") in edges
        assert edges[("node1", "node2")]["attrs"]["style"] == "dashed"
        assert ("node2", "node3") in edges
        assert edges[("node2", "node3")]["attrs"]["style"] == "dashed"
        assert ("node3", "node4") in edges

    def test_terminal_measurement_static_dyn_mix(self):
        """Tests that a terminal measurement on a mix of dynamic and static wires connects properly."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow(x, y):
            qml.X(0)
            qml.Y(0)
            for i in range(3):
                qml.H(i)
            return qml.expval(qml.Z(x))

        args = (1, 2)
        module = my_workflow(*args)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        # node0 -> NullQubit

        # Check all nodes
        assert "X" in nodes["node1"]["label"]
        assert "Y" in nodes["node2"]["label"]
        assert "H" in nodes["node3"]["label"]
        assert "expval(Z)" in nodes["node4"]["label"]

        # Check all edges
        assert len(edges) == 4
        assert ("node0", "node1") in edges
        assert ("node1", "node2") in edges
        assert ("node2", "node3") in edges
        assert edges[("node2", "node3")]["attrs"]["style"] == "dashed"
        assert ("node3", "node4") in edges
        assert edges[("node3", "node4")]["attrs"]["style"] == "dashed"

    def test_terminal_measurement_dyn_after_static(self):
        """Tests that a terminal measurement on a mix of dynamic and static wires connects properly."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow(x, y):
            qml.X(x)
            qml.Y(0)
            return qml.expval(qml.Z(y))

        args = (1, 2)
        module = my_workflow(*args)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        # node0 -> NullQubit

        # Check all nodes
        assert "X" in nodes["node1"]["label"]
        assert "Y" in nodes["node2"]["label"]
        assert "expval(Z)" in nodes["node3"]["label"]

        # Check all edges
        assert len(edges) == 3
        assert ("node0", "node1") in edges
        assert ("node1", "node2") in edges
        assert ("node2", "node3") in edges
        assert edges[("node2", "node3")]["attrs"]["style"] == "dashed"

    def test_no_term_meas_interconnectivity(self):
        """Tests that terminal measurements don't connect amongst themselves."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow(x, y):
            return qml.probs(), qml.expval(qml.Z(0)), qml.expval(qml.X(x))

        args = (1, 2)
        module = my_workflow(*args)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        # Check all edges
        # Node0 is NullQubit
        assert len(edges) == 3
        assert ("node0", "node1") in edges
        assert edges[("node0", "node1")]["attrs"]["style"] == "dashed"
        assert ("node0", "node2") in edges
        assert ("node0", "node3") in edges
        assert edges[("node0", "node3")]["attrs"]["style"] == "dashed"
