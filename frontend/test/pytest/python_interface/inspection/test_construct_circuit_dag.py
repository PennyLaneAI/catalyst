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

import jax
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
from catalyst.python_interface.conversion import parse_generic_to_xdsl_module, xdsl_from_qjit
from catalyst.python_interface.inspection.construct_circuit_dag import (
    ConstructCircuitDAG,
    get_label,
)
from catalyst.python_interface.inspection.dag_builder import DAGBuilder


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


def assert_dag_structure(nodes, edges, expected_edges):
    """
    Validates graph structure using gate names instead of UIDs - this will make
    the tests more agnostic to the order the operations appear in the IR which
    is sensitive to how it was captured and lowered.

    """
    name_to_uid = {}

    # Pattern explanation:
    # 1. Look for <name> then capture everything up to a |
    # 2. OR: if no <name> is found, capture the entire string
    gate_pattern = re.compile(r"<name>\s+([^|]+)|(^[^|]+$)")

    # NOTE: Specifically requires that all nodes (operators / measurements) are unique in the test
    for uid, node_info in nodes.items():
        label = node_info["label"]
        match = gate_pattern.search(label)

        if match:
            # group(1) is the <name> match (e.g. X, H, Toffoli)
            # group(2) is the standard string match (e.g. NullQubit)
            gate_name = (match.group(1) or match.group(2)).strip()
            name_to_uid[gate_name] = uid
        else:
            name_to_uid[label] = uid

    assert len(edges) == len(
        expected_edges
    ), f"Expected {len(expected_edges)} edges, got {len(edges)}."

    for edge in expected_edges:
        start_name, end_name = edge[0], edge[1]
        u_start = name_to_uid.get(start_name, start_name)
        u_end = name_to_uid.get(end_name, end_name)

        # Check connection
        assert (
            u_start,
            u_end,
        ) in edges, f"Missing expected edge {start_name} -> {end_name} ({u_start}, {u_end})."

        # Check attrs
        edge_attrs = {}
        if len(edge) > 2:
            edge_attrs: dict = edge[2]
        if edge_attrs:
            actual_edge_attrs = edges[(u_start, u_end)]["attrs"]
            for attr_key, expected_val in edge_attrs.items():
                actual_val = actual_edge_attrs.get(attr_key)
                assert (
                    actual_val == expected_val
                ), f"Expected {attr_key}='{expected_val}', got '{actual_val}'."


@pytest.mark.usefixtures("use_both_frontend")
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


@pytest.mark.usefixtures("use_both_frontend")
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


@pytest.mark.usefixtures("use_both_frontend")
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


@pytest.mark.usefixtures("use_both_frontend")
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


@pytest.mark.usefixtures("use_both_frontend")
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
        assert clusters["conditional_cluster2"]["label"] == "conditional"
        assert clusters["conditional_cluster2"]["parent_cluster_uid"] == "cluster1"

        # Check three clusters live within cluster2 (conditional)
        assert clusters["cluster3"]["label"] == "if"
        assert clusters["cluster3"]["parent_cluster_uid"] == "conditional_cluster2"
        assert clusters["cluster4"]["label"] == "else"
        assert clusters["cluster4"]["parent_cluster_uid"] == "conditional_cluster2"

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
        assert clusters["conditional_cluster2"]["label"] == "conditional"
        assert clusters["conditional_cluster2"]["parent_cluster_uid"] == "cluster1"

        # Check three clusters live within conditional
        assert clusters["cluster3"]["label"] == "if"
        assert clusters["cluster3"]["parent_cluster_uid"] == "conditional_cluster2"
        assert clusters["cluster4"]["label"] == "elif"
        assert clusters["cluster4"]["parent_cluster_uid"] == "conditional_cluster2"
        assert clusters["cluster5"]["label"] == "else"
        assert clusters["cluster5"]["parent_cluster_uid"] == "conditional_cluster2"

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
        assert clusters["conditional_cluster2"]["label"] == "conditional"
        assert clusters["conditional_cluster2"]["parent_cluster_uid"] == "cluster1"

        # Check 'if' cluster of first conditional has another conditional
        assert clusters["cluster3"]["label"] == "if"
        assert clusters["cluster3"]["parent_cluster_uid"] == "conditional_cluster2"

        # Second conditional
        assert clusters["conditional_cluster4"]["label"] == "conditional"
        assert clusters["conditional_cluster4"]["parent_cluster_uid"] == "cluster3"
        # Check 'if' and 'else' in second conditional
        assert clusters["cluster5"]["label"] == "if"
        assert clusters["cluster5"]["parent_cluster_uid"] == "conditional_cluster4"
        assert clusters["cluster6"]["label"] == "else"
        assert clusters["cluster6"]["parent_cluster_uid"] == "conditional_cluster4"

        # Check nested if / else is within the first if cluster
        assert clusters["cluster7"]["label"] == "else"
        assert clusters["cluster7"]["parent_cluster_uid"] == "conditional_cluster2"

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
        assert clusters["conditional_cluster2"]["label"] == "conditional"
        assert clusters["conditional_cluster2"]["parent_cluster_uid"] == "cluster1"
        assert clusters["cluster3"]["label"] == "if"
        assert clusters["cluster3"]["parent_cluster_uid"] == "conditional_cluster2"
        assert clusters["cluster4"]["label"] == "elif"
        assert clusters["cluster4"]["parent_cluster_uid"] == "conditional_cluster2"
        assert clusters["cluster5"]["label"] == "else"
        assert clusters["cluster5"]["parent_cluster_uid"] == "conditional_cluster2"

        # Nested conditional (2) inside conditional (1)
        assert clusters["conditional_cluster6"]["label"] == "conditional"
        assert clusters["conditional_cluster6"]["parent_cluster_uid"] == "cluster5"
        assert clusters["cluster7"]["label"] == "if"
        assert clusters["cluster7"]["parent_cluster_uid"] == "conditional_cluster6"
        assert clusters["cluster8"]["label"] == "elif"
        assert clusters["cluster8"]["parent_cluster_uid"] == "conditional_cluster6"
        assert clusters["cluster9"]["label"] == "else"
        assert clusters["cluster9"]["parent_cluster_uid"] == "conditional_cluster6"

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
        assert clusters["conditional_cluster2"]["label"] == "conditional"
        assert clusters["conditional_cluster2"]["parent_cluster_uid"] == "cluster1"
        assert clusters["cluster3"]["label"] == "if"
        assert clusters["cluster3"]["parent_cluster_uid"] == "conditional_cluster2"
        assert clusters["cluster4"]["label"] == "elif"
        assert clusters["cluster4"]["parent_cluster_uid"] == "conditional_cluster2"
        assert clusters["cluster5"]["label"] == "else"
        assert clusters["cluster5"]["parent_cluster_uid"] == "conditional_cluster2"

        # Nested conditional (2) inside conditional (1)
        assert clusters["cluster6"]["label"] == "for loop"
        assert clusters["cluster6"]["parent_cluster_uid"] == "cluster5"

        assert clusters["conditional_cluster7"]["label"] == "conditional"
        assert clusters["conditional_cluster7"]["parent_cluster_uid"] == "cluster5"
        assert clusters["cluster8"]["label"] == "if"
        assert clusters["cluster8"]["parent_cluster_uid"] == "conditional_cluster7"
        assert clusters["cluster9"]["label"] == "elif"
        assert clusters["cluster9"]["parent_cluster_uid"] == "conditional_cluster7"
        assert clusters["cluster10"]["label"] == "else"
        assert clusters["cluster10"]["parent_cluster_uid"] == "conditional_cluster7"


class TestGetLabel:
    """Tests the get_label utility."""

    @pytest.mark.parametrize(
        "op, label",
        [
            (qml.H(0), "<name> Hadamard|<wire> [0]"),
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
        assert get_label(qml.GlobalPhase(0.5)) == "<name> GlobalPhase|<wire> all"

    @pytest.mark.parametrize(
        "meas, label",
        [
            (qml.state(), "<name> state|<wire> all"),
            (qml.expval(qml.Z(0)), "<name> expval(PauliZ)|<wire> [0]"),
            (qml.var(qml.Z(0)), "<name> var(PauliZ)|<wire> [0]"),
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


@pytest.mark.usefixtures("use_both_frontend")
class TestCreateStaticOperatorNodes:
    """Tests that operators with static parameters can be created and visualized as nodes."""

    def test_custom_op(self):
        """Tests that the CustomOp operation node can be created and visualized."""

        # Build module with only a CustomOp
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            qml.H(0)
            qml.SWAP([0, 1])

        module = my_circuit()

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        # Ensure DAG only has one node
        nodes = utility.dag_builder.nodes
        assert len(nodes) == 3  # Device node + operators

        # Make sure label has relevant info
        assert nodes["node1"]["label"] == get_label(qml.H(0))
        assert nodes["node2"]["label"] == get_label(qml.SWAP([0, 1]))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"wires": 0},
            {"wires": [0, 1]},
        ],
    )
    def test_global_phase_op(self, kwargs):
        """Test that GlobalPhase can be handled."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            qml.GlobalPhase(0.5, **kwargs)

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
            qml.QubitUnitary(jax.numpy.array([[0, 1], [1, 0]]), wires=0)

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

        if qml.capture.enabled():
            fn = qml.measure
        else:
            fn = measure

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            fn(0)

        module = my_circuit()

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + operator

        assert nodes["node1"]["label"] == "<name> MidMeasureMP|<wire> [0]"

    @pytest.mark.usefixtures("use_capture")
    def test_ppm(self):
        """Test that PPMs can be captured as nodes."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            qml.pauli_measure("X", wires=[0])
            qml.pauli_measure(pauli_word="XY", wires=[0, 1])

        module = my_circuit()

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 3  # Device node + operator

        assert nodes["node1"]["label"] == "<name> PPM-X|<wire> [0]"
        assert nodes["node1"]["attrs"]["fillcolor"] == "#70B3F5"
        assert nodes["node2"]["label"] == "<name> PPM-XY|<wire> [0, 1]"
        assert nodes["node2"]["attrs"]["fillcolor"] == "#70B3F5"

    @pytest.mark.usefixtures("use_capture")
    def test_ppr(self):
        """Tests that a PPR node can be created."""
        pipe = [("pipe", ["quantum-compilation-stage"])]

        @qml.qjit(pipelines=pipe, target="mlir")
        @qml.transform(pass_name="to-ppr")
        @qml.qnode(qml.device("null.qubit", wires=3))
        def cir():
            qml.PauliRot(jax.numpy.pi, pauli_word="YZ", wires=[0, 1])
            qml.PauliRot(jax.numpy.pi / 4, pauli_word="X", wires=[0])
            qml.PauliRot(jax.numpy.pi / 2, pauli_word="XYZ", wires=[0, 1, 2])

        module = parse_generic_to_xdsl_module(cir.mlir_opt)

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 4  # Device node + operator

        assert nodes["node1"]["label"] == "<name> PPR-YZ (π/2)|<wire> [0, 1]"
        assert nodes["node1"]["attrs"]["fillcolor"] == "#D9D9D9"
        assert nodes["node2"]["label"] == "<name> PPR-X (π/8)|<wire> [0]"
        assert nodes["node2"]["attrs"]["fillcolor"] == "#E3FFA1"
        assert nodes["node3"]["label"] == "<name> PPR-XYZ (π/4)|<wire> [0, 1, 2]"
        assert nodes["node3"]["attrs"]["fillcolor"] == "#F5BD70"

    @pytest.mark.usefixtures("use_capture")
    def test_ppr_arbitary(self):
        """Tests that a PPR node can be created."""

        pipe = [("pipe", ["quantum-compilation-stage"])]

        @qml.qjit(pipelines=pipe, target="mlir")
        @qml.transform(pass_name="to-ppr")
        @qml.qnode(qml.device("null.qubit", wires=3))
        def cir():
            # NOTE: use angle != pi / <something>
            # to get an qec.ppr.arbitary in the IR
            qml.PauliRot(1.0, pauli_word="X", wires=[0])
            qml.PauliRot(1.0, pauli_word="XYZ", wires=[0, 1, 2])

        module = parse_generic_to_xdsl_module(cir.mlir_opt)

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 3  # Device node + operator

        assert nodes["node1"]["label"] == "<name> PPR-X (φ)|<wire> [0]"
        assert nodes["node1"]["attrs"]["fillcolor"] == "#E3FFA1"
        assert nodes["node2"]["label"] == "<name> PPR-XYZ (φ)|<wire> [0, 1, 2]"
        assert nodes["node2"]["attrs"]["fillcolor"] == "#E3FFA1"

    @pytest.mark.usefixtures("use_capture")
    def test_pauli_rot(self):
        """Tests that a PauliRot node can be created."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            qml.PauliRot(0.5, "X", wires=0)
            qml.PauliRot(1.5, "XYZ", wires=[0, 1, 2])

        module = my_circuit()

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 3  # Device node + operator

        assert nodes["node1"]["label"] == "<name> PauliRot|<wire> [0]"
        assert nodes["node2"]["label"] == "<name> PauliRot|<wire> [0, 1, 2]"

    @pytest.mark.skipif(not qml.capture.enabled(), reason="Only works with capture enabled.")
    def test_complex_measurements(self):
        """Tests that complex measurements can be created."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True)
        @qml.qnode(dev)
        def my_workflow():
            coeffs = [0.2, -0.543]
            obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2)]
            ham = qml.ops.LinearCombination(coeffs, obs)

            return (
                qml.expval(ham),
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
            )

        module = my_workflow()

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 3  # Device node + measurements

        assert nodes["node1"]["label"] == "<name> LinearCombination|<wire> [0, 1, 2]"
        assert nodes["node2"]["label"] == "<name> Prod|<wire> [0, 1]"

    @pytest.mark.parametrize(
        "param, wires",
        (
            (jax.numpy.array([1, 0]), [0]),
            (jax.numpy.array([1, 0, 0, 0]), [0, 1]),
            (jax.numpy.array([1, 0, 0, 0, 1, 0, 0, 0]), [0, 1, 2]),
        ),
    )
    def test_state_prep(self, param, wires):
        """Tests that state preparation operators can be captured as nodes."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            qml.StatePrep(param, wires=wires)

        module = my_circuit()

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + operator

        assert nodes["node1"]["label"] == f"<name> StatePrep|<wire> {wires}"

    @pytest.mark.parametrize(
        "param, wires",
        (
            (jax.numpy.array([1]), [0]),
            (jax.numpy.array([1, 0]), [0, 1]),
            (jax.numpy.array([1, 0, 0]), [0, 1, 2]),
        ),
    )
    def test_basis_state(self, param, wires):
        """Tests that basis state operators can be captured as nodes."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            qml.BasisState(param, wires=wires)

        module = my_circuit()

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + operator

        assert nodes["node1"]["label"] == f"<name> BasisState|<wire> {wires}"


@pytest.mark.usefixtures("use_both_frontend")
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

        assert nodes["node1"]["label"] == "<name> SWAP|<wire> [0, arg0]"

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

        assert nodes["node1"]["label"] == "<name> Hadamard|<wire> [arg0]"
        assert nodes["node2"]["label"] == "<name> PauliX|<wire> [arg1]"

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

        assert nodes["node1"]["label"] == "<name> Hadamard|<wire> [arg0]"

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

        assert nodes["node1"]["label"] == "<name> Hadamard|<wire> [arg0]"

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

        assert nodes["node1"]["label"] == "<name> Hadamard|<wire> [arg0]"

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

        assert nodes["node1"]["label"] == "<name> Hadamard|<wire> [arg0]"
        assert nodes["node2"]["label"] == "<name> PauliX|<wire> [arg1]"

    def test_visualize_pythonic_operators(self):
        """Tests that we can use operators like +,-,%"""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow(x):
            qml.RX(x % 3, wires=x % 3)
            qml.RY(x - 3, wires=x - 3)
            qml.RZ(x + 3, wires=x + 3)

        args = (1,)
        module = my_workflow(*args)

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 4  # Device node + ops

        assert nodes["node1"]["label"] == "<name> RX|<wire> [(arg5 % 3)]"
        assert nodes["node2"]["label"] == "<name> RY|<wire> [(arg5 - 3)]"
        assert nodes["node3"]["label"] == "<name> RZ|<wire> [(arg5 + 3)]"

    @pytest.mark.usefixtures("use_capture")
    def test_ppm_dynamic(self):
        """Test that PPMs can be captured as nodes."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit(x, y):
            qml.pauli_measure("X", wires=[x])
            qml.pauli_measure(pauli_word="XY", wires=[y, 0])

        module = my_circuit(1, 2)

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 3  # Device node + operator

        assert nodes["node1"]["label"] == "<name> PPM-X|<wire> [arg0]"
        assert nodes["node1"]["attrs"]["fillcolor"] == "#70B3F5"
        assert nodes["node2"]["label"] == "<name> PPM-XY|<wire> [arg1, 0]"
        assert nodes["node2"]["attrs"]["fillcolor"] == "#70B3F5"


@pytest.mark.usefixtures("use_both_frontend")
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
        "kwargs",
        [
            {},
            {"wires": 0},
            {"wires": [0, 1]},
        ],
    )
    def test_probs_measurement_op(self, kwargs):
        """Tests that the probs measurement function can be captured as a node."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_circuit():
            return qml.probs(**kwargs)

        module = my_circuit()

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + probs

        assert nodes["node1"]["label"] == get_label(qml.probs(**kwargs))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"wires": 0},
            {"wires": [0, 1]},
        ],
    )
    def test_valid_sample_measurement_op(self, kwargs):
        """Tests that the sample measurement function can be captured as a node."""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.set_shots(10)
        @qml.qnode(dev)
        def my_circuit():
            return qml.sample(**kwargs)

        module = my_circuit()

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + sample

        assert nodes["node1"]["label"] == get_label(qml.sample(**kwargs))


@pytest.mark.usefixtures("use_both_frontend")
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

        assert nodes["node1"]["label"] == "<name> probs|<wire> [0, arg0]"

    def test_qnode_argument(self):
        """Tests that qnode arguments can be used as wires."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.set_shots(10)
        @qml.qnode(dev)
        def my_circuit(x, y):
            return (
                qml.probs(wires=x),
                qml.expval(qml.Z(x)),
                qml.var(qml.X(y)),
                qml.sample(wires=x),
            )

        args = (1, 2)
        module = my_circuit(*args)

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 5  # Device node + probs + expval + var + sample

        assert nodes["node1"]["label"] == "<name> probs|<wire> [arg0]"
        assert nodes["node2"]["label"] == "<name> expval(PauliZ)|<wire> [arg0]"
        assert nodes["node3"]["label"] == "<name> var(PauliX)|<wire> [arg1]"
        assert nodes["node4"]["label"] == "<name> sample|<wire> [arg0]"

    def test_visualize_pythonic_operators_on_meas(self):
        """Tests that we can use operators like +,-,%"""
        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.set_shots(3)
        @qml.qnode(dev)
        def my_workflow(x):
            return qml.probs(wires=[x % 3, x - 3, x + 3]), qml.sample(wires=[x % 3, x - 3, x + 3])

        args = (1,)
        module = my_workflow(*args)

        # Construct DAG
        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 3  # Device node + meas

        assert nodes["node1"]["label"] == "<name> probs|<wire> [(arg5 % 3), (arg5 - 3), (arg5 + 3)]"
        assert (
            nodes["node2"]["label"] == "<name> sample|<wire> [(arg5 % 3), (arg5 - 3), (arg5 + 3)]"
        )


@pytest.mark.usefixtures("use_both_frontend")
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

        # Check edges
        #           -> X -> Y
        # NullQubit -> Z -> H -> S
        #           -> T
        expected_edges = (
            ("NullQubit", "PauliX"),
            ("NullQubit", "PauliZ"),
            ("NullQubit", "T"),
            ("PauliX", "PauliY"),
            ("PauliZ", "Hadamard"),
            ("Hadamard", "S"),
        )
        assert_dag_structure(nodes, edges, expected_edges)

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
            qml.Z(0)

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes
        # node0 -> NullQubit

        # Check all nodes
        assert "PauliX" in nodes["node1"]["label"]
        assert "PauliY" in nodes["node2"]["label"]

        # Check edges
        #                   for loop
        # NullQubit -> X ----> Y ----> Z
        expected_edges = (
            ("NullQubit", "PauliX"),
            ("PauliX", "PauliY"),
            ("PauliY", "PauliZ"),
        )
        assert_dag_structure(nodes, edges, expected_edges)

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
            qml.Z(0)

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes
        # node0 -> NullQubit

        # Check all nodes
        assert "PauliX" in nodes["node1"]["label"]
        assert "PauliY" in nodes["node2"]["label"]

        # Check edges
        #                    while loop
        # NullQubit -> X ------> Y ------> Z
        expected_edges = (
            ("NullQubit", "PauliX"),
            ("PauliX", "PauliY"),
            ("PauliY", "PauliZ"),
        )
        assert_dag_structure(nodes, edges, expected_edges)

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

        expected_edges = (
            ("NullQubit", "PauliX"),
            ("NullQubit", "T"),
            ("T", "S"),
            ("PauliX", "RX"),
            ("PauliX", "RY"),
            ("PauliX", "RZ"),
            ("RX", "Hadamard"),
            ("RY", "Hadamard"),
            ("RZ", "Hadamard"),
        )
        assert_dag_structure(nodes, edges, expected_edges)

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

        expected_edges = (
            ("NullQubit", "PauliX"),
            ("NullQubit", "T"),
            ("T", "PauliY"),
            ("PauliX", "Hadamard"),
            ("PauliX", "RZ"),
            ("PauliX", "PauliZ"),
            ("RZ", "Hadamard"),
            ("PauliZ", "Hadamard"),
        )
        assert_dag_structure(nodes, edges, expected_edges)

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

        expected_edges = (
            ("NullQubit", "RX"),
            ("NullQubit", "RY"),
            ("NullQubit", "RZ"),
            ("RX", "CNOT"),
            ("RY", "CNOT"),
            ("CNOT", "Toffoli"),
            ("RZ", "Toffoli"),
        )
        assert_dag_structure(nodes, edges, expected_edges)

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
            qml.RY(0, y)

        args = (1, 2)
        module = my_workflow(*args)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        expected_edges = (
            ("NullQubit", "PauliX"),
            ("NullQubit", "PauliY"),
            ("NullQubit", "PauliZ"),
            # choke into the dynamic hadamard
            ("PauliX", "Hadamard", {"style": "dashed"}),
            ("PauliY", "Hadamard", {"style": "dashed"}),
            ("PauliZ", "Hadamard", {"style": "dashed"}),
            # fan out to static ops
            ("Hadamard", "S"),
            ("Hadamard", "T"),
            # choke again
            ("S", "RY", {"style": "dashed"}),
            ("T", "RY", {"style": "dashed"}),
        )
        assert_dag_structure(nodes, edges, expected_edges)

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
        assert "Hadamard" in nodes["node1"]["label"]

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
        assert "PauliX" in nodes["node1"]["label"]
        assert "PauliY" in nodes["node2"]["label"]
        assert "PauliZ" in nodes["node3"]["label"]

        # Check all edges
        assert len(edges) == 3
        assert ("node0", "node1") in edges
        assert ("node1", "node2") in edges
        assert ("node2", "node3") in edges
        assert edges[("node2", "node3")]["attrs"]["style"] == "dashed"

    def test_complex_connectivity_for_loop(self):
        """Tests a complicated connectivity through a for loop."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow(a, b):
            qml.X(a)
            for i in range(3):
                qml.H(0)
                qml.Y(i)
            qml.S(0)
            qml.Z(b)

        module = my_workflow(1, 2)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        expected_edges = (
            ("NullQubit", "PauliX", {"style": "dashed"}),
            ("PauliX", "Hadamard"),
            ("Hadamard", "PauliY", {"style": "dashed"}),
            ("PauliY", "S"),
            ("S", "PauliZ", {"style": "dashed"}),
        )
        assert_dag_structure(nodes, edges, expected_edges)

    def test_complex_connectivity_while_loop(self):
        """Tests a complicated connectivity through a while loop."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow(a, b):
            qml.X(a)
            counter = 0
            while counter < 5:
                qml.H(0)
                qml.Y(counter)
                counter += 1
            qml.S(0)
            qml.Z(b)

        module = my_workflow(1, 2)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        expected_edges = (
            ("NullQubit", "PauliX", {"style": "dashed"}),
            ("PauliX", "Hadamard"),
            ("Hadamard", "PauliY", {"style": "dashed"}),
            ("PauliY", "S"),
            ("S", "PauliZ", {"style": "dashed"}),
        )
        assert_dag_structure(nodes, edges, expected_edges)

    def test_basic_static_conditional(self):
        """Tests a basic static example of a conditional."""

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=3))
        def my_workflow(a):
            if a == 2:
                qml.X(0)
            elif a == 3:
                qml.Y(0)
            else:
                qml.Z(0)
            qml.H(a)

        module = my_workflow(1)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        expected_edges = (
            ("NullQubit", "PauliX"),
            ("NullQubit", "PauliY"),
            ("NullQubit", "PauliZ"),
            ("PauliX", "Hadamard", {"style": "dashed"}),
            ("PauliY", "Hadamard", {"style": "dashed"}),
            ("PauliZ", "Hadamard", {"style": "dashed"}),
        )
        assert_dag_structure(nodes, edges, expected_edges)

    def test_basic_static_conditional_in_between_dynamic(self):
        """Tests a basic static example of a conditional."""

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=3))
        def my_workflow(a, b):
            qml.S(b)
            if a == 2:
                qml.X(0)
            elif a == 3:
                qml.Y(0)
            else:
                qml.Z(0)
            qml.H(a)

        module = my_workflow(1, 2)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        expected_edges = (
            ("NullQubit", "S", {"style": "dashed"}),
            ("S", "PauliX"),
            ("S", "PauliY"),
            ("S", "PauliZ"),
            ("PauliX", "Hadamard", {"style": "dashed"}),
            ("PauliY", "Hadamard", {"style": "dashed"}),
            ("PauliZ", "Hadamard", {"style": "dashed"}),
        )
        assert_dag_structure(nodes, edges, expected_edges)

    def test_basic_dynamic_conditional(self):
        """Tests a basic example of a conditional with dynamic wires."""

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=3))
        def my_workflow(a, b, c):
            qml.H(0)
            if a == 2:
                qml.X(a)
            elif a == b:
                qml.Y(c)
            else:
                qml.Z(c)
            qml.S(0)

        module = my_workflow(1, 2, 3)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        expected_edges = (
            ("NullQubit", "Hadamard"),
            ("Hadamard", "PauliX", {"style": "dashed"}),
            ("Hadamard", "PauliY", {"style": "dashed"}),
            ("Hadamard", "PauliZ", {"style": "dashed"}),
            ("PauliX", "S"),
            ("PauliY", "S"),
            ("PauliZ", "S"),
        )
        assert_dag_structure(nodes, edges, expected_edges)

    def test_connectivity_through_simple_if(self):
        """Tests that a simple if can be visualized."""

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=3))
        def my_workflow(a):
            qml.H(0)
            if a == 2:
                qml.X(a)
                qml.Y(0)
            qml.Z(0)

        module = my_workflow(1)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        expected_edges = (
            ("NullQubit", "Hadamard"),
            ("Hadamard", "PauliX", {"style": "dashed"}),
            ("PauliX", "PauliY"),
            ("PauliY", "PauliZ"),
        )
        assert_dag_structure(nodes, edges, expected_edges)

    def test_complex_connectivity_if_elif_else(self):
        """Tests that complex connectivity can go through a conditional."""

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=3))
        def my_workflow(a, b, c):
            qml.X(a)
            if a == 2:
                qml.H(0)
                qml.Y(1)
            elif a == b:
                qml.T(c)
            else:
                qml.Z(b)
            qml.S(0)
            qml.RZ(0, b)

        module = my_workflow(1, 2, 3)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        expected_edges = (
            ("NullQubit", "PauliX", {"style": "dashed"}),
            ("PauliX", "Hadamard"),
            ("PauliX", "PauliY"),
            ("PauliX", "T", {"style": "dashed"}),
            ("PauliX", "PauliZ", {"style": "dashed"}),
            ("Hadamard", "S"),
            ("T", "S"),
            ("PauliZ", "S"),
            ("S", "RZ", {"style": "dashed"}),
            ("PauliY", "RZ", {"style": "dashed"}),
        )
        assert_dag_structure(nodes, edges, expected_edges)

    def test_complex_connectivity_nested_if_with_else(self):
        """Tests that complex connectivity can go through a conditional."""

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=3))
        def my_workflow(a, b):
            qml.X(a)
            qml.S(0)
            if a == 2:
                if b == 2:
                    qml.RX(0, 0)
            else:
                qml.Y(0)

        module = my_workflow(1, 2)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        expected_edges = (
            ("NullQubit", "PauliX", {"style": "dashed"}),
            ("PauliX", "S"),
            ("S", "RX"),
            ("S", "PauliY"),
        )
        assert_dag_structure(nodes, edges, expected_edges)

    def test_complex_connectivity_nested_if(self):
        """Tests that complex connectivity can go through a conditional."""

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=3))
        def my_workflow(a, b):
            qml.X(a)
            if a == 2:
                qml.H(0)
                if b == 2:
                    qml.RX(0, 0)
                qml.Y(1)
            qml.S(0)
            return qml.probs()

        module = my_workflow(1, 2)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        expected_edges = (
            ("NullQubit", "PauliX", {"style": "dashed"}),
            ("PauliX", "Hadamard"),
            ("Hadamard", "RX"),
            ("RX", "S"),
            ("PauliX", "PauliY"),
            ("PauliY", "probs", {"style": "dashed"}),
            ("S", "probs", {"style": "dashed"}),
        )
        assert_dag_structure(nodes, edges, expected_edges)

    def test_complex_connectivity_conditional_inside_control_flow(self):
        """Tests the interaction with conditional inside of control flow"""

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=3))
        def my_workflow(x, y):
            qml.X(0)
            qml.Y(1)
            qml.H(x)

            for i in range(3):
                qml.S(0)
                if i == 3:
                    qml.T(0)

            qml.RY(0, x)

        module = my_workflow(1, 2)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        expected_edges = (
            ("NullQubit", "PauliX"),
            ("NullQubit", "PauliY"),
            ("PauliX", "Hadamard", {"style": "dashed"}),
            ("PauliY", "Hadamard", {"style": "dashed"}),
            ("Hadamard", "S"),
            ("S", "T"),
            ("T", "RY", {"style": "dashed"}),
        )
        assert_dag_structure(nodes, edges, expected_edges)

    def test_complex_connectivity_conditional_dynamic_branching_static_node_after(self):
        """Tests that complex connectivity can go through a conditional."""

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=3))
        def my_workflow(x, y):
            if x == y:
                qml.Y(0)
            else:
                qml.Z(x)
            qml.H(0)

        module = my_workflow(1, 2)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        expected_edges = (
            ("NullQubit", "PauliY"),
            ("NullQubit", "PauliZ", {"style": "dashed"}),
            ("PauliY", "Hadamard"),
            ("PauliZ", "Hadamard"),
        )
        assert_dag_structure(nodes, edges, expected_edges)

    def test_complex_connectivity_conditional_dynamic_branching_static_and_dyn_node_after(self):
        """Tests that complex connectivity can go through a conditional."""

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=3))
        def my_workflow(x, y):
            if x == y:
                qml.Y(0)
            else:
                qml.Z(x)
            qml.H(0)
            qml.X(x)

        module = my_workflow(1, 2)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        expected_edges = (
            ("NullQubit", "PauliY"),
            ("NullQubit", "PauliZ", {"style": "dashed"}),
            ("PauliY", "Hadamard"),
            ("PauliZ", "Hadamard"),
            ("Hadamard", "PauliX", {"style": "dashed"}),
        )
        assert_dag_structure(nodes, edges, expected_edges)

    def test_complex_connectivity_conditional_dynamic_branching_no_node_before(self):
        """Tests that complex connectivity can go through a conditional."""

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=3))
        def my_workflow(x, y):
            if x == y:
                qml.Y(0)
            else:
                qml.Z(x)
            qml.H(y)

        module = my_workflow(1, 2)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        expected_edges = (
            ("NullQubit", "PauliY"),
            ("NullQubit", "PauliZ", {"style": "dashed"}),
            ("PauliY", "Hadamard", {"style": "dashed"}),
            ("PauliZ", "Hadamard", {"style": "dashed"}),
        )
        assert_dag_structure(nodes, edges, expected_edges)

    def test_complex_connectivity_conditional_dynamic_branching(self):
        """Tests that complex connectivity can go through a conditional."""

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=3))
        def my_workflow(x, y):
            qml.X(x)
            if x == y:
                qml.Y(0)
            else:
                qml.Z(x)
            qml.H(y)

        module = my_workflow(1, 2)

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        edges = utility.dag_builder.edges
        nodes = utility.dag_builder.nodes

        expected_edges = (
            ("NullQubit", "PauliX", {"style": "dashed"}),
            ("PauliX", "PauliY"),
            ("PauliX", "PauliZ", {"style": "dashed"}),
            ("PauliY", "Hadamard", {"style": "dashed"}),
            ("PauliZ", "Hadamard", {"style": "dashed"}),
        )
        assert_dag_structure(nodes, edges, expected_edges)


@pytest.mark.usefixtures("use_both_frontend")
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
        assert "PauliX" in nodes["node1"]["label"]
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
        assert "PauliX" in nodes["node1"]["label"]
        assert "PauliY" in nodes["node2"]["label"]
        assert "PauliZ" in nodes["node3"]["label"]
        assert "Hadamard" in nodes["node4"]["label"]
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
        assert "PauliX" in nodes["node1"]["label"]
        assert "PauliY" in nodes["node2"]["label"]
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
        """Tests that a terminal measurement on a mix of dynamic and static wires connects."""

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
        assert "PauliX" in nodes["node1"]["label"]
        assert "PauliY" in nodes["node2"]["label"]
        assert "PauliZ" in nodes["node3"]["label"]
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
        """Tests that a terminal measurement on a mix of dynamic and static wires connects."""

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
        assert "PauliX" in nodes["node1"]["label"]
        assert "PauliY" in nodes["node2"]["label"]
        assert "Hadamard" in nodes["node3"]["label"]
        assert "expval(PauliZ)" in nodes["node4"]["label"]

        # Check all edges
        assert len(edges) == 4
        assert ("node0", "node1") in edges
        assert ("node1", "node2") in edges
        assert ("node2", "node3") in edges
        assert edges[("node2", "node3")]["attrs"]["style"] == "dashed"
        assert ("node3", "node4") in edges
        assert edges[("node3", "node4")]["attrs"]["style"] == "dashed"

    def test_terminal_measurement_dyn_after_static(self):
        """Tests that a terminal measurement on a mix of dynamic and static wires connects."""

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
        assert "PauliX" in nodes["node1"]["label"]
        assert "PauliY" in nodes["node2"]["label"]
        assert "expval(PauliZ)" in nodes["node3"]["label"]

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


@pytest.mark.usefixtures("use_both_frontend")
class TestCtrl:
    """Tests that the ctrl transform is visualized correctly."""

    def test_ctrl_function(self):
        """Test that the ctrl of a function works."""

        dev = qml.device("null.qubit", wires=1)

        def op():
            return qml.H(0)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            qml.ctrl(op, control=1)()

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes

        # cluster0 -> qjit
        # cluster1 -> my_workflow
        assert "CH" in nodes["node1"]["label"]
        assert "[1, 0]" in nodes["node1"]["label"]
        assert nodes["node1"]["parent_cluster_uid"] == "cluster1"

    def test_ctrl_operator_instance(self):
        """Test that the ctrl of an operator instance works."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            qml.ctrl(qml.H(0), control=1)

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes

        # cluster0 -> qjit
        # cluster1 -> my_workflow
        assert "CH" in nodes["node1"]["label"]
        assert "[1, 0]" in nodes["node1"]["label"]
        assert nodes["node1"]["parent_cluster_uid"] == "cluster1"

    def test_ctrl_operator_type(self):
        """Test that the ctrl of an operator instance works."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            qml.ctrl(qml.H, control=1)(0)

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes

        # cluster0 -> qjit
        # cluster1 -> my_workflow
        assert "CH" in nodes["node1"]["label"]
        assert "[1, 0]" in nodes["node1"]["label"]
        assert nodes["node1"]["parent_cluster_uid"] == "cluster1"

    def test_ctrl_operator_without_alias(self):
        """Test that the ctrl of an operator instance that doesn't have an alias works."""

        dev = qml.device("null.qubit", wires=2)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            # Use two control wires so we avoid the CH alias
            qml.ctrl(qml.H(0), control=[1, 2])
            qml.ctrl(qml.H, control=[1, 2])(0)

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        nodes = utility.dag_builder.nodes

        # cluster0 -> qjit
        # cluster1 -> my_workflow
        assert "C(Hadamard)" in nodes["node1"]["label"]
        assert "[1, 2, 0]" in nodes["node1"]["label"]
        assert nodes["node1"]["parent_cluster_uid"] == "cluster1"
        assert "C(Hadamard)" in nodes["node2"]["label"]
        assert "[1, 2, 0]" in nodes["node2"]["label"]
        assert nodes["node2"]["parent_cluster_uid"] == "cluster1"


@pytest.mark.usefixtures("use_both_frontend")
class TestAdjoint:
    """Tests that the ctrl transform is visualized correctly."""

    def test_adjoint_function(self):
        """Test that the adjoint of a function works."""

        dev = qml.device("null.qubit", wires=1)

        def op():
            return qml.H(0)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            qml.adjoint(op)()

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        clusters = utility.dag_builder.clusters
        nodes = utility.dag_builder.nodes

        # cluster0 -> qjit
        # cluster1 -> my_workflow
        assert clusters["cluster2"]["label"] == "adjoint"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"
        assert "Hadamard" in nodes["node1"]["label"]
        assert nodes["node1"]["parent_cluster_uid"] == "cluster2"

    def test_adjoint_operator_instance(self):
        """Test that the adjoint of an operator instance works."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            qml.adjoint(qml.H(0))

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        clusters = utility.dag_builder.clusters
        nodes = utility.dag_builder.nodes

        # cluster0 -> qjit
        # cluster1 -> my_workflow

        # Because it is an operator instance, no cluster needed
        assert "Adjoint(Hadamard)" in nodes["node1"]["label"]
        assert nodes["node1"]["parent_cluster_uid"] == "cluster1"

    def test_adjoint_operator_type(self):
        """Test that the adjoint of an operator instance works."""

        dev = qml.device("null.qubit", wires=1)

        @xdsl_from_qjit
        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(dev)
        def my_workflow():
            qml.adjoint(qml.H)(0)

        module = my_workflow()

        utility = ConstructCircuitDAG(FakeDAGBuilder())
        utility.construct(module)

        clusters = utility.dag_builder.clusters
        nodes = utility.dag_builder.nodes

        # cluster0 -> qjit
        # cluster1 -> my_workflow
        assert clusters["cluster2"]["label"] == "adjoint"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"
        assert "Hadamard" in nodes["node1"]["label"]
        assert nodes["node1"]["parent_cluster_uid"] == "cluster2"
