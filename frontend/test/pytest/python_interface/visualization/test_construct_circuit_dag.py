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
        # NOTE: node1 is the qml.H(0) in my_qnode1
        assert graph_nodes["node2"]["parent_cluster_uid"] == "cluster2"

        # Assert label is as expected
        assert graph_nodes["node2"]["label"] == "LightningSimulator"


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
        assert clusters["cluster2"]["node_label"] == "for loop"
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
        assert clusters["cluster2"]["node_label"] == "for loop"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"
        assert clusters["cluster3"]["node_label"] == "for loop"
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
        assert clusters["cluster2"]["node_label"] == "while loop"
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
        assert clusters["cluster2"]["node_label"] == "while loop"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"
        assert clusters["cluster3"]["node_label"] == "while loop"
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
        assert clusters["cluster3"]["node_label"] == "if"
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
        assert clusters["cluster3"]["node_label"] == "if"
        assert clusters["cluster3"]["parent_cluster_uid"] == "cluster2"
        assert clusters["cluster4"]["node_label"] == "elif"
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
        assert clusters["cluster2"]["cluster_label"] == "conditional"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"

        # Check 'if' cluster of first conditional has another conditional
        assert clusters["cluster3"]["node_label"] == "if"
        assert clusters["cluster3"]["parent_cluster_uid"] == "cluster2"

        # Second conditional
        assert clusters["cluster4"]["cluster_label"] == "conditional"
        assert clusters["cluster4"]["parent_cluster_uid"] == "cluster3"
        # Check 'if' and 'else' in second conditional
        assert clusters["cluster5"]["node_label"] == "if"
        assert clusters["cluster5"]["parent_cluster_uid"] == "cluster4"
        assert clusters["cluster6"]["node_label"] == "else"
        assert clusters["cluster6"]["parent_cluster_uid"] == "cluster4"

        # Check nested if / else is within the first if cluster
        assert clusters["cluster7"]["node_label"] == "else"
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
        assert clusters["cluster2"]["cluster_label"] == "conditional"
        assert clusters["cluster2"]["parent_cluster_uid"] == "cluster1"
        assert clusters["cluster3"]["node_label"] == "if"
        assert clusters["cluster3"]["parent_cluster_uid"] == "cluster2"
        assert clusters["cluster4"]["node_label"] == "elif"
        assert clusters["cluster4"]["parent_cluster_uid"] == "cluster2"
        assert clusters["cluster5"]["node_label"] == "else"
        assert clusters["cluster5"]["parent_cluster_uid"] == "cluster2"

        # Nested conditional (2) inside conditional (1)
        assert clusters["cluster6"]["cluster_label"] == "conditional"
        assert clusters["cluster6"]["parent_cluster_uid"] == "cluster5"
        assert clusters["cluster7"]["node_label"] == "if"
        assert clusters["cluster7"]["parent_cluster_uid"] == "cluster6"
        assert clusters["cluster8"]["node_label"] == "elif"
        assert clusters["cluster8"]["parent_cluster_uid"] == "cluster6"
        assert clusters["cluster9"]["node_label"] == "else"
        assert clusters["cluster9"]["parent_cluster_uid"] == "cluster6"


class TestGetLabel:
    """Tests the get_label utility."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "op", [qml.H(0), qml.QubitUnitary([[0, 1], [1, 0]], 0), qml.SWAP([0, 1])]
    )
    def test_standard_operator(self, op):
        """Tests against an operator instance."""
        wires = list(op.wires.labels)
        if wires == []:
            wires_str = "all"
        else:
            wires_str = f"[{', '.join(map(str, wires))}]"

        assert get_label(op) == f"<name> {op.name}|<wire> {wires_str}"

    def test_global_phase_operator(self):
        """Tests against a GlobalPhase operator instance."""
        assert get_label(qml.GlobalPhase(0.5)) == f"<name> GlobalPhase|<wire> all"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "meas",
        [
            qml.state(),
            qml.expval(qml.Z(0)),
            qml.var(qml.Z(0)),
            qml.probs(),
            qml.probs(wires=0),
            qml.probs(wires=[0, 1]),
            qml.sample(),
            qml.sample(wires=0),
            qml.sample(wires=[0, 1]),
        ],
    )
    def test_standard_measurement(self, meas):
        """Tests against an operator instance."""

        assert get_label(meas) == str(meas)


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

        # Ensure DAG only has one node
        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + operator

        # Make sure label has relevant info
        assert nodes["node1"]["label"] == get_label(op)

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

        # Ensure DAG only has one node
        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + operator

        # Compiler throws out the wires and they get converted to wires=[] no matter what
        assert nodes["node1"]["label"] == get_label(qml.GlobalPhase(0.5))

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

        # Ensure DAG only has one node
        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + operator

        assert nodes["node1"]["label"] == get_label(qml.QubitUnitary([[0, 1], [1, 0]], wires=0))

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

        # Ensure DAG only has one node
        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + operator

        assert nodes["node1"]["label"] == get_label(qml.MultiRZ(0.5, wires=[0]))

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

        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + operator

        assert nodes["node1"]["label"] == f"<name> MidMeasureMP|<wire> [0]"


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

        # Ensure DAG only has one node
        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + operator

        assert nodes["node1"]["label"] == get_label(qml.state())

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

        # Ensure DAG only has one node
        nodes = utility.dag_builder.nodes
        assert len(nodes) == 2  # Device node + operator

        assert nodes["node1"]["label"] == get_label(meas_fn(qml.Z(0)))

    @pytest.mark.unit
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
        assert len(nodes) == 2  # Device node + operator

        assert nodes["node1"]["label"] == get_label(op)

    @pytest.mark.unit
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
        assert len(nodes) == 2  # Device node + operator

        assert nodes["node1"]["label"] == get_label(op)
