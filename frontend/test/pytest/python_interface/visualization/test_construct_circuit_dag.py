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

        clusters = utility.dag_builder.get_clusters()
        # 1 = "jit_my_workflow"
        # 2 = "my_workflow"
        # 3 = "setup"
        # 4 = "teardown"
        assert len(clusters) == 4

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

        clusters = utility.dag_builder.get_clusters()
        # 1 = "jit_my_workflow"
        # 2 = "my_qnode1"
        # 3 = "my_qnode2"
        # 4 = "setup"
        # 5 = "teardown"
        assert len(clusters) == 3
