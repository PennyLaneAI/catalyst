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
"""Unit tests for the PyDotDAGBuilder subclass."""

from unittest.mock import MagicMock

import pytest

pydot = pytest.importorskip("pydot")

pytestmark = pytest.mark.xdsl
xdsl = pytest.importorskip("xdsl")
# pylint: disable=wrong-import-position
from catalyst.python_interface.inspection.pydot_dag_builder import PyDotDAGBuilder


def test_initialization_defaults():
    """Tests the default graph attributes are as expected."""

    dag_builder = PyDotDAGBuilder()

    assert isinstance(dag_builder.graph, pydot.Dot)
    # Ensure it's a directed graph
    assert dag_builder.graph.get_graph_type() == "digraph"
    # Ensure that it flows top to bottom
    assert dag_builder.graph.get_rankdir() == "TB"
    # Ensure edges can be connected directly to clusters / subgraphs
    assert dag_builder.graph.get_compound() == "true"
    # Ensure duplicated edges cannot be added
    assert dag_builder.graph.obj_dict["strict"] is True
    # Ensure edges are correct
    assert dag_builder.graph.obj_dict["attributes"]["splines"] == "polyline"


class TestExceptions:
    """Tests the various exceptions defined in the class."""

    def test_duplicate_node_ids(self):
        """Tests that a ValueError is raised for duplicate nodes."""

        dag_builder = PyDotDAGBuilder()

        dag_builder.add_node("0", "node0")
        with pytest.raises(ValueError, match="Node ID 0 already present in graph."):
            dag_builder.add_node("0", "node1")

    def test_edge_duplicate_source_destination(self):
        """Tests that a ValueError is raised when an edge is created with the
        same source and destination"""

        dag_builder = PyDotDAGBuilder()

        dag_builder.add_node("0", "node0")
        with pytest.raises(ValueError, match="Edges must connect two unique IDs."):
            dag_builder.add_edge("0", "0")

    def test_edge_missing_ids(self):
        """Tests that an error is raised if IDs are missing."""

        dag_builder = PyDotDAGBuilder()

        dag_builder.add_node("0", "node0")
        with pytest.raises(ValueError, match="Destination is not found in the graph."):
            dag_builder.add_edge("0", "1")

        dag_builder = PyDotDAGBuilder()

        dag_builder.add_node("1", "node1")
        with pytest.raises(ValueError, match="Source is not found in the graph."):
            dag_builder.add_edge("0", "1")

    def test_duplicate_cluster_id(self):
        """Tests that an exception is raised if an ID is already present."""

        dag_builder = PyDotDAGBuilder()

        dag_builder.add_cluster("0")
        with pytest.raises(ValueError, match="Cluster ID 0 already present in graph."):
            dag_builder.add_cluster("0")


class TestAddMethods:
    """Test that elements can be added to the graph."""

    def test_add_node(self):
        """Unit test the `add_node` method."""

        dag_builder = PyDotDAGBuilder()

        dag_builder.add_node("0", "node0")
        node_list = dag_builder.graph.get_node_list()
        assert len(node_list) == 1
        assert node_list[0].get_label() == "node0"

    def test_add_edge(self):
        """Unit test the `add_edge` method."""

        dag_builder = PyDotDAGBuilder()
        dag_builder.add_node("0", "node0")
        dag_builder.add_node("1", "node1")
        dag_builder.add_edge("0", "1")

        assert len(dag_builder.graph.get_edges()) == 1
        edge = dag_builder.graph.get_edges()[0]
        assert edge.get_source() == "0"
        assert edge.get_destination() == "1"

    def test_add_cluster(self):
        """Unit test the 'add_cluster' method."""

        dag_builder = PyDotDAGBuilder()
        dag_builder.add_cluster("0")

        assert len(dag_builder.graph.get_subgraphs()) == 1
        assert dag_builder.graph.get_subgraphs()[0].get_name() == "cluster_0"

    def test_add_node_to_parent_graph(self):
        """Tests that you can add a node to a parent graph."""
        dag_builder = PyDotDAGBuilder()

        # Create node
        dag_builder.add_node("0", "node0")

        # Create cluster
        dag_builder.add_cluster("c0")

        # Create node inside cluster
        dag_builder.add_node("1", "node1", cluster_uid="c0")

        # Verify graph structure
        root_graph = dag_builder.graph

        # Make sure the base graph has node0
        assert root_graph.get_node("0"), "Node 0 not found in root graph"

        # Get the cluster and verify it has node1 and not node0
        cluster_list = root_graph.get_subgraph("cluster_c0")
        assert cluster_list, "Subgraph 'cluster_c0' not found"
        cluster_graph = cluster_list[0]  # Get the actual subgraph object

        assert cluster_graph.get_node("1"), "Node 1 not found in cluster 'c0'"
        assert not cluster_graph.get_node("0"), (
            "Node 0 was incorrectly added to cluster"
        )

        assert not root_graph.get_node("1"), "Node 1 was incorrectly added to root"

    def test_add_cluster_to_parent_graph(self):
        """Test that you can add a cluster to a parent graph."""
        dag_builder = PyDotDAGBuilder()

        # Level 0 (Root): Adds cluster on top of base graph
        dag_builder.add_node("n_root", "node_root")

        # Level 1 (c0): Add node on outer cluster
        dag_builder.add_cluster("c0")
        dag_builder.add_node("n_outer", "node_outer", cluster_uid="c0")

        # Level 2 (c1): Add node on inner cluster
        dag_builder.add_cluster("c1", cluster_uid="c0")
        dag_builder.add_node("n_inner", "node_inner", cluster_uid="c1")

        root_graph = dag_builder.graph

        outer_cluster_list = root_graph.get_subgraph("cluster_c0")
        assert outer_cluster_list, "Outer cluster 'c0' not found in root"
        c0 = outer_cluster_list[0]

        inner_cluster_list = c0.get_subgraph("cluster_c1")
        assert inner_cluster_list, "Inner cluster 'c1' not found in 'c0'"
        c1 = inner_cluster_list[0]

        # Check Level 0 (Root)
        assert root_graph.get_node("n_root"), "n_root not found in root"
        assert root_graph.get_subgraph("cluster_c0"), "c0 not found in root"
        assert not root_graph.get_node("n_outer"), "n_outer incorrectly found in root"
        assert not root_graph.get_node("n_inner"), "n_inner incorrectly found in root"
        assert not root_graph.get_subgraph("cluster_c1"), "c1 incorrectly found in root"

        # Check Level 1 (c0)
        assert c0.get_node("n_outer"), "n_outer not found in c0"
        assert c0.get_subgraph("cluster_c1"), "c1 not found in c0"
        assert not c0.get_node("n_root"), "n_root incorrectly found in c0"
        assert not c0.get_node("n_inner"), "n_inner incorrectly found in c0"

        # Check Level 2 (c1)
        assert c1.get_node("n_inner"), "n_inner not found in c1"
        assert not c1.get_node("n_root"), "n_root incorrectly found in c1"
        assert not c1.get_node("n_outer"), "n_outer incorrectly found in c1"


class TestAttributes:
    """Tests that the attributes for elements in the graph are overridden correctly."""

    def test_default_graph_attrs(self):
        """Test that default graph attributes can be set."""

        dag_builder = PyDotDAGBuilder(attrs={"fontname": "Times"})

        dag_builder.add_node("0", "node0")
        node0 = dag_builder.graph.get_node("0")[0]
        assert node0.get("fontname") == "Times"

        dag_builder.add_cluster("1")
        cluster = dag_builder.graph.get_subgraphs()[0]
        assert cluster.get("fontname") == "Times"

    def test_add_node_with_attrs(self):
        """Tests that default attributes are applied and can be overridden."""
        dag_builder = PyDotDAGBuilder(attrs={"fillcolor": "lightblue", "penwidth": 3})

        # Defaults
        dag_builder.add_node("0", "node0")
        node0 = dag_builder.graph.get_node("0")[0]
        assert node0.get("fillcolor") == "lightblue"
        assert node0.get("penwidth") == 3

        # Make sure we can override
        dag_builder.add_node("1", "node1", fillcolor="red", penwidth=4)
        node1 = dag_builder.graph.get_node("1")[0]
        assert node1.get("fillcolor") == "red"
        assert node1.get("penwidth") == 4

    def test_add_edge_with_attrs(self):
        """Tests that default attributes are applied and can be overridden."""
        dag_builder = PyDotDAGBuilder(attrs={"color": "lightblue4", "penwidth": 3})

        dag_builder.add_node("0", "node0")
        dag_builder.add_node("1", "node1")
        dag_builder.add_edge("0", "1")
        edge = dag_builder.graph.get_edges()[0]
        # Defaults defined earlier
        assert edge.get("color") == "lightblue4"
        assert edge.get("penwidth") == 3

        # Make sure we can override
        dag_builder.add_edge("0", "1", color="red", penwidth=4)
        edge = dag_builder.graph.get_edges()[1]
        assert edge.get("color") == "red"
        assert edge.get("penwidth") == 4

    def test_add_cluster_with_attrs(self):
        """Tests that default cluster attributes are applied and can be overridden."""
        dag_builder = PyDotDAGBuilder(
            attrs={
                "style": "solid",
                "fillcolor": None,
                "penwidth": 2,
                "fontname": "Helvetica",
            }
        )

        dag_builder.add_cluster("0")
        cluster1 = dag_builder.graph.get_subgraph("cluster_0")[0]

        # Defaults
        assert cluster1.get("style") == "solid"
        assert cluster1.get("fillcolor") is None
        assert cluster1.get("penwidth") == 2
        assert cluster1.get("fontname") == "Helvetica"

        dag_builder.add_cluster("1", style="filled", penwidth=10, fillcolor="red")
        cluster2 = dag_builder.graph.get_subgraph("cluster_1")[0]

        # Make sure we can override
        assert cluster2.get("style") == "filled"
        assert cluster2.get("penwidth") == 10
        assert cluster2.get("fillcolor") == "red"

        # Check that other defaults are still present
        assert cluster2.get("fontname") == "Helvetica"


class TestProperties:
    """Tests the properties."""

    def test_nodes(self):
        """Tests that nodes works."""
        dag_builder = PyDotDAGBuilder()

        dag_builder.add_node("0", "node0", fillcolor="red")
        dag_builder.add_cluster("c0")
        dag_builder.add_node("1", "node1", cluster_uid="c0")

        nodes = dag_builder.nodes

        assert len(nodes) == 2
        assert len(nodes["0"]) == 4

        assert nodes["0"]["uid"] == "0"
        assert nodes["0"]["label"] == "node0"
        assert nodes["0"]["cluster_uid"] == None
        assert nodes["0"]["attrs"]["fillcolor"] == "red"

        assert nodes["1"]["uid"] == "1"
        assert nodes["1"]["label"] == "node1"
        assert nodes["1"]["cluster_uid"] == "c0"

    def test_edges(self):
        """Tests that edges works."""

        dag_builder = PyDotDAGBuilder()
        dag_builder.add_node("0", "node0")
        dag_builder.add_node("1", "node1")
        dag_builder.add_edge("0", "1", penwidth=10)

        edges = dag_builder.edges

        assert len(edges) == 1

        assert edges[0]["from_uid"] == "0"
        assert edges[0]["to_uid"] == "1"
        assert edges[0]["attrs"]["penwidth"] == 10

    def test_clusters(self):
        """Tests that clusters property works."""

        dag_builder = PyDotDAGBuilder()
        dag_builder.add_cluster("0", "my_cluster", penwidth=10)

        clusters = dag_builder.clusters

        dag_builder.add_cluster(
            "1",
            "my_nested_cluster",
            cluster_uid="0",
        )
        clusters = dag_builder.clusters
        assert len(clusters) == 2

        assert len(clusters["0"]) == 4
        assert clusters["0"]["uid"] == "0"
        assert clusters["0"]["label"] == "my_cluster"
        assert clusters["0"]["cluster_uid"] == None
        assert clusters["0"]["attrs"]["penwidth"] == 10

        assert len(clusters["1"]) == 4
        assert clusters["1"]["uid"] == "1"
        assert clusters["1"]["label"] == "my_nested_cluster"
        assert clusters["1"]["cluster_uid"] == "0"


class TestOutput:
    """Test that the graph can be outputted correctly."""

    @pytest.mark.parametrize(
        "filename, file_format",
        [("my_graph", None), ("my_graph", "png"), ("prototype.trial1", "png")],
    )
    def test_to_file(self, monkeypatch, filename, file_format):
        """Tests that the `to_file` method works correctly."""
        dag_builder = PyDotDAGBuilder()

        # mock out the graph writing functionality
        mock_write = MagicMock()
        monkeypatch.setattr(dag_builder.graph, "write", mock_write)
        dag_builder.to_file(filename + "." + (file_format or "png"))

        # make sure the function handles extensions correctly
        mock_write.assert_called_once_with(
            filename + "." + (file_format or "png"), format=file_format or "png"
        )

    @pytest.mark.parametrize("file_format", ["pdf", "svg", "jpeg"])
    def test_other_supported_formats(self, monkeypatch, file_format):
        """Tests that the `to_file` method works with other formats."""
        dag_builder = PyDotDAGBuilder()

        # mock out the graph writing functionality
        mock_write = MagicMock()
        monkeypatch.setattr(dag_builder.graph, "write", mock_write)
        dag_builder.to_file(f"my_graph.{file_format}")

        # make sure the function handles extensions correctly
        mock_write.assert_called_once_with(
            f"my_graph.{file_format}", format=file_format
        )

    def test_to_string(self):
        """Tests that the `to_string` method works correclty."""

        dag_builder = PyDotDAGBuilder()
        dag_builder.add_node("n0", "node0")
        dag_builder.add_node("n1", "node1")
        dag_builder.add_edge("n0", "n1")

        string = dag_builder.to_string()
        assert isinstance(string, str)

        # make sure important things show up in the string
        assert "digraph" in string
        assert "n0" in string
        assert "n1" in string
        assert "n0 -> n1" in string
