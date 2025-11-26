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
"""File that defines the PyDotDAGBuilder subclass of DAGBuilder."""

import pathlib
from collections import ChainMap
from typing import Any

from .dag_builder import DAGBuilder

has_pydot = True
try:
    import pydot
    from pydot import Cluster, Dot, Edge, Graph, Node, Subgraph
except ImportError:
    has_pydot = False


class PyDotDAGBuilder(DAGBuilder):
    """A Directed Acyclic Graph builder for the PyDot backend."""

    def __init__(
        self,
        attrs: dict | None = None,
        node_attrs: dict | None = None,
        edge_attrs: dict | None = None,
        cluster_attrs: dict | None = None,
    ) -> None:
        """Initialize PyDotDAGBuilder instance.

        Args:
            attrs (dict | None): User default attributes to be used for all elements (nodes, edges, clusters) in the graph.
            node_attrs (dict | None): User default attributes for a node.
            edge_attrs (dict | None): User default attributes for an edge.
            cluster_attrs (dict | None): User default attributes for a cluster.

        """
        # Initialize the pydot graph:
        # - graph_type="digraph": Create a directed graph (edges have arrows).
        # - rankdir="TB": Set layout direction from Top to Bottom.
        # - compound="true": Allow edges to connect directly to clusters/subgraphs.
        # - strict=True: Prevent duplicate edges (e.g., A -> B added twice).
        self.graph: Dot = Dot(
            graph_type="digraph", rankdir="TB", compound="true", strict=True
        )

        # Use internal cache that maps cluster ID to actual pydot (Dot or Cluster) object
        # NOTE: This is needed so we don't need to traverse the graph to find the relevant
        # cluster object to modify
        self._subgraph_cache: dict[str, Graph] = {}

        # Internal state for graph structure
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._clusters: dict[str, dict[str, Any]] = {}

        _default_attrs: dict = (
            {"fontname": "Helvetica", "penwidth": 2} if attrs is None else attrs
        )
        self._default_node_attrs: dict = (
            {
                **_default_attrs,
                "shape": "ellipse",
                "style": "filled",
                "fillcolor": "lightblue",
                "color": "lightblue4",
                "penwidth": 3,
            }
            if node_attrs is None
            else node_attrs
        )
        self._default_edge_attrs: dict = (
            {
                "color": "lightblue4",
                "penwidth": 3,
            }
            if edge_attrs is None
            else edge_attrs
        )
        self._default_cluster_attrs: dict = (
            {
                **_default_attrs,
                "shape": "rectangle",
                "style": "solid",
            }
            if cluster_attrs is None
            else cluster_attrs
        )

    def add_node(
        self,
        id: str,
        label: str,
        cluster_id: str | None = None,
        **attrs: Any,
    ) -> None:
        """Add a single node to the graph.

        Args:
            id (str): Unique node ID to identify this node.
            label (str): The text to display on the node when rendered.
            cluster_id (str | None): Optional ID of the cluster this node belongs to.
            **attrs (Any): Any additional styling keyword arguments.

        """
        # Use ChainMap so you don't need to construct a new dictionary
        node_attrs: ChainMap = ChainMap(attrs, self._default_node_attrs)
        node = Node(id, label=label, **node_attrs)

        # Add node to cluster
        if cluster_id is None:
            self.graph.add_node(node)
        else:
            parent_cluster = self._subgraph_cache[cluster_id].add_node(node)

        self._nodes[id] = {
            "id": id,
            "label": label,
            "cluster_id": cluster_id,
            "attrs": dict(node_attrs),
        }

    def add_edge(self, from_id: str, to_id: str, **attrs: Any) -> None:
        """Add a single directed edge between nodes in the graph.

        Args:
            from_id (str): The unique ID of the source node.
            to_id (str): The unique ID of the destination node.
            **attrs (Any): Any additional styling keyword arguments.

        """
        # Use ChainMap so you don't need to construct a new dictionary
        edge_attrs: ChainMap = ChainMap(attrs, self._default_edge_attrs)
        edge = Edge(from_id, to_id, **edge_attrs)

        self.graph.add_edge(edge)

        self._edges.append(
            {"from_id": from_id, "to_id": to_id, "attrs": dict(edge_attrs)}
        )

    def add_cluster(
        self,
        id: str,
        node_label: str | None = None,
        cluster_id: str | None = None,
        **attrs: Any,
    ) -> None:
        """Add a single cluster to the graph.

        A cluster is a specific type of subgraph where the nodes and edges contained
        within it are visually and logically grouped.

        Args:
            id (str): Unique cluster ID to identify this cluster.
            node_label (str | None): The text to display on the information node within the cluster when rendered.
            cluster_id (str | None): Optional ID of the cluster this cluster belongs to. If `None`, the cluster will be positioned on the base graph.
            **attrs (Any): Any additional styling keyword arguments.

        """
        # Use ChainMap so you don't need to construct a new dictionary
        cluster_attrs: ChainMap = ChainMap(attrs, self._default_cluster_attrs)
        cluster = Cluster(id, **cluster_attrs)

        # Puts the label in a node within the cluster.
        # Ensures that any edges connecting nodes through the cluster
        # boundary don't block the label.
        # ┌───────────┐
        # │ ┌───────┐ │
        # │ │ label │ │
        # │ └───────┘ │
        # │           │
        # └───────────┘
        if node_label:
            node_id = f"{cluster_id}_info_node"
            rank_subgraph = Subgraph()
            node = Node(
                node_id,
                label=node_label,
                shape="rectangle",
                style="dashed",
                fontname="Helvetica",
                penwidth=2,
            )
            rank_subgraph.add_node(node)
            cluster.add_subgraph(rank_subgraph)
            cluster.add_node(node)

        # Add node to cluster
        if cluster_id is None:
            self.graph.add_subgraph(cluster)
        else:
            parent_cluster = self._subgraph_cache[cluster_id].add_node(cluster)

        self._clusters[id] = {
            "id": id,
            "cluster_label": cluster_attrs.get("label"),
            "node_label": node_label,
            "cluster_id": cluster_id,
            "attrs": dict(cluster_attrs),
        }

    @property
    def nodes(self) -> dict[str, dict[str, Any]]:
        """Retrieve the current set of nodes in the graph.

        Returns:
            nodes (dict[str, dict[str, Any]]): A dictionary that maps the node's ID to it's node information.
        """
        return self._nodes

    @property
    def edges(self) -> list[dict[str, Any]]:
        """Retrieve the current set of edges in the graph.

        Returns:
            edges (list[dict[str, Any]]): A list of edges where each element in the list contains a dictionary of edge information.
        """
        return self._edges

    @property
    def clusters(self) -> dict[str, dict[str, Any]]:
        """Retrieve the current set of clusters in the graph.

        Returns:
            clusters (dict[str, dict[str, Any]]): A dictionary that maps the cluster's ID to it's cluster information.
        """
        return self._clusters

    def to_file(self, output_filename: str) -> None:
        """Save the graph to a file.

        This method will infer the file's format (e.g., 'png', 'svg') from this filename's extension.
        If no extension is provided, the 'png' format will be the default.

        Args:
            output_filename (str): Desired filename for the graph. File extension can be included
                and if no file extension is provided, it will default to a `.png` file.

        """
        output_filename_path: pathlib.Path = pathlib.Path(output_filename)
        if not output_filename_path.suffix:
            output_filename_path = output_filename_path.with_suffix(".png")

        format = output_filename_path.suffix[1:].lower()

        self.graph.write(str(output_filename_path), format=format)

    def to_string(self) -> str:
        """Return the graph as a string.

        This is typically used to get the graph's representation in a standard string format like DOT.

        Returns:
            str: A string representation of the graph.
        """
        return self.graph.to_string()
