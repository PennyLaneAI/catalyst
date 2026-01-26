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
from shutil import which
from typing import Any

from .dag_builder import DAGBuilder

HAS_PYDOT = True
try:
    from pydot import Cluster, Dot, Edge, Graph, Node
except ImportError:
    HAS_PYDOT = False


HAS_GRAPHVIZ = True
if which("dot") is None:
    HAS_GRAPHVIZ = False

# pylint: disable=too-many-instance-attributes
class PyDotDAGBuilder(DAGBuilder):
    """A Directed Acyclic Graph builder for the PyDot backend.

    Args:
        attrs (dict | None): User default attributes to be used for all elements 
            (nodes, edges, clusters) in the graph.
        node_attrs (dict | None): User default attributes for a node.
        edge_attrs (dict | None): User default attributes for an edge.
        cluster_attrs (dict | None): User default attributes for a cluster.

    Example:
        >>> builder = PyDotDAGBuilder()
        >>> builder.add_node("n0", "node 0")
        >>> builder.add_cluster("c0")
        >>> builder.add_node("n1", "node 1", cluster_uid="c0")
        >>> print(builder.to_string())
        strict digraph G {
        rankdir=TB;
        compound=true;
        n0 [...];
        subgraph cluster_c0 {
            ...
        }
        n1 [...];
        }

    """

    def __init__(
        self,
        attrs: dict | None = None,
        node_attrs: dict | None = None,
        edge_attrs: dict | None = None,
        cluster_attrs: dict | None = None,
    ) -> None:
        if not HAS_GRAPHVIZ:
            raise ImportError(
                "The 'Graphviz' package is not found. Please install it for your system by "
                "following the instructions found here: https://graphviz.org/download/"
            )
        if not HAS_PYDOT:
            raise ImportError(
                "The 'pydot' package is not found. Please install with 'pip install pydot'."
            )


        # Initialize the pydot graph:
        # - graph_type="digraph": Create a directed graph (edges have arrows).
        # - rankdir="TB": Set layout direction from Top to Bottom.
        # - compound="true": Allow edges to connect directly to clusters/subgraphs.
        # - strict=True: Prevent duplicate edges (e.g., A -> B added twice).
        # - splines="polyline": Edges connecting clusters are polyline

        # NOTE: splines="ortho" have an open issue
        # on graphviz: https://gitlab.com/graphviz/graphviz/-/issues/1408
        self.graph: Dot = Dot(
            graph_type="digraph", rankdir="TB", compound="true", strict=True, splines="polyline"
        )

        # Use internal cache that maps cluster ID to actual pydot (Dot or Cluster) object
        # NOTE: This is needed so we don't need to traverse the graph to find the relevant
        # cluster object to modify
        self._subgraph_cache: dict[str, Graph] = {}

        # Internal state for graph structure
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._clusters: dict[str, dict[str, Any]] = {}

        _default_attrs: dict = {"fontname": "Helvetica"} if attrs is None else attrs
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
                "arrowsize": 0.5,
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
                "penwidth": 2,
            }
            if cluster_attrs is None
            else cluster_attrs
        )

    def add_node(
        self,
        uid: str,
        label: str,
        cluster_uid: str | None = None,
        **attrs: Any,
    ) -> None:
        """Add a single node to the graph.

        Args:
            uid (str): Unique node ID to identify this node.
            label (str): The text to display on the node when rendered.
            cluster_uid (str | None): Optional unique ID of the cluster this node belongs to.
            **attrs (Any): Any additional styling keyword arguments.

        Raises:
            ValueError: Node ID is already present in the graph.

        """
        if uid in self.nodes:
            raise ValueError(f"Node ID {uid} already present in graph.")

        # Use ChainMap so you don't need to construct a new dictionary
        node_attrs: ChainMap = ChainMap(attrs, self._default_node_attrs)
        node = Node(uid, label=label, **node_attrs)

        # Add node to cluster
        if cluster_uid is None:
            self.graph.add_node(node)
        else:
            self._subgraph_cache[cluster_uid].add_node(node)

        self._nodes[uid] = {
            "uid": uid,
            "label": label,
            "cluster_uid": cluster_uid,
            "attrs": dict(node_attrs),
        }

    def add_edge(self, from_uid: str, to_uid: str, **attrs: Any) -> None:
        """Add a single directed edge between nodes in the graph.

        Args:
            from_uid (str): The unique ID of the source node.
            to_uid (str): The unique ID of the destination node.
            **attrs (Any): Any additional styling keyword arguments.

        Raises:
            ValueError: Source and destination have the same ID
            ValueError: Source is not found in the graph.
            ValueError: Destination is not found in the graph.

        """
        if from_uid.split(":")[0] == to_uid.split(":")[0]:
            raise ValueError("Edges must connect two unique IDs.")
        if from_uid.split(":")[0] not in self.nodes:
            raise ValueError("Source is not found in the graph.")
        if to_uid.split(":")[0] not in self.nodes:
            raise ValueError("Destination is not found in the graph.")

        # Use ChainMap so you don't need to construct a new dictionary
        edge_attrs: ChainMap = ChainMap(attrs, self._default_edge_attrs)
        edge = Edge(from_uid, to_uid, **edge_attrs)

        self.graph.add_edge(edge)

        self._edges.append(
            {"from_uid": from_uid, "to_uid": to_uid, "attrs": dict(edge_attrs)}
        )

    def add_cluster(
        self,
        uid: str,
        label: str | None = None,
        cluster_uid: str | None = None,
        **attrs: Any,
    ) -> None:
        """Add a single cluster to the graph.

        A cluster is a specific type of subgraph where the nodes and edges contained
        within it are visually and logically grouped.

        Args:
            uid (str): Unique cluster ID to identify this cluster.
            label (str | None): Optional text to display as a label on the cluster when rendered.
            cluster_uid (str | None): Optional unique ID of the cluster this cluster belongs to. 
                If `None`, the cluster will be positioned on the base graph.
            **attrs (Any): Any additional styling keyword arguments.

        Raises:
            ValueError: Cluster ID is already present in the graph.
        """
        if uid in self.clusters:
            raise ValueError(f"Cluster ID {uid} already present in graph.")

        # Use ChainMap so you don't need to construct a new dictionary
        cluster_attrs: ChainMap = ChainMap(attrs, self._default_cluster_attrs)
        cluster = Cluster(uid, label=label, **cluster_attrs)

        # Record new cluster
        self._subgraph_cache[uid] = cluster

        # Add node to cluster
        if cluster_uid is None:
            self.graph.add_subgraph(cluster)
        else:
            self._subgraph_cache[cluster_uid].add_subgraph(cluster)

        self._clusters[uid] = {
            "uid": uid,
            "label": label,
            "cluster_uid": cluster_uid,
            "attrs": dict(cluster_attrs),
        }

    @property
    def nodes(self) -> dict[str, dict[str, Any]]:
        """Retrieve the current set of nodes in the graph.

        Returns:
            nodes (dict[str, dict[str, Any]]): A dictionary that maps the
                node's ID to its node information.
        """
        return self._nodes

    @property
    def edges(self) -> list[dict[str, Any]]:
        """Retrieve the current set of edges in the graph.

        Returns:
            edges (list[dict[str, Any]]): A list of edges where each element in the list 
                contains a dictionary of edge information.
        """
        return self._edges

    @property
    def clusters(self) -> dict[str, dict[str, Any]]:
        """Retrieve the current set of clusters in the graph.

        Returns:
            clusters (dict[str, dict[str, Any]]): A dictionary that maps the cluster's ID 
                to its cluster information.
        """
        return self._clusters

    def to_file(self, output_filename: str) -> None:
        """Save the graph to a file.

        This method will infer the file's format (e.g., 'png', 'svg') from this filename's 
        extension. If no extension is provided, the 'png' format will be the default.

        Args:
            output_filename (str): Desired filename for the graph. File extension can be included
                and if no file extension is provided, it will default to a `.png` file.

        """
        output_filename_path: pathlib.Path = pathlib.Path(output_filename)
        if not output_filename_path.suffix:
            output_filename_path = output_filename_path.with_suffix(".png")

        file_format = output_filename_path.suffix[1:].lower()

        self.graph.write(str(output_filename_path), format=file_format)

    def to_string(self) -> str:
        """Return the graph as a string.

        This is used to get the graph's representation in a standard string format like DOT.

        Returns:
            str: A string representation of the graph.
        """
        return self.graph.to_string()
