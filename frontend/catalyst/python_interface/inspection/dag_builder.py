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
"""File that defines the DAGBuilder abstract base class."""

from abc import ABC, abstractmethod
from typing import Any, TypeAlias

ClusterUID: TypeAlias = str
NodeUID: TypeAlias = str


class DAGBuilder(ABC):
    """An abstract base class for building Directed Acyclic Graphs (DAGs).

    This class provides a simple interface with three core methods 
    (`add_node`, `add_edge` and `add_cluster`). You can override these methods to implement any 
    backend, like `pydot` or `graphviz` or even `matplotlib`.

    Outputting your graph can be done by overriding `to_file` and `to_string`.
    """

    @abstractmethod
    def add_node(
        self,
        uid: NodeUID,
        label: str,
        *,
        cluster_uid: ClusterUID | None = None,
        **attrs: Any,
    ) -> None:
        """Add a single node to the graph.

        Args:
            uid (str): Unique node ID to identify this node.
            label (str): The text to display on the node when rendered.
            cluster_uid (str | None): Optional unique ID of the cluster this node belongs to. 
                If `None`, this node gets added on the base graph.
            **attrs (Any): Any additional styling keyword arguments.

        """
        raise NotImplementedError

    @abstractmethod
    def add_edge(self, from_uid: NodeUID, to_uid: NodeUID, **attrs: Any) -> None:
        """Add a single directed edge between nodes in the graph.

        Args:
            from_uid (str): The unique ID of the source node.
            to_uid (str): The unique ID of the destination node.
            **attrs (Any): Any additional styling keyword arguments.

        """
        raise NotImplementedError

    @abstractmethod
    def add_cluster(
        self,
        uid: ClusterUID,
        *,
        label: str | None = None,
        cluster_uid: ClusterUID | None = None,
        **attrs: Any,
    ) -> None:
        """Add a single cluster to the graph.

        A cluster is a specific type of subgraph where the nodes and edges contained
        within it are visually and logically grouped.

        Args:
            uid (str): Unique cluster ID to identify this cluster.
            label (str | None): Optional text to display as a label on the cluster when rendered.
            cluster_uid (str | None): Optional unique ID of the cluster this cluster belongs to.
                If `None`, the cluster will be placed on the base graph.
            **attrs (Any): Any additional styling keyword arguments.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def nodes(self) -> dict[NodeUID, dict[str, Any]]:
        """Retrieve the current set of nodes in the graph.

        Returns:
            nodes (dict[str, dict[str, Any]]): A dictionary that maps the 
                node's ID to its node information.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def edges(self) -> list[dict[str, Any]]:
        """Retrieve the current set of edges in the graph.

        Returns:
            edges (list[dict[str, Any]]): A list of edges where each element in the list contains 
                a dictionary of edge information.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def clusters(self) -> dict[ClusterUID, dict[str, Any]]:
        """Retrieve the current set of clusters in the graph.

        Returns:
            clusters (dict[str, dict[str, Any]]): A dictionary that maps the cluster's ID to 
                its cluster information.
        """
        raise NotImplementedError

    @abstractmethod
    def to_file(self, output_filename: str) -> None:
        """Save the graph to a file.

        The implementation should ideally infer the output format
        (e.g., 'png', 'svg') from this filename's extension.

        Args:
            output_filename (str): Desired filename for the graph.

        """
        raise NotImplementedError

    @abstractmethod
    def to_string(self) -> str:
        """Return the graph as a string.

        This is used to get the graph's representation in a standard string format like DOT.

        Returns:
            str: A string representation of the graph.
        """
        raise NotImplementedError
