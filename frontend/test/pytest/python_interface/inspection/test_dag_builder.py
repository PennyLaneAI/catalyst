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
"""Unit tests for the DAGBuilder abstract base class."""

from typing import Any

import pytest

from catalyst.python_interface.inspection.dag_builder import DAGBuilder

pytestmark = pytest.mark.xdsl


def test_concrete_implementation_works():
    """Unit test for concrete implementation of abc."""

    # pylint: disable=unused-argument,missing-function-docstring
    class ConcreteDAGBuilder(DAGBuilder):
        """Concrete subclass of an ABC for testing purposes."""

        def add_node(
            self,
            uid: str,
            label: str,
            cluster_id: str | None = None,
            **attrs: Any,
        ) -> None:
            return

        def add_edge(self, from_uid: str, to_uid: str, **attrs: Any) -> None:
            return

        def add_cluster(
            self,
            uid: str,
            label: str | None = None,
            cluster_id: str | None = None,
            **attrs: Any,
        ) -> None:
            return

        @property
        def nodes(self) -> dict[str, dict[str, Any]]:
            return {}

        @property
        def edges(self) -> list[dict[str, Any]]:
            return []

        @property
        def clusters(self) -> dict[str, dict[str, Any]]:
            return {}

        def to_file(self, output_filename: str) -> None:
            return

        def to_string(self) -> str:
            return "test"

    dag_builder = ConcreteDAGBuilder()
    # pylint: disable = assignment-from-none
    node = dag_builder.add_node("0", "node0")
    edge = dag_builder.add_edge("0", "1")
    cluster = dag_builder.add_cluster("0")
    nodes = dag_builder.nodes
    edges = dag_builder.edges
    clusters = dag_builder.clusters
    render = dag_builder.to_file("test.png")
    string = dag_builder.to_string()

    # pylint: disable=use-implicit-booleaness-not-comparison
    assert node is None
    assert nodes == {}
    assert edge is None
    assert edges == []
    assert cluster is None
    assert clusters == {}
    assert render is None
    assert string == "test"


def test_abc_cannot_be_instantiated():
    """Tests that the DAGBuilder ABC cannot be instantiated."""

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        # pylint: disable=abstract-class-instantiated
        DAGBuilder()


def test_incomplete_subclass():
    """Tests that an incomplete subclass will fail"""

    # pylint: disable=too-few-public-methods,missing-function-docstring
    class IncompleteDAGBuilder(DAGBuilder):
        """Incomplete dag builder dummy class."""

        def add_node(self, *args, **kwargs):
            pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        # pylint: disable=abstract-class-instantiated
        IncompleteDAGBuilder()
