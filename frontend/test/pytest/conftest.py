# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pytest configuration file for Catalyst test suite.
"""

import os
from tempfile import TemporaryDirectory
from textwrap import dedent

import pennylane as qml
import pytest


@pytest.fixture(scope="function")
def create_temporary_toml_file(request) -> str:
    """Create a temporary TOML file with the given content."""
    content = request.param
    with TemporaryDirectory() as temp_dir:
        toml_file = os.path.join(temp_dir, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(dedent(content))
        request.node.toml_file = toml_file
        yield


@pytest.fixture(scope="function")
def disable_capture():
    """Safely disable capture after a test, even on failure."""
    try:
        yield
    finally:
        if qml.capture.enabled():
            qml.capture.disable()


@pytest.fixture(scope="function")
def use_capture():
    """Enable capture before and disable capture after the test."""
    qml.capture.enable()
    try:
        yield
    finally:
        qml.capture.disable()


@pytest.fixture(scope="function")
def use_capture_dgraph():
    """Enable capture and graph-decomposition before and disable them both after the test."""
    qml.capture.enable()
    qml.decomposition.enable_graph()
    try:
        yield
    finally:
        qml.decomposition.disable_graph()
        qml.capture.disable()


@pytest.fixture(params=["capture", "no_capture"], scope="function")
def use_both_frontend(request):
    """Runs the test once with capture enabled and once with it disabled."""
    if request.param == "capture":
        if 'capture_todo' in request.keywords:
            pytest.xfail("capture todo's do not yet work with program capture.")
        if "old_frontend" in request.keywords:
            pytest.skip("this test should not be run with the old frontend.")
        qml.capture.enable()
        try:
            yield
        finally:
            qml.capture.disable()
    else:
        yield


@pytest.fixture(scope="function")
def requires_xdsl():
    """Fixture that ensures xdsl is available. It skips the test if xdsl is not installed."""
    pytest.importorskip("xdsl", reason="xdsl is not installed, skipping test")
    pytest.importorskip("xdsl_jax", reason="xdsl-jax is not installed, skipping test")


def pytest_collection_modifyitems(items, config):  # pylint: disable=unused-argument
    """Modify collected items as needed."""
    # Tests that do not have a specific suite marker are marked `core`
    for item in items:
        markers = {mark.name for mark in item.iter_markers()}
        if "xdsl" in markers and "requires_xdsl" not in item.fixturenames:
            item.fixturenames.append("requires_xdsl")
