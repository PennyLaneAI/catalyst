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
from importlib.util import find_spec
from tempfile import TemporaryDirectory
from textwrap import dedent
from warnings import warn

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
        qml.capture.enable()
        try:
            yield
        finally:
            qml.capture.disable()
    else:
        yield


def pytest_collection_modifyitems(items, config):  # pylint: disable=unused-argument
    """Modify collected items as needed."""
    # Tests that do not have a specific suite marker are marked `core`
    for item in items:
        markers = {mark.name for mark in item.iter_markers()}
        if "xdsl" in markers:
            # If filecheck is not installed, the xDSL lit tests get skipped silently. This
            # warning will provide verbosity to testers.
            if not find_spec("filecheck"):
                warn(
                    "The 'filecheck' Python package must be installed to use fixtures for "
                    "lit testing xDSL features. Otherwise, tests using the 'run_filecheck' "
                    "or 'run_filecheck_qjit' fixtures will be skipped.",
                    UserWarning,
                )
                break
