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
        if "old_frontend" in request.keywords:
            pytest.skip("Test is specific to the old frontend and should not run with capture.")
        if "capture_todo" in request.keywords:
            pytest.xfail("capture todo's do not yet work with program capture.")
        qml.capture.enable()
        try:
            yield
        finally:
            qml.capture.disable()
    else:
        yield


@pytest.fixture(params=[True, False], ids=["capture=True", "capture=False"])
def capture_mode(request):
    """Parametrize tests to run with capture=True and capture=False.

    This fixture returns a boolean that should be passed to @qjit(capture=...).
    Unlike use_both_frontend, this does NOT toggle the global capture state,
    allowing more isolated and explicit testing.

    Usage:
        def test_example(backend, capture_mode):
            @qjit(capture=capture_mode)
            @qml.qnode(qml.device(backend, wires=1))
            def circuit():
                ...

    Markers:
        @pytest.mark.old_frontend - Skip when capture_mode=True
        @pytest.mark.capture_todo - xfail when capture_mode=True
    """
    if request.param:  # capture=True
        if "old_frontend" in request.keywords:
            pytest.skip("Test is specific to the old frontend and should not run with capture.")
        if "capture_todo" in request.keywords:
            pytest.xfail("Not expected to work yet with program capture.")
    return request.param


def pytest_collection_modifyitems(items, config):  # pylint: disable=unused-argument
    """Modify collected items as needed."""
    xdsl_tests_skipped = "not xdsl" in config.getoption("markexpr")

    for item in items:
        markers = {mark.name for mark in item.iter_markers()}
        # The nested conditional can be merged with this one, but we don't do that so that we can
        # break right after the first xDSL test is found. Otherwise, we will have unnecessary
        # iterations if filecheck is installed or xDSL tests are skipped.
        if "xdsl" in markers:
            # If filecheck is not installed, the xDSL lit tests get skipped silently. This
            # warning will provide verbosity to testers.
            if not (xdsl_tests_skipped or find_spec("filecheck")):
                warn(
                    "The 'filecheck' Python package must be installed to use fixtures for "
                    "lit testing xDSL features. Otherwise, tests using the 'run_filecheck' "
                    "or 'run_filecheck_qjit' fixtures will be skipped.",
                    UserWarning,
                )

            break
