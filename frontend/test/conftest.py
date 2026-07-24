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

# OMP env vars have to be set before importing numpy in order to have an effect
os.environ["OMP_PROC_BIND"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"

# pylint: disable=unused-import,wrong-import-position
import pytest

# Default from PennyLane
TOL_STOCHASTIC = 0.05


@pytest.fixture(scope="session")
def tol_stochastic():
    """Numerical tolerance for equality tests of stochastic values."""
    return TOL_STOCHASTIC


def pytest_addoption(parser):
    """Add pytest custom options."""

    parser.addoption(
        "--backend",
        action="store",
        default="lightning.qubit",
        help="Name of the backend device",
    )

    parser.addoption(
        "--runbraket",
        action="store",
        default="NONE",
        help="Run AWS Braket ['ALL', 'LOCAL', 'REMOTE'] tests;"
        " runtime must be compiled with `ENABLE_OPENQASM=ON`",
    )

    parser.addoption(
        "--debug-pipeline",
        action="store_true",
        help=(
            "For tests that use the run_filecheck fixture, display the full xDSL module IR before "
            "and after applying a compilation pipeline. This option should generally be used with "
            "either the '--capture=no' or '-s' option (or similar) in order to display the output "
            "to the terminal."
        ),
    )


def pytest_generate_tests(metafunc):
    """A pytest fixture to define custom parametrization"""

    if "backend" in metafunc.fixturenames:
        metafunc.parametrize("backend", [metafunc.config.getoption("--backend")])


def pytest_configure(config):
    """A pytest configure helper method"""

    config.addinivalue_line(
        "markers",
        "braketlocal: run on local aws-braket devices backed by `OpenQasmDevice` in the runtime",
    )

    config.addinivalue_line(
        "markers",
        "braketremote: run on remote aws-braket devices backed by `OpenQasmDevice` in the runtime",
    )


def pytest_collection_modifyitems(config, items):
    """A pytest items modifier method"""

    # skip braket tests
    skipper = pytest.mark.skip()
    braket_val = config.getoption("--runbraket")
    if braket_val in ["ALL", "LOCAL", "REMOTE"]:
        # only runs test with the braket marker
        braket_tests = []
        for item in items:
            if (braket_val in ["ALL", "LOCAL"] and item.get_closest_marker("braketlocal")) or (
                braket_val in ["ALL", "REMOTE"] and item.get_closest_marker("braketremote")
            ):
                braket_tests.append(item)
        items[:] = braket_tests
    else:
        for item in items:
            if "braketlocal" in item.keywords or "braketremote" in item.keywords:
                item.add_marker(skipper)
