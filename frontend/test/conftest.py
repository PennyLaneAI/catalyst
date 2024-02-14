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
# pylint: disable=unused-import
import platform
import pytest

try:
    import cudaq
except (ImportError, ModuleNotFoundError) as e:
    cudaq_available = False
else:
    cudaq_available = True

try:
    import catalyst
    import tensorflow as tf
except (ImportError, ModuleNotFoundError) as e:
    tf_available = False
else:
    tf_available = True

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
        "--cuda",
        action="store",
        default=True,
        help="Run cuda tests.",
    )

    parser.addoption(
        "--runbraket",
        action="store",
        default="NONE",
        help="Run AWS Braket ['ALL', 'LOCAL', 'REMOTE'] tests;"
        " runtime must be compiled with `ENABLE_OPENQASM=ON`",
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

    config.addinivalue_line(
        "markers",
        "cuda: run cuda tests",
    )


def pytest_runtest_setup(item):
    """Automatically skip tests if interfaces are not installed"""
    interfaces = {"tf", "cuda"}
    available_interfaces = {
        "tf": tf_available,
        "cuda": cudaq_available,
    }

    allowed_interfaces = [
        allowed_interface
        for allowed_interface in interfaces
        if available_interfaces[allowed_interface] is True
    ]

    # load the marker specifying what the interface is
    all_interfaces = {"tf", "cuda"}
    marks = {mark.name for mark in item.iter_markers() if mark.name in all_interfaces}

    for b in marks:
        if b not in allowed_interfaces:
            pytest.skip(
                f"\nTest {item.nodeid} only runs with {allowed_interfaces}"
                f" interfaces(s) but {b} interface provided",
            )


def pytest_collection_modifyitems(config, items):
    """A pytest items modifier method"""

    # skip braket tests
    skipper = pytest.mark.skip()
    for item in items:
        is_apple = platform.system() == "Darwin"
        # CUDA quantum is not supported in apple silicon.
        run_cuda_tests = "cuda" in item.keywords
        skip_cuda = run_cuda_tests and (item.get_closest_marker("cuda") == "True" or is_apple)
        if skip_cuda:
            item.add_marker(skipper)

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
