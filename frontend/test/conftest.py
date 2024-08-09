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

# pylint: disable=unused-import
import platform

import numpy as np
import pennylane as qml
import pytest


def is_cuda_available():
    """Checks if cuda is available by trying an import.

    Do not run from top level!
    We do not want to import cudaq unless we absolutely need to.
    This is because cudaq prevents kokkos kernels from executing properly.
    See https://github.com/PennyLaneAI/catalyst/issues/513
    """
    try:
        # pylint: disable=import-outside-toplevel
        import cudaq
    except (ImportError, ModuleNotFoundError):
        cudaq_available = False
    else:
        cudaq_available = True
    return cudaq_available


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


def skip_cuda_tests(config, items):
    """Skip cuda tests according to the following logic:
    By default: RUN
      except: if apple
      except: if kokkos
      except: is cuda-quantum not installed

    Important! We should only check if cuda-quantum is installed
    as a last resort. We don't want to check if cuda-quantum is
    installed at all when we are running kokkos.
    """
    skipper = pytest.mark.skip()
    is_kokkos = config.getoption("backend") == "lightning.kokkos"
    is_apple = platform.system() == "Darwin"
    # CUDA quantum is not supported in apple silicon.
    # CUDA quantum cannot run with kokkos
    skip_cuda_tests = is_kokkos or is_apple
    if not skip_cuda_tests and not is_cuda_available():
        # Only check this conditionally as it imports cudaq.
        # And we don't even want to succeed with kokkos.
        skip_cuda_tests = True
    for item in items:
        is_cuda_test = "cuda" in item.keywords
        skip_cuda = is_cuda_test and skip_cuda_tests
        if skip_cuda:
            item.add_marker(skipper)


def pytest_collection_modifyitems(config, items):
    """A pytest items modifier method"""

    skip_cuda_tests(config, items)

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
