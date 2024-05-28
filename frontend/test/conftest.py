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

import pennylane as qml
import numpy as np
from functools import reduce
from typing import Iterable, Sequence


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


def validate_counts(shots, results1, results2, batch_size=None):
    """Compares two counts.

    If the results are ``Sequence``s, loop over entries.

    Fails if a key of ``results1`` is not found in ``results2``.
    Passes if counts are too low, chosen as ``100``.
    Otherwise, fails if counts differ by more than ``20`` plus 20 percent.
    """
    if isinstance(shots, Sequence):
        assert isinstance(results1, tuple)
        assert isinstance(results2, tuple)
        assert len(results1) == len(results2) == len(shots)
        for s, r1, r2 in zip(shots, results1, results2):
            validate_counts(s, r1, r2, batch_size=batch_size)
        return

    if batch_size is not None:
        assert isinstance(results1, Iterable)
        assert isinstance(results2, Iterable)
        assert len(results1) == len(results2) == batch_size
        for r1, r2 in zip(results1, results2):
            validate_counts(shots, r1, r2, batch_size=None)
        return

    for key1, val1 in results1.items():
        val2 = results2[key1]
        if abs(val1 + val2) > 100:
            assert np.allclose(val1, val2, atol=20, rtol=0.2)


def validate_samples(shots, results1, results2, batch_size=None):
    """Compares two samples.

    If the results are ``Sequence``s, loop over entries.

    Fails if the results do not have the same shape, within ``20`` entries plus 20 percent.
    This is to handle cases when post-selection yields variable shapes.
    Otherwise, fails if the sums of samples differ by more than ``20`` plus 20 percent.
    """
    if isinstance(shots, Sequence):
        assert isinstance(results1, tuple)
        assert isinstance(results2, tuple)
        assert len(results1) == len(results2) == len(shots)
        for s, r1, r2 in zip(shots, results1, results2):
            validate_samples(s, r1, r2, batch_size=batch_size)
        return

    if batch_size is not None:
        assert isinstance(results1, Iterable)
        assert isinstance(results2, Iterable)
        assert len(results1) == len(results2) == batch_size
        for r1, r2 in zip(results1, results2):
            validate_samples(shots, r1, r2, batch_size=None)
        return

    sh1, sh2 = results1.shape[0], results2.shape[0]
    assert np.allclose(sh1, sh2, atol=20, rtol=0.2)
    assert results1.ndim == results2.ndim
    if results2.ndim > 1:
        assert results1.shape[1] == results2.shape[1]
    np.allclose(qml.math.sum(results1), qml.math.sum(results2), atol=20, rtol=0.2)


def validate_expval(shots, results1, results2, batch_size=None):
    """Compares two expval, probs or var.

    If the results are ``Sequence``s, validate the average of items.

    If ``shots is None``, validate using ``np.allclose``'s default parameters.
    Otherwise, fails if the results do not match within ``0.01`` plus 20 percent.
    """
    if isinstance(shots, Sequence):
        assert isinstance(results1, tuple)
        assert isinstance(results2, tuple)
        assert len(results1) == len(results2) == len(shots)
        results1 = reduce(lambda x, y: x + y, results1) / len(results1)
        results2 = reduce(lambda x, y: x + y, results2) / len(results2)
        validate_expval(sum(shots), results1, results2, batch_size=batch_size)
        return

    if shots is None:
        assert np.allclose(results1, results2)
        return

    if batch_size is not None:
        assert len(results1) == len(results2) == batch_size
        for r1, r2 in zip(results1, results2):
            validate_expval(shots, r1, r2, batch_size=None)

    assert np.allclose(results1, results2, atol=0.01, rtol=0.2)


def validate_measurements(func, shots, results1, results2, batch_size=None):
    """Calls the correct validation function based on measurement type."""
    if func is qml.counts:
        validate_counts(shots, results1, results2, batch_size=batch_size)
        return

    if func is qml.sample:
        validate_samples(shots, results1, results2, batch_size=batch_size)
        return

    validate_expval(shots, results1, results2, batch_size=batch_size)
