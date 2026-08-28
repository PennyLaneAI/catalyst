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
import subprocess
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from warnings import warn

import pennylane as qp
import pytest

from catalyst.utils.runtime_environment import get_lib_path


def _detect_gpu_triton_platform():
    """Return a Triton ``platform`` string matching this runner's GPU, or ``None``.

    Auto-detects NVIDIA (via ``nvidia-smi``) and AMD (via ``rocminfo``) so the same test
    body runs against whichever hardware is attached. Preferring NVIDIA when both are
    somehow present is arbitrary but consistent across runners.
    """
    # NVIDIA: nvidia-smi reports compute cap as e.g. "8.0". Warp size is always 32.
    try:
        cc = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .splitlines()[0]
            .strip()
        )
        major, minor = cc.split(".")
        return f"cuda:{major}{minor}:32"
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        pass

    # AMD: rocminfo lists ``Name: gfxNNN`` for each agent. Wave size is 64 on gfx9/10,
    # some gfx11 SKUs use 32; default to 64 which matches the demo examples.
    try:
        rocm = subprocess.check_output(["rocminfo"], text=True, stderr=subprocess.DEVNULL)
        for line in rocm.splitlines():
            stripped = line.strip()
            if stripped.startswith("Name:") and "gfx" in stripped:
                arch = stripped.split(":", 1)[1].strip()
                return f"hip:{arch}:64"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return None


def _triton_driver_active():
    """Whether Triton can actually initialize a backend driver on this runner.

    ``nvidia-smi`` / ``rocminfo`` detect the GPU as hardware, but Triton has its own
    activation path (``triton.runtime.driver.active`` -> ``is_active()`` on the CUDA or
    HIP driver), which needs ``libcuda.so`` / ``libamdhip64.so`` and their python bindings.
    A runner with an NVIDIA GPU visible via nvidia-smi but no CUDA runtime installed lands
    at ``0 active drivers`` when Triton probes.
    """
    try:
        import triton  # noqa: F401  -- pylint: disable=import-outside-toplevel
        from triton.runtime.driver import (
            driver as _driver,  # pylint: disable=import-outside-toplevel
        )
    except (ImportError, RuntimeError):
        return False
    try:
        _driver.active  # forces driver selection; raises RuntimeError on 0 or >1 active
    except RuntimeError:
        return False
    return True


@pytest.fixture(scope="module")
def gpu_triton_platform():
    """Skip cleanly if no usable Triton backend is attached to this runner.

    Two gates: hardware detected via ``nvidia-smi`` / ``rocminfo``, and Triton's own
    driver activation. The second catches the "GPU present, CUDA runtime missing"
    case that surfaces as ``RuntimeError: 0 active drivers`` deep in ``triton.jit``.
    """
    platform = _detect_gpu_triton_platform()
    if platform is None:
        pytest.skip("No NVIDIA or AMD GPU detected on this runner")
    if not _triton_driver_active():
        pytest.skip(
            "GPU detected but Triton has no active driver "
            "(missing CUDA / HIP runtime for the installed triton wheel)"
        )
    return platform


@pytest.fixture(scope="module")
def gpu_transport_backend():
    """Skip unless the GPU coprocessor transport backend was actually built.

    ``catalyst_transport_memcpy_gpu_coprocessor`` sits inside an ``if(TRANSPORT_HAS_HIP)`` guard
    in ``runtime/lib/transport/memcpy/CMakeLists.txt``, so a runtime built without a HIP
    toolchain produces no such library at all. That matters because
    ``backline._resolve_backend_lib`` requires it on disk and raises ``ValueError`` when it is
    missing -- a hard failure, not a skip -- so a test that executes a GPU coprocessor has to
    check for it up front.

    Distinct from :func:`gpu_triton_platform`, which asks whether a GPU and a Triton driver are
    present. Both gates are needed by the executing cases and they fail for unrelated reasons: a
    runner can have a working Triton driver and still lack HIP. A compile-only case that attaches
    its own ``backend_lib`` needs neither this nor the library.
    """
    lib_dir = Path(get_lib_path("runtime", "RUNTIME_LIB_DIR"))
    names = [f"libcatalyst_transport_memcpy_gpu_coprocessor.{ext}" for ext in ("so", "dylib")]
    if not any((lib_dir / name).exists() for name in names):
        pytest.skip(
            f"GPU coprocessor transport backend not built: none of {names} found in {lib_dir}. "
            "The runtime needs a HIP toolchain (HIP-on-CUDA on an NVIDIA host) for CMake to "
            "set TRANSPORT_HAS_HIP and build it."
        )


@pytest.fixture(scope="function")
def stop_node_executors():
    """Stop the executors the compiler launched for the backline nodes registered here.

    A node carrying ``executor_options`` never hands the test its executor: the compiler builds,
    launches and caches one on the node itself (``backline._realize_executor``). Catalyst's
    ``_SessionRegistry`` does clean up, but only at process exit, which is too late inside a
    pytest session -- the ``catalyst-executor`` subprocess stays up holding the coprocessor's
    out-of-band TCP port, so a rerun, ``pytest-xdist``, or a second out-of-process case in the
    same session would meet ``EADDRINUSE``.

    Register each node that asked for an executor; ``Executor.stop()`` is idempotent and runs on
    teardown whatever the test's outcome, releasing the port and removing the deploy workspace.
    """
    nodes = []
    try:
        yield nodes.append
    finally:
        for node in nodes:
            executor = getattr(node, "executor", None)
            if executor is not None:
                executor.stop()


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
        if qp.capture.enabled():
            qp.capture.disable()


@pytest.fixture(scope="function")
def use_capture():
    """Enable capture before and disable capture after the test."""
    qp.capture.enable()
    try:
        yield
    finally:
        qp.capture.disable()


@pytest.fixture(scope="function")
def use_capture_dgraph():
    """Enable capture and graph-decomposition before and disable them both after the test."""
    qp.capture.enable()
    qp.decomposition.enable_graph()
    try:
        yield
    finally:
        qp.decomposition.disable_graph()
        qp.capture.disable()


@pytest.fixture(params=["capture", "no_capture"], scope="function")
def use_both_frontend(request):
    """Runs the test once with capture enabled and once with it disabled."""
    if request.param == "capture":
        if "capture_todo" in request.keywords:
            pytest.xfail("capture todo's do not yet work with program capture.")
        qp.capture.enable()
        try:
            yield
        finally:
            qp.capture.disable()
    else:
        yield


@pytest.fixture(
    params=[
        pytest.param(True, marks=pytest.mark.capture),
        pytest.param(False, marks=pytest.mark.old_frontend),
    ],
    ids=["capture=True", "capture=False"],
)
def capture_mode(request):
    """Parametrize tests to run with capture=True and capture=False.

    This fixture returns a boolean that should be passed to @qjit(capture=...).
    Unlike use_both_frontend, this does NOT toggle the global capture state,
    allowing more isolated and explicit testing.

    Usage:
        def test_example(backend, capture_mode):
            @qjit(capture=capture_mode)
            @qp.qnode(qp.device(backend, wires=1))
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
