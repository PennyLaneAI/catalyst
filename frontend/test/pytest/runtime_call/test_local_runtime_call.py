# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for a local ``runtime_call``: a direct C ABI call into a native library."""

# pylint: disable=redefined-outer-name

import platform
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pennylane as qp
import pytest
from pennylane.backline.runtime import runtime_call
from pennylane.backline.runtime.signature import declare

from catalyst.utils.exceptions import CompileError


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    """``native.cpp``, built into a shared library."""
    if shutil.which("c++") is None:
        pytest.skip("no C++ compiler")
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library = tmp_path_factory.mktemp("native") / f"libnative{suffix}"
    source = Path(__file__).with_name("native.cpp")
    # Optimized, as a library would ship: an optimized callee relies on the caller's ABI duties.
    command = ["c++", "-std=c++17", "-O2", "-shared", "-fPIC", "-o", library, source]
    subprocess.run(command, check=True)
    return str(library)


@pytest.mark.parametrize(
    "symbol, spec, args, expected",
    [
        ("native_add", "(i32, i32) -> i32", (np.int32(20), np.int32(22)), 42),
        ("native_scale", "(f64, f32) -> f64", (np.float64(1.5), np.float32(4.0)), 6.0),
        ("native_echo", "(ptr) -> ptr", (np.uint64(0x7FAB1234),), 0x7FAB1234),
        ("native_sum", "(buf, u64) -> i32", (np.array([3, 5, 7, 11], np.int8), 4), 26),
        ("_ZN4demo3addEii", "(i32, i32) -> i32", (np.int32(19), np.int32(23)), 42),
        ("demo::add", "(i32, i32) -> i32", (np.int32(19), np.int32(23)), 42),
        ("demo::twice(double)", "(f64) -> f64", (np.float64(1.5),), 3.0),
    ],
    ids=["integers", "floats", "pointer", "buffer", "symbol", "c++-name", "c++-overload"],
)
def test_call(native, symbol, spec, args, expected):
    """Arguments reach the function as the declared C types."""
    declare(symbol, spec, library=native)

    @qp.qjit
    def prog(*args):
        return runtime_call(symbol, *args)

    assert prog(*args) == expected


def test_string(native):
    """A compile-time string is passed as a ``const char *``."""
    declare("native_strlen", "(str) -> u64", library=native)
    assert qp.qjit(lambda: runtime_call("native_strlen", "direct C ABI"))() == 12


def test_out_buffer(native):
    """An ``out`` buffer is allocated for the call, and returned after its result."""
    declare("native_fill", "(out, u64, u8) -> i32", library=native)
    status, output = qp.qjit(lambda: runtime_call("native_fill", 4, 10, out_bytes=4))()
    assert status == 0
    np.testing.assert_array_equal(output, [10, 11, 12, 13])


def test_void_calls_keep_their_order(native):
    """Store and load tests."""
    declare("native_store", "(u64)", library=native)
    declare("native_load", "() -> u64", library=native)

    @qp.qjit
    def prog(value):
        runtime_call("native_store", value)
        return runtime_call("native_load")

    assert prog(np.uint64(73)) == 73


def test_narrow_integers_are_extended(native):
    """Narrow integers are zero- or sign-extended by the caller."""
    declare("native_widen_u8", "(u8) -> u32", library=native)
    declare("native_widen_i16", "(i16) -> i32", library=native)

    @qp.qjit
    def prog(value):
        return (
            runtime_call("native_widen_u8", value.astype(np.uint8)),
            runtime_call("native_widen_i16", value.astype(np.int16)),
        )

    assert prog(np.int64(0x1FFFF)) == (0xFF, -1)


@pytest.mark.parametrize(
    "callee, library, error",
    [
        ("demo::missing", True, "because no library exports it"),
        ("demo::nowhere", False, "because no library was specified"),
        (
            "demo::twice",
            True,
            "because it is overloaded; give its full signature. Libraries exporting "
            "'demo::twice': demo::twice(double); demo::twice(int)",
        ),
    ],
    ids=["missing", "no-library", "overloaded"],
)
def test_unresolved_cpp_name(native, callee, library, error):
    """It should raise an error for an unresolved C++ function."""
    declare(callee, "(i32) -> i32", library=native if library else None)

    @qp.qjit
    def prog(value):
        return runtime_call(callee, value)

    with pytest.raises(CompileError, match=re.escape(error)):
        prog(np.int32(1))
