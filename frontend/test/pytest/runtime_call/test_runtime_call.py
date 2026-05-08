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

"""Tests for kernel.declare and kernel.runtime_call."""

import itertools
import os
import platform
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import pytest
from jax import ShapeDtypeStruct

from catalyst import kernel, qjit
from catalyst.kernel import KernelDescriptor

_C_SOURCE = os.path.join(os.path.dirname(__file__), "libxor_ref.c")
_EXT = ".so" if platform.system() != "Darwin" else ".dylib"


@pytest.fixture(scope="module")
def libxor_ref():
    """Compile libxor_ref.c to a shared library; return its absolute path."""
    if not os.path.isfile(_C_SOURCE):
        pytest.skip(f"Reference C source not found: {_C_SOURCE}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        lib_path = os.path.join(tmp_dir, f"libxor_ref{_EXT}")
        subprocess.run(
            ["cc", "-shared", "-fPIC", "-o", lib_path, _C_SOURCE],
            capture_output=True,
            text=True,
            check=True,    
        )
        yield lib_path


@pytest.fixture(scope="module")
def xor_reduce(libxor_ref):
    return kernel.declare(
        "xor_reduce",
        artifact=libxor_ref,
        outputs=ShapeDtypeStruct((1,), jnp.int32),
    )


class TestKernelDeclare:
    def test_basic(self, xor_reduce, libxor_ref):
        assert isinstance(xor_reduce, KernelDescriptor)
        assert xor_reduce.name == "xor_reduce"
        assert xor_reduce.artifact == libxor_ref
        assert xor_reduce.output_spec == (((1,), "int32"),)

    def test_missing_artifact(self):
        with pytest.raises(FileNotFoundError, match="artifact not found"):
            kernel.declare(
                "xor_reduce",
                artifact="/blah/libxor.so",
                outputs=ShapeDtypeStruct((1,), jnp.int32),
            )

    def test_dynamic_shape_rejected(self, libxor_ref):
        with pytest.raises(ValueError, match="dynamic shapes unsupported"):
            kernel.declare(
                "xor_reduce",
                artifact=libxor_ref,
                outputs=ShapeDtypeStruct((None,), jnp.int32),
            )

    def test_descriptor_is_hashable(self, xor_reduce):
        assert {xor_reduce: 1}[xor_reduce] == 1

    def test_multiple_outputs(self, libxor_ref):
        desc = kernel.declare(
            "xor_reduce",
            artifact=libxor_ref,
            outputs=(ShapeDtypeStruct((1,), jnp.int32), ShapeDtypeStruct((3,), jnp.float32)),
        )
        assert len(desc.output_spec) == 2


class TestRuntimeCallIntegration:
    def test_xor_truth_table(self, xor_reduce):
        @qjit
        def circuit(x):
            (result,) = kernel.runtime_call(xor_reduce, x)
            return result

        import itertools  # pylint: disable=import-outside-toplevel

        for bits in itertools.product([0, 1], repeat=3):
            x = jnp.array(bits, dtype=jnp.int8)
            assert int(circuit(x)[0]) == bits[0] ^ bits[1] ^ bits[2]
