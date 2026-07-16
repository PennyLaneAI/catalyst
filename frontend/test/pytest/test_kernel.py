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

"""Unit tests for the ``catalyst.kernel`` declaration API (declare/define)."""

import os

import jax
import jax.numpy as jnp
import pytest

from catalyst import kernel
from catalyst.kernel import KernelDescriptor


def test_declare(tmp_path):
    """declare resolves the artifact to an absolute path and records the output spec."""
    artifact = tmp_path / "lib.so"
    artifact.write_bytes(b"")
    desc = kernel.declare("sym", str(artifact), jax.ShapeDtypeStruct((3,), jnp.int32))
    assert isinstance(desc, KernelDescriptor)
    assert desc.name == "sym"
    assert desc.artifact == os.path.abspath(str(artifact))
    assert desc.output_spec == (((3,), "int32"),)


def test_declare_missing_artifact():
    """A non-existent artifact is rejected at declare time."""
    with pytest.raises(FileNotFoundError):
        kernel.declare("sym", "/no/such/lib.so", jax.ShapeDtypeStruct((1,), jnp.float32))


def test_define(tmp_path):
    """define builds via the backend builder and declares, inferring the symbol name."""
    artifact = tmp_path / "k.so"
    artifact.write_bytes(b"")

    class Builder:
        def build(self, kernel_fn, *, name):  # pylint: disable=unused-argument
            assert name == "my_kernel"
            return str(artifact)

    @kernel.define(Builder(), outputs=jax.ShapeDtypeStruct((1,), jnp.int32))
    def my_kernel():  # pragma: no cover 
        pass

    assert isinstance(my_kernel, KernelDescriptor)
    assert my_kernel.name == "my_kernel"
    assert my_kernel.artifact == str(artifact)
