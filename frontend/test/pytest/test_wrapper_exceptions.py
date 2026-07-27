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
"""
Test that a RuntimeError thrown from the JIT'd setup function be caught by the wrapper.
"""

import pytest
from jax.interpreters.mlir import ir

from catalyst import qjit
from catalyst.utils import gen_mlir


@pytest.fixture
def gen_setup_with_failing_assert(monkeypatch):
    """
    Patch the gen_setup function to return an MLIR module with a setup function that
    executes a catalyst.assert false after quantum.init.
    """

    def patched_gen_setup(ctx, seed):
        seed_attr = f" {{seed = {seed} : i32}}" if seed is not None else ""
        txt = f"""
func.func @setup() -> () {{
    "quantum.init"(){seed_attr} : () -> ()
    %f = arith.constant false
    "catalyst.assert"(%f) <{{error = "triggered from setup"}}> : (i1) -> ()
    return
}}
"""
        return ir.Module.parse(txt, ctx)

    monkeypatch.setattr(gen_mlir, "gen_setup", patched_gen_setup)


def test_setup_failure_surfaces_as_runtime_error(
    gen_setup_with_failing_assert,
):  # pylint: disable=redefined-outer-name,unused-argument
    """Test that a RuntimeError thrown from the JIT'd setup function be caught by the wrapper."""

    @qjit
    def circuit():
        return 0

    with pytest.raises(RuntimeError, match="triggered from setup"):
        circuit()
