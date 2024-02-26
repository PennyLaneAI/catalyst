# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Catalyst's tracing module."""

import jax
import pytest

from catalyst.jax_tracer import lower_jaxpr_to_mlir


def test_jaxpr_lowering_without_dynshapes():
    """Test that the lowering function can be used without Catalyst's dynamic shape support."""

    def f():
        return 0

    jaxpr = jax.make_jaxpr(f)()
    result, _ = lower_jaxpr_to_mlir(jaxpr, "test_fn")

    assert "@jit_test_fn() -> tensor<i64>" in str(result)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
