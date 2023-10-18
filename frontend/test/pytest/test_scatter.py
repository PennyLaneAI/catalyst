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

"""Test integration for the lowering of MHLO scatter."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from catalyst import qjit


def test_add_multiply():
    """Test to index a jax array and have operations on it."""

    @qjit
    def add_multiply(l: jax.core.ShapedArray((3,), dtype=float), idx: int):
        res = l.at[idx].multiply(3)
        res2 = l.at[idx].add(2)
        return res + res2

    res = add_multiply(jnp.array([0, 1, 2]), 2)
    assert np.allclose(res, [0, 2, 10])


def test_multiple_index():
    """Test to index a jax array and have operations on it."""

    @qjit
    def multiple_index_multiply(l: jax.core.ShapedArray((3,), dtype=float)):
        res = l.at[1:3].multiply(3)
        return res

    res = multiple_index_multiply(jnp.array([0, 1, 2]))
    assert np.allclose(res, [0, 3, 6])


def test_matrix():
    """Test to index a jax array and have operations on it."""

    @qjit
    def multiple_index_multiply(l: jax.core.ShapedArray((3, 3), dtype=float)):
        res = l.at[1:3].multiply(jnp.array(3))
        return res

    res = multiple_index_multiply(jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    assert np.allclose(res, jnp.array([[0, 1, 2], [9, 12, 15], [18, 21, 24]]))


if __name__ == "__main__":
    pytest.main(["-x", __file__])
