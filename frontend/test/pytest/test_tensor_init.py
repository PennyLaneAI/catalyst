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

import numpy as np
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

from catalyst import empty, ones, qjit, zeros

DTYPES = [float, int, jnp.float32, jnp.float64, jnp.int8, jnp.int16, np.float32, np.float64]
SHAPES = [(2, 3, 1), ()]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_ones(dtype, shape):
    """Test catalyst.ones primitive"""

    @qjit
    def f(s):
        return ones(shape=s, dtype=dtype)

    assert_allclose(f(shape), jnp.ones(shape, dtype=dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_zeros(dtype, shape):
    """Test catalyst.zeros primitive"""

    @qjit
    def f(s):
        return zeros(shape=s, dtype=dtype)

    assert_allclose(f(shape), jnp.zeros(shape, dtype=dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_empty(dtype, shape):
    """Test catalyst.empty primitive"""

    @qjit
    def f(s):
        return empty(shape=s, dtype=dtype)

    res = f(shape)
    assert res.shape == shape
    assert res.dtype == dtype


if __name__ == "__main__":
    pytest.main(["-x", __file__])
