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
"""Tests memref descriptor functions with JAX's ShapedArrays."""

import ctypes

import jax.numpy as jnp
import numpy as np
import pytest
from jax.core import ShapedArray
from mlir_quantum.runtime import (
    C64,
    C128,
    as_ctype,
)
from mlir_quantum.runtime import (
    get_unranked_memref_descriptor as mlir_get_unranked_memref_descriptor,
)
from mlir_quantum.runtime import (
    make_nd_memref_descriptor,
    make_zero_d_memref_descriptor,
)

from catalyst.utils.jnp_to_memref import (
    get_ranked_memref_descriptor,
    get_unranked_memref_descriptor,
)


@pytest.mark.parametrize(
    "inp, exp",
    [
        (jnp.dtype(jnp.float64), ctypes.c_double),
        (jnp.dtype(jnp.float32), ctypes.c_float),
        (jnp.dtype(jnp.int64), ctypes.c_long),
        (jnp.dtype(jnp.bool_), ctypes.c_bool),
        (jnp.dtype(jnp.complex128), C128),
        (jnp.dtype(jnp.complex64), C64),
    ],
)
def test_as_ctype(inp, exp):
    """Tests that JAX's dtypes behave the same as numpy's dtypes"""
    obs = as_ctype(inp)
    assert exp == obs


@pytest.mark.parametrize(
    "inp, exp",
    [
        (1, make_zero_d_memref_descriptor(ctypes.c_long)),
        (ShapedArray([], float), make_zero_d_memref_descriptor(ctypes.c_double)),
        (ShapedArray([1], float), make_nd_memref_descriptor(1, ctypes.c_double)),
        (ShapedArray([2, 2], float), make_nd_memref_descriptor(2, ctypes.c_double)),
    ],
)
def test_get_ranked_memref_descriptor(inp, exp):
    """Tests that the structure has the expected fields."""
    obs = get_ranked_memref_descriptor(inp)
    assert exp._fields_ == obs._fields_  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "inp, exp",
    [
        (1, mlir_get_unranked_memref_descriptor(np.array(1))),
        (np.array(1), mlir_get_unranked_memref_descriptor(np.array(1))),
    ],
)
def test_get_unranked_meref_descriptor(inp, exp):
    """Test unranked_memref_descriptor"""
    obs = get_unranked_memref_descriptor(inp)
    assert exp._fields_ == obs._fields_  # pylint: disable=protected-access
