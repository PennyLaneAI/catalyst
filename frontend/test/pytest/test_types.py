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
"""Tests for type conversions"""

import inspect

import numpy as np
import pytest
from jax._src.lib.mlir import ir
from jax.core import ShapedArray

from catalyst.utils.types import (
    convert_numpy_dtype_to_mlir,
    convert_pytype_to_shaped_array,
)

ctx = ir.Context()
f64 = ir.F64Type.get(ctx)
f32 = ir.F32Type.get(ctx)
complex128 = ir.ComplexType.get(f64)
complex64 = ir.ComplexType.get(f32)
i64 = ir.IntegerType.get_signless(64, ctx)
i32 = ir.IntegerType.get_signless(32, ctx)
i16 = ir.IntegerType.get_signless(16, ctx)
i8 = ir.IntegerType.get_signless(8, ctx)
i1 = ir.IntegerType.get_signless(1, ctx)


@pytest.mark.parametrize(
    "inp,exp",
    [
        (np.dtype(np.complex128), complex128),
        (np.dtype(np.complex64), complex64),
        (np.dtype(np.float64), f64),
        (np.dtype(np.float32), f32),
        (np.dtype(np.bool_), i1),
        (np.dtype(np.int8), i8),
        (np.dtype(np.int16), i16),
        (np.dtype(np.int32), i32),
        (np.dtype(np.int64), i64),
    ],
)
def test_convert_numpy_dtype_to_mlir(inp, exp):
    """Converting numpy dtype to MLIR types"""
    with ctx:
        assert convert_numpy_dtype_to_mlir(inp) == exp


def test_convert_numpy_dtype_to_mlir_error():
    """Test errors"""
    with pytest.raises(ValueError, match="Requested type conversion not available."):
        with ctx:
            convert_numpy_dtype_to_mlir(np.dtype(object))


@pytest.mark.parametrize(
    "inp,exp",
    [
        (inspect.Signature.empty, None),
        (int, ShapedArray([], int)),
        (float, ShapedArray([], float)),
        (bool, ShapedArray([], bool)),
        (ShapedArray([2], bool), ShapedArray([2], bool)),
    ],
)
def test_convert_pytype_to_shaped_array(inp, exp):
    """Test conversion to shaped arrays"""
    assert convert_pytype_to_shaped_array(inp) == exp


if __name__ == "__main__":
    pytest.main(["-x", __file__])
