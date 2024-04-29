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
"""Tests for type conversion"""

import inspect
from collections.abc import Sequence

import numpy as np
from jax._src.api_util import shaped_abstractify
from jax._src.lib.mlir import ir

from catalyst.jax_extras import ShapedArray


def convert_shaped_arrays_to_tensors(sarrays: Sequence[ShapedArray]):
    """Convert a sequence of ShapedArrays to a sequence of tensors"""
    return map(convert_shaped_array_to_tensor, sarrays)


def convert_shaped_array_to_tensor(sarray):
    """Convert an invidual shaped array to an MLIR tensor"""
    numpy_dtype = sarray.dtype
    py_shape = sarray.shape
    mlir_dtype = convert_numpy_dtype_to_mlir(numpy_dtype)
    return ir.RankedTensorType.get(py_shape, mlir_dtype)


def convert_pytype_to_shaped_array(ty):
    """Maps types from the type signature or otherwise to shaped_arrays without weak type."""
    if ty == inspect.Signature.empty:
        return None
    if isinstance(ty, ShapedArray):
        return ty.strip_weak_type()
    return shaped_abstractify(ty).strip_weak_type()


# pylint: disable=too-many-return-statements
def convert_numpy_dtype_to_mlir(dtp):
    """Convert dtype to MLIR. Raise ValueError if no conversion is possible"""
    if dtp == np.dtype(np.complex128):
        base = ir.F64Type.get()
        return ir.ComplexType.get(base)
    elif dtp == np.dtype(np.complex64):
        base = ir.F32Type.get()
        return ir.ComplexType.get(base)
    elif dtp == np.dtype(np.float64):
        return ir.F64Type.get()
    elif dtp == np.dtype(np.float32):
        return ir.F32Type.get()
    elif dtp == np.dtype(np.bool_):
        return ir.IntegerType.get_signless(1)
    elif dtp == np.dtype(np.int8):
        return ir.IntegerType.get_signless(8)
    elif dtp == np.dtype(np.int16):
        return ir.IntegerType.get_signless(16)
    elif dtp == np.dtype(np.int32):
        return ir.IntegerType.get_signless(32)
    elif dtp == np.dtype(np.int64):
        return ir.IntegerType.get_signless(64)
    raise ValueError("Requested type conversion not available.")
