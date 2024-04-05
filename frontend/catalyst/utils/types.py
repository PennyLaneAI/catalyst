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

from collections.abc import Sequence

import numpy as np
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


# This variable needs to be lazily-initialized. The values
# stored in this dictionary are MLIR types, which depend on the
# existance of an MLIRContext object.
# Upon import, there is no MLIRContext object.
NUMPY_DTYPE_TO_MLIR = None


def convert_numpy_dtype_to_mlir_safe(dtp):
    """Convert dtype to MLIR. Return None if no type conversion is possible"""
    global NUMPY_DTYPE_TO_MLIR  # pylint: disable=global-statement
    if not NUMPY_DTYPE_TO_MLIR:
        NUMPY_DTYPE_TO_MLIR = {
            np.dtype(np.complex128): ir.ComplexType.get(ir.F64Type.get()),
            np.dtype(np.complex64): ir.ComplexType.get(ir.F32Type.get()),
            np.dtype(np.float64): ir.F64Type.get(),
            np.dtype(np.float32): ir.F32Type.get(),
            np.dtype(np.bool_): ir.IntegerType.get_signless(1),
            np.dtype(np.int8): ir.IntegerType.get_signless(8),
            np.dtype(np.int16): ir.IntegerType.get_signless(16),
            np.dtype(np.int32): ir.IntegerType.get_signless(32),
            np.dtype(np.int64): ir.IntegerType.get_signless(64),
        }
    return NUMPY_DTYPE_TO_MLIR.get(dtp)


def convert_numpy_dtype_to_mlir(dtp):
    """Convert dtype to MLIR. Raise ValueError if no conversion is possible"""
    retval = convert_numpy_dtype_to_mlir_safe(dtp)
    if not retval:
        raise ValueError("Requested type conversion not available.")
    return retval
