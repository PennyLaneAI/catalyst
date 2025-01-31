#  This file is essentially a re-implementation of np_to_memref.py
#  But we make sure that all the functions also work with JAX's numpy.
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#  This file contains functions to convert between Memrefs and NumPy arrays and vice-versa.

"""
This file is a wrapper around MLIR's np_to_memref to allow for abstract types and JAX arrays
to be converted to and from memrefs.
"""

import jax
import numpy as np
from mlir_quantum.runtime import (
    as_ctype,
)
from mlir_quantum.runtime import (
    get_ranked_memref_descriptor as mlir_get_ranked_memref_descriptor,
)
from mlir_quantum.runtime import (
    get_unranked_memref_descriptor as mlir_get_unranked_memref_descriptor,
)
from mlir_quantum.runtime import ranked_memref_to_numpy as mlir_ranked_memref_to_numpy
from mlir_quantum.runtime.np_to_memref import (
    make_nd_memref_descriptor,
    make_zero_d_memref_descriptor,
    move_aligned_ptr_by_offset,
    to_numpy,
)

from catalyst.jax_extras import DynamicJaxprTracer, ShapedArray


def get_ranked_memref_descriptor_from_shaped_array(array: ShapedArray):
    """Get a ranked memref descriptor from a shaped array.

    Unlike MLIR's implementation, all values are left uninitialized.
    This is because the values are not yet known. We only have a description
    of the type.
    """

    ctp = as_ctype(array.dtype)
    if array.ndim == 0:
        return make_zero_d_memref_descriptor(ctp)()

    return make_nd_memref_descriptor(array.ndim, ctp)()


def get_ranked_memref_descriptor(array):
    """Wrapper around MLIR's get_ranked_memref_descriptor."""

    if isinstance(array, DynamicJaxprTracer):
        array = array.aval

    if isinstance(array, (int, float, bool, complex)):
        # This is necessary for keyword arguments
        array = np.array(array)

    if isinstance(array, jax.Array):
        array = np.asarray(array)

    if isinstance(array, ShapedArray):
        # If input is ShapedArray
        return get_ranked_memref_descriptor_from_shaped_array(array)

    # Use default implementation from MLIR's library.
    return mlir_get_ranked_memref_descriptor(array)


def get_unranked_memref_descriptor(array):
    """Wrapper around MLIR's get_unranked_memref_descriptor."""

    if isinstance(array, (int, float, bool, complex)):
        # Convenience
        array = np.array(array)

    if isinstance(array, jax.Array):
        array = np.asarray(array)

    # Use default implementation from MLIR's library.
    return mlir_get_unranked_memref_descriptor(array)


def ranked_memref_to_numpy(ranked_memref):
    """Wrapper around MLIR's ranked_memref_to_numpy.

    This wrapper succeeds when the ranked_memref is a scalar tensor.
    """
    try:
        return mlir_ranked_memref_to_numpy(ranked_memref)
    except AttributeError:
        # zero dimensional tensor...
        content_ptr = move_aligned_ptr_by_offset(ranked_memref[0].aligned, ranked_memref[0].offset)
        np_arr = np.ctypeslib.as_array(content_ptr, shape=[])
        return to_numpy(np_arr)
