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

"""Type handling utilities for decomposition rule lowering."""

import jax.numpy as jnp
import pennylane as qp
from jax.core import ShapedArray

_MLIR_DTYPES_TO_PY_DTYPES = {
    "i1": jnp.bool_,
    "i8": jnp.int8,
    "i16": jnp.int16,
    "i32": jnp.int32,
    "i64": jnp.int64,
    "f16": jnp.float16,
    "f32": jnp.float32,
    "f64": jnp.float64,
    "complex<f64>": jnp.complex64,
    "complex<f128>": jnp.complex128,
}

_PY_DTYPES_TO_MLIR_DTYPES = {v: k for k, v in _MLIR_DTYPES_TO_PY_DTYPES.items()}


def _stringify_shaped_type(shape: tuple, dim: int, element_type) -> str:
    """Return a string representation of the given shaped data type."""
    if dim + 1 == len(shape):
        inner_content = _PY_DTYPES_TO_MLIR_DTYPES[element_type]
    else:
        inner_content = _stringify_shaped_type(shape, dim + 1, element_type)
    length = shape[dim]
    return f"[{','.join([inner_content] * length)}]"


def mlir_stringify_type(dtype: qp.typing.AbstractArray):
    """Return a string representation of the given data type."""
    assert isinstance(
        dtype, qp.typing.AbstractArray
    ), f"Expected an AbstractArray to stringify, got {dtype}"
    element_type = dtype.dtype.type
    if dtype.shape == ():
        return f"[{_PY_DTYPES_TO_MLIR_DTYPES[element_type]}]"
    else:
        return _stringify_shaped_type(dtype.shape, 0, element_type)


def get_dummy_values_for_container(container):
    """
    Given a container of python or MLIR types, replace the types with corresponding dummy values.

    Each item in the container must be representible as an MLIR tensor with at most one layer of
    nesting, i.e. cannot be nested and all elements must be of the same type.
    Ex.
    [[float, float], [int, int, int], [int32, int32, int32, int32]]
    """
    if isinstance(container, str):
        return jnp.zeros((), dtype=_MLIR_DTYPES_TO_PY_DTYPES[container])

    def handle_item(item):
        if isinstance(item, (list, tuple)):
            return jnp.zeros(len(item), dtype=handle_item(item[0]).dtype)
        if isinstance(item, ShapedArray):
            return jnp.zeros(item.shape[0], dtype=item.dtype)
        elif isinstance(item, str):
            return jnp.zeros((), dtype=_MLIR_DTYPES_TO_PY_DTYPES[item])
        elif isinstance(item, (type, jnp.dtype)):
            try:
                return jnp.zeros((), jnp.dtype(item))
            except TypeError:
                raise TypeError(
                    f"Unexpected type in container when creating dummy values: {type(item)}"
                )

    return tuple(handle_item(item) for item in container)
