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

import jax.numpy as jnp
import numpy as np
import pennylane as qp

_MLIR_DTYPES_TO_PY_DTYPES = {
    "i1": np.bool_,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
    "complex<f64>": np.complex64,
    "complex<f128>": np.complex128,
}

_PY_DTYPES_TO_MLIR_DTYPES = {v: k for k, v in _MLIR_DTYPES_TO_PY_DTYPES.items()}


def _process_shaped_type(shape: tuple, dim: int, element_type, formatter):
    """Recursively processes the shape, applying the formatter at each dimension."""
    if dim + 1 == len(shape):
        inner_content = _PY_DTYPES_TO_MLIR_DTYPES[element_type]
    else:
        inner_content = _process_shaped_type(shape, dim + 1, element_type, formatter)

    return formatter(inner_content, shape[dim])


def _convert_type(dtype: qp.typing.AbstractArray, formatter):
    """Base function to handle scalar checks before processing the shape."""
    assert isinstance(dtype, qp.typing.AbstractArray)
    element_type = dtype.dtype.type

    if dtype.shape == ():
        return _PY_DTYPES_TO_MLIR_DTYPES[element_type]

    return _process_shaped_type(dtype.shape, 0, element_type, formatter)


def mlir_stringify_type(dtype: qp.typing.AbstractArray):
    string_formatter = lambda content, length: f"[{','.join([content] * length)}]"
    return _convert_type(dtype, string_formatter)


def listify_type(dtype: qp.typing.AbstractArray):
    list_formatter = lambda content, length: [content] * length
    return _convert_type(dtype, list_formatter)


def get_dummy_values_for_container(container):
    """
    Converts a nested list of dtype strings into a matching nested list of jnp.zeros.
    """
    if isinstance(container, (list, tuple)):
        return [get_dummy_values_for_container(item) for item in container]
    elif isinstance(container, str):
        return jnp.zeros((), dtype=_MLIR_DTYPES_TO_PY_DTYPES[container])
    else:
        raise TypeError(
            f"Unexpected type in container when creating dummy values: {type(container)}"
        )
