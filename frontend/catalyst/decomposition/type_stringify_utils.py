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
import pennylane as qp

from catalyst.utils.exceptions import CompileError


def _py_dtype_to_mlir_type_string(python_dtype: type):
    match python_dtype:
        case jnp.float64:
            return "f64"
        case _:
            raise CompileError("Unknown data type")


def _stringify_shaped_type(shape: tuple, dim: int, element_type):
    if dim + 1 == len(shape):
        inner_content = _py_dtype_to_mlir_type_string(element_type)
    else:
        inner_content = _stringify_shaped_type(shape, dim + 1, element_type)
    length = shape[dim]
    return f"[{','.join([inner_content] * length)}]"


def mlir_stringify_type(dtype: qp.typing.AbstractArray):
    assert isinstance(dtype, qp.typing.AbstractArray)
    element_type = dtype.dtype.type
    if dtype.shape == ():
        return _py_dtype_to_mlir_type_string(element_type)
    else:
        return _stringify_shaped_type(dtype.shape, 0, element_type)
