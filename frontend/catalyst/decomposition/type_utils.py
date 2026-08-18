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

import copy

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
    "complex<f32>": jnp.complex64,
    "complex<f64>": jnp.complex128,
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


def get_dummy_values_for_arg(arg):
    """
    Given a container of python or MLIR types, replace the types with corresponding dummy values.

    Each item in the container must be representible as an MLIR tensor with at most one layer of
    nesting, i.e. cannot be nested and all elements must be of the same type.
    Ex.
    [[float, float], [int, int, int], [int32, int32, int32, int32]]
    """
    if isinstance(arg, str):
        return jnp.zeros((), dtype=_MLIR_DTYPES_TO_PY_DTYPES[arg])
    elif isinstance(arg, (list, tuple)):
        return jnp.zeros(len(arg), dtype=get_dummy_values_for_arg(arg[0]).dtype)
    elif isinstance(arg, ShapedArray):
        return jnp.zeros(arg.shape[0], dtype=arg.dtype)
    elif isinstance(arg, str):
        return jnp.zeros((), dtype=_MLIR_DTYPES_TO_PY_DTYPES[arg])
    elif isinstance(arg, (type, jnp.dtype)):
        try:
            return jnp.zeros((), jnp.dtype(arg))
        except TypeError:
            pass

    raise TypeError(f"Unexpected type in container when creating dummy values: {type(arg)}")


def replace_abstract_wires_with_concrete_wires(node):
    if isinstance(node, qp.core.Operator2):
        return _replace_op_abstract_wires_with_concrete_wires(node)

    if isinstance(node, list):
        return [replace_abstract_wires_with_concrete_wires(item) for item in node]
    elif isinstance(node, dict):
        return {k: replace_abstract_wires_with_concrete_wires(v) for k, v in node.items()}
    elif isinstance(node, tuple):
        return tuple(replace_abstract_wires_with_concrete_wires(item) for item in node)
    else:
        if isinstance(node, qp.typing.AbstractWires):
            return qp.wires.Wires(range(node.num_wires))
        else:
            return node


def _replace_op_abstract_wires_with_concrete_wires(op2):
    """
    Given an Operator2 instance, return a copy of the same instance but with all fields whose value
    is an `AbstractWires` replaced with concrete `Wires`.
    """
    new_op = copy.deepcopy(op2)
    for wire_arg in new_op.wire_argnames:
        if isinstance(new_op.arguments[wire_arg], qp.typing.AbstractWires):
            num_wires = new_op.arguments[wire_arg].num_wires
            new_op.arguments[wire_arg] = qp.wires.Wires(range(-1, -num_wires - 1, -1))
    for hybrid_arg in new_op.hybrid_argnames:
        if isinstance(new_op.arguments[hybrid_arg], qp.core.Operator2):
            new_op.arguments[hybrid_arg] = _replace_op_abstract_wires_with_concrete_wires(
                new_op.arguments[hybrid_arg]
            )

    return new_op


def post_process_concretize_leaves(leaves):
    """
    Given a list of pytree leaf values, change all `AbstractArray`s to `ShapedArray`s of the same
    shape and dtype, and change all negative integers to `AbstractQubit()`s.
    """
    out_leaves = []
    for leaf in leaves:
        if isinstance(leaf, int) and leaf < 0:
            out_leaves.append(qp.wires.AbstractQubit())
        elif isinstance(leaf, qp.typing.AbstractArray):
            out_leaves.append(
                ShapedArray(shape=leaf.shape, dtype=leaf.dtype, weak_type=leaf._weak_type)
            )
        else:
            out_leaves.append(leaf)

    return out_leaves
