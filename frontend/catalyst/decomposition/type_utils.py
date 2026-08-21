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
import numpy as np
import pennylane as qp
from jax._src.lib.mlir import ir
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

_PY_DTYPES_TO_MLIR_DTYPES = {v: k for k, v in _MLIR_DTYPES_TO_PY_DTYPES.items()} | {
    float: "f64",
    int: "i64",
    complex: "complex<f64>",
    (ir.IntegerType, 1): "i1",
    (ir.IntegerType, 8): "i8",
    (ir.IntegerType, 16): "i16",
    (ir.IntegerType, 32): "i32",
    (ir.IntegerType, 64): "i64",
    ir.F16Type: "f16",
    ir.F32Type: "f32",
    ir.F64Type: "f64",
    (ir.ComplexType, ir.F64Type): "complex<f64>",
}


def get_mlir_tensor_type_map_key(mlir_type):
    if isinstance(mlir_type, ir.ComplexType):
        return (type(mlir_type), type(mlir_type.element_type))
    if isinstance(mlir_type, ir.IntegerType):
        return (type(mlir_type), mlir_type.width)
    return type(mlir_type)


def convert_shaped_type_to_mlir_string(shaped_type, current_dim=0):
    """Convert a shape of arbitrary dimension to a string with MLIR type strings for values."""
    if isinstance(shaped_type, (ShapedArray, qp.typing.AbstractArray)):
        if current_dim == shaped_type.ndim:
            return _PY_DTYPES_TO_MLIR_DTYPES[shaped_type.dtype.type]

        return [
            convert_shaped_type_to_mlir_string(shaped_type, current_dim + 1)
        ] * shaped_type.shape[current_dim]
    elif isinstance(shaped_type, ir.RankedTensorType):
        if current_dim == shaped_type.rank:
            return _PY_DTYPES_TO_MLIR_DTYPES[get_mlir_tensor_type_map_key(shaped_type.element_type)]

        return [
            convert_shaped_type_to_mlir_string(shaped_type, current_dim + 1)
        ] * shaped_type.shape[current_dim]


def convert_types_to_mlir_strings(d: dict) -> dict:
    """Convert the values of a dictionary to MLIR type strings."""

    def handle_item(item):
        match item:
            case str():
                return item
            case type():
                return _PY_DTYPES_TO_MLIR_DTYPES[item]
            case ir.RankedTensorType():
                if len(item.shape) == 0:
                    return [
                        _PY_DTYPES_TO_MLIR_DTYPES[get_mlir_tensor_type_map_key(item.element_type)]
                    ]
                return convert_shaped_type_to_mlir_string(item)
            case float() | int() | complex():
                # these need to be wrapped in an additional list to account for the tensor creation in lowering
                return [_PY_DTYPES_TO_MLIR_DTYPES[type(item)]]
            case list() | tuple():
                return [handle_item(i) for i in item]
            case ShapedArray() | qp.typing.AbstractArray():
                if item.shape == ():
                    return [_PY_DTYPES_TO_MLIR_DTYPES[item.dtype.type]]
                return convert_shaped_type_to_mlir_string(item)
            case _ if type(item) in _PY_DTYPES_TO_MLIR_DTYPES:
                return _PY_DTYPES_TO_MLIR_DTYPES[type(item)]
            case _:
                raise TypeError(
                    f"encountered unknown type {type(item)} of item {item} when converting to mlir strings."
                )

    return {k: handle_item(v) for k, v in d.items()}


def format_dynamic_params_for_id(d):
    """Format a structure for ID, after calling convert_types_to_mlir_string on it."""

    def handle_item(item):
        match item:
            case str():
                return item
            case list() | tuple():
                return "[" + ",".join(handle_item(i) for i in item) + "]"

    return (
        "{"
        + ",".join(
            k + ":" + "[" + ",".join(handle_item(item) for item in v) + "]" for k, v in d.items()
        )
        + "}"
    )


def get_dummy_values_for_arg(arg):
    """
    Given a container of python or MLIR types, replace the types with corresponding dummy values.

    Each item in the container must be representible as an MLIR tensor with at most one layer of
    nesting, i.e. cannot be nested and all elements must be of the same type.
    Ex.
    [[float, float], [int, int, int], [int32, int32, int32, int32]]
    """
    match arg:
        case str():
            return jnp.zeros((), dtype=_MLIR_DTYPES_TO_PY_DTYPES[arg])
        case list() | tuple():
            dtype = get_dummy_values_for_arg(arg[0]).dtype
            # NOTE: numpy is required since jax won't create an array of strings
            return jnp.zeros(np.array(arg, str).shape, dtype)
        case ShapedArray():
            return jnp.zeros(arg.shape[0], dtype=arg.dtype)
        case type() | jnp.dtype():
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
            num_wires = len(new_op.arguments[wire_arg])
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
