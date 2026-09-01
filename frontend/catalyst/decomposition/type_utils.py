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
import re

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
    ir.F16Type: "f16",
    ir.F32Type: "f32",
    ir.F64Type: "f64",
    (ir.ComplexType, ir.F32Type): "complex<f32>",
    (ir.ComplexType, ir.F64Type): "complex<f64>",
    np.dtype("bool_"): "i1",
    np.dtype("int8"): "i8",
    np.dtype("int16"): "i16",
    np.dtype("int32"): "i32",
    np.dtype("int64"): "i64",
    np.dtype("float16"): "f16",
    np.dtype("float32"): "f32",
    np.dtype("float64"): "f64",
    np.dtype("complex64"): "complex<f32>",
    np.dtype("complex128"): "complex<f64>",
}


def convert_item_to_mlir_type(item, is_special_lowering=False):
    """Convert a string or PennyLane AbstractArray to an mlir type annotation."""
    if isinstance(item, str):
        return item

    if item.shape == ():
        if is_special_lowering:
            return _PY_DTYPES_TO_MLIR_DTYPES[item.dtype]
        return "tensor<" + _PY_DTYPES_TO_MLIR_DTYPES[item.dtype] + ">"

    return (
        "tensor<"
        + "x".join(str(dim_size) for dim_size in item.shape)
        + "x"
        + _PY_DTYPES_TO_MLIR_DTYPES[item.dtype]
        + ">"
    )


def format_dynamic_params_for_id(d):
    """Format a structure for ID."""

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
    """Given a container of python or MLIR types, replace the types with corresponding dummy values.

    The types are expected to be formatted for ``GraphOpId``s. Lists/Tuples must contain homogeneous
    data types (this is true for any operator).
    """
    match arg:
        case str():
            if arg.startswith("tensor"):
                # Captures the optional dimensions (e.g., '2x2x') in group 1, and the
                # element type in group 2
                match = re.match(r"^tensor<((?:\d+x)*)(.*)>$", arg)
                dim_str, dtype = match.groups()
                ranks = tuple(int(d) for d in dim_str.split("x") if d)
                return jnp.zeros(ranks, dtype=_MLIR_DTYPES_TO_PY_DTYPES[dtype])
            else:
                return jnp.zeros((), dtype=_MLIR_DTYPES_TO_PY_DTYPES[arg])
        case list() | tuple():
            if all(isinstance(e, str) for e in arg) and arg[0].startswith("tensor"):
                # if arg is something like [tensor<...>], i.e. a single tensor but carrying the
                # layer of brackets from StringMap<Vector<Type>>, np str parsing fails to realize
                # the actual tensor shape, and we need to do it manually
                assert len(arg) == 1, "cannot create a tensor of tensors"
                return get_dummy_values_for_arg(arg[0])
            else:
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


def replace_wires_with_placeholder_wires(node):
    """
    Replace every wire container (abstract or concrete) in ``node`` with placeholder wires.

    Wire labels never affect which decomposition rules apply to an operator: at lowering time
    wires always show up as (abstract) qubit operands. Hence both `AbstractWires` and concrete
    `Wires` are replaced with negative placeholder labels, so that operators only differing in
    their (concrete) wire labels reduce to the same ``GraphOpId``.
    """
    if isinstance(node, qp.core.Operator2):
        return _replace_op_wires_with_placeholder_wires(node)

    if isinstance(node, list):
        return [replace_wires_with_placeholder_wires(item) for item in node]
    elif isinstance(node, dict):
        return {k: replace_wires_with_placeholder_wires(v) for k, v in node.items()}
    elif isinstance(node, tuple):
        return tuple(replace_wires_with_placeholder_wires(item) for item in node)
    else:
        if isinstance(node, (qp.typing.AbstractWires, qp.wires.Wires)):
            return _placeholder_wires(len(node))
        else:
            return node


def _placeholder_wires(num_wires):
    """Return ``num_wires`` placeholder wires, labelled with negative integers."""
    return qp.wires.Wires(range(-1, -num_wires - 1, -1))


def _replace_op_wires_with_placeholder_wires(op2):
    """
    Given an Operator2 instance, return a copy of the same instance but with all fields whose value
    is an ``AbstractWires`` or a concrete ``Wires`` replaced with placeholder ``Wires``.
    """
    new_op = copy.deepcopy(op2)
    for wire_arg in new_op.wire_argnames:
        wire_val = new_op.arguments[wire_arg]
        if isinstance(wire_val, (qp.typing.AbstractWires, qp.wires.Wires)):
            new_op.arguments[wire_arg] = _placeholder_wires(len(wire_val))
    for hybrid_arg in new_op.hybrid_argnames:
        if isinstance(new_op.arguments[hybrid_arg], qp.core.Operator2):
            new_op.arguments[hybrid_arg] = _replace_op_wires_with_placeholder_wires(
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
