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
from jax._src.interpreters.mlir import dtype_to_ir_type
from jax._src.lib.mlir import ir
from jax.core import ShapedArray
from pennylane.pytrees import flatten, unflatten

from catalyst.jax_extras.lowering import mlir_build_context

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


def convert_item_to_mlir_type(item, is_special_lowering=False):
    """Convert a string or PennyLane AbstractArray to an mlir type annotation.

    The type is spelled by MLIR's own printer, applied to the type the value lowers to, which is
    what ``printDynamicShape`` in mlir/lib/Quantum/IR/QuantumInterfaces.cpp does as well. One
    printer spelling both sides is what keeps a rule compiled here findable by the
    ``graph-decomposition`` pass.
    """
    if isinstance(item, str):
        return item

    with mlir_build_context():
        element_type = dtype_to_ir_type(np.dtype(item.dtype))
        if is_special_lowering and item.shape == ():
            return str(element_type)
        return str(ir.RankedTensorType.get(item.shape, element_type))


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


def _is_wires(node):
    """Return whether ``node`` is a container of wires, abstract or concrete."""
    return isinstance(node, (qp.typing.AbstractWires, qp.wires.Wires))


def replace_wires_with_placeholder_wires(node):
    """
    Return a copy of the pytree ``node`` in which every wire container (abstract or concrete,
    including the ones nested inside ``Operator2`` instances) is replaced by placeholder wires
    labelled with negative integers.

    Wire labels never affect which decomposition rules apply to an operator: at lowering time
    wires always show up as (abstract) qubit operands. Replacing them with placeholders makes
    operators that only differ in their (concrete) wire labels reduce to the same ``GraphOpId``.

    Note that the result is a deep copy.
    """
    # Wires is a pytree itself, so it has to be marked as a leaf to be replaced as a whole.
    leaves, tree = flatten(copy.deepcopy(node), is_leaf=_is_wires)
    leaves = [
        qp.wires.Wires(range(-1, -len(leaf) - 1, -1)) if _is_wires(leaf) else leaf
        for leaf in leaves
    ]
    return unflatten(leaves, tree)


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
