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
"""
UID generation logic for compiling operators with non-compilable data.
"""
from functools import singledispatch
from typing import Any

from pennylane import math
from pennylane.core import Operator2
from pennylane.pytrees import PyTreeStructure, unflatten


# pylint: disable=too-many-arguments,too-many-positional-arguments
def generate_uid(
    all_dynamic_args: tuple[Any, ...],
    op_cls: type[Operator2],
    wire_lens: tuple[int, ...],
    hybrid_lens: tuple[int, ...],
    hybrid_trees: tuple[PyTreeStructure, ...],
    static_args: dict[str, Any],
):
    """Generate a unique identifier that allows us to distinguish between
    operators with unique non-compilable arguments."""
    reduced = []

    # Flat dynamic arguments
    dynamic_args = all_dynamic_args[: len(op_cls.dynamic_argnames)]
    dynamic_shapes, dynamic_dtypes = [], []
    for val in dynamic_args:
        dynamic_shapes.append(math.shape(val))
        dynamic_dtypes.append(math.get_dtype_name(val))

    # Hybrid wire arguments
    hybrid_wires = tuple(name for name in op_cls.hybrid_argnames if name in op_cls.wire_argnames)

    # Non-wire hybrid arguments
    hybrid_start = (
        len(op_cls.dynamic_argnames) + sum(wire_lens) + sum(hybrid_lens[: len(hybrid_wires)])
    )
    hybrid_args = all_dynamic_args[hybrid_start:]
    hybrid_shapes, hybrid_dtypes = [], []
    i = 0
    for len_ in hybrid_lens:
        cur_shapes, cur_dtypes = [], []
        for val in hybrid_args[i : i + len_]:
            cur_shapes.append(math.shape(val))
            cur_dtypes.append(math.get_dtype_name(val))

        hybrid_shapes.append(tuple(cur_shapes))
        hybrid_dtypes.append(tuple(cur_dtypes))
        i += len_

    # Static arguments
    reduced_static_values = tuple(
        _serialize_static(unflatten(*static_args[name]), name) for name in op_cls.static_argnames
    )

    reduced += ("dynamic", tuple(dynamic_shapes), tuple(dynamic_dtypes))
    reduced += ("wires", wire_lens)
    reduced += ("hybrid_wires", hybrid_trees[: len(hybrid_wires)], hybrid_lens[: len(hybrid_wires)])
    reduced += ("hybrid", hybrid_trees[len(hybrid_wires) :], hybrid_shapes, hybrid_dtypes)
    reduced += ("static", reduced_static_values)

    return hash(tuple(reduced))


@singledispatch
def _serialize_static(val: Any, name: str | None):
    """Create a reduced representation of a value that can be used to easily
    create a UID for it.

    The reduced representation will be a tuple with the following format:

    .. code-block:: python

        (name, type, hashable_reduction)
    """
    # For arbitrary unhashable data that is opaque, just use the id
    return (name, type(val), id(val))


# pylint: disable=unused-argument
@_serialize_static.register(type(None))
def _serialize_none(val, name):
    return (name, type(None), None)


@_serialize_static.register(bool)
def _serialize_bool(val, name):
    return (name, bool, val)


@_serialize_static.register(int)
def _serialize_int(val, name):
    return (name, int, val)


@_serialize_static.register(float)
def _serialize_float(val, name):
    return (name, float, repr(val))


@_serialize_static.register(complex)
def _serialize_complex(val, name):
    return (name, complex, (repr(val.real), repr(val.imag)))


@_serialize_static.register(str)
def _serialize_str(val, name):
    return (name, str, val)


@_serialize_static.register(list)
def _serialize_list(val, name):
    return (name, list, tuple(_serialize_static(item, None) for item in val))


@_serialize_static.register(tuple)
def _serialize_tuple(val, name):
    return (name, tuple, tuple(_serialize_static(item, None) for item in val))


@_serialize_static.register(dict)
def _serialize_dict(val, name):
    return (
        name,
        dict,
        frozenset((_serialize_static(k, None), _serialize_static(v, None)) for k, v in val.items()),
    )


@_serialize_static.register(set | frozenset)
def _serialize_set(val, name):
    return (name, type(val), frozenset(_serialize_static(item, None) for item in val))
