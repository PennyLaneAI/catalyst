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

from pennylane.core import Operator2
from pennylane.pytrees import PyTreeStructure
from pennylane.wires import AbstractQubit


# pylint: disable=too-many-arguments,too-many-positional-arguments
def generate_uid(
    *avals_in: tuple[Any, ...],
    op_cls: type[Operator2],
    wire_lens: tuple[int, ...],
    hybrid_lens: tuple[int, ...],
    hybrid_trees: tuple[PyTreeStructure, ...],
    adjoint: bool,
    n_ctrls: int,
    static_args: dict[str, Any],
):
    """Generate a unique identifier that allows us to distinguish between
    operators with unique non-compilable arguments."""

    # Flat dynamic arguments
    dynamic_args = avals_in[: len(op_cls.dynamic_argnames)]
    dynamic_avals = []
    for val in dynamic_args:
        dynamic_avals.append((val.shape, val.dtype.name))

    # Hybrid arguments (wire and non-wire)
    args_idx = len(op_cls.dynamic_argnames) + sum(wire_lens)
    hybrid_avals = []
    for hname, hsize in zip(op_cls.hybrid_argnames, hybrid_lens):
        if hname in op_cls.wire_argnames:
            hybrid_avals.append(hsize)

        else:
            cur_avals = []
            for val in avals_in[args_idx : args_idx + hsize]:
                aval = val if isinstance(val, AbstractQubit) else (val.shape, val.dtype.name)
                cur_avals.append(aval)
            hybrid_avals.append(tuple(cur_avals))

        args_idx += hsize

    # Static arguments
    reduced_static_args = tuple(_serialize_static(val, name) for name, val in static_args.items())

    reduced = [op_cls]
    reduced.append(("dynamic", tuple(dynamic_avals)))
    reduced.append(("wires", wire_lens))
    reduced.append(("hybrid", hybrid_trees, tuple(hybrid_avals)))
    reduced.append(("static", reduced_static_args))
    reduced.append(("adjoint", adjoint))
    reduced.append(("n_ctrls", n_ctrls))

    return hash(tuple(reduced))


@singledispatch
def _serialize_static(val: Any, name: str | None):
    """Create a reduced representation of a value that can be used to easily
    create a UID for it.

    The reduced representation will be a tuple with the following format:

    .. code-block:: python

        (name, type, hashable_reduction)
    """
    # For arbitrary opaque data that may be unhashable, just use the id
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
