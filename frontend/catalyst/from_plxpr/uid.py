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

import hashlib
from functools import singledispatch
from typing import Any

from pennylane.core import Operator2
from pennylane.pytrees import PyTreeStructure
from pennylane.wires import AbstractQubit


# pylint: disable=too-many-arguments,too-many-positional-arguments
def generate_uid(
    *avals_in: tuple[Any, ...],
    op_cls: type[Operator2],
    hybrid_lens: tuple[int, ...],
    hybrid_trees: tuple[PyTreeStructure, ...],
    static_args: dict[str, Any],
):
    """Generate a unique identifier that allows us to distinguish between
    operators with unique non-compilable arguments."""

    # Hybrid arguments (wire and non-wire)
    arg_idx = 0
    hybrid_avals = []
    for hname, hsize in zip(op_cls.hybrid_argnames, hybrid_lens):
        if hname in op_cls.wire_argnames:
            hybrid_avals.append(hsize)

        else:
            cur_avals = tuple(
                val if isinstance(val, AbstractQubit) else (val.shape, val.dtype.name)
                for val in avals_in[arg_idx : arg_idx + hsize]
            )
            hybrid_avals.append(cur_avals)

        arg_idx += hsize

    serialized_hybrid = (hybrid_trees, tuple(hybrid_avals))
    serialized_static = tuple(
        (name, type(val), _serialize(val)) for name, val in static_args.items()
    )

    encoded_bytes = str((serialized_hybrid, serialized_static)).encode("utf-8")
    sha_hash = hashlib.sha256(encoded_bytes).hexdigest()

    # hexdigest() returns the hexadecimal hash in string format
    # Take 16 hexadecimals, since UID on Operator op is I64Attr. Right-shift
    # to stay in the safe range of positive signed 64-bit integers
    return int(sha_hash[:16], 16) >> 1


@singledispatch
def _serialize(val: Any):
    """Create a serialized representation of a value that can be used to easily
    create a UID for it.
    """
    return str(val)


@_serialize.register(list | tuple)
def _serialize_sequence(val):
    return tuple(_serialize(item) for item in val)


@_serialize.register(dict)
def _serialize_dict(val):
    serialized = ((str(_serialize(k)), _serialize(v)) for k, v in val.items())
    return tuple(sorted(serialized, key=lambda item: item[0]))


@_serialize.register(set | frozenset)
def _serialize_set(val):
    return tuple(sorted(str(_serialize(v)) for v in val))
