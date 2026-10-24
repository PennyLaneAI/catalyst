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
The positional layout of the operands of an ``Operator2`` lowered to ``qref.operator``.

This module is the single definition of the order in which the named arguments of an operator
appear as operands. Everything that relies on that order positionally (the lowering of the
operator, the GraphOpID parser, the UID, and the operand layout of compiled decomposition rules)
must derive it from the functions here, so they cannot disagree.

The order is the operand order produced by ``Operator2._primitive_bind_call`` in PennyLane:

* parameters: ``dynamic_argnames`` in declaration order, followed by the numeric leaves of each
  non-wire hybrid argument, in ``hybrid_argnames`` order;
* qubits: every pure (non-hybrid) wire argument in ``wire_argnames`` order, including empty ones,
  followed by every hybrid argument that contributes qubits, in ``hybrid_argnames`` order.

Names are used only as identity. The GraphOpID sorts them when it is printed, and the
``param_map``/``qubit_map`` dictionary attributes are sorted by MLIR; neither of those is an
operand order. To recover the operand order from the IR, order those entries by operand index.
"""

from collections.abc import Iterable, Mapping
from typing import Any


def pure_wire_argnames(op_cls) -> tuple[str, ...]:
    """Return the wire arguments lowered as flat qubit operands, in operand order.

    These are the names in ``wire_argnames`` that are not hybrid arguments. Hybrid wire arguments
    are lowered with the other hybrid arguments, after all of these.
    """
    return tuple(name for name in op_cls.wire_argnames if name not in op_cls.hybrid_argnames)


def qubit_layout(
    op_cls, pure_wire_lens: Iterable[int], hybrid_qubit_counts: Iterable[int]
) -> dict[str, int]:
    """Return the qubit arguments of an operator mapped to their number of qubits, in operand order.

    Pure wire arguments are always present, even when empty, because the lowered op and its
    GraphOpID must name them. Hybrid arguments are present only if they contribute qubits.

    Args:
        op_cls (type): the ``Operator2`` class
        pure_wire_lens (Iterable[int]): the length of each of ``pure_wire_argnames(op_cls)``
        hybrid_qubit_counts (Iterable[int]): the number of qubits in each of
            ``op_cls.hybrid_argnames``

    Returns:
        dict[str, int]: argument names to numbers of qubits, in operand order
    """
    layout = dict(zip(pure_wire_argnames(op_cls), pure_wire_lens, strict=True))
    for name, count in zip(op_cls.hybrid_argnames, hybrid_qubit_counts, strict=True):
        if count:
            layout[name] = count
    return layout


def param_layout(
    op_cls, dynamic_params: Mapping[str, Any], hybrid_params: Mapping[str, list]
) -> dict[str, list]:
    """Return the parameter arguments of an operator mapped to their values, in operand order.

    Args:
        op_cls (type): the ``Operator2`` class
        dynamic_params (Mapping[str, Any]): a value per name in ``op_cls.dynamic_argnames``
        hybrid_params (Mapping[str, list]): the numeric leaves of the non-wire hybrid arguments;
            arguments missing from it, or without leaves, are left out

    Returns:
        dict[str, list]: argument names to their list of parameter values, in operand order
    """
    layout = {name: [dynamic_params[name]] for name in op_cls.dynamic_argnames}
    for name in op_cls.hybrid_argnames:
        if name not in op_cls.wire_argnames and hybrid_params.get(name):
            layout[name] = list(hybrid_params[name])
    return layout
