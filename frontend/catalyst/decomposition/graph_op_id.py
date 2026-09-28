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

"""Python implementation of Graph Operator ID."""

from collections.abc import Mapping, Sequence
from typing import Any

import pennylane as qp
from pennylane.ops.op_math.adjoint2 import Adjoint2
from pennylane.ops.op_math.controlled2 import ControlledOp2
from pennylane.pytrees import flatten

from catalyst.decomposition.type_utils import (
    convert_item_to_mlir_type,
    post_process_concretize_leaves,
    replace_wires_with_placeholder_wires,
)
from catalyst.from_plxpr.uid import generate_uid
from catalyst.jax_extras.lowering import get_mlir_attribute_from_pyval, mlir_build_context

_SPECIAL_LOWERINGS = {}


def _is_custom_op(op_cls, avals_in):
    """Return whether an operator lowers to ``qref.custom`` rather than ``qref.operator``.

    Callers reached through a special lowering are handled before this is consulted, so it does
    not exclude ``_SPECIAL_LOWERINGS``; :meth:`GraphOpID.parse_is_custom_op` adds that itself.
    """
    if op_cls.static_argnames or op_cls.hybrid_argnames or op_cls.compilable_argnames:
        return False
    if op_cls.wire_argnames != ("wires",):
        return False
    if list(op_cls._sig.parameters.keys())[-1] != "wires":
        return False
    # Params are widened to f64 by `safe_cast_to_f64`; complex cannot be cast safely.
    return all(p.shape == () and p.dtype.kind in "ifu" for p in avals_in)


def _is_wires(item) -> bool:
    """Return whether a pytree leaf is a group of wires."""
    return isinstance(item, (qp.typing.AbstractWires, qp.wires.Wires))


def _is_op_or_wires(item) -> bool:
    """Return whether a pytree leaf is a nested operator or a group of wires."""
    return isinstance(item, qp.core.Operator2) or _is_wires(item)


def format_static_data_dict_for_id(static_data):
    """Format the static-data group of a GraphOpID with MLIR's attribute printer."""
    with mlir_build_context():
        return str(get_mlir_attribute_from_pyval(static_data))


def format_dynamic_params_for_id(dynamic_shape):
    """Format the dynamic-parameter group of a GraphOpID."""

    def handle_item(item):
        match item:
            case str():
                return item
            case list() | tuple():
                return "[" + ",".join(handle_item(i) for i in item) + "]"

    return (
        "{"
        + ",".join(
            name + ":" + "[" + ",".join(handle_item(item) for item in types) + "]"
            for name, types in dynamic_shape.items()
        )
        + "}"
    )


def build_graph_op_id(
    operator_name: str,
    dynamic_shape: Mapping[str, Sequence[str]],
    wire_lens: Mapping[str, int],
    static_data: Mapping[str, Any],
    *,
    adjoint: bool = False,
    num_controls: int = 0,
    uid: int | None = None,
) -> str:
    """Build a canonical frontend GraphOpID from its identity components."""
    if num_controls < 0:
        raise ValueError("GraphOpID control count cannot be negative")

    name = f"Adjoint({operator_name})" if adjoint else operator_name
    if num_controls:
        prefix = "C" if num_controls == 1 else f"{num_controls}C"
        name = f"{prefix}({name})"

    dynamic_id = format_dynamic_params_for_id(dict(sorted(dynamic_shape.items())))
    wire_id = "{" + ",".join(f"{key}:{value}" for key, value in sorted(wire_lens.items())) + "}"
    static_id = format_static_data_dict_for_id(dict(static_data))
    uid_id = f"[{uid}]" if uid is not None else ""
    return name + dynamic_id + wire_id + static_id + uid_id


class GraphOpID:
    """
    A parser object to compute the graph operator id for an abstract operator2 instance `op`.

    The format of the computed graph op ID string is as follows:
        op_name{param_shaped_type_dictionary}{wire_lens_dictionary}{static_data_dictionary}[UID]

    The types in the dynamic shape dictionary should be represented as a list of MLIR-style type annotations.
    The UID is computed from the shapes, dtypes and pytree structures of the `hybrid_args` of
    the Operator2 instance.

    For example, an Operator2 instance with class name `HybridOpArg`, taking in one float param
    argument named `angle`, one wire argument named `cwires`, one static data argument
    `label="hello"`, and a computed UID of 10 would be parsed to the following graph op ID:
        HybridOpArg{angle:[tensor<f64>]}{cwires:1}{label = "hello"}[10]

    The static data group is spelled by MLIR's own attribute printer, applied to the attributes the
    data lowers to, so each entry reads as it would inside the `static_data` dictionary on the op.

    The defining trait of a graph op ID is that it has unique correspondence to decomposition rules.
    In other words, different graph op IDs have different sets of decomposition rules.

    For example,
        PauliRot{angle:[f64]}{wires:1}{pauli_word = "X"}
    and
        PauliRot{angle:[f64]}{wires:2}{pauli_word = "XX"}
    will have different decomposition rules.

    Note that this function should not be updated without updating the corresponding methods on the
    DecomposableGate interface in mlir/lib/quantum/IR/QuantumInterfaces.cpp and the corresponding
    DecomposableGate interface in mlir/include/QRef/IR/QRefInterfaces.h.
    """

    def __init__(self, op: qp.core.Operator2):
        """Create a new GraphOpId."""
        assert isinstance(
            op, qp.core.Operator2
        ), f"Graph-based decomposition expects an Operator2 instance, got {op} of type {type(op)}"
        op, self.adjoint, self.num_controls = self.peel_modifiers(op)
        self.op = op
        self.is_custom_op = self.parse_is_custom_op()

        self.operator_name = op.name
        self.dynamic_shape = self.parse_dynamic_shape()
        self.wire_lens = self.parse_wire_lens()
        self.static_data = self.parse_static_data()
        self.extra_data, self.uid = self.parse_extra_data()

    @staticmethod
    def peel_modifiers(op: qp.core.Operator2):
        """Return the innermost base of ``op`` together with the modifiers wrapping it.

        A *generic* symbolic operator (``Adjoint2``/``ControlledOp2``, as opposed to a concrete
        class such as ``CH``) reports its wrapper's own arguments, e.g.
        ``Adjoint(S){}{}{}[1141126509488406748]``. That is not how the compiler spells a modified
        operator: ``wrapModifiers`` folds the modifier into the *base* operator's id. So the
        wrappers are peeled off here and counted, and :meth:`getGraphOpId` puts them back in
        canonical position (control outermost) rather than splicing them into a finished id.

        The walk terminates structurally: each step moves to ``op.base``, one wrapper shallower,
        and the chain ends at the first non-symbolic operator.

        Args:
            op (Operator2): the operator to unwrap

        Returns:
            Operator2: the innermost base operator
            bool: whether the base is adjointed
            int: how many controls wrap the base
        """
        adjoint, num_controls = False, 0
        while True:
            if isinstance(op, Adjoint2):
                adjoint = not adjoint
            elif isinstance(op, ControlledOp2):
                num_controls += len(op.control_wires)
            else:
                return op, adjoint, num_controls
            op = op.base

    def parse_dynamic_shape(self) -> dict:
        """Return a dictionary of dynamic arg names to list of dtypes."""
        # enters as {name: dtype}, we want the format {name: list[dtype]}
        if self.is_custom_op:
            return {str(i): ["f64"] for i in range(len(self.op.dynamic_args))}
        elif isinstance(self.op, qp.QubitUnitary):
            # `qref.unitary` always takes a complex matrix, so a real one is converted on the way
            # in and the id must spell the converted type, not the one the user passed.
            name, matrix = next(iter(self.op.dynamic_args.items()))
            spec = qp.typing.AbstractArray(qp.math.shape(matrix), complex)
            return {name: [convert_item_to_mlir_type(spec, is_special_lowering=True)]}
        elif issubclass(type(self.op), tuple(_SPECIAL_LOWERINGS.keys())):  # special cases
            return {
                argname: [convert_item_to_mlir_type(argtype, is_special_lowering=True)]
                for argname, argtype in sorted(self.op.dynamic_args.items())
            }
        else:
            dynamic_shape = {
                argname: [convert_item_to_mlir_type(argtype)]
                for argname, argtype in sorted(self.op.dynamic_args.items())
            }
            # Collect additional dynamic params from hybrid args (have to match _process_params).
            for argname, value in sorted(self.op.hybrid_args.items()):
                if argname in self.op.wire_argnames:  # do not include wires
                    continue
                # skip operators
                leaves, _ = flatten(value, is_leaf=_is_op_or_wires)
                params = [leaf for leaf in leaves if not _is_op_or_wires(leaf)]
                if params:
                    dynamic_shape[argname] = [convert_item_to_mlir_type(param) for param in params]
            return dynamic_shape

    def parse_wire_lens(self) -> dict[str, int]:
        """Return a dictionary of wire arg names to lengths."""
        wire_lens = {}
        for wire_name, wire_arg in sorted(self.op.wire_args.items()):
            if wire_name not in self.op.hybrid_argnames:
                wire_lens[wire_name] = len(wire_arg)
        # match hybrid arg wires collection from _process_qubits (have to match generated IR op)
        for argname, value in sorted(self.op.hybrid_args.items()):
            # Descend through nested operators: their wires are qubit operands of this operator.
            leaves, _ = flatten(value, is_leaf=_is_wires)
            count = sum(len(leaf) for leaf in leaves if _is_wires(leaf))
            if count:
                wire_lens[argname] = count
        return wire_lens

    def parse_static_data(self) -> dict[str, Any]:
        """Return a dictionary of (compiler-)static data names to values."""
        if isinstance(self.op, qp.QubitUnitary):
            # `unitary_check` is a validation-only flag, the lowering drops it so we must too
            return {}
        return {
            static_argname: getattr(self.op, static_argname)
            for static_argname in sorted(self.op.compilable_argnames)
        }

    def parse_extra_data(self):
        """Return the UID computed from this Operator2 instance."""
        if self.op.static_args or self.op.hybrid_args:
            hybrid_lens = []
            hybrid_trees = []
            hybrid_args = []
            filtered_wire_lens = tuple(
                length
                for name, length in self.wire_lens.items()
                if name not in self.op.hybrid_argnames
            )
            for _, hybrid_argval in self.op.hybrid_args.items():
                leaves, tree = flatten(replace_wires_with_placeholder_wires(hybrid_argval))
                leaves = post_process_concretize_leaves(leaves)
                hybrid_lens.append(len(leaves))
                hybrid_trees.append(tree)
                hybrid_args.extend(leaves)
            uid = generate_uid(
                *tuple(self.op.dynamic_args.values()),  # dynamic args
                *(None,) * sum(filtered_wire_lens),
                # non hybrid wires, unused during uid generation, so just give empty values
                *hybrid_args,
                op_cls=type(self.op),
                wire_lens=filtered_wire_lens,
                hybrid_lens=tuple(hybrid_lens),
                hybrid_trees=tuple(hybrid_trees),
                static_args=self.op.static_args,
            )
            return self.op.static_args | self.op.hybrid_args, uid
        else:
            return {}, -1  # uid is unsigned, so use -1 for invalid uid

    def parse_is_custom_op(self) -> bool:
        """
        Return whether the Operator2 instance is considered a custom op in MLIR.

        Defers to the same :func:`_is_custom_op` the lowering uses, so the two cannot drift. The
        lowering dispatches a special-lowered operator before reaching that check, so the extra
        ``_SPECIAL_LOWERINGS`` exclusion is applied here instead.
        """
        return _is_custom_op(
            type(self.op), tuple(self.op.dynamic_args.values())
        ) and not issubclass(type(self.op), tuple(_SPECIAL_LOWERINGS.keys()))

    def get_operator_name(self) -> str:
        """Return the name of the operator."""
        return self.operator_name

    def _build_id(self, adjoint: bool, num_controls: int) -> str:
        """Return the GraphOpId of the base operator under the given modifiers."""
        uid = None
        if self.extra_data:
            assert self.uid >= 0, f"Failed to compute UID for operator {self.op}"
            uid = self.uid
        return build_graph_op_id(
            self.get_operator_name(),
            self.dynamic_shape,
            self.wire_lens,
            self.static_data,
            adjoint=adjoint,
            num_controls=num_controls,
            uid=uid,
        )

    def getGraphOpId(self, adjoint: bool = False, num_controls: int = 0) -> str:
        """
        Return the GraphOpId as a string.

        The arguments are the modifiers the *caller* applies on top of the operator, e.g. a rule
        being distributed over adjoint. They compose with the modifiers the operator carries
        itself, so an already-symbolic operator spells canonically instead of being wrapped twice.

        NOTE: do not modify this method without also modifying the corresponding DecomposableGate
        interface in MLIR.
        """
        return self._build_id(adjoint != self.adjoint, num_controls + self.num_controls)

    def getBaseGraphOpId(self) -> str:
        """Return the GraphOpId of the base operator alone, leaving off the modifiers wrapping it.

        The base is what owns the decomposition rules, so this is the id to look them up under.
        """
        return self._build_id(False, 0)
