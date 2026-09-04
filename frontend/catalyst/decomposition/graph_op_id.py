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

from contextlib import contextmanager
from typing import Any

import pennylane as qp
from jax._src.lib.mlir import ir
from pennylane.pytrees import flatten

from catalyst.decomposition.type_utils import (
    convert_item_to_mlir_type,
    post_process_concretize_leaves,
    replace_wires_with_placeholder_wires,
)
from catalyst.from_plxpr.uid import generate_uid
from catalyst.jax_extras.lowering import get_mlir_attribute_from_pyval

_SPECIAL_LOWERINGS = {}


@contextmanager
def _mlir_context():
    """Provide an MLIR context and location for attribute construction."""
    if current := ir.Context.current:
        with ir.Location.unknown(context=current):
            yield current
    else:
        with ir.Context() as context, ir.Location.unknown(context=context):
            yield context


def build_graph_op_key(
    base_name: str,
    dynamic_types: dict[str, list[str]],
    wire_lengths: dict[str, int],
    static_data: dict[str, Any],
    *,
    adjoint: bool = False,
    num_controls: int = 0,
    uid: int | None = None,
) -> str:
    """Build the canonical structured GraphOpID string.

    Parameter and wire groups are identified by their position, so both mappings must be in
    frontend signature order.
    """
    if num_controls < 0:
        raise ValueError("GraphOpID control count cannot be negative")

    with _mlir_context():
        fields = {"op": ir.StringAttr.get(base_name)}
        traits = {}
        if adjoint:
            traits["adj"] = ir.BoolAttr.get(True)
        if num_controls:
            traits["controls"] = get_mlir_attribute_from_pyval(num_controls)
        if dynamic_types:
            fields["params"] = ir.ArrayAttr.get(
                [
                    ir.ArrayAttr.get(
                        [ir.TypeAttr.get(ir.Type.parse(type_name)) for type_name in types]
                    )
                    for types in dynamic_types.values()
                ]
            )
        if static_data:
            fields["static"] = get_mlir_attribute_from_pyval(static_data)
        if traits:
            fields["traits"] = ir.DictAttr.get(traits)
        if uid is not None:
            fields["uid"] = get_mlir_attribute_from_pyval(uid)
        if wire_lengths:
            fields["wires"] = get_mlir_attribute_from_pyval(list(wire_lengths.values()))
        return str(ir.DictAttr.get(fields))


def _parse_graph_op_id(op_id: str) -> ir.DictAttr:
    """Parse and minimally validate a structured GraphOpID."""
    parsed = ir.Attribute.parse(op_id)
    if not isinstance(parsed, ir.DictAttr):
        raise ValueError(f"Malformed GraphOpID, expected a dictionary attribute: {op_id!r}")

    fields = {entry.name for entry in parsed}
    if "op" not in fields:
        raise ValueError(f"Malformed GraphOpID, missing field 'op': {op_id!r}")
    if unknown := fields.difference({"op", "params", "static", "traits", "uid", "wires"}):
        raise ValueError(f"Malformed GraphOpID, unknown fields {sorted(unknown)}: {op_id!r}")
    if "traits" in parsed:
        traits = ir.DictAttr(parsed["traits"])
        if unknown := {entry.name for entry in traits}.difference({"adj", "controls"}):
            raise ValueError(
                f"Malformed GraphOpID, unknown trait fields {sorted(unknown)}: {op_id!r}"
            )
    return parsed


def graph_op_id_modifiers(op_id: str) -> tuple[str, bool, int]:
    """Return ``(base_name, adjoint, num_controls)`` from a GraphOpID."""
    with _mlir_context():
        parsed = _parse_graph_op_id(op_id)
        base_name = ir.StringAttr(parsed["op"]).value
        traits = ir.DictAttr(parsed["traits"]) if "traits" in parsed else None
        adjoint = ir.BoolAttr(traits["adj"]).value if traits and "adj" in traits else False
        num_controls = (
            ir.IntegerAttr(traits["controls"]).value if traits and "controls" in traits else 0
        )
        return base_name, adjoint, num_controls


def with_graph_op_id_modifiers(
    op_id: str, *, adjoint: bool | None = None, num_controls: int | None = None
) -> str:
    """Return ``op_id`` with selected modifier fields replaced."""
    if num_controls is not None and num_controls < 0:
        raise ValueError("GraphOpID control count cannot be negative")

    with _mlir_context():
        parsed = _parse_graph_op_id(op_id)
        fields = {entry.name: entry.attr for entry in parsed}
        traits = (
            {entry.name: entry.attr for entry in ir.DictAttr(parsed["traits"])}
            if "traits" in parsed
            else {}
        )
        if adjoint is not None:
            if adjoint:
                traits["adj"] = ir.BoolAttr.get(True)
            else:
                traits.pop("adj", None)
        if num_controls is not None:
            if num_controls:
                traits["controls"] = get_mlir_attribute_from_pyval(num_controls)
            else:
                traits.pop("controls", None)
        if traits:
            fields["traits"] = ir.DictAttr.get(traits)
        else:
            fields.pop("traits", None)
        return str(ir.DictAttr.get(fields))


def is_custom_op(op_cls: type[qp.core.Operator2], dynamic_args) -> bool:
    """Return whether an Operator2 lowers to ``qref.custom``."""
    if op_cls.static_argnames or op_cls.hybrid_argnames or op_cls.compilable_argnames:
        return False
    if op_cls.wire_argnames != ("wires",):
        return False
    if list(op_cls._sig.parameters.keys())[-1] != "wires":
        return False
    return all(
        arg.shape == () and arg.dtype.kind in "ifu" for arg in dynamic_args
    ) and not issubclass(op_cls, tuple(_SPECIAL_LOWERINGS.keys()))


class GraphOpID:
    """
    A parser object to compute the graph operator id for an abstract operator2 instance `op`.

    The ID is the canonical MLIR text of a dictionary containing the operator's base name and
    non-default modifiers, dynamic types, wire lengths, static data, and UID.

    The types in the dynamic shape dictionary should be represented as a list of MLIR-style type
    annotations. Dynamic and wire dictionaries preserve frontend signature order; their names are
    not part of the ID.
    The UID is computed from the shapes, dtypes and pytree structures of the `hybrid_args` of
    the Operator2 instance.

    The defining trait of a graph op ID is that it has unique correspondence to decomposition rules.
    In other words, different graph op IDs have different sets of decomposition rules.

    For example,
        {op = "PauliRot", ..., wires = [1]}
    and
        {op = "PauliRot", ..., wires = [2]}
    will have different decomposition rules.

    Note that this function should not be updated without updating the corresponding method on the
    DecomposableGate interface in mlir/lib/quantum/IR/QuantumInterfaces.cpp.
    """

    def __init__(self, op: qp.core.Operator2):
        """Create a new GraphOpId."""
        assert isinstance(
            op, qp.core.Operator2
        ), f"Graph-based decomposition expects an Operator2 instance, got {op} of type {type(op)}"
        self.op = op
        self.is_custom_op = is_custom_op(type(op), op.dynamic_args.values())

        self.operator_name = op.name
        self.dynamic_shape = self.parse_dynamic_shape()
        self.wire_lens = self.parse_wire_lens()
        self.static_data = self.parse_static_data()
        self.extra_data, self.uid = self.parse_extra_data()

    def parse_dynamic_shape(self) -> dict:
        """Return dynamic argument type groups in frontend signature order."""
        # enters as {name: dtype}, we want the format {name: list[dtype]}
        if self.is_custom_op:
            return {str(i): ["f64"] for i in range(len(self.op.dynamic_args))}
        elif issubclass(type(self.op), tuple(_SPECIAL_LOWERINGS.keys())):  # special cases
            return {
                argname: [
                    convert_item_to_mlir_type(
                        self.op.dynamic_args[argname], is_special_lowering=True
                    )
                ]
                for argname in self.op.dynamic_argnames
            }
        else:
            return {
                argname: [convert_item_to_mlir_type(self.op.dynamic_args[argname])]
                for argname in self.op.dynamic_argnames
            }

    def parse_wire_lens(self) -> dict[str, int]:
        """Return wire argument lengths in frontend signature order."""
        wire_lens = {}
        for wire_name in self.op.wire_argnames:
            if wire_name not in self.op.hybrid_argnames:
                wire_lens[wire_name] = len(self.op.wire_args[wire_name])
        return wire_lens

    def parse_static_data(self) -> dict[str, Any]:
        """Return a dictionary of (compiler-)static data names to values."""
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
            for _, hybrid_argval in self.op.hybrid_args.items():
                leaves, tree = flatten(replace_wires_with_placeholder_wires(hybrid_argval))
                leaves = post_process_concretize_leaves(leaves)
                hybrid_lens.append(len(leaves))
                hybrid_trees.append(tree)
                hybrid_args.extend(leaves)
            uid = generate_uid(
                *tuple(self.op.dynamic_args.values()),  # dynamic args
                *(None,)
                * sum(
                    self.wire_lens.values()
                ),  # non hybrid wires, unused during uid generation, so just give empty values
                *hybrid_args,
                op_cls=type(self.op),
                wire_lens=tuple(self.wire_lens.values()),
                hybrid_lens=tuple(hybrid_lens),
                hybrid_trees=tuple(hybrid_trees),
                adjoint=False,
                n_ctrls=0,
                static_args=self.op.static_args,
            )
            return self.op.static_args | self.op.hybrid_args, uid
        else:
            return {}, -1  # uid is unsigned, so use -1 for invalid uid

    def get_operator_name(self) -> str:
        """Return the name of the operator."""
        return self.operator_name

    def getGraphOpId(self, adjoint: bool = False, num_controls: int = 0) -> str:
        """
        Return the GraphOpId as a string.

        NOTE: do not modify this method without also modifying the corresponding DecomposableGate
        interface in MLIR.
        """
        uid = None
        if self.extra_data:
            assert self.uid >= 0, f"Failed to compute UID for operator {self.op}"
            uid = self.uid
        return build_graph_op_key(
            self.get_operator_name(),
            self.dynamic_shape,
            self.wire_lens,
            self.static_data,
            adjoint=adjoint,
            num_controls=num_controls,
            uid=uid,
        )
