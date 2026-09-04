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

import jax.numpy as jnp
import pennylane as qp
from pennylane.pytrees import flatten

from catalyst.decomposition.type_utils import (
    convert_item_to_mlir_type,
    format_dynamic_params_for_id,
    format_static_data_dict_for_id,
    post_process_concretize_leaves,
    replace_wires_with_placeholder_wires,
)
from catalyst.from_plxpr.uid import generate_uid

_SPECIAL_LOWERINGS = {}


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

    Note that this function should not be updated without updating the corresponding method on the
    DecomposableGate interface in mlir/lib/quantum/IR/QuantumInterfaces.cpp.
    """

    def __init__(self, op: qp.core.Operator2):
        """Create a new GraphOpId."""
        assert isinstance(
            op, qp.core.Operator2
        ), f"Graph-based decomposition expects an Operator2 instance, got {op} of type {type(op)}"
        self.op = op
        self.is_custom_op = self.parse_is_custom_op()

        self.operator_name = op.name
        self.dynamic_shape = self.parse_dynamic_shape()
        self.wire_lens = self.parse_wire_lens()
        self.static_data = self.parse_static_data()
        self.extra_data, self.uid = self.parse_extra_data()

    def parse_dynamic_shape(self) -> dict:
        """Return a dictionary of dynamic arg names to list of dtypes."""
        # enters as {name: dtype}, we want the format {name: list[dtype]}
        if self.is_custom_op:
            return {str(i): ["f64"] for i in range(len(self.op.dynamic_args))}
        elif issubclass(type(self.op), tuple(_SPECIAL_LOWERINGS.keys())):  # special cases
            return {
                argname: [convert_item_to_mlir_type(argtype, is_special_lowering=True)]
                for argname, argtype in sorted(self.op.dynamic_args.items())
            }
        else:
            return {
                argname: [convert_item_to_mlir_type(argtype)]
                for argname, argtype in sorted(self.op.dynamic_args.items())
            }

    def parse_wire_lens(self) -> dict[str, int]:
        """Return a dictionary of wire arg names to lengths."""
        wire_lens = {}
        for wire_name, wire_arg in sorted(self.op.wire_args.items()):
            if wire_name not in self.op.hybrid_argnames:
                wire_lens[wire_name] = len(wire_arg)
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

    def parse_is_custom_op(self) -> bool:
        """
        Return whether the Operator2 instance is considered a custom op in MLIR.

        The source of truth for the criteria is in qref_operator2_primitives.py,
        in the _is_custom_op() helper function. However, that function cannot be directly used here
        as that is a lowering time util, and only works with JAX-MLIR types.
        """
        if self.op.static_argnames or self.op.hybrid_argnames or self.op.compilable_argnames:
            return False
        if self.op.wire_argnames != ("wires",):
            return False
        if list(self.op._sig.parameters.keys())[-1] != "wires":
            return False
        return all(
            arg.shape == () and arg.dtype.type == jnp.float64
            for arg in self.op.dynamic_args.values()
        ) and not issubclass(type(self.op), tuple(_SPECIAL_LOWERINGS.keys()))

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
        return build_graph_op_id(
            self.get_operator_name(),
            self.dynamic_shape,
            self.wire_lens,
            self.static_data,
            adjoint=adjoint,
            num_controls=num_controls,
            uid=uid,
        )
