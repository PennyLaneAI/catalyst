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

import warnings
from collections import deque
from functools import partial
from typing import Any

import jax.numpy as jnp
import pennylane as qp
from pennylane.pytrees import flatten

from catalyst.decomposition.type_utils import (
    convert_types_to_mlir_strings,
    format_dynamic_params_for_id,
    get_dummy_values_for_arg,
    post_process_concretize_leaves,
    replace_abstract_wires_with_concrete_wires,
)
from catalyst.from_plxpr.uid import generate_uid
from catalyst.jax_extras.lowering import get_mlir_attribute_from_pyval


class GraphOpID:
    """
    A parser object to compute the graph operator id for an abstract operator2 instance `op`.

    The format of the computed graph op ID string is as follows:
        op_name{dynamic_data_dictionary}{wire_lens_dictionary}{static_data_dictionary}[UID (optional)]

    For example, an Operator2 instance with class name `HybridOpArg`, taking in one float param
    argument named `angle`, one wire argument named `cwires`, one static data argument
    `label="hello"`, and UID 10 would be parsed to the following graph op ID:
        HybridOpArg{angle:[f64]}{cwires:1}{label:hello}[10]

    The defining trait of a graph op ID is that it has unique correspondence to decomposition rules.
    In other words, different graph op IDs have different sets of decomposition rules.

    For example,
        PauliRot{angle:[f64]}{wires:1}{pauli_word:X}
    and
        PauliRot{angle:[f64]}{wires:2}{pauli_word:XX}
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

        self.operator_name = op.name
        self.dynamic_shape = self.parse_dynamic_shape()
        self.wire_lens = self.parse_wire_lens()
        self.static_data = self.parse_static_data()
        self.extra_data, self.uid = self.parse_extra_data()

    def parse_dynamic_shape(self) -> dict:
        """Return a dictionary of dynamic arg names to list of dtypes."""
        # enters as {name: dtype}, we want the format {name: list[dtype]}
        return {argname: [argtype] for argname, argtype in sorted(self.op.dynamic_args.items())}

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
                leaves, tree = flatten(replace_abstract_wires_with_concrete_wires(hybrid_argval))
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

    def get_dynamic_shape(self) -> dict:
        """Return a dictionary of names to dynamic shapes."""
        return self.dynamic_shape

    def get_dynamic_shape_id_format(self) -> str:
        """Return the dynamic shape formatted for GraphOpId."""
        return format_dynamic_params_for_id(convert_types_to_mlir_strings(self.dynamic_shape))

    def get_wire_lens_id_format(self) -> str:
        """Return the wire lengths formatted for GraphOpId."""
        return "{" + ",".join(f"{name}:{shape}" for name, shape in self.wire_lens.items()) + "}"

    def get_static_data_id_format(self) -> str:
        """Return the static data formatted for GraphOpId."""
        return "{" + ",".join(f"{k}:{v}" for k, v in self.static_data.items()) + "}"

    def getID(self) -> str:
        """
        Return the GraphOpId as a string.

        NOTE: do not modify this method without also modifying the corresponding DecomposableGate
        interface in MLIR.
        """
        ID_string = (
            self.get_operator_name()
            + self.get_dynamic_shape_id_format()
            + self.get_wire_lens_id_format()
            + self.get_static_data_id_format()
        )
        if self.extra_data:
            assert self.uid >= 0, f"Failed to compute UID for operator {self.op}"
            ID_string += "[" + str(self.uid) + "]"
        return ID_string
