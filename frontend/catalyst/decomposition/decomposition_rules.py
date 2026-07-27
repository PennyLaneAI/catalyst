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
This module provides infrastructure for lowering decomposition rules via python.
"""

# pylint: disable=protected-access,bare-except

from functools import partial

import jax.numpy as jnp
import pennylane as qp
from jax._src.lib.mlir import ir
from jaxlib.mlir.dialects.builtin import ModuleOp

from catalyst.decomposition.type_utils import (
    _MLIR_DTYPES_TO_PY_DTYPES,
    _PY_DTYPES_TO_MLIR_DTYPES,
    get_dummy_values_for_container,
    mlir_stringify_type,
)
from catalyst.jax_extras.lowering import get_mlir_attribute_from_pyval


class GraphOpID:
    """
    Return the graph operator id for the operator2 instance `op`.

    The FuncOp decomposition rules in the returned string satisfy the following requirements:
        - Are named `{rule name}_{op graph ID}`.
        - Are MLIR representations of the PennyLane decomposition rules associated with the
          specified operator.
        - Are instantiated with the static data provided, and all other data remains dynamic.
        - Are self-contained, and do not contain any device initialization, setup/teardown etc.
        - Are compatible with the `decompose-lowering` and `graph-decomposition` passes, meaning
          the following:
            - Their `target_gate` attribute is set to the provided graph operator ID
            - They have a resources attribute containing an operations attribute which maps graph
              operator IDs to counts of their occurrences in the rule.
            - Their arguments are mappable to the operator they decompose via `decompose-lowering`.

    Note that this function should not be updated without updating the corresponding method on the
    DecomposableGate interface in mlir/lib/quantum/IR/QuantumInterfaces.cpp.
    """

    def __init__(self, op: qp.core.Operator2, uid=None):
        """Create a new GraphOpId."""
        assert isinstance(
            op, qp.core.Operator2
        ), "Graph-based decomposition expects an Operator2 instance"
        self.op = op

        self.operator_name = op.name
        self.dynamic_shape = self.parse_dynamic_shape()
        self.wire_lens = self.parse_wire_lens()
        self.static_data = self.parse_static_data()
        self.extra_data = uid

    def parse_dynamic_shape(self) -> list:
        """Return the dynamic shape as a list of dtypes."""
        return {
            argname: mlir_stringify_type(argtype)
            for argname, argtype in sorted(self.op.dynamic_args.items())
        }

    def parse_wire_lens(self) -> list[int]:
        """Return the length of each of the wire args."""
        return {
            wire_name: len(wire_arg) for wire_name, wire_arg in sorted(self.op.wire_args.items())
        }

    def parse_static_data(self) -> dict:
        """Return a dictionary of names to static data values."""
        return {
            static_argname: getattr(self.op, static_argname)
            for static_argname in sorted(self.op.compilable_argnames)
        }

    def get_operator_name(self) -> str:
        """Return the name of the operator."""
        return self.operator_name

    def get_dynamic_shape_id_format(self) -> str:
        """Return the dynamic shape formatted for GraphOpId."""
        return f"{{{','.join(f"{name}:{shape}" for name, shape in self.dynamic_shape.items())}}}"

    def get_wire_lens_id_format(self) -> str:
        """Return the wire lengths formatted for GraphOpId."""
        return f"{{{','.join(f"{name}:{shape}" for name, shape in self.wire_lens.items())}}}"

    def get_static_data_id_format(self) -> str:
        """Return the static data formatted for GraphOpId."""
        return f"{{{','.join(f'{k}:{v}' for k, v in self.static_data.items())}}}"

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
            ID_string += "[" + str(self.extra_data) + "]"
        return ID_string


def collect_resources_for_op(op_name, kwargs):
    """
    Return resource data for all decomposition rules associated to op_name.

    This includes a dictionary
    """
    decomp_rules = list(qp.decomposition.list_decomps(op_name))

    # map rules to resource resources, in a more generic format
    name_to_resource_ids = {}
    name_to_resources = {}
    for rule in decomp_rules:
        # The `compute_resources` function's signature is the same as the Operator2 signature
        # for the original op of the rule
        resources = rule.compute_resources(**kwargs)
        name_to_resources[rule.name] = resources.gate_counts
        name_to_resource_ids[rule.name] = {
            GraphOpID(op).getID(): count for op, count in resources.gate_counts.items()
        }

    return name_to_resources, name_to_resource_ids, decomp_rules


def compile_decomposition_rules(op_name, op_id, dynamic_shape, wire_lens, static_data) -> ModuleOp:
    """
    Return a ModuleOp containing the decomposition rules for an operator instance.

    The decomposition rules will be decorated with appropriate resource and target_gate attributes.
    """
    kwargs = {}
    device = qp.device("null.qubit", wires=sum(wire_lens.values()))
    for wire_name, wire_len in wire_lens.items():
        kwargs[wire_name] = jnp.array(range(wire_len), dtype=int)
    for arg_name, arg_shape in dynamic_shape.items():
        kwargs[arg_name] = get_dummy_values_for_container(arg_shape)
    kwargs.update(static_data)

    _, name_to_resource_ids, decomp_rules = collect_resources_for_op(op_name, kwargs)

    # The static_data was only needed to query the correct decomp rule
    # Once we have the correct rules, don't send them into qjit
    def rule_to_subroutine(rule):
        def decomp_rule(*args, **kwargs):
            rule._impl(*args, **kwargs)

        decomp_rule_no_static_args = partial(decomp_rule, **static_data)

        # keep the frontend name for readability, append target op_id for symbol uniqueness
        decomp_rule_no_static_args.__name__ = rule._impl.__name__ + "_" + op_id

        return qp.capture.subroutine(decomp_rule_no_static_args)

    subroutines = [rule_to_subroutine(rule) for rule in decomp_rules]

    for static_argname in static_data.keys():
        del kwargs[static_argname]

    @qp.qjit(
        target="mlir",
        capture=True,
    )
    @qp.qnode(device=device)
    def circuit():
        for subroutine in subroutines:
            subroutine(**kwargs)

    module = circuit.mlir_module

    def update_funcop_attributes(op):
        """Update the decomposition rule attributes if op is a decomposition rule.

        For use with module.walk

        This function updates the following attributes:
            - Adds the `target_gate` attribute.
            - Adds the `resources` attribute.
        """
        if op.name == "func.func":
            rule_name = ir.StringAttr(op.attributes["sym_name"]).value.removesuffix("_" + op_id)
            if rule_name in name_to_resource_ids:
                op.attributes["resources"] = get_mlir_attribute_from_pyval(
                    {"operations": name_to_resource_ids[rule_name]}
                )
                op.attributes["target_gate"] = ir.StringAttr.get(op_id)

        return ir.WalkResult.ADVANCE

    with module.context:
        module.operation.walk(update_funcop_attributes)

    return module


def compile_decomposition_rules_wrapper(
    op_name, op_id, dynamic_shape, wire_lens, static_data
) -> str:
    """Return a string MLIR module containing the decomposition rules for an operator instance."""
    return str(compile_decomposition_rules(op_name, op_id, dynamic_shape, wire_lens, static_data))
