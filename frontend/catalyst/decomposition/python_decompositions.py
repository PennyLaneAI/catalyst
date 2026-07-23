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
This module provides infrastructure for compile-time lowering of decomposition rules via python.
"""

# pylint: disable=protected-access,bare-except

import jax.numpy as jnp
import pennylane as qp
from jax._src.lib.mlir import ir
from jaxlib.mlir.dialects.builtin import ModuleOp

from catalyst.decomposition.type_utils import (
    _MLIR_DTYPES_TO_PY_DTYPES,
    _PY_DTYPES_TO_MLIR_DTYPES,
    mlir_stringify_type,
)
from catalyst.jax_extras.lowering import get_mlir_attribute_from_pyval


def get_dummy_values_for_container(container):
    """Given a container of python types, replace the types with corresponding dummy values."""
    dummy_args = []
    for dtype in container:
        if isinstance(dtype, str):
            if dtype in _MLIR_DTYPES_TO_PY_DTYPES:
                count = 1
                dtype = _MLIR_DTYPES_TO_PY_DTYPES[dtype]
            elif dtype.startswith("tensor"):
                # tensor<{number}x{type}>
                dtype = dtype.removeprefix("tensor<")
                dtype = dtype.remove_suffice(">")
                count, dtype = dtype.split("x")
            else:
                raise ValueError(f"Unknown dtype {dtype}.")
        else:
            count = 1
            dtype = jnp.dtype(dtype)

        dummy_args.append(jnp.zeros((count,), dtype=dtype))

    return tuple(dummy_args)


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
        assert isinstance(
            op, qp.core.Operator2
        ), "Graph-based decomposition expects an Operator2 instance"
        self.op = op

        self.operator_name = op.name
        self.dynamic_shape = self.parse_dynamic_shape()
        self.wire_lens = self.parse_wire_lens()
        self.static_data = self.parse_static_data()
        self.extra_data = uid

    def parse_dynamic_shape(self):
        return list(self.op.dynamic_args.values())

    def parse_wire_lens(self):
        return list(map(len, self.op.wire_args.values()))

    def parse_static_data(self):
        return {
            static_argname: getattr(self.op, static_argname)
            for static_argname in self.op.compilable_argnames
        }

    def get_operator_name(self):
        return self.operator_name

    def get_dynamic_shape_id_format(self):
        return f"[{','.join(map(mlir_stringify_type, self.dynamic_shape))}]"

    def get_wire_lens_id_format(self):
        return f"[{','.join(map(str, self.wire_lens))}]"

    def get_static_data_id_format(self):
        return f"{{{','.join(f'{k}:{v}' for k, v in self.static_data.items())}}}"

    def getID(self):
        ID_string = (
            self.get_operator_name()
            + self.get_dynamic_shape_id_format()
            + self.get_wire_lens_id_format()
            + self.get_static_data_id_format()
        )
        if self.extra_data:
            ID_string += "[" + str(self.extra_data) + "]"
        return ID_string


def collect_resources_for_op(op_name, dummy_dynamic_args, dummy_wires, static_data):
    decomp_rules = list(qp.decomposition.list_decomps(op_name))

    # map rules to resource resources, in a more generic format
    name_to_resource_ids = {}
    name_to_resources = {}
    for rule in decomp_rules:
        # The `compute_resources` function's signature is the same as the Operator2 signature
        # for the original op of the rule
        resources = rule.compute_resources(*dummy_dynamic_args, *dummy_wires, **static_data)
        name_to_resources[rule.name] = resources.gate_counts
        name_to_resource_ids[rule.name] = {
            GraphOpID(op).getID(): count for op, count in resources.gate_counts.items()
        }

    return name_to_resources, name_to_resource_ids, decomp_rules


def python_decomposition(op_name, op_id, dynamic_shape, wire_lens, static_data) -> ModuleOp:
    """Python decomposition rule lowering."""
    # TODO update docstring
    device = qp.device("null.qubit", wires=sum(wire_lens))
    dummy_wires = tuple(jnp.array(range(length), dtype=int) for length in wire_lens)
    dummy_dynamic_args = get_dummy_values_for_container(dynamic_shape)

    _, name_to_resource_ids, decomp_rules = collect_resources_for_op(
        op_name, dummy_dynamic_args, dummy_wires, static_data
    )

    def rule_to_subroutine(rule):
        def decomp_rule(*args, **kwargs):
            rule._impl(*args, **kwargs)

        # keep the frontend name for readability, append target op_id for symbol uniqueness
        decomp_rule.__name__ = rule._impl.__name__ + "_" + op_id

        return qp.capture.subroutine(decomp_rule)

    subroutines = [rule_to_subroutine(rule) for rule in decomp_rules]

    @qp.qjit(
        target="mlir",
        capture=True,
    )
    @qp.qnode(device=device)
    def circuit():
        for subroutine in subroutines:
            subroutine(*dummy_dynamic_args, wires=dummy_wires)

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


def python_decomposition_wrapper(op_name, op_id, dynamic_shape, wire_lens, static_data) -> str:
    """Generic decomposition wrapper."""
    return str(python_decomposition(op_name, op_id, dynamic_shape, wire_lens, static_data))
