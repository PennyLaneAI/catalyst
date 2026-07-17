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

from catalyst.jax_extras.lowering import get_mlir_attribute_from_pyval

_MLIR_DTYPES = {
    "i1": jnp.bool_,
    "i8": jnp.int8,
    "i16": jnp.int16,
    "i32": jnp.int32,
    "i64": jnp.int64,
    "f16": jnp.float16,
    "f32": jnp.float32,
    "f64": jnp.float64,
    "complex<f64>": jnp.complex64,
    "complex<f128>": jnp.complex128,
}


def get_dummy_values_for_container(container):
    """Given a container of python types, replace the types with corresponding dummy values."""
    dummy_args = []
    for dtype in container:
        if isinstance(dtype, str):
            if dtype in _MLIR_DTYPES:
                count = 1
                dtype = _MLIR_DTYPES[dtype]
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


def get_graph_op_id(op: qp.decomposition.CompressedResourceOp | qp.Operator2):
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
    if isinstance(op, qp.decomposition.CompressedResourceOp):
        # NOTE: handling this the old-fashioned way, remove once Operator2 migration is complete
        op_type = op.op_type
    else:
        op_type = op

    if issubclass(op_type, qp.core.operator.Operator):
        name = op_type.__name__
        num_params = op_type.num_params
        num_wires = str(op_type.num_wires) if op_type.num_wires else "0"
        return name + "[" + ",".join(["f64"] * num_params) + "][" + num_wires + "]{}"
    elif isinstance(op_type, qp.core.operator.Operator2):
        # TODO: use real getters here
        name = op_type.__name__
        dynamic_shape = op_type.getDynamicShape()
        wire_lens = op_type.getWireLens()
        static_data = op_type.getStaticData()
        extra_data = op_type.uid
        return (
            name
            + ("[" + dynamic_shape + "]")
            + ("[" + wire_lens + "]")
            + ("{" + static_data + "}")
            + ("[" + extra_data + "]")
        )
    else:
        raise ValueError(
            "Only AbstractOperator and CompressedResourceOp types are supported for generating a "
            f"graph ID, got {op} of type {type(op)}"
        )


def python_decomposition(op_name, op_id, dynamic_shape, wire_lens, static_data) -> ModuleOp:
    """Python decomposition rule lowering."""
    # TODO update docstring
    device = qp.device("null.qubit", wires=sum(wire_lens))
    wires = tuple(jnp.array(range(length), dtype=int) for length in wire_lens)

    decomp_rules = list(qp.decomposition.list_decomps(op_name))

    # map rules to resource resources, in a more generic format

    name_to_resources = {}
    for rule in decomp_rules:
        # TODO: not all PL ops have been migrated to the operator 2 format expected by mlir graph
        # decomp This means some rules will fail the python callback compilation. When migration is
        # complete, remove the try-except.
        try:
            name_to_resources[rule.name] = {
                get_graph_op_id(op): count
                for op, count in rule.compute_resources(**static_data).gate_counts.items()
            }
        except:  # pylint: disable=bare-except
            decomp_rules.remove(rule)

    def rule_to_subroutine(rule):
        def decomp_rule(*params, wires):
            rule._impl(*params, *wires, **static_data)

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
            subroutine(*get_dummy_values_for_container(dynamic_shape), wires=wires)

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
            if rule_name in name_to_resources:
                op.attributes["resources"] = get_mlir_attribute_from_pyval(
                    {"operations": name_to_resources[rule_name]}
                )
                op.attributes["target_gate"] = ir.StringAttr.get(op_id)

        return ir.WalkResult.ADVANCE

    with module.context:
        module.operation.walk(update_funcop_attributes)

    return module


def python_decomposition_wrapper(op_name, op_id, dynamic_shape, wire_lens, static_data) -> str:
    """Generic decomposition wrapper."""
    return str(python_decomposition(op_name, op_id, dynamic_shape, wire_lens, static_data))
