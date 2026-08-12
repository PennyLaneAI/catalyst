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

import warnings
from collections import deque
from functools import partial

import jax.numpy as jnp
import pennylane as qp
from jax._src.lib.mlir import ir
from jaxlib.mlir.dialects.builtin import ModuleOp
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
    A parser object to compute the graph operator id for the operator2 instance `op`.

    The format of the computed graph op ID string is as follows:
        op_name{param_shaped_type_dictionary}{wire_lens_dictionary}{static_data_dictionary}[UID]

    For example, an Operator2 instance with class name `HybridOpArg`, taking in one float param
    argument named `angle`, one wire argument named `cwires`, one static data argument
    `label="hello"`, and UID 10 would be parsed to the following graph op ID:
        HybridOpArg{angle:[f64]}{cwires:1}{label:hello}[10]

    The defining trait of a graph op ID is that it has unique correspondence to decomposition rules.
    In other words, different graph op IDs have different sets of decomposition rules.

    For example,
        PauliRot{angle:[f64]}{wires:1}{pauli_word:X}[]
    and
        PauliRot{angle:[f64]}{wires:2}{pauli_word:XX}[]
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
        """Return the dynamic shape as a dictionary of dtypes from the dynamic arg names."""
        return {argname: argtype for argname, argtype in sorted(self.op.dynamic_args.items())}

    def parse_wire_lens(self) -> dict:
        """Return the length of each of the wire args as a dictionary from the wire arg names."""
        wire_lens = {}
        for wire_name, wire_arg in sorted(self.op.wire_args.items()):
            if wire_name not in self.op.hybrid_argnames:
                wire_lens[wire_name] = len(wire_arg)
        return wire_lens

    def parse_static_data(self) -> dict:
        """Return a dictionary of names to static data values."""
        return {
            static_argname: getattr(self.op, static_argname)
            for static_argname in sorted(self.op.compilable_argnames)
        }

    def parse_extra_data(self) -> dict:
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
            return {}, -1  # uid is always unsigned, so use -1 for invalid uid

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
            assert self.uid >= 0
            ID_string += "[" + str(self.uid) + "]"
        return ID_string


def get_rule_strings_from_module(module: ir.Module) -> list[str]:
    raw_funcOps = []

    def find_condition(op):
        if op.name == "func.func":
            if "target_gate" in op.attributes:
                raw_funcOps.append(op)
                return ir.WalkResult.SKIP
        return ir.WalkResult.ADVANCE

    module.operation.walk(find_condition)

    # If we simply rename the rule func op in the original module (from the qjit that compiles the
    # rule), the call op to the rule subroutine from the main qjit function will complain that its
    # callee doesn't exist.
    # We have to do a clone, and rename the clone.
    # And to clone safely, we must set the insertion point to a separate sandbox module
    funcOps = []
    ctx = module.context
    with ctx, ir.Location.unknown(ctx):
        sandbox_module = ir.Module.create()
        with ir.InsertionPoint(sandbox_module.body):
            for op in raw_funcOps:
                clone = op.clone()

                old_attr = clone.attributes["sym_name"]
                clean_name = old_attr.value.strip('"')

                if not clean_name.startswith("__builtin_"):
                    clone.attributes["sym_name"] = ir.StringAttr.get(
                        "__builtin_" + clean_name, context=ctx
                    )

                funcOps.append(str(clone))

    return funcOps


def get_rules_from_module(module: ir.Module) -> str:
    """
    Parse and modify decomposition rules from a ModuleOp.

    Args:
        module: an MLIR module object containing a FuncOp named `rule_wrapper` to be extracted

    Returns:
        str: The string representation of any decomposition rules from `module`, pre-pending the
             `__builtin_` prefix to their names.
    """
    funcOps = get_rule_strings_from_module(module)
    return "\n".join(str(funcOp) for funcOp in funcOps) if funcOps else ""


def inject_new_rules_into_module(module: ir.Module, decomp_rules: list[str]):
    with ir.InsertionPoint(module.body):
        for decomp_rule in decomp_rules:
            if not decomp_rule:
                continue

            decomp_rule_op = ir.Operation.parse(decomp_rule)
            rule_already_exists = False

            def find_condition(op):
                nonlocal rule_already_exists
                if op.name == "func.func":
                    if "target_gate" in op.attributes:
                        target_gate = op.attributes["target_gate"]
                        resources = op.attributes["resources"]

                        current_rule_target_gate = decomp_rule_op.attributes["target_gate"]
                        current_rule_resources = decomp_rule_op.attributes["resources"]
                        if (
                            target_gate == current_rule_target_gate
                            and resources == current_rule_resources
                        ):
                            rule_already_exists = True
                            return ir.WalkResult.INTERRUPT
                        return ir.WalkResult.SKIP
                return ir.WalkResult.ADVANCE

            module.operation.walk(find_condition)
            if not rule_already_exists:
                decomp_rule_op.clone()


def collect_resources_for_op(op_name, kwargs, is_custom_op=False):
    """Return resource data for all decomposition rules associated to op_name."""
    decomp_rules = list(qp.decomposition.list_decomps(op_name))
    args = ()

    # map rules to resource resources, in a more generic format
    name_to_resource_ids = {}
    name_to_resources = {}
    for rule in decomp_rules:
        try:
            # The `compute_resources` function's signature is the same as the Operator2 signature
            # for the original op of the rule
            if is_custom_op:
                args = tuple(val for key, val in kwargs.items() if key != "wires")
                kwargs = {"wires": kwargs["wires"]}
            resources = rule.compute_resources(*args, **kwargs)
            name_to_resources[rule.name] = resources.gate_counts
            name_to_resource_ids[rule.name] = {
                GraphOpID(op).getID(): count for op, count in resources.gate_counts.items()
            }
        except Exception as e:
            warnings.warn(f"Failed to get resources for the {rule.name} decomposition rule: {e}")

    return name_to_resources, name_to_resource_ids, decomp_rules


def prepare_dynamic_op_kwargs(dynamic_shape, wire_lens) -> dict:
    kwargs = {}
    for wire_name, wire_len in wire_lens.items():
        kwargs[wire_name] = jnp.array(range(wire_len), dtype=int)
    for arg_name, arg_shape in dynamic_shape.items():
        kwargs[arg_name] = get_dummy_values_for_arg(arg_shape)
    return kwargs


def compile_decomposition_rules(
    op_name,
    op_id,
    dynamic_shape,
    wire_lens,
    static_data,
    extra_data=None,
    is_custom_op=False,
) -> ModuleOp:
    """
    Return a ModuleOp containing the decomposition rules for an operator instance.

    The decomposition rules will be decorated with appropriate resource and target_gate attributes.
    """
    kwargs = prepare_dynamic_op_kwargs(dynamic_shape, wire_lens)
    extra_data = extra_data or {}
    device = qp.device("null.qubit", wires=sum(wire_lens.values()))

    _, name_to_resource_ids, decomp_rules = collect_resources_for_op(
        op_name, kwargs | static_data | extra_data, is_custom_op
    )

    # The static_data was only needed to instantiate the correct decomp rule
    # Once we have the correct rules, don't send them into qjit
    def rule_to_subroutine(rule):
        def decomp_rule(*_args, **_kwargs):
            rule._impl(*_args, **_kwargs)

        decomp_rule_no_static_args = partial(decomp_rule, **static_data)
        if extra_data:
            decomp_rule_no_static_args = partial(decomp_rule_no_static_args, **extra_data)

        # keep the frontend name for readability, append target op_id for symbol uniqueness
        decomp_rule_no_static_args.__name__ = rule._impl.__name__ + "_" + op_id

        return qp.capture.subroutine(decomp_rule_no_static_args)

    subroutines = [rule_to_subroutine(rule) for rule in decomp_rules]

    @qp.qjit(target="mlir", capture=True, skip_decomp_rules=True)
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
    op_name,
    op_id,
    dynamic_shape,
    wire_lens,
    static_data,
    extra_data=None,
    is_custom_op=False,
) -> str:
    """Return a string MLIR module containing the decomposition rules for an operator instance."""
    return str(
        compile_decomposition_rules(
            op_name,
            op_id,
            dynamic_shape,
            wire_lens,
            static_data,
            extra_data=extra_data,
            is_custom_op=is_custom_op,
        )
    )


def fetch_all_reachable_decomposition_rules_from_op(
    op_name, op_id, dynamic_shape, wire_lens, static_data, extra_data=None
):
    extra_data = extra_data or {}
    queue = deque()
    start = (op_name, dynamic_shape, wire_lens, static_data, extra_data)
    queue.append(start)
    visited = [start]

    rules = get_rule_strings_from_module(
        compile_decomposition_rules(
            op_name, op_id, dynamic_shape, wire_lens, static_data, extra_data=extra_data
        )
    )

    while len(queue) != 0:
        this_name, this_dynamic_shape, this_wire_lens, this_static_data, this_extra_data = (
            queue.popleft()
        )
        this_extra_data = this_extra_data or {}
        this_kwargs = prepare_dynamic_op_kwargs(this_dynamic_shape, this_wire_lens)
        resources, _, _ = collect_resources_for_op(
            this_name, this_kwargs | this_static_data | this_extra_data
        )
        for _rule_name, resource in resources.items():
            try:
                for op, _count in resource.items():
                    graph_op_id = GraphOpID(op)
                    probe = (
                        graph_op_id.get_operator_name(),
                        convert_types_to_mlir_strings(graph_op_id.get_dynamic_shape()),
                        graph_op_id.wire_lens,
                        graph_op_id.static_data,
                        graph_op_id.extra_data,
                    )

                    if not probe in visited:
                        visited.append(probe)
                        queue.append(probe)
                        module = compile_decomposition_rules(
                            probe[0],
                            graph_op_id.getID(),
                            probe[1],
                            probe[2],
                            probe[3],
                            probe[4],
                        )
                        rules.extend(get_rule_strings_from_module(module))
            except Exception as e:
                warnings.warn(
                    f"Failed to lower the {_rule_name} decomposition rule for {this_name}: {e}"
                )
                continue
    return rules
