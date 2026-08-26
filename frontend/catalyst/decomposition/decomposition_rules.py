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

from catalyst.decomposition.graph_op_id import GraphOpID
from catalyst.decomposition.type_utils import (
    convert_types_to_mlir_strings,
    get_dummy_values_for_arg,
)
from catalyst.jax_extras.lowering import get_mlir_attribute_from_pyval

# Ops that make a decomposition body non-invertible:
_NON_INVERTIBLE_MARKERS = ("qref.measure", "quantum.measure")


# Canonical nesting order for op-level modifiers, listed OUTERMOST first. The compiler's
# ``wrapModifiers`` (mlir/lib/Quantum/IR/QuantumInterfaces.cpp) folds modifiers into a graphOpId
# in this exact order: 1) control outermost, 2) adjoint innermost. So a single op that is both
# controlled and adjointed always spells as ``C(Adjoint(Op))``, never ``Adjoint(C(Op))``. This is
# a canonicalization. So the two spellings denote the same operator and MUST map to the same graph
# node, or the solver would treat them as distinct gates to match a rule against the op.
# The parser (``parseOperator``) is a structural round-trip and does NOT re-order, so canonicity has
# to be guaranteed here at the producer.
# We add future modifiers to this tuple at their canonical depth.
_MODIFIER_CANONICAL_ORDER = ("C", "Adjoint")


def _modifier_kind(modifier: str) -> str:
    """Normalise a modifier token to its canonical form."""
    return "C" if modifier.endswith("C") else modifier


def _leading_modifier_kind(op_id: str) -> str | None:
    """Return the canonical kind of ``op_id``'s current outermost modifier, or None if bare."""
    if op_id.startswith("Adjoint("):
        return "Adjoint"
    i = 0
    while i < len(op_id) and op_id[i].isdigit():
        i += 1
    if op_id[i:].startswith("C("):
        return "C"
    return None


def wrap_modifier_id(op_id: str, modifier: str) -> str:
    """Name-wrap an op-level ``modifier`` around a graphOpId's operator name.

    The modifier decorates the operator name only; the ``{param}{wire}{static}`` groups (and an
    optional ``[uid]``) follow it, matching the compiler's ``defaultGetGraphOpId``. This applies to
    any modifier (e.g. ``"Adjoint"``, ``"C"``), so nested ids compose as ``C(Adjoint(RX)){...}``.
    Extend callers here to support future op-level modifiers.
    """
    new_kind = _modifier_kind(modifier)
    inner_kind = _leading_modifier_kind(op_id)
    # `modifier is added as the new *outermost* layer. To keep graphOpIds canonical (see
    # MODIFIER_CANONICAL_ORDER), the new outer modifier must not belong *inside* one that is
    # already applied (e.g. wrapping Adjoint around an already-controlled C(RX){...} would
    # produce the non-canonical Adjoint(C(RX)) and is rejected, since the canonical form is
    # C(Adjoint(RX))).
    if inner_kind is not None:
        new_rank = _MODIFIER_CANONICAL_ORDER.index(new_kind)
        inner_rank = _MODIFIER_CANONICAL_ORDER.index(inner_kind)
        if new_rank > inner_rank:
            raise ValueError(
                f"Non-canonical modifier order: cannot wrap {modifier!r} (canonically inner) "
                f"around {op_id!r} whose outermost modifier {inner_kind!r} is canonically outer. "
                f"Apply modifiers outermost-last in the order {_MODIFIER_CANONICAL_ORDER}."
            )

    split = len(op_id)
    for i, char in enumerate(op_id):
        if char in "{[":
            split = i
            break
    return f"{modifier}({op_id[:split]}){op_id[split:]}"


def name_wrap_adjoint(op_id: str) -> str:
    """Name-wrap the adjoint modifier around a graphOpId (``RX{...}`` -> ``Adjoint(RX){...}``)."""
    return wrap_modifier_id(op_id, "Adjoint")


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


def split_call_args(kwargs, is_custom_op):
    """Split prepared kwargs into (args, kwargs) for calling a rule or resource function.

    Custom-op dynamic params are keyed positionally ("0", "1", ...) but the rule callables expect
    them by their real argnames, so they are passed positionally with only "wires" kept as keyword.
    Custom-op params are always scalar f64, so scalar ``0.0`` dummies are used (a shape-(1,) array
    would instead lower the gate through the general ``qref.operator`` path rather than ``qref.custom``).
    """
    if is_custom_op:
        args = tuple(0.0 for key in kwargs if key != "wires")
        return args, {"wires": kwargs["wires"]}
    return (), kwargs


def collect_resources_for_op(op_name, kwargs, is_custom_op=False):
    """Return resource data for all decomposition rules associated to op_name."""
    decomp_rules = list(qp.decomposition.list_decomps(op_name))
    args, kwargs = split_call_args(kwargs, is_custom_op)

    # map rules to resource resources, in a more generic format
    name_to_resource_ids = {}
    name_to_resources = {}
    for rule in decomp_rules:
        try:
            # The `compute_resources` function's signature is the same as the Operator2 signature
            # for the original op of the rule
            resources = rule.compute_resources(*args, **kwargs)
            name_to_resources[rule.name] = resources.gate_counts
            name_to_resource_ids[rule.name] = {
                GraphOpID(op).getGraphOpId(): count for op, count in resources.gate_counts.items()
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
    wrap_adjoint=False,
) -> ModuleOp:
    """
    Return a ModuleOp containing the decomposition rules for an operator instance.

    The decomposition rules will be decorated with appropriate resource and target_gate attributes.

    When ``wrap_adjoint`` is True, the rules registered on the base op ``op_name`` are instead
    synthesized into rules for ``Adjoint(op_name)`` (aka the "distribution" pathway). Each base
    rule body is wrapped in a ``qml.adjoint`` region (reduced to op-level modified gates by
    ``adjoint-lowering`` within the decomposition pass), the ``target_gate`` updates to the adjoint
    id, and each produced op in the resources is wrapped in ``Adjoint(...)``.
    """
    kwargs = prepare_dynamic_op_kwargs(dynamic_shape, wire_lens)
    extra_data = extra_data or {}
    device = qp.device("null.qubit", wires=sum(wire_lens.values()))

    _, name_to_resource_ids, decomp_rules = collect_resources_for_op(
        op_name, kwargs | static_data | extra_data, is_custom_op
    )

    # The distribution pathway targets `Adjoint(op)` and produces adjointed resource gates.
    target_id = name_wrap_adjoint(op_id) if wrap_adjoint else op_id
    if wrap_adjoint:
        name_to_resource_ids = {
            rule_name: {name_wrap_adjoint(produced_id): count for produced_id, count in ids.items()}
            for rule_name, ids in name_to_resource_ids.items()
        }

    # The static_data was only needed to instantiate the correct decomp rule
    # Once we have the correct rules, don't send them into qjit
    def rule_to_subroutine(rule):
        def decomp_rule(*_args, **_kwargs):
            if wrap_adjoint:
                qp.adjoint(rule._impl)(*_args, **_kwargs)
            else:
                rule._impl(*_args, **_kwargs)

        decomp_rule_no_static_args = partial(decomp_rule, **static_data)
        if extra_data:
            decomp_rule_no_static_args = partial(decomp_rule_no_static_args, **extra_data)

        # Keep the frontend name for readability, append target op_id for symbol uniqueness:
        decomp_rule_no_static_args.__name__ = rule._impl.__name__ + "_" + target_id

        return qp.capture.subroutine(decomp_rule_no_static_args)

    call_args, call_kwargs = split_call_args(kwargs, is_custom_op)

    subroutines = []
    for rule in decomp_rules:
        if rule.is_applicable(*call_args, **call_kwargs):
            subroutines.append(rule_to_subroutine(rule))

    @qp.qjit(target="mlir", capture=True, collect_decomp_rules=False)
    @qp.qnode(device=device)
    def circuit():
        for subroutine in subroutines:
            subroutine(*call_args, **call_kwargs)

    module = circuit.mlir_module

    def update_funcop_attributes(op):
        """Update the decomposition rule attributes if op is a decomposition rule.

        For use with module.walk

        This function updates the following attributes:
            - Adds the `target_gate` attribute.
            - Adds the `resources` attribute.
        """
        if op.name == "func.func":
            rule_name = ir.StringAttr(op.attributes["sym_name"]).value.removesuffix("_" + target_id)
            if rule_name in name_to_resource_ids:
                op.attributes["resources"] = get_mlir_attribute_from_pyval(
                    {"operations": name_to_resource_ids[rule_name]}
                )
                op.attributes["target_gate"] = ir.StringAttr.get(target_id)

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
    op_name, op_id, dynamic_shape, wire_lens, static_data, extra_data=None, is_custom_op=False
):
    extra_data = extra_data or {}
    queue = deque()
    start = (op_name, dynamic_shape, wire_lens, static_data, extra_data, is_custom_op)
    queue.append(start)
    visited = [start]

    def compile_variants(
        name, op_id, dynamic_shape, wire_lens, static_data, extra_data, is_custom_op
    ):
        # CQRs/Adjoint: For an op `name` capture the rules for
        #   1. the base op `name`,
        #   2. the adjoint op `Adjoint(name)` rules, and
        #   3. the adjoint op synthesized by distributing each base rule over adjoint.
        # Note: a rule whose body or resources can't be captured is skipped with a warning.
        out = get_rule_strings_from_module(
            compile_decomposition_rules(
                name,
                op_id,
                dynamic_shape,
                wire_lens,
                static_data,
                extra_data=extra_data,
                is_custom_op=is_custom_op,
            )
        )
        if not name.startswith("Adjoint("):
            adj_name = f"Adjoint({name})"
            # Rules registered directly against Adjoint(name):
            try:
                out.extend(
                    get_rule_strings_from_module(
                        compile_decomposition_rules(
                            adj_name,
                            name_wrap_adjoint(op_id),
                            dynamic_shape,
                            wire_lens,
                            static_data,
                            extra_data=extra_data,
                            is_custom_op=is_custom_op,
                        )
                    )
                )
            except Exception as e:  # pylint: disable=broad-except
                warnings.warn(f"Failed to lower the decomposition rules for {adj_name}: {e}")
            # Rules for Adjoint(name) synthesized by adjointing each base rule of `name`:
            try:
                distributed = get_rule_strings_from_module(
                    compile_decomposition_rules(
                        name,
                        op_id,
                        dynamic_shape,
                        wire_lens,
                        static_data,
                        extra_data=extra_data,
                        is_custom_op=is_custom_op,
                        wrap_adjoint=True,
                    )
                )
                # Suppress a distribution rule whose body is non-invertible:
                distributed = [
                    rule
                    for rule in distributed
                    if not any(marker in rule for marker in _NON_INVERTIBLE_MARKERS)
                ]
                out.extend(distributed)
            except Exception as e:  # pylint: disable=broad-except
                warnings.warn(f"Failed to synthesize distributed adjoint rules for {adj_name}: {e}")
        return out

    rules = compile_variants(
        op_name, op_id, dynamic_shape, wire_lens, static_data, extra_data, is_custom_op
    )

    while len(queue) != 0:
        (
            this_name,
            this_dynamic_shape,
            this_wire_lens,
            this_static_data,
            this_extra_data,
            this_is_custom_op,
        ) = queue.popleft()
        this_extra_data = this_extra_data or {}
        this_kwargs = prepare_dynamic_op_kwargs(this_dynamic_shape, this_wire_lens)
        # Explore ops reachable through the rules of both this op and its adjoint:
        for explore_name, _ in _op_variants(this_name, ""):
            resources, _, _ = collect_resources_for_op(
                explore_name, this_kwargs | this_static_data | this_extra_data, this_is_custom_op
            )
            for _rule_name, resource in resources.items():
                try:
                    for op, _count in resource.items():
                        graph_op_id = GraphOpID(op)
                        probe = (
                            graph_op_id.get_operator_name(),
                            convert_types_to_mlir_strings(graph_op_id.dynamic_shape),
                            graph_op_id.wire_lens,
                            graph_op_id.static_data,
                            graph_op_id.extra_data,
                            graph_op_id.is_custom_op,
                        )

                        if not probe in visited:
                            visited.append(probe)
                            queue.append(probe)
                            rules.extend(
                                compile_variants(
                                    probe[0],
                                    graph_op_id.getGraphOpId(),
                                    probe[1],
                                    probe[2],
                                    probe[3],
                                    probe[4],
                                    probe[5],
                                )
                            )
                except Exception as e:
                    warnings.warn(
                        f"Failed to lower the {_rule_name} decomposition rule for {this_name}: {e}"
                    )
                continue
    return rules


def _op_variants(op_name, op_id):
    """Yield the (name, id) for an operator, unless it is already adjointed.

    For a gate `Op` we lower the rules registered against both `Op` and `Adjoint(Op)`.
    """
    yield op_name, op_id
    if not op_name.startswith("Adjoint("):
        yield f"Adjoint({op_name})", name_wrap_adjoint(op_id)
