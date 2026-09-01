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

"""This module provides infrastructure for lowering decomposition rules via python."""

# pylint: disable=protected-access,bare-except

import warnings
from collections import deque
from functools import partial

import jax.numpy as jnp
import pennylane as qp
from jax._src.lib.mlir import ir
from jaxlib.mlir.dialects.builtin import ModuleOp

from catalyst.decomposition.graph_op_id import GraphOpID
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.decomposition.type_utils import get_dummy_values_for_arg
from catalyst.jax_extras.lowering import get_mlir_attribute_from_pyval

# Ops that make a decomposition body non-invertible
_NON_INVERTIBLE_MARKERS = (
    "qref.measure",
    "quantum.measure",
    "measure_in_basis",
    ".ppm",  # pbc.ppm / pbc.ref.ppm / pbc.select.ppm
)


# Canonical nesting order for op-level modifiers, listed OUTERMOST first. The compiler's
# ``wrapModifiers`` (mlir/lib/Quantum/IR/QuantumInterfaces.cpp) folds modifiers into a graphOpId
# in this exact order:
# 1. control outermost
# 2. adjoint innermost
# So a single op that is both controlled and adjointed always spells as ``C(Adjoint(Op))``,
# never ``Adjoint(C(Op))``. This is a canonicalization. So the two spellings denote the same
# operator and MUST map to the same graph node, or the solver would treat them as distinct gates
# to match a rule against the op.
# The parser (``parseOperator``) is a structural round-trip and does NOT re-order, so canonicity
# has to be guaranteed here at the producer. We add future modifiers to this tuple at their
# canonical depth.
_MODIFIER_CANONICAL_ORDER = ("C", "Adjoint")


def _modifier_kind(modifier: str) -> str:
    """Normalise a modifier token to its canonical form."""
    return "C" if modifier.endswith("C") else modifier


def _control_modifier(n_ctrl: int) -> str:
    """Return the graphOpId control-modifier token for ``n_ctrl`` controls.

    A single control is written ``C`` and ``n > 1`` controls ``<n>C``, mirroring the compiler's
    ``wrapModifiers``. Used with :func:`wrap_modifier_id`, e.g. ``wrap_modifier_id(op_id, "2C")``.
    """
    assert n_ctrl >= 1, "control modifier requires at least one control"
    return "C" if n_ctrl == 1 else f"{n_ctrl}C"


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
    # The modifier is added as the new *outermost* layer. To keep graphOpIds canonical (see
    # _MODIFIER_CANONICAL_ORDER), the new outer modifier must not belong *inside* one that is
    # already applied (e.g. wrapping Adjoint around an already-controlled C(RX){...} would produce
    # the non-canonical Adjoint(C(RX)) and is rejected, since the canonical form is C(Adjoint(RX))).
    if inner_kind is not None:
        new_rank = _MODIFIER_CANONICAL_ORDER.index(new_kind)
        inner_rank = _MODIFIER_CANONICAL_ORDER.index(inner_kind)
        if new_rank > inner_rank:
            raise ValueError(
                f"Non-canonical modifier order: cannot wrap {modifier!r} (canonically inner) "
                f"around {op_id!r} whose outermost modifier {inner_kind!r} is canonically outer. "
                f"Apply modifiers outermost-last in the order {_MODIFIER_CANONICAL_ORDER}."
            )

    # Only the operator name is wrapped; the first '{' begins the {param}{wire}{static}[uid] suffix
    # (the dynamic-shape group is always present), which is carried through untouched.
    assert "{" in op_id, f"Malformed op id for graph decomposition, got {op_id}"
    split = op_id.find("{")
    return f"{modifier}({op_id[:split]}){op_id[split:]}"


def name_wrap_adjoint(op_id: str) -> str:
    """Name-wrap the adjoint modifier around a graphOpId (``RX{...}`` -> ``Adjoint(RX){...}``)."""
    return wrap_modifier_id(op_id, "Adjoint")


def name_unwrap_adjoint(op_name: str, op_id: str) -> str:
    """Inverse of :func:`name_wrap_adjoint` given the base ``op_name``."""
    prefix = f"Adjoint({op_name})"
    if not op_id.startswith(prefix):
        raise ValueError(f"{op_id!r} is not an adjoint id for base op {op_name!r}")
    return op_name + op_id[len(prefix) :]


def name_unwrap_control(op_name: str, op_id: str):
    """Inverse of control name-wrapping given the base ``op_name``.

    ``("RX", "2C(RX){...}")`` -> ``("RX{...}", 2)`` and ``("RX", "C(RX){...}")`` -> ``("RX{...}", 1)``.
    Raises if ``op_id`` is not a control id for ``op_name``.
    """
    i = 0
    while i < len(op_id) and op_id[i].isdigit():
        i += 1
    digits = op_id[:i]
    n_ctrl = int(digits) if digits else 1
    prefix = f"{digits}C({op_name})"
    if not op_id.startswith(prefix):
        raise ValueError(f"{op_id!r} is not a control id for base op {op_name!r}")
    return op_name + op_id[len(prefix) :], n_ctrl


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
        module: an MLIR module object; every FuncOp carrying a `target_gate` attribute is
                extracted as a decomposition rule.

    Returns:
        str: The string representation of any decomposition rules from `module`, pre-pending the
             `__builtin_` prefix to their names.
    """
    funcOps = get_rule_strings_from_module(module)
    return "\n".join(str(funcOp) for funcOp in funcOps) if funcOps else ""


def inject_new_rules_into_module(module: ir.Module, decomp_rules: list[str]):
    with ir.InsertionPoint(module.body):
        for decomp_rule in decomp_rules:
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


def collect_resources_for_op(op_name, kwargs, is_custom_op=False, adjoint_resources=False):
    """Return resource data for all decomposition rules associated to op_name."""
    decomp_rules = list(qp.decomposition.list_decomps(op_name))
    args, kwargs = split_call_args(kwargs, is_custom_op)

    # map each rule to its resources, in a more generic format
    name_to_resource_ids = {}
    name_to_resources = {}
    for rule in decomp_rules:
        try:
            # The `compute_resources` function's signature is the same as the Operator2 signature
            # for the original op of the rule
            resources = rule.compute_resources(*args, **kwargs)
            name_to_resources[rule.name] = resources.gate_counts
            # When adjoint_resources is True, each produced resource's graphOpId is generated in its
            # adjoint form (Adjoint(<name>){...}) directly from the resource op instance via
            # GraphOpID.getGraphOpId, rather than string.
            name_to_resource_ids[rule.name] = {
                GraphOpID(op).getGraphOpId(adjoint=adjoint_resources): count
                for op, count in resources.gate_counts.items()
            }
        except Exception as e:
            warnings.warn(
                f"Failed to get resources for the {rule.name} decomposition rule: {e}",
                category=RuleLoweringWarning,
            )

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
    wrap_control=False,
    n_ctrl=1,
) -> ModuleOp:
    """
    Return a ModuleOp containing the decomposition rules for an operator instance.

    The decomposition rules will be decorated with appropriate resource and target_gate attributes.

    When ``wrap_adjoint`` is True, the rules registered on the base op ``op_name`` are instead
    synthesized into rules for ``Adjoint(op_name)`` (aka the "distribution" pathway). Each base
    rule body is wrapped in a ``qp.adjoint`` region (reduced to op-level modified gates by
    ``adjoint-lowering`` within the decomposition pass), the ``target_gate`` updates to the adjoint
    id, and each produced op in the resources is wrapped in ``Adjoint(...)``.

    When ``wrap_control`` is True, the analogous "distribution" pathway is applied for control: each
    base rule body is wrapped in ``qp.ctrl(..., control=<n_ctrl wires>)`` (reduced to op-level
    controlled gates by ``ctrl-lowering`` within the decomposition pass), the ``target_gate`` becomes
    ``<n>C(op_name)`` and each produced resource op is wrapped in the same ``<n>C(...)`` modifier.

    Note that ``wrap_adjoint`` and ``wrap_control`` may be combined to synthesize the nested modifier
    ``<n>C(Adjoint(op_name))``: adjoint is applied innermost and control outermost (the canonical
    order matching the compiler's ``wrapModifiers``).
    """
    kwargs = prepare_dynamic_op_kwargs(dynamic_shape, wire_lens)
    extra_data = extra_data or {}
    n_base_wires = sum(wire_lens.values())
    device = qp.device("null.qubit", wires=n_base_wires + (n_ctrl if wrap_control else 0))

    _, name_to_resource_ids, decomp_rules = collect_resources_for_op(
        op_name, kwargs | static_data | extra_data, is_custom_op, adjoint_resources=wrap_adjoint
    )

    # TODO: The modified target id and the wrapped resource ids are derived here by string-wrapping
    # the graphOpId (via wrap_modifier_id). Ideally they would be generated via
    # GraphOpID.getGraphOpId which requires missing steps in the GraphOpID object.
    # Note it needs changes not just in this function, also in the string id and across
    # the on-demand C++ loader boundary.
    target_id = name_wrap_adjoint(op_id) if wrap_adjoint else op_id
    if wrap_control:
        ctrl_mod = _control_modifier(n_ctrl)
        target_id = wrap_modifier_id(target_id, ctrl_mod)
        name_to_resource_ids = {
            rule_name: {
                wrap_modifier_id(produced_id, ctrl_mod): count for produced_id, count in ids.items()
            }
            for rule_name, ids in name_to_resource_ids.items()
        }

    # The static_data was only needed to instantiate the correct decomp rule
    # Once we have the correct rules, don't send them into qjit
    def rule_to_subroutine(rule):
        def decomp_rule(*_args, _ctrl_wires=None, **_kwargs):
            # Apply adjoint innermost, control outermost (canonical `C(Adjoint(Op))`).
            body = qp.adjoint(rule._impl) if wrap_adjoint else rule._impl
            if wrap_control:
                qp.ctrl(body, control=list(_ctrl_wires))(*_args, **_kwargs)
            else:
                body(*_args, **_kwargs)

        decomp_rule_no_static_args = partial(decomp_rule, **static_data)
        if extra_data:
            decomp_rule_no_static_args = partial(decomp_rule_no_static_args, **extra_data)

        # Keep the frontend name for readability, append target op_id for symbol uniqueness:
        decomp_rule_no_static_args.__name__ = rule._impl.__name__ + "_" + target_id

        return qp.capture.subroutine(decomp_rule_no_static_args)

    condition_args, condition_kwargs = split_call_args(
        kwargs | static_data | extra_data, is_custom_op
    )

    subroutines = []
    for rule in decomp_rules:
        if rule.name in name_to_resource_ids and rule.is_applicable(
            *condition_args, **condition_kwargs
        ):
            subroutines.append(rule_to_subroutine(rule))

    # For control distribution, the extra control wires are
    # added to each rule body via the `_ctrl_wires` keyword argument.
    ctrl_wires = (
        jnp.array(range(n_base_wires, n_base_wires + n_ctrl), dtype=int) if wrap_control else None
    )

    call_args, call_kwargs = split_call_args(kwargs, is_custom_op)

    @qp.qjit(target="mlir", capture=True, collect_decomp_rules=False)
    @qp.qnode(device=device)
    def circuit():
        for subroutine in subroutines:
            if wrap_control:
                subroutine(*call_args, _ctrl_wires=ctrl_wires, **call_kwargs)
            else:
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


def adjoint_variant_rule_strings(
    op_name, op_id, dynamic_shape, wire_lens, static_data, extra_data=None, is_custom_op=False
):
    """Return the rule strings whose ``target_gate`` is ``Adjoint(op_name)``.

    ``op_id`` is the *base* op's graphOpId (e.g. ``"S{...}"``). Two pathways contribute:
      1. rules registered directly against ``Adjoint(op_name)`` (``list_decomps("Adjoint(S)")``), and
      2. rules synthesized by distributing each base rule of ``op_name`` over adjoint
         (the ``wrap_adjoint`` pathway), dropping any whose body is non-invertible.

    Shared by the eager lowering-time closure (:func:`fetch_all_reachable_decomposition_rules_from_op`)
    and the compiler's on-demand loader (:func:`compile_decomposition_rules_wrapper`) so both build
    adjoint rules identically.
    """
    out = []
    adj_name = f"Adjoint({op_name})"
    # (1) Rules registered directly against Adjoint(op_name):
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
        warnings.warn(
            f"Failed to lower the decomposition rules for {adj_name}: {e}",
            category=RuleLoweringWarning,
        )
    # (2) Rules for Adjoint(op_name) synthesized by adjointing each base rule of op_name:
    try:
        distributed = get_rule_strings_from_module(
            compile_decomposition_rules(
                op_name,
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
        warnings.warn(
            f"Failed to synthesize distributed adjoint rules for {adj_name}: {e}",
            category=RuleLoweringWarning,
        )
    return out


def control_variant_rule_strings(
    op_name,
    op_id,
    ctrl_counts,
    dynamic_shape,
    wire_lens,
    static_data,
    extra_data=None,
    is_custom_op=False,
):
    """Return the rule strings whose ``target_gate`` is ``<n>C(op_name)`` for each ``n`` in
    ``ctrl_counts``.

    The control analogue of :func:`adjoint_variant_rule_strings`. ``op_id`` is the *base* op's
    graphOpId (e.g. ``"RX{...}"``). For each control count ``n`` three pathways contribute:
      1. rules registered directly against ``<n>C(op_name)`` (``list_decomps("C(RX)")``),
      2. rules synthesized by distributing each base rule of ``op_name`` over ``n`` controls
         (the ``wrap_control`` pathway), and
      3. rules for the nested modifier ``<n>C(Adjoint(op_name))`` synthesized by controlling each
         *adjointed* base rule (``wrap_adjoint`` + ``wrap_control``), so controlled-adjoint ops are
         reachable too.
    Distribution rules whose body is non-controllable are dropped.
    """
    out = []
    for n in ctrl_counts:
        ctrl_mod = _control_modifier(n)
        ctrl_name = f"{ctrl_mod}({op_name})"
        # (1) Rules registered directly against <n>C(op_name):
        try:
            out.extend(
                get_rule_strings_from_module(
                    compile_decomposition_rules(
                        ctrl_name,
                        wrap_modifier_id(op_id, ctrl_mod),
                        dynamic_shape,
                        wire_lens,
                        static_data,
                        extra_data=extra_data,
                        is_custom_op=is_custom_op,
                    )
                )
            )
        except Exception as e:  # pylint: disable=broad-except
            warnings.warn(
                f"Failed to lower the decomposition rules for {ctrl_name}: {e}",
                category=RuleLoweringWarning,
            )
        # (2) <n>C(op_name) by controlling each base rule, and
        # (3) <n>C(Adjoint(op_name)) by controlling each adjointed base rule.
        for wrap_adjoint, label in ((False, ctrl_name), (True, f"{ctrl_mod}(Adjoint({op_name}))")):
            try:
                controlled = get_rule_strings_from_module(
                    compile_decomposition_rules(
                        op_name,
                        op_id,
                        dynamic_shape,
                        wire_lens,
                        static_data,
                        extra_data=extra_data,
                        is_custom_op=is_custom_op,
                        wrap_adjoint=wrap_adjoint,
                        wrap_control=True,
                        n_ctrl=n,
                    )
                )
                # Suppress a distribution rule whose body is non-controllable (e.g. a measurement):
                controlled = [
                    rule
                    for rule in controlled
                    if not any(marker in rule for marker in _NON_INVERTIBLE_MARKERS)
                ]
                out.extend(controlled)
            except Exception as e:  # pylint: disable=broad-except
                warnings.warn(
                    f"Failed to synthesize distributed control rules for {label}: {e}",
                    category=RuleLoweringWarning,
                )
    return out


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


def compile_reachable_decomposition_rules_wrapper(
    op_name,
    op_id,
    dynamic_shape,
    wire_lens,
    static_data,
    extra_data=None,
    is_custom_op=False,
) -> str:
    """Return an MLIR module with the full reachable decomposition-rule closure for an operator.

    This is the entry point for the compiler's *on-demand* rule loader (``loadPythonDecomps`` ->
    ``pythonRuleLowering``), which passes the op's *base* name (``getOperatorName()``) together with
    its full graphOpId (``getGraphOpId()``). Two things matter here:

    * **Modifier ids.** For a plain op the name and id agree (``"S"`` / ``"S{...}"``). For an
      op-level modifier the graphOpId is name-wrapped (``"Adjoint(S){...}"``) while the name stays
      the base (``"S"``). We recover the base id so the closure explores the base op *and* its
      adjoint variants; otherwise ``Adjoint(S)`` would be decomposed as if it were ``S``.
    * **The whole closure, not just this op's direct rules.** The loader does not recurse into a
      rule's resource ops, so it needs every rule reachable from this op down to the gate set in one
      shot. For ``Adjoint(S)`` that includes ``Adjoint(S) -> Adjoint(PhaseShift)`` *and*
      ``Adjoint(PhaseShift) -> PhaseShift``; returning only the first would leave the solver unable
      to complete a path. :func:`fetch_all_reachable_decomposition_rules_from_op` builds that closure
      (base + adjoint-registered + distributed-adjoint rules, transitively) and each returned func
      keeps its own ``target_gate``, which is how the loader registers them.
    """
    base_id = op_id
    n_ctrls = 0
    if op_id.startswith("Adjoint(") and not op_name.startswith("Adjoint("):
        base_id = name_unwrap_adjoint(op_name, op_id)
    elif _leading_modifier_kind(op_id) == "C" and not op_name.startswith("Adjoint("):
        # A controlled op-id (`C(op)` / `<n>C(op)`): recover the base id and control count so the
        # closure synthesizes the matching `<n>C(...)` rules.
        base_id, n_ctrls = name_unwrap_control(op_name, op_id)

    rule_strings = fetch_all_reachable_decomposition_rules_from_op(
        op_name=op_name,
        op_id=base_id,
        dynamic_shape=dynamic_shape,
        wire_lens=wire_lens,
        static_data=static_data,
        extra_data=extra_data,
        is_custom_op=is_custom_op,
        n_ctrls=n_ctrls,
    )
    # Wrap the rule funcs in a module: the compiler parses this string with
    # `parseSourceString<ModuleOp>`, which requires a single top-level op.
    return "module {\n" + "\n".join(rule_strings) + "\n}"


def fetch_all_reachable_decomposition_rules_from_op(
    op_name,
    op_id,
    dynamic_shape,
    wire_lens,
    static_data,
    extra_data=None,
    is_custom_op=False,
    n_ctrls=0,
):
    extra_data = extra_data or {}
    queue = deque()
    start = (op_name, dynamic_shape, wire_lens, static_data, extra_data, is_custom_op)
    queue.append(start)
    visited = [start]

    # Control counts to synthesize `<n>C(...)` rules for. A single control is always captured
    # proactively; a multi-controlled instance (`n_ctrls > 1`) additionally needs its own count.
    ctrl_counts = [1] if n_ctrls <= 1 else [1, n_ctrls]

    def compile_variants(
        name, op_id, dynamic_shape, wire_lens, static_data, extra_data, is_custom_op
    ):
        # CQRs (Adjoint/Control): For an op `name` capture the rules for
        #   1. the base op `name`,
        #   2. the adjoint op `Adjoint(name)`: registered + distributed-over-adjoint rules, and
        #   3. the controlled op `<n>C(name)` for each `n` in `ctrl_counts`: registered +
        #      distributed-over-control rules.
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
        # Only synthesize adjoint/control variants of a base op. If `op_id` already carries an
        # outermost modifier (e.g. `Adjoint(...)` or `C(...)`, reached as another rule's resource),
        # its own modifier variants are synthesized from its base op instead:
        if _leading_modifier_kind(op_id) is None:
            out.extend(
                adjoint_variant_rule_strings(
                    name,
                    op_id,
                    dynamic_shape,
                    wire_lens,
                    static_data,
                    extra_data=extra_data,
                    is_custom_op=is_custom_op,
                )
            )
            out.extend(
                control_variant_rule_strings(
                    name,
                    op_id,
                    ctrl_counts,
                    dynamic_shape,
                    wire_lens,
                    static_data,
                    extra_data=extra_data,
                    is_custom_op=is_custom_op,
                )
            )
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
        for explore_name in _op_variant_names(this_name):
            resources, _, _ = collect_resources_for_op(
                explore_name, this_kwargs | this_static_data | this_extra_data, this_is_custom_op
            )
            for _rule_name, resource in resources.items():
                try:
                    for op, _count in resource.items():
                        graph_op_id = GraphOpID(op)
                        probe = (
                            graph_op_id.get_operator_name(),
                            graph_op_id.dynamic_shape,
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
                        f"Failed to lower the {_rule_name} decomposition rule for {this_name}: {e}",
                        category=RuleLoweringWarning,
                    )
                continue
    return rules


def _op_variant_names(op_name):
    """Yield the operator names to explore for `op_name`: the op itself and, unless it is already
    adjointed, its adjoint `Adjoint(op_name)`. Rules registered against both are collected."""
    yield op_name
    if not op_name.startswith("Adjoint("):
        yield f"Adjoint({op_name})"
