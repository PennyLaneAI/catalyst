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

"""Look up decomposition rules, collect resources, and prepare callable batches.

Each batch describes one target and its tracing inputs. Argument handling lives in
``rule_arguments``; ``rule_discovery`` follows the batches' outgoing resource edges.
"""

# pylint: disable=protected-access,bare-except

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import pennylane as qp
from pennylane.core.operator import Operator2, abstractify
from pennylane.decomposition import DecompositionRule
from pennylane.decomposition.utils import to_name

from catalyst.decomposition.capture_session import OpDecompRequest
from catalyst.decomposition.graph_op_id import (
    CTRL_WORK_WIRES,
    GraphOpID,
)
from catalyst.decomposition.modifier_state import ModifierState
from catalyst.decomposition.rule_arguments import (
    RuleTraceInputs,
    _dummy_work_wires,
    build_base_op,
    flatten_hybrid_args,
    ordered_kwarg_names,
    prepare_dummy_rule_arguments,
    prepare_dynamic_op_kwargs,
    symbolic_arguments,
    unpack_rule_operands,
)
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.decomposition.type_utils import SerializedGraphOpID, _RuleBuilderArgs
from catalyst.utils.exceptions import CompileError

_NON_INVERTIBLE_RESOURCE_TYPES = (qp.ops.MidMeasure, qp.ops.PauliMeasure)


@dataclass
class CollectedRule:
    """A source rule whose resource counts were successfully collected.

    Attributes:
        source_rule: PennyLane rule, including its implementation and frontend name.
        resource_counts_by_op: Raw resource operations and their counts, used for graph traversal.
        resource_counts_by_id: Serialized resource IDs and their counts, adjusted for the
            target's modifiers during collection or preparation.
    """

    source_rule: DecompositionRule
    resource_counts_by_op: dict[Operator2, int]
    resource_counts_by_id: dict[SerializedGraphOpID, int]


@dataclass
class DecompRuleSpec:
    """A decomposition rule adapted for tracing, with resource counts for its consumers.

    The prepared callable unpacks flat operands, supplies closed-over arguments, and applies
    any distributed modifiers before invoking the source rule.

    Attributes:
        prepared_rule_fn: Adapted rule body accepting the flattened operands described by its
            :class:`RuleTraceInputs`.
        frontend_name: Name of the PennyLane decomposition rule, retained for fixed and excluded
            rule selection.
        resource_counts_by_id: GraphOpID-to-count mapping attached to the lowered function.
        resource_counts_by_op: Operator2-to-count mapping used to traverse outgoing resource edges.
    """

    prepared_rule_fn: Callable[..., None]
    frontend_name: str
    resource_counts_by_id: dict[SerializedGraphOpID, int]
    resource_counts_by_op: dict[Operator2, int]


@dataclass
class DecompTargetSpec:
    """A batch of prepared rules sharing a target graph op ID and tracing inputs.

    Registered symbolic rules and rules synthesized by distributing modifiers can form
    separate batches for the same target.

    Attributes:
        rules: Applicable rule alternatives for this target.
        trace_inputs: Shared dummy arguments and operand layout used to trace the rules.
        target_id: Full GraphOpID that every rule in the set decomposes.
        num_qreg_qubits: Register size required by the legacy standalone materializer.
        modifier_context: Additional modifiers propagated to resource operations when
            discovering child rules. May differ from the target's modifiers because
            registered rules can already account for them in their resource operations.
        reject_noninvertible: Whether modifier distribution requires filtering noninvertible bodies.
    """

    rules: list[DecompRuleSpec]
    trace_inputs: RuleTraceInputs
    target_id: SerializedGraphOpID
    num_qreg_qubits: int
    modifier_context: ModifierState
    reject_noninvertible: bool = False


def _resources_have_measurement(gate_counts) -> bool:
    """Return whether a rule's declared resources contain a mid-circuit measurement."""
    return any(isinstance(op, _NON_INVERTIBLE_RESOURCE_TYPES) for op in gate_counts)


def _adjoint_folds_to_base(
    resource_ids: dict[SerializedGraphOpID, int], base_id: SerializedGraphOpID
) -> bool:
    """Whether an ``Adjoint(op)`` rule's resources are a single unmodified copy of the base op.

    This holds for ``adjoint_rotation`` and ``self_adjoint`` to simplify the rule registry
    and avoid the solver having to match a rule that produces a modified gate back to the base op.
    Note that there are other rules that produce a single op of the same class but with
    modified static data; such rules are not considered a fold.

    Args:
        resource_ids (dict): the rule's resources as ``{resource graphOpId: count}``
        base_id (SerializedGraphOpID): the graphOpId of the base operator the rule was probed with

    Returns:
        bool: whether the rule folds ``Adjoint(op)`` to a single unmodified ``op``
    """
    if len(resource_ids) != 1:
        return False
    ((rid, count),) = resource_ids.items()
    return count == 1 and rid == base_id


def symbolic_op_name(op_name, kind) -> str:
    """Return the name PennyLane's registry holds ``op_name``'s symbolic rules under.

    A controlled operator is named ``C(Op)`` whatever its control count, so the count never appears
    here.

    Args:
        op_name (str): the base operator's name
        kind (str): ``"adjoint"`` or ``"control"``

    Returns:
        str: the registry name

    Raises:
        CompileError: if ``kind`` is not a known symbolic kind
    """
    if kind == "adjoint":
        return f"Adjoint({op_name})"
    if kind == "control":
        return f"C({op_name})"
    raise CompileError(f"Unknown symbolic kind: {kind}")  # pragma: no cover


def _rule_allocates_work_wires(rule, *args, **kwargs) -> bool:
    """Whether a decomposition rule dynamically allocates work wires.

    TODO: remove when --decompose-lowering can handle multiple registers
    """
    return rule.get_work_wire_spec(*args, **kwargs).total > 0


def _rule_is_compatible(op_name: str, rule: DecompositionRule, *args, **kwargs) -> bool:
    """Return whether a rule applies and passes the work-wire support check.

    Rules requiring work-wire allocation are excluded until decomposition lowering supports
    multiple registers. Exceptions during either check emit a warning and exclude the rule.
    """
    try:
        if not bool(rule.is_applicable(*args, **kwargs)):
            return False
    except Exception as e:  # pylint: disable=broad-except
        warnings.warn(
            f"Excluded the {rule.name} decomposition rule for {op_name}; raised '{e}'",
            category=RuleLoweringWarning,
        )
        return False

    # TODO: remove when decompose lowering pass can handle multiple registers
    try:
        allocates_work_wires = _rule_allocates_work_wires(rule, *args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        warnings.warn(
            f"Excluded the {rule.name} decomposition rule for {op_name}; could not read its "
            f"work-wire spec, raised '{e}'",
            category=RuleLoweringWarning,
        )
        return False

    if allocates_work_wires:
        warnings.warn(
            f"Excluded the {rule.name} decomposition rule for {op_name} since "
            "--decompose-lowering cannot work with multiple registers yet",
            category=RuleLoweringWarning,
        )
        return False

    return True


def collect_resources_for_op(
    op_name,
    kwargs,
    is_custom_op=False,
    op_cls=None,
    modifiers=ModifierState(),
) -> list[CollectedRule]:
    """Collect resources for eligible rules, omitting rules whose resource collection fails.

    Args:
        op_name (str): the operator's name, as PennyLane registers its rules
        kwargs (dict): the arguments to compute the resources with
        is_custom_op (bool): whether the operator lowers to ``qref.custom``
        op_cls (type): operator class used for registry lookup, when available
        modifiers (ModifierState): the modifiers to spell on each produced id

    Returns:
        list[CollectedRule]: source rules and their resource counts, in registry order
    """
    decomp_rules = list(qp.decomposition.list_decomps(op_cls or op_name))
    args, kwargs = prepare_dummy_rule_arguments(kwargs, is_custom_op)

    collected_rules = []
    for rule in decomp_rules:
        if not _rule_is_compatible(op_name, rule, *args, **kwargs):
            continue

        try:
            # The `compute_resources` function's signature is the same as the Operator2 signature
            # for the original op of the rule
            with qp.capture.toggle_ctx(True):
                resources = rule.compute_resources(*args, **kwargs)
            # A distributed rule produces modified gates, so each id is *generated* in its
            # modified form straight from the resource op instance -- the modifiers are placed
            # canonically by `build_graph_op_id`, never spliced into a finished id string.
            resource_counts_by_id = {
                GraphOpID(op).getGraphOpId(modifiers): count
                for op, count in resources.gate_counts.items()
            }
            collected_rules.append(
                CollectedRule(
                    source_rule=rule,
                    resource_counts_by_op=resources.gate_counts,
                    resource_counts_by_id=resource_counts_by_id,
                )
            )
        except Exception as e:
            warnings.warn(
                f"Failed to get resources for the {rule.name} decomposition rule: {e}",
                category=RuleLoweringWarning,
            )

    return collected_rules


def build_decomp_target_spec(
    op_name: str,
    op_id: SerializedGraphOpID,
    dynamic_shape,
    wire_lens,
    static_data,
    extra_data=None,
    is_custom_op=False,
    op_cls=None,
    modifiers=ModifierState(),
) -> DecompTargetSpec:
    """Prepare decomposition callables, operands, and attributes without materializing MLIR.

    When ``modifiers`` is adjointed, the rules registered on the base op ``op_name`` are instead
    synthesized into rules for ``Adjoint(op_name)`` (aka the "distribution" pathway). Each base
    rule body is wrapped in a ``qp.adjoint`` region (reduced to op-level modified gates by
    ``adjoint-lowering`` within the decomposition pass), the ``target_gate`` updates to the adjoint
    id, and each produced op in the resources is wrapped in ``Adjoint(...)``.

    When ``modifiers`` carries controls, the analogous "distribution" pathway is applied for
    control: each base rule body is wrapped in ``qp.ctrl(..., control=<n wires>)`` (reduced to
    op-level controlled gates by ``ctrl-lowering`` within the decomposition pass), the
    ``target_gate`` becomes ``<n>C(op_name)`` and each produced resource op is wrapped in the same
    ``<n>C(...)`` modifier. The control work wires are handed to the region and spelled on the
    target and resource ids alike.

    The two may be combined to synthesize the nested modifier ``<n>C(Adjoint(op_name))``: adjoint is
    applied innermost and control outermost (the canonical order matching the compiler's
    ``wrapModifiers``).

    Args:
        op_name (str): the operator's GraphOpID name and fallback registry name
        op_id (SerializedGraphOpID): the operator's graphOpId
        dynamic_shape (dict): dynamic argument names to their MLIR types
        wire_lens (dict): wire argument names to their lengths
        static_data (dict): compiler-static argument names to their values
        extra_data (dict): argument values the graphOpId identifies by UID instead of spelling
        is_custom_op (bool): whether the operator lowers to ``qref.custom``
        op_cls (type): operator class used for registry lookup, when available
        modifiers (ModifierState): the modifiers to distribute the base rules over

    Returns:
        DecompTargetSpec: callables and metadata consumable by either eager JAXPR capture or the
        legacy standalone MLIR materializer
    """
    kwargs = prepare_dynamic_op_kwargs(dynamic_shape, wire_lens)
    extra_data = extra_data or {}
    n_base_wires = sum(wire_lens.values())
    n_ctrl = modifiers.control_count
    num_work_wires = modifiers.num_work_wires
    signature_wire_lens = (
        {**wire_lens, CTRL_WORK_WIRES: num_work_wires} if num_work_wires else wire_lens
    )

    collected_rules = collect_resources_for_op(
        op_name,
        kwargs | static_data | extra_data,
        is_custom_op,
        op_cls=op_cls,
        modifiers=modifiers,
    )

    # The *target* id is still derived by string-wrapping, because this is the one identity we are
    # given rather than holding the operator for: the compiler's on-demand loader hands us a
    # finished graphOpId string (`getGraphOpId()`), and re-generating it from the components would
    # only agree as long as both spellings stay in step. Splicing keeps whatever came in verbatim.
    # TODO: generate it as well, once the loader passes the identity components (or the modifiers)
    # instead of a pre-wrapped id.
    target_id = modifiers.wrap_id(op_id)

    call_args, call_kwargs = prepare_dummy_rule_arguments(kwargs, is_custom_op)
    kwarg_names = ordered_kwarg_names(call_kwargs, dynamic_shape)

    hybrid_specs, hybrid_leaves, closed_hybrid = flatten_hybrid_args(extra_data)

    # The static_data was only needed to instantiate the correct decomp rule
    # Once we have the correct rules, don't send them into qjit: they are closed over instead.
    def rule_to_callable(rule: DecompositionRule) -> Callable[..., None]:
        def decomp_rule(*_operands) -> None:
            _args, _kwargs, _ctrl_wires, _ctrl_values, _work_wires = unpack_rule_operands(
                _operands,
                len(call_args),
                kwarg_names,
                signature_wire_lens,
                modifiers.is_controlled,
                hybrid_specs,
            )
            # Operand-passed hybrid args are rebuilt from operands in unpack_rule_operands; static
            # data and closed-over hybrid args are supplied directly.
            _kwargs |= static_data | closed_hybrid
            modifiers.wrap_callable(rule._impl, _ctrl_wires, _ctrl_values, _work_wires)(
                *_args, **_kwargs
            )

        # JAX's symbol table disambiguates rules with the same frontend name.
        decomp_rule.__name__ = rule.name

        return decomp_rule

    rule_specs = []
    for collected in collected_rules:
        rule = collected.source_rule
        if modifiers.is_modified and _resources_have_measurement(collected.resource_counts_by_op):
            warnings.warn(
                f"Skipped the {rule.name} decomposition rule for {target_id}: it contains a "
                "mid-circuit measurement, which is not supported with adjoint or control regions.",
                category=RuleLoweringWarning,
            )
            continue
        rule_specs.append(
            DecompRuleSpec(
                prepared_rule_fn=rule_to_callable(rule),
                frontend_name=rule.name,
                resource_counts_by_id=collected.resource_counts_by_id,
                resource_counts_by_op=collected.resource_counts_by_op,
            )
        )

    # Control wires trail the grouped registers. Control work wires are one of those registers.
    ctrl_wires = (
        jnp.array(range(n_base_wires, n_base_wires + n_ctrl), dtype=int)
        if modifiers.is_controlled
        else None
    )
    ctrl_values = jnp.ones(n_ctrl, dtype=bool) if modifiers.is_controlled else None
    if num_work_wires:
        call_kwargs = {
            **call_kwargs,
            CTRL_WORK_WIRES: _dummy_work_wires(n_base_wires + n_ctrl, num_work_wires),
        }

    return DecompTargetSpec(
        rules=rule_specs,
        trace_inputs=RuleTraceInputs(
            call_args=call_args,
            call_kwargs=call_kwargs,
            kwarg_names=kwarg_names,
            wire_lens=signature_wire_lens,
            ctrl_wires=ctrl_wires,
            ctrl_values=ctrl_values,
            hybrid_leaves=tuple(hybrid_leaves),
        ),
        target_id=target_id,
        num_qreg_qubits=n_base_wires + n_ctrl + num_work_wires,
        modifier_context=modifiers,
        reject_noninvertible=modifiers.is_modified,
    )


def collect_symbolic_resources(
    op_cls,
    op_name,
    kwargs,
    is_custom_op,
    *,
    kind,
    ctrl_wires=(),
    ctrl_values=(),
    work_wires=(),
    work_wire_type="borrowed",
) -> tuple[list[CollectedRule], dict]:
    """Collect resources for eligible rules registered against ``Adjoint(op_name)``/``C(op_name)``.

    PennyLane names a controlled operator ``C(Op)`` whatever its control count, so the registry is
    always queried under that single name; ``ctrl_wires`` carries the count.
    Rules whose resource collection fails are omitted.

    Args:
        op_cls (type): the base operator's class
        op_name (str): the base operator's GraphOpID name
        kwargs (dict): the arguments to build the base operator with
        is_custom_op (bool): whether the operator lowers to ``qref.custom``
        kind (str): ``"adjoint"`` or ``"control"``
        ctrl_wires: the control wires, for ``kind="control"``
        ctrl_values: the control values, for ``kind="control"``
        work_wires: the work wires available to the controls, for ``kind="control"``
        work_wire_type (str): the type of those work wires

    Returns:
        list[CollectedRule]: source rules and their resource counts, in registry order
        dict: the probe arguments, or an empty dictionary if no rules are registered
    """
    lookup_name = symbolic_op_name(to_name(op_cls), kind)
    rules = list(qp.decomposition.list_decomps(lookup_name))
    if not rules:
        return [], {}

    # The base op is only a carrier of the dummy parameters and wires: the graph reasons about
    # resources in terms of abstract operators, matching `_get_kwargs` in PennyLane's graph.
    with qp.capture.pause():  # do not bind more primitives from abstract instantiation
        base_op = abstractify(build_base_op(op_cls, kwargs, is_custom_op))
    probe_args = symbolic_arguments(
        base_op, kind, ctrl_wires, ctrl_values, work_wires, work_wire_type
    )

    collected_rules = []
    for rule in rules:
        if not _rule_is_compatible(op_name, rule, **probe_args):
            continue

        try:
            with qp.capture.toggle_ctx(True):
                resources = rule.compute_resources(**probe_args)
            # The rule body names the ops it produces itself, so unlike the distribution pathway
            # these ids carry no added modifier; a resource that is itself symbolic is spelled the
            # way the compiler spells it (see :meth:`ModifierState.peel_modifiers`).
            resource_counts_by_id = {
                GraphOpID(op).getGraphOpId(): count for op, count in resources.gate_counts.items()
            }
            collected_rules.append(
                CollectedRule(
                    source_rule=rule,
                    resource_counts_by_op=resources.gate_counts,
                    resource_counts_by_id=resource_counts_by_id,
                )
            )
        except Exception as e:  # pylint: disable=broad-except
            warnings.warn(
                f"Failed to get resources for the {rule.name} decomposition rule: {e}",
                category=RuleLoweringWarning,
            )
    return collected_rules, probe_args


def build_registered_decomp_target_spec(
    op_name: str,
    op_id: SerializedGraphOpID,
    dynamic_shape,
    wire_lens,
    static_data,
    extra_data=None,
    is_custom_op=False,
    op_cls=None,
    *,
    modifiers,
) -> DecompTargetSpec | None:
    """Prepare rules registered against ``Adjoint(op_name)``/``C(op_name)`` that follow
    PennyLane's symbolic-argument convention.

    These rules take a base operator instance rather than the base op's parameters, so the rule
    body rebuilds the operator from its own traced arguments.

    An adjointed target is decomposed by the ``Adjoint(op)`` rules, and any controls are wrapped
    around each rule body to synthesize the composed modifier ``C(Adjoint(op))``. This reduces the
    adjoint under control (``C(Adjoint(RZ)) -> C(RZ)``), which the registered ``C(op)`` rules then
    terminate (``C(RZ) -> CRZ``) -- something plain distribution cannot reach, since a registered
    ``C(op)`` rule reads op-specific attributes and cannot take an adjoint base. A target that is
    only controlled is decomposed by the ``C(op)`` rules, which take the controls as arguments.

    Args:
        op_name (str): the base operator's name
        op_id (SerializedGraphOpID): the base operator's graphOpId
        dynamic_shape (dict): dynamic argument names to their MLIR types
        wire_lens (dict): wire argument names to their lengths
        static_data (dict): compiler-static argument names to their values
        extra_data (dict): argument values the graphOpId identifies by UID instead of spelling
        is_custom_op (bool): whether the operator lowers to ``qref.custom``
        op_cls (type[Operator2]): the base operator's class, required to rebuild it
        modifiers (ModifierState): the modifiers of the target the rules decompose

    Returns:
        DecompTargetSpec or None: callable specifications and metadata, or None if there are no rules

    Raises:
        ValueError: if ``op_cls`` is not given, or ``modifiers`` denotes the unmodified operator
    """
    target_id = modifiers.wrap_id(op_id)
    if not modifiers.is_modified:
        raise ValueError(f"No symbolic rules are registered against the unmodified {target_id}")
    if op_cls is None:
        raise ValueError(
            f"The operator class of {op_name!r} is needed to lower the decomposition rules "
            f"registered against {target_id}"
        )
    assert issubclass(op_cls, Operator2), f"Expected an Operator2 subclass, got {op_cls}"

    extra_data = extra_data or {}
    static_and_extra = static_data | extra_data
    kwargs = prepare_dynamic_op_kwargs(dynamic_shape, wire_lens)
    n_base_wires = sum(wire_lens.values())
    kind = "adjoint" if modifiers.adjoint else "control"
    controls = modifiers.without_adjoint()
    wrap_control = modifiers.adjoint and controls.is_controlled
    n_wires = n_base_wires + controls.control_count
    signature_wire_lens = (
        {**wire_lens, CTRL_WORK_WIRES: controls.num_work_wires}
        if controls.num_work_wires
        else wire_lens
    )

    # A kind="control" rule takes the control and work wires as its own arguments; a wrap_control
    # adjoint rule instead has the control applied around its body so it is probed without them.
    is_control = kind == "control"
    collect_ctrl_wires = range(n_base_wires, n_wires) if is_control else ()
    collect_ctrl_values = [True] * len(collect_ctrl_wires)
    collected_rules, probe_args = collect_symbolic_resources(
        op_cls,
        op_name,
        kwargs | static_and_extra,
        is_custom_op,
        kind=kind,
        ctrl_wires=collect_ctrl_wires,
        ctrl_values=collect_ctrl_values,
        work_wires=range(n_wires, n_wires + controls.num_work_wires) if is_control else (),
        work_wire_type=controls.work_wire_type,
    )
    if not collected_rules:
        return None

    # Rewrite the ids for an adjoint target:
    #  - adjoint_rotation/self_adjoint folds Adjoint(X) to X, so declare X in its source spelling.
    #       This matches the op the rule body emits at apply time (f64), not the tensor<1xf64> the
    #       abstract-probe resource carries, so the solver does not key a rule on an unproduced
    #       spelling.
    #  - Under ctrl, every other produced op additionally gains the control modifier.
    if kind == "adjoint":
        base_id = GraphOpID(probe_args["base"]).getGraphOpId()
        for collected in collected_rules:
            ids = collected.resource_counts_by_id
            if _adjoint_folds_to_base(ids, base_id):
                collected.resource_counts_by_id = {controls.wrap_id(op_id): 1}
            elif wrap_control:
                collected.resource_counts_by_id = {
                    controls.wrap_id(rid): count for rid, count in ids.items()
                }

    call_args, call_kwargs = prepare_dummy_rule_arguments(kwargs, is_custom_op)
    kwarg_names = ordered_kwarg_names(call_kwargs, dynamic_shape)

    # Rebuild hybrid arguments containing parameters or wires from operands.
    # Fully static arguments are closed over.
    hybrid_specs, hybrid_leaves, closed_hybrid = flatten_hybrid_args(extra_data)

    def rule_to_callable(rule: DecompositionRule) -> Callable[..., None]:
        def decomp_rule(*_operands) -> None:
            _args, _kwargs, _ctrl_wires, _ctrl_values, _work_wires = unpack_rule_operands(
                _operands,
                len(call_args),
                kwarg_names,
                signature_wire_lens,
                modifiers.is_controlled,
                hybrid_specs,
            )
            # The base is rebuilt from the traced arguments of the rule function, so the wires and
            # parameters the rule body reads off it are this function's own operands. Operand-passed
            # hybrid args are rebuilt into ``_kwargs``; static data and closed-over hybrid args are
            # supplied directly.
            # Do not bind this carrier instance as an operation in the rule trace.
            with qp.capture.pause():
                base = op_cls(*_args, **_kwargs, **static_data, **closed_hybrid)
            # TODO: Call the rule itself instead of its _impl after merging
            # https://github.com/PennyLaneAI/pennylane/pull/10144
            if wrap_control:
                # Reduce the adjoint under control: C(Adjoint(op)) -> C(<adjoint decomposition>).
                controls.wrap_callable(
                    lambda: rule._impl(**symbolic_arguments(base, kind, None)),
                    _ctrl_wires,
                    _ctrl_values,
                    _work_wires,
                )()
            else:
                rule._impl(
                    **symbolic_arguments(
                        base, kind, _ctrl_wires, _ctrl_values, _work_wires, controls.work_wire_type
                    )
                )

        decomp_rule.__name__ = rule.name
        return decomp_rule

    rule_specs = []
    for collected in collected_rules:
        rule = collected.source_rule

        # A rule whose body is not region-wrapped (plain adjoint/control) may legitimately
        # contain a measurement and will be kept. Otherwise, the rule is skipped because
        # the compiler cannot place a mid-circuit measurement in a control or adjoint region.
        if wrap_control and _resources_have_measurement(collected.resource_counts_by_op):
            warnings.warn(
                f"Skipped the {rule.name} decomposition rule for {target_id}: it contains a "
                "mid-circuit measurement, which cannot be placed in a control region.",
                category=RuleLoweringWarning,
            )
            continue
        rule_specs.append(
            DecompRuleSpec(
                prepared_rule_fn=rule_to_callable(rule),
                frontend_name=rule.name,
                resource_counts_by_id=collected.resource_counts_by_id,
                resource_counts_by_op=collected.resource_counts_by_op,
            )
        )

    if not rule_specs:
        return None

    ctrl_wires = (
        jnp.array(range(n_base_wires, n_wires), dtype=int) if modifiers.is_controlled else None
    )
    ctrl_values = jnp.ones(controls.control_count, dtype=bool) if modifiers.is_controlled else None
    if controls.num_work_wires:
        call_kwargs = {
            **call_kwargs,
            CTRL_WORK_WIRES: _dummy_work_wires(n_wires, controls.num_work_wires),
        }

    return DecompTargetSpec(
        rules=rule_specs,
        trace_inputs=RuleTraceInputs(
            call_args=call_args,
            call_kwargs=call_kwargs,
            kwarg_names=kwarg_names,
            wire_lens=signature_wire_lens,
            ctrl_wires=ctrl_wires,
            ctrl_values=ctrl_values,
            hybrid_leaves=tuple(hybrid_leaves),
        ),
        target_id=target_id,
        num_qreg_qubits=n_wires + controls.num_work_wires,
        modifier_context=controls if wrap_control else ModifierState(),
        reject_noninvertible=wrap_control,
    )


def _append_decomp_target_spec(out, description, factory):
    """Append one optional target specification, reporting exclusions consistently."""

    try:
        target_spec = factory()
        if target_spec is not None:
            out.append(target_spec)
    except Exception as exc:  # pylint: disable=broad-except
        warnings.warn(
            f"Failed to {description}: {exc}",
            category=RuleLoweringWarning,
        )


def build_decomp_rule_variants(
    request: OpDecompRequest, modifier_states: set[ModifierState]
) -> list[DecompTargetSpec]:
    """Build the requested modifier variants for one base identity.

    Registered symbolic rules describe their produced modifiers directly on their resource
    operators. Distributed rules instead inherit the target's modifier context, recorded on the
    resulting target specification for use by the reachability walk.
    """

    common: _RuleBuilderArgs = {
        "op_name": request.op_name,
        "op_id": request.base_id,
        "dynamic_shape": request.dynamic_shape,
        "wire_lens": request.wire_lens,
        "static_data": request.static_data,
        "extra_data": request.extra_data,
        "is_custom_op": request.is_custom_op,
        "op_cls": request.op_cls,
    }
    target_specs = []
    ordered_states = sorted(
        modifier_states,
        key=lambda s: (s.control_count, s.num_work_wires, s.work_wire_type, s.adjoint),
    )
    for state in ordered_states:
        target_id = state.wrap_id(request.base_id)
        if state.is_modified and request.op_cls is not None:
            _append_decomp_target_spec(
                target_specs,
                f"lower the registered decomposition rules for {target_id}",
                lambda: build_registered_decomp_target_spec(**common, modifiers=state),
            )
        _append_decomp_target_spec(
            target_specs,
            f"synthesize the distributed decomposition rules for {target_id}",
            lambda: build_decomp_target_spec(**common, modifiers=state),
        )

    return target_specs
