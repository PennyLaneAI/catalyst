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


"""Discover decomposition rules and prepare batches for tracing.

This module owns registry lookup, resource collection, argument preparation, and
reachability traversal. Both QNode capture and standalone compilation consume its
``DecompTargetSpec`` batches. Operand packing runs in the consumer's tracing context.
"""

# pylint: disable=protected-access,bare-except

import itertools
import warnings
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from jax import Array
from pennylane.core.operator import Operator2, abstractify
from pennylane.decomposition import DecompositionRule
from pennylane.decomposition.utils import to_name
from pennylane.wires import Wires

from catalyst.decomposition.capture_session import OpDecompRequest
from catalyst.decomposition.graph_op_id import (
    CTRL_WORK_WIRE_TYPE,
    CTRL_WORK_WIRES,
    GraphOpID,
)
from catalyst.decomposition.modifier_state import ModifierState
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.decomposition.type_utils import (
    SerializedGraphOpID,
    _RuleBuilderArgs,
    get_dummy_values_for_dynamic_shape,
)
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


@dataclass(kw_only=True)
class RuleTraceInputs:
    """Dummy arguments and operand layout shared by a target's prepared rules.

    :func:`rule_call_operands` packs these inputs as positional and named parameters, numeric
    hybrid leaves, one grouped base-wire operand, then optional control wires and values.
    Packing creates JAX arrays, so it must happen in the caller's tracing context.

    ``ctrl_wires`` and ``ctrl_values`` must be supplied together with matching lengths,
    or both be ``None``.

    Attributes:
        call_args: Positional dummy arguments used to trace custom-operation rules.
        call_kwargs: Named dummy parameters and wires used to trace general rules.
        kwarg_names: Stable parameter-first, wire-second ordering for flattened keyword operands.
        wire_lens: Qubit argument lengths, in operand order, used to split the grouped wire
            operand.
        ctrl_wires: Optional dummy control-wire tensor for a controlled target.
        ctrl_values: Optional dummy control-value tensor for a controlled target.
        hybrid_leaves: Numeric hybrid leaves passed as rule operands instead of being closed over.
    """

    call_args: tuple[Array | np.float64, ...]
    call_kwargs: dict[str, Array]
    kwarg_names: list[str]
    wire_lens: dict[str, int]
    ctrl_wires: Array | None = None
    ctrl_values: Array | None = None
    hybrid_leaves: tuple[Array, ...] = ()


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


def build_base_op(op_cls, kwargs, is_custom_op):
    """Instantiate the base operator of a PennyLane's symbolic rule from prepared rule kwargs.

    Note that we have to pause capture here, because the base operator is only a carrier of
    the traced parameters and wires, and the rule body is what decides whether (and how) it
    gets applied.

    Args:
        op_cls (type): the base operator's class
        kwargs (dict): the prepared rule kwargs
        is_custom_op (bool): whether the operator lowers to ``qref.custom``

    Returns:
        Operator2: the base operator, built without binding its primitive
    """
    args, kwargs = prepare_dummy_rule_arguments(kwargs, is_custom_op)
    with qp.capture.pause():
        return op_cls(*args, **kwargs)


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


def symbolic_arguments(
    base_op, kind, ctrl_wires=None, ctrl_values=None, work_wires=None, work_wire_type="borrowed"
) -> dict:
    """Return the arguments a rule registered against a symbolic operator expects.

    PennyLane calls such a rule with the symbolic operator's own ``arguments``: ``base`` alone for
    an adjoint, and the five entries below for a controlled operator. Control values are plain
    Python booleans so that a rule wrapped in ``flip_zero_control`` statically resolves to no ``X``
    flips, matching the all-ones ``ctrlvals`` a ``<n>C(...)`` graphOpId denotes.

    Args:
        base_op (Operator2): the base operator the rule acts on
        kind (str): ``"adjoint"`` or ``"control"``
        ctrl_wires: the control wires, for ``kind="control"``
        ctrl_values: the control values, for ``kind="control"``
        work_wires: the work wires available to the controls, for ``kind="control"``
        work_wire_type (str): the type of those work wires, ``"zeroed"`` or ``"borrowed"``

    Returns:
        dict: the arguments to call the rule with

    Raises:
        CompileError: if ``kind`` is not a known symbolic kind
    """
    if kind == "adjoint":
        return {"base": base_op}
    if kind == "control":
        assert ctrl_wires is not None
        assert ctrl_values is not None
        ctrl_wires = Wires(ctrl_wires)
        work_wires = Wires([] if work_wires is None else work_wires)
        return {
            "base": base_op,
            "control_wires": ctrl_wires,
            "control_values": ctrl_values,
            "work_wires": work_wires,
            "work_wire_type": work_wire_type if work_wires else "borrowed",
        }
    raise CompileError(f"Unknown symbolic kind: {kind}")  # pragma: no cover


def ordered_kwarg_names(call_kwargs, dynamic_shape) -> list[str]:
    """Order a rule's keyword operands params-first, followed by wires.

    The params keep the order of ``dynamic_shape``, which is the operand order of the operator, so
    that the rule's inputs line up with the operands the compiler passes to it. They must not be
    sorted.
    """
    params = [name for name in dynamic_shape if name in call_kwargs]
    wires = [name for name in call_kwargs if name not in dynamic_shape]
    return params + wires


class HybridLeafSource(Enum):
    """Where a reconstructed hybrid argument gets each leaf."""

    ORDINARY_PARAMETER = auto()
    OPERATOR_PARAMETER = auto()
    WIRE = auto()
    STATIC = auto()


@dataclass
class HybridArgumentSpec:
    """Describe how to rebuild a hybrid argument from rule inputs."""

    name: str
    argument_tree: object
    original_leaves: list
    leaf_sources: list[HybridLeafSource]

    def rebuild(self, ordinary_parameters, operator_parameters, wires):
        """Replace abstract leaves with the rule's parameter and wire inputs."""
        reconstructed_leaves = []
        wire_offset = 0
        for leaf, source in zip(self.original_leaves, self.leaf_sources, strict=True):
            if source == HybridLeafSource.ORDINARY_PARAMETER:
                reconstructed_leaves.append(next(ordinary_parameters))
            elif source == HybridLeafSource.OPERATOR_PARAMETER:
                reconstructed_leaves.append(next(operator_parameters))
            elif source == HybridLeafSource.WIRE:
                wire_slice = wires[wire_offset : wire_offset + len(leaf)]
                reconstructed_leaves.append(Wires(wire_slice))
                wire_offset += len(leaf)
            else:
                reconstructed_leaves.append(leaf)

        # Constructing the carrier object must not apply another gate.
        with qp.capture.pause():
            return qp.pytrees.unflatten(reconstructed_leaves, self.argument_tree)


def _is_wire_leaf(value):
    return isinstance(value, (Wires, qp.typing.AbstractWires))


def flatten_hybrid_args(extra_data) -> tuple:
    """Record how to rebuild hybrid arguments from a decomposition rule's inputs.

    Track each argument's parameters, wires, and fixed values so nested operators can be
    rebuilt with the supplied inputs. Create dummy numeric arrays for tracing; dummy wires
    are created separately by ``prepare_dynamic_op_kwargs``.

    Args:
        extra_data (dict): Hybrid arguments with dynamic values already abstractified.

    Returns:
        tuple: Reconstruction specs, dummy numeric arrays, and fully static arguments.
        Ordinary parameters come before nested-operator parameters in the dummy arrays.

    Example:
        Here, ``dummies`` contains the data array followed by the RX angle. The RX wire
        is recorded in ``specs`` and supplied separately when rebuilding the operator.

        >>> extra_data = {
        ...     "unitary": abstractify(qp.RX(0.7, wires=2)),
        ...     "data": qp.typing.Float[2],
        ...     "label": "rotation",
        ... }
        >>> specs, dummies, static = flatten_hybrid_args(extra_data)
        >>> {spec.name: [source.name for source in spec.leaf_sources] for spec in specs}
        {'unitary': ['OPERATOR_PARAMETER', 'WIRE'], 'data': ['ORDINARY_PARAMETER']}
        >>> [dummy.shape for dummy in dummies]
        [(2,), ()]
        >>> static
        {'label': 'rotation'}
    """
    argument_specs = []
    ordinary_parameter_dummies = []
    operator_parameter_dummies = []
    static_arguments = {}

    numeric_types = (np.ndarray, np.generic, jnp.ndarray, qp.typing.AbstractArray)

    for argument_name, argument in extra_data.items():
        # Keep the complete structure so we can rebuild the argument.
        original_leaves, argument_tree = qp.pytrees.flatten(argument, is_leaf=_is_wire_leaf)

        # Stop at operators to identify which parameters belong to them.
        argument_parts, _ = qp.pytrees.flatten(
            argument,
            is_leaf=lambda value: isinstance(value, Operator2) or _is_wire_leaf(value),
        )

        leaf_sources = []
        for argument_part in argument_parts:
            is_operator = isinstance(argument_part, Operator2)
            part_leaves, _ = qp.pytrees.flatten(argument_part, is_leaf=_is_wire_leaf)

            for leaf in part_leaves:
                if isinstance(leaf, numeric_types):
                    dummy_parameter = jnp.zeros(leaf.shape, dtype=leaf.dtype)
                    if is_operator:
                        source = HybridLeafSource.OPERATOR_PARAMETER
                        operator_parameter_dummies.append(dummy_parameter)
                    else:
                        source = HybridLeafSource.ORDINARY_PARAMETER
                        ordinary_parameter_dummies.append(dummy_parameter)
                elif _is_wire_leaf(leaf):
                    source = HybridLeafSource.WIRE
                else:
                    source = HybridLeafSource.STATIC

                leaf_sources.append(source)

        if any(source != HybridLeafSource.STATIC for source in leaf_sources):
            argument_specs.append(
                HybridArgumentSpec(
                    name=argument_name,
                    argument_tree=argument_tree,
                    original_leaves=original_leaves,
                    leaf_sources=leaf_sources,
                )
            )
        else:
            static_arguments[argument_name] = argument

    # Match gate lowering: ordinary parameters precede operator parameters.
    dummy_parameters = ordinary_parameter_dummies + operator_parameter_dummies
    return argument_specs, dummy_parameters, static_arguments


def rule_call_operands(trace_inputs: RuleTraceInputs) -> list[Array | np.float64]:
    """Pack dummy inputs into the positional operand layout used to trace a rule.

    The order is positional parameters, named parameters, numeric hybrid leaves, one grouped
    base-wire array, then optional control wires and values. Named parameters follow
    ``trace_inputs.kwarg_names``; wire groups follow ``trace_inputs.wire_lens`` insertion order.
    Control work wires are included as a ``ctrl_work_wires`` register in the grouped
    wire array. The grouped wire array is present even when empty.

    Packing creates JAX arrays and must run in the caller's tracing context. All operands are
    positional so ``jax.jit`` (used by ``qp.capture.subroutine``) cannot reorder keyword arguments
    alphabetically.

    Args:
        trace_inputs (RuleTraceInputs): the rule's dummy arguments and operand layout

    Returns:
        list: positional operands in the layout consumed by :func:`unpack_rule_operands`
    """
    param_names = [name for name in trace_inputs.kwarg_names if name not in trace_inputs.wire_lens]
    wire_names = list(trace_inputs.wire_lens)
    grouped_wires = (
        jnp.concatenate(tuple(trace_inputs.call_kwargs[name] for name in wire_names))
        if wire_names
        else jnp.array([], dtype=int)
    )
    operands = [
        *trace_inputs.call_args,
        *(trace_inputs.call_kwargs[name] for name in param_names),
        *trace_inputs.hybrid_leaves,
        grouped_wires,
    ]
    if trace_inputs.ctrl_wires is not None:
        operands.append(trace_inputs.ctrl_wires)
    if trace_inputs.ctrl_values is not None:
        operands.append(trace_inputs.ctrl_values)
    return operands


def unpack_rule_operands(
    operands,
    n_params,
    kwarg_names,
    wire_lens,
    has_ctrl_wires,
    hybrid_specs=(),
):
    """Recover params, kwargs, control wires, control values, and work wires from
    :func:`rule_call_operands`' flattening.

    Reconstructs each hybrid argument from its traced parameters and wires so the rule body
    receives the same object the user passed, but built from the traced operands. A
    ``ctrl_work_wires`` group is taken back out of the grouped wires: it belongs to the control
    modifier, not to the base operator the rule body rebuilds.

    Args:
        operands (tuple): the traced operands the rule body received
        n_params (int): how many of them are positional parameters
        kwarg_names (list[str]): parameter names first, followed by wire names
        wire_lens (dict[str, int]): wire names in operand order and their lengths
        has_ctrl_wires (bool): whether trailing control-wire and control-value operands are present
        hybrid_specs (list[HybridArgumentSpec]): reconstruction specs for hybrid arguments whose
            numeric leaves sit between the parameter kwargs and the grouped wire operand

    Returns:
        tuple: the rule's positional arguments
        dict: the rule's keyword arguments (including the rebuilt hybrid arguments)
        the control wires, or None
        the control values, or None
        the control work wires, or None
    """
    params = tuple(operands[:n_params])
    param_names = [name for name in kwarg_names if name not in wire_lens]
    n_kwarg_params = len(param_names)
    named = dict(zip(param_names, operands[n_params : n_params + n_kwarg_params], strict=True))
    idx = n_params + n_kwarg_params
    num_ordinary_parameters = sum(
        spec.leaf_sources.count(HybridLeafSource.ORDINARY_PARAMETER) for spec in hybrid_specs
    )
    num_operator_parameters = sum(
        spec.leaf_sources.count(HybridLeafSource.OPERATOR_PARAMETER) for spec in hybrid_specs
    )
    ordinary_parameters = iter(operands[idx : idx + num_ordinary_parameters])
    idx += num_ordinary_parameters
    operator_parameters = iter(operands[idx : idx + num_operator_parameters])
    idx += num_operator_parameters

    grouped_wires = operands[idx]
    idx += 1
    offset = 0
    for name, length in wire_lens.items():
        named[name] = grouped_wires[offset : offset + length]
        offset += length

    for spec in hybrid_specs:
        named[spec.name] = spec.rebuild(
            ordinary_parameters, operator_parameters, wires=named.get(spec.name, ())
        )

    work_wires = named.pop(CTRL_WORK_WIRES, None)
    ctrl_wires = operands[idx] if has_ctrl_wires else None
    ctrl_values = operands[idx + 1] if has_ctrl_wires and len(operands) > idx + 1 else None
    if has_ctrl_wires:
        assert len(ctrl_values) == len(
            ctrl_wires
        ), "number of control wires and control values must be the same"
    return params, named, ctrl_wires, ctrl_values, work_wires


def prepare_dummy_rule_arguments(kwargs, is_custom_op: bool):
    """Prepare dummy positional and keyword arguments for a rule or resource function.

    Custom-op dynamic params are keyed positionally ("0", "1", ...) but the rule callables expect
    them by their real argnames, so they are passed positionally with only "wires" kept as keyword.
    Custom-op params are always scalar f64, so scalar ``0.0`` dummies are used (a shape-(1,) array
    would instead lower the gate through the general ``qref.operator`` path rather than ``qref.custom``).

    Args:
        kwargs (dict): the prepared rule kwargs
        is_custom_op (bool): whether the operator lowers to ``qref.custom``

    Returns:
        tuple: the positional arguments to call the rule with
        dict: the keyword arguments to call the rule with
    """
    if is_custom_op:
        # stand-in values with shape & dtype information
        args = tuple(np.float64(0.0) for key in kwargs if key != "wires")
        return args, {"wires": kwargs["wires"]}
    return (), kwargs


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


def prepare_dynamic_op_kwargs(dynamic_shape, wire_lens) -> dict[str, Array]:
    """Build the dummy arguments an operator's decomposition rules are called with.

    Args:
        dynamic_shape (dict): dynamic argument names to their MLIR types
        wire_lens (dict): wire argument names to their lengths

    Returns:
        dict: argument names to dummy values
    """
    kwargs = {}

    # NOTE: Run an accumulator to generate unique negative wire labels.
    # Otherwise, operators with control and target wires as their arguments
    # will receive overlapping wires.
    wire_counter = itertools.count(-1, -1)
    for wire_name, wire_len in wire_lens.items():
        kwargs[wire_name] = jnp.array([next(wire_counter) for _ in range(wire_len)], dtype=int)
    for arg_name, arg_shape in dynamic_shape.items():
        kwargs[arg_name] = get_dummy_values_for_dynamic_shape(arg_shape)
    return kwargs


def _dummy_work_wires(offset, num_work_wires):
    """Return the control work-wire operand trailing ``offset`` rule wires, or None without any."""
    if not num_work_wires:
        return None
    return jnp.array(range(offset, offset + num_work_wires), dtype=int)


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


def walk_reachable_decomp_rule_sets(
    initial_requests: list[tuple[OpDecompRequest, set[ModifierState]]],
) -> list[DecompTargetSpec]:
    """Build every reachable decomposition rule set in one contextual worklist traversal.

    Every modifier state is demand-driven and propagated through resource edges. Revisiting an
    identity processes only states not handled by an earlier queue entry.
    """

    queue = deque(
        (
            request,
            set(required_states),
            {
                request.base_id: (
                    (
                        request.modifier_state.control_count,
                        request.modifier_state.num_work_wires,
                    ),
                )
            },
        )
        for request, required_states in initial_requests
    )

    processed_states: dict[SerializedGraphOpID, set[ModifierState]] = {}
    target_specs = []

    def child_requests(rule_sets):
        children = []
        for target_spec in rule_sets:
            context = target_spec.modifier_context
            for rule in target_spec.rules:
                for op in rule.resource_counts_by_op:
                    if isinstance(op, Operator2):
                        children.append(OpDecompRequest.from_operation(op, context))
        return children

    while queue:
        request, demanded_states, path_control_counts = queue.popleft()
        demanded_states = set(demanded_states)
        previous_states = processed_states.setdefault(request.base_id, set())
        missing_states = demanded_states - previous_states
        if not missing_states:
            continue

        previous_states.update(missing_states)
        new_specs = build_decomp_rule_variants(request, missing_states)
        target_specs.extend(new_specs)

        for child in child_requests(new_specs):
            # A modifier-changing cycle can otherwise manufacture an unbounded sequence of control
            # and control work-wire counts, e.g. C(SemiAdder) passes its spare work wires on to
            # ctrl(CNOT). Compare with every state of the same base identity earlier on this path,
            # rather than the direct parent: intermediate operators may absorb their controls into
            # their identity, or trade a control for a work wire. Prune a revisit whose counts are
            # all at least those of an earlier controlled state and differ from it; any infinite
            # sequence of count pairs contains such a revisit, so every path is finite. The first
            # uncontrolled-to-controlled revisit remains valid, and an equal revisit is left to
            # ``processed_states``.
            child_counts = (child.modifier_state.control_count, child.modifier_state.num_work_wires)
            earlier_counts = path_control_counts.get(child.base_id, ())
            if any(
                controls > 0
                and child_counts != (controls, work_wires)
                and child_counts[0] >= controls
                and child_counts[1] >= work_wires
                for controls, work_wires in earlier_counts
            ):
                continue

            child_path_control_counts = path_control_counts.copy()
            child_path_control_counts[child.base_id] = (*earlier_counts, child_counts)
            queue.append((child, {child.modifier_state}, child_path_control_counts))

    return target_specs
