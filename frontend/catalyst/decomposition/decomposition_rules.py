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

import itertools
import warnings
from collections import deque

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from jax._src.lib.mlir import ir
from jax.tree_util import tree_flatten, tree_unflatten
from pennylane.core.operator import Operator2, abstractify
from pennylane.decomposition.utils import to_name
from pennylane.wires import Wires

from catalyst.compiler import _quantum_opt
from catalyst.decomposition.graph_op_id import GraphOpID
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.decomposition.type_utils import get_dummy_values_for_arg
from catalyst.jax_extras.lowering import get_mlir_attribute_from_pyval
from catalyst.utils.exceptions import CompileError

# Ops that make a decomposition body non-invertible
_NON_INVERTIBLE_MARKERS = (
    "qref.measure",
    "quantum.measure",
    "measure_in_basis",
    ".ppm",  # pbc.ppm / pbc.ref.ppm / pbc.select.ppm
)

_NON_INVERTIBLE_RESOURCE_TYPES = (qp.ops.MidMeasure, qp.ops.PauliMeasure)


def _resources_have_measurement(gate_counts) -> bool:
    """Return whether a rule's declared resources contain a mid-circuit measurement."""
    return any(isinstance(op, _NON_INVERTIBLE_RESOURCE_TYPES) for op in gate_counts)


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
    args, kwargs = split_call_args(kwargs, is_custom_op)
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


def symbolic_arguments(base_op, kind, ctrl_wires=None) -> dict:
    """Return the arguments a rule registered against a symbolic operator expects.

    PennyLane calls such a rule with the symbolic operator's own ``arguments``: ``base`` alone for
    an adjoint, and the five entries below for a controlled operator. Control values are plain
    Python booleans so that a rule wrapped in ``flip_zero_control`` statically resolves to no ``X``
    flips, matching the all-ones ``ctrlvals`` a ``<n>C(...)`` graphOpId denotes.

    Args:
        base_op (Operator2): the base operator the rule acts on
        kind (str): ``"adjoint"`` or ``"control"``
        ctrl_wires: the control wires, for ``kind="control"``

    Returns:
        dict: the arguments to call the rule with

    Raises:
        CompileError: if ``kind`` is not a known symbolic kind
    """
    if kind == "adjoint":
        return {"base": base_op}
    if kind == "control":
        # FIXME: the all-ones assumption is unsound for an operator that carries a zero control
        # value. A `<n>C(...)` graphOpId records only the control count, so such an operator maps
        # to the same node as an all-ones one and the solver hands it this rule, which was traced
        # without the `X` flips the zero value needs. It's silently giving the wrong state.
        #
        # The values cannot be spelled in the id (they may be dynamic), so the fix is to normalize
        # them away where they are known, by emitting the flips around the operator during capture
        # so that every controlled operator the decomposition sees really does denote all-ones
        # controls.
        ctrl_wires = Wires(ctrl_wires)
        return {
            "base": base_op,
            "control_wires": ctrl_wires,
            "control_values": [True] * len(ctrl_wires),
            "work_wires": Wires([]),
            "work_wire_type": "borrowed",
        }
    raise CompileError(f"Unknown symbolic kind: {kind}")  # pragma: no cover


def ordered_kwarg_names(call_kwargs, dynamic_shape) -> list:
    """Order a rule's keyword operands params-first, followed by wires in declaration order."""
    params = sorted(name for name in call_kwargs if name in dynamic_shape)
    wires = [name for name in call_kwargs if name not in dynamic_shape]
    return params + wires


def flatten_hybrid_args(extra_data) -> tuple:
    """Split hybrid arguments into numeric ones (passed as operands) and the rest (closed over).

    A hybrid argument whose pytree is entirely numeric arrays (e.g. a ``CDFHamiltonian``) is
    flattened so its leaves can be passed into the rule call as operands, keeping the concrete
    values out of the rule body (where they would otherwise bake in as constants).

    Args:
        extra_data (dict): the operator's hybrid arguments, keyed by name

    Returns:
        list[tuple]: ``(name, treedef, num_leaves)`` per operand-passed hybrid argument
        list: the flat leaf dummies (shape/dtype of the concrete leaves), to pass as operands
        dict: the hybrid arguments to close over rather than pass as operands, keyed by name
    """
    specs, leaf_dummies, closed_over = [], [], {}
    for name, value in extra_data.items():
        leaves, treedef = tree_flatten(value)
        # TODO: this "all leaves are numeric arrays" test is a second source of truth for which
        # hybrid leaves become rule-function inputs, and it must agree with the operator lowering's
        # own partition (`_process_params` in from_plxpr/qref_operator2_primitives.py), which uses a
        # per-leaf `forward_mask` to split leaves into `param_map` vs `forward_args` (excluding
        # qubits).
        #
        # Note right now the two agree for all-numeric args (numeric hamiltonians) and for args
        # with a non-array leaf (an operator carrying Wires), but a mixed hybrid arg would diverge
        # causing the lowered op's params to outnumber the rule's inputs.
        # Drive this off `_process_params`/`forward_mask` instead, once the operator's mask is
        # passed through to rule compilation.
        if leaves and all(
            isinstance(leaf, (np.ndarray, np.generic, jnp.ndarray)) for leaf in leaves
        ):
            specs.append((name, treedef, len(leaves)))
            leaf_dummies.extend(
                jnp.zeros(jnp.shape(leaf), dtype=jnp.asarray(leaf).dtype) for leaf in leaves
            )
        else:
            closed_over[name] = value
    return specs, leaf_dummies, closed_over


def rule_call_operands(
    call_args, call_kwargs, kwarg_names, wire_lens, ctrl_wires=None, hybrid_leaves=()
) -> list:
    """Flatten a rule call as params, operand-passed hybrid-arg leaves, one grouped base-wire
    operand, then control wires.

    Everything is passed positionally because ``qp.capture.subroutine`` traces through ``jax.jit``,
    which would otherwise flatten keyword arguments in sorted-name order.

    Args:
        call_args (tuple): the rule's positional arguments
        call_kwargs (dict): the rule's keyword arguments
        kwarg_names (list[str]): parameter names first, followed by wire names
        wire_lens (dict[str, int]): wire names in declaration order and their lengths
        ctrl_wires: the control wires to append, or None when the rule is not controlled
        hybrid_leaves (list): flattened hybrid-argument leaves, inserted after the parameter kwargs

    Returns:
        list: the operands to call the traced rule with
    """
    param_names = [name for name in kwarg_names if name not in wire_lens]
    wire_names = list(wire_lens)
    grouped_wires = (
        jnp.concatenate(tuple(call_kwargs[name] for name in wire_names))
        if wire_names
        else jnp.array([], dtype=int)
    )
    operands = [
        *call_args,
        *(call_kwargs[name] for name in param_names),
        *hybrid_leaves,
        grouped_wires,
    ]
    if ctrl_wires is not None:
        operands.append(ctrl_wires)
    return operands


def unpack_rule_operands(
    operands, n_params, kwarg_names, wire_lens, has_ctrl_wires, hybrid_specs=()
):
    """Recover ``(params, kwargs, control wires)`` from :func:`rule_call_operands`' flattening.

    Reconstructs each hybrid argument from its traced leaves (via its ``treedef``) so the rule body
    receives the same object the user passed, but built from the traced operands.

    Args:
        operands (tuple): the traced operands the rule body received
        n_params (int): how many of them are positional parameters
        kwarg_names (list[str]): parameter names first, followed by wire names
        wire_lens (dict[str, int]): wire names in declaration order and their lengths
        has_ctrl_wires (bool): whether a trailing control-wire operand is present
        hybrid_specs (list[tuple]): ``(name, treedef, num_leaves)`` per hybrid argument, sitting
            between the parameter kwargs and the grouped wire operand

    Returns:
        tuple: the rule's positional arguments
        dict: the rule's keyword arguments (including the rebuilt hybrid arguments)
        the control wires, or None
    """
    params = tuple(operands[:n_params])
    param_names = [name for name in kwarg_names if name not in wire_lens]
    n_kwarg_params = len(param_names)
    named = dict(zip(param_names, operands[n_params : n_params + n_kwarg_params], strict=True))
    idx = n_params + n_kwarg_params
    for name, treedef, num_leaves in hybrid_specs:
        named[name] = tree_unflatten(treedef, operands[idx : idx + num_leaves])
        idx += num_leaves

    grouped_wires = operands[idx]
    idx += 1
    offset = 0
    for name, length in wire_lens.items():
        named[name] = grouped_wires[offset : offset + length]
        offset += length

    ctrl_wires = operands[idx] if has_ctrl_wires else None
    return params, named, ctrl_wires


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

    Args:
        op_id (str): the graphOpId to wrap
        modifier (str): the modifier token, e.g. ``"Adjoint"``, ``"C"`` or ``"2C"``

    Returns:
        str: the modified graphOpId

    Raises:
        ValueError: if the modifier would nest inside one that is canonically outer
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
    """Inverse of :func:`name_wrap_adjoint` given the base ``op_name``.

    Args:
        op_name (str): the base operator's name
        op_id (str): the adjoint graphOpId to unwrap

    Returns:
        str: the base operator's graphOpId

    Raises:
        ValueError: if ``op_id`` is not an adjoint id for ``op_name``
    """
    prefix = f"Adjoint({op_name})"
    if not op_id.startswith(prefix):
        raise ValueError(f"{op_id!r} is not an adjoint id for base op {op_name!r}")
    return op_name + op_id[len(prefix) :]


def name_unwrap_control(op_name: str, op_id: str):
    """Inverse of control name-wrapping given the base ``op_name``.

    ``("RX", "2C(RX){...}")`` -> ``("RX{...}", 2)`` and ``("RX", "C(RX){...}")`` -> ``("RX{...}", 1)``.

    Args:
        op_name (str): the base operator's name
        op_id (str): the controlled graphOpId to unwrap

    Returns:
        str: the base operator's graphOpId
        int: the number of controls

    Raises:
        ValueError: if ``op_id`` is not a control id for ``op_name``
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
    """Extract the decomposition rules held by a module as MLIR strings.

    Every FuncOp carrying a ``target_gate`` attribute is a decomposition rule.

    Args:
        module (ir.Module): the module a rule-compiling qjit produced

    Returns:
        list[str]: one string per rule, with the ``__builtin_`` prefix added to its name
    """
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
        module (ir.Module): an MLIR module object; every FuncOp carrying a `target_gate`
            attribute is extracted as a decomposition rule.

    Returns:
        str: The string representation of any decomposition rules from `module`, pre-pending the
             `__builtin_` prefix to their names.
    """
    funcOps = get_rule_strings_from_module(module)
    return "\n".join(str(funcOp) for funcOp in funcOps) if funcOps else ""


def inject_new_rules_into_module(module: ir.Module, decomp_rules: list[str]):
    """Add decomposition rules to a module, skipping the ones it already holds.

    A rule counts as already held when both its ``target_gate`` and its ``resources`` match.

    Args:
        module (ir.Module): the module to add the rules to
        decomp_rules (list[str]): the rules to add, as MLIR strings
    """
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

    Args:
        kwargs (dict): the prepared rule kwargs
        is_custom_op (bool): whether the operator lowers to ``qref.custom``

    Returns:
        tuple: the positional arguments to call the rule with
        dict: the keyword arguments to call the rule with
    """
    if is_custom_op:
        args = tuple(0.0 for key in kwargs if key != "wires")
        return args, {"wires": kwargs["wires"]}
    return (), kwargs


def _rule_allocates_work_wires(rule, *args, **kwargs) -> bool:
    """Whether a decomposition rule dynamically allocates work wires.

    TODO: remove when --decompose-lowering can handle multiple registers
    """
    return rule.get_work_wire_spec(*args, **kwargs).total > 0


def _rule_is_applicable(op_name, rule, *args, **kwargs) -> bool:
    """Return resource data for the decomposition rules that apply to ``op_name``."""
    try:
        if not bool(rule.is_applicable(*args, **kwargs)):
            return False
    except Exception as e:  # pylint: disable=broad-except
        warnings.warn(
            f"Excluded the {rule.name} decomposition rule for {op_name}; raised '{e}'",
            category=RuleLoweringWarning,
        )
        return False

    try:
        allocates_work_wires = _rule_allocates_work_wires(rule, *args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        warnings.warn(
            f"Could not read the work-wire spec of the {rule.name} decomposition rule for "
            f"{op_name}",
            category=RuleLoweringWarning,
        )
        allocates_work_wires = True

    if allocates_work_wires:
        warnings.warn(
            f"Excluded the {rule.name} decomposition rule for {op_name} since --decompose-lowering cannot work with multiple registers yet",
            category=RuleLoweringWarning,
        )
        return False

    return True


def collect_resources_for_op(
    op_name, kwargs, is_custom_op=False, adjoint_resources=False, num_controls=0
):
    """Return resource data for all decomposition rules associated to op_name.

    Args:
        op_name (str): the operator's name, as PennyLane registers its rules
        kwargs (dict): the arguments to compute the resources with
        is_custom_op (bool): whether the operator lowers to ``qref.custom``
        adjoint_resources (bool): whether to spell each produced id in its adjoint form
        num_controls (int): how many controls to spell on each produced id

    Returns:
        dict: rule name to the resources it produces
        dict: rule name to the graphOpId of each resource
        list: the rules that apply to the probed operator
    """
    decomp_rules = list(qp.decomposition.list_decomps(op_name))
    args, kwargs = split_call_args(kwargs, is_custom_op)

    # map each rule to its resources, in a more generic format
    name_to_resource_ids = {}
    name_to_resources = {}
    applicable_rules = []
    for rule in decomp_rules:
        if not _rule_is_applicable(op_name, rule, *args, **kwargs):
            continue

        applicable_rules.append(rule)
        try:
            # The `compute_resources` function's signature is the same as the Operator2 signature
            # for the original op of the rule
            with qp.capture.toggle_ctx(True):
                resources = rule.compute_resources(*args, **kwargs)
            name_to_resources[rule.name] = resources.gate_counts
            # A distributed rule produces modified gates, so each id is *generated* in its
            # modified form straight from the resource op instance -- the modifiers are placed
            # canonically by `build_graph_op_id`, never spliced into a finished id string.
            name_to_resource_ids[rule.name] = {
                GraphOpID(op).getGraphOpId(
                    adjoint=adjoint_resources, num_controls=num_controls
                ): count
                for op, count in resources.gate_counts.items()
            }
        except Exception as e:
            warnings.warn(
                f"Failed to get resources for the {rule.name} decomposition rule: {e}",
                category=RuleLoweringWarning,
            )

    return name_to_resources, name_to_resource_ids, applicable_rules


def prepare_dynamic_op_kwargs(dynamic_shape, wire_lens) -> dict:
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
) -> ir.Operation:
    """
    Return the top-level ``builtin.module`` operation containing the decomposition rules for an
    operator instance.

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

    Args:
        op_name (str): the operator's name, as PennyLane registers its rules
        op_id (str): the operator's graphOpId
        dynamic_shape (dict): dynamic argument names to their MLIR types
        wire_lens (dict): wire argument names to their lengths
        static_data (dict): compiler-static argument names to their values
        extra_data (dict): argument values the graphOpId identifies by UID instead of spelling
        is_custom_op (bool): whether the operator lowers to ``qref.custom``
        wrap_adjoint (bool): whether to distribute the base rules over adjoint
        wrap_control (bool): whether to distribute the base rules over control
        n_ctrl (int): the number of controls to distribute over

    Returns:
        ir.Operation: the ``builtin.module`` holding the rules
    """
    kwargs = prepare_dynamic_op_kwargs(dynamic_shape, wire_lens)
    extra_data = extra_data or {}
    n_base_wires = sum(wire_lens.values())
    device = qp.device("null.qubit", wires=n_base_wires + (n_ctrl if wrap_control else 0))

    name_to_resources, name_to_resource_ids, decomp_rules = collect_resources_for_op(
        op_name,
        kwargs | static_data | extra_data,
        is_custom_op,
        adjoint_resources=wrap_adjoint,
        num_controls=n_ctrl if wrap_control else 0,
    )

    # The *target* id is still derived by string-wrapping, because this is the one identity we are
    # given rather than holding the operator for: the compiler's on-demand loader hands us a
    # finished graphOpId string (`getGraphOpId()`), and re-generating it from the components would
    # only agree as long as both spellings stay in step. Splicing keeps whatever came in verbatim.
    # TODO: generate it as well, once the loader passes the identity components (or the modifiers)
    # instead of a pre-wrapped id.
    target_id = name_wrap_adjoint(op_id) if wrap_adjoint else op_id
    if wrap_control:
        target_id = wrap_modifier_id(target_id, _control_modifier(n_ctrl))

    call_args, call_kwargs = split_call_args(kwargs, is_custom_op)
    kwarg_names = ordered_kwarg_names(call_kwargs, dynamic_shape)

    hybrid_specs, hybrid_leaves, closed_hybrid = flatten_hybrid_args(extra_data)

    # The static_data was only needed to instantiate the correct decomp rule
    # Once we have the correct rules, don't send them into qjit: they are closed over instead.
    def rule_to_subroutine(rule):
        def decomp_rule(*_operands):
            _args, _kwargs, _ctrl_wires = unpack_rule_operands(
                _operands, len(call_args), kwarg_names, wire_lens, wrap_control, hybrid_specs
            )
            # Operand-passed hybrid args are rebuilt from operands in unpack_rule_operands; static
            # data and closed-over hybrid args are supplied directly.
            _kwargs |= static_data | closed_hybrid
            # Apply adjoint innermost, control outermost (canonical C(Adjoint(Op))).
            body = qp.adjoint(rule._impl) if wrap_adjoint else rule._impl
            if wrap_control:
                qp.ctrl(body, control=list(_ctrl_wires))(*_args, **_kwargs)
            else:
                body(*_args, **_kwargs)

        # keep the frontend name for readability, append target op_id for symbol uniqueness
        decomp_rule.__name__ = rule.name + "_" + target_id

        return qp.capture.subroutine(decomp_rule)

    subroutines = []
    for rule in decomp_rules:
        if rule.name not in name_to_resource_ids:
            continue
        if (wrap_adjoint or wrap_control) and _resources_have_measurement(
            name_to_resources[rule.name]
        ):
            warnings.warn(
                f"Skipped the {rule.name} decomposition rule for {target_id}: it contains a "
                "mid-circuit measurement, which is not supported with adjoint or control regions.",
                category=RuleLoweringWarning,
            )
            continue
        subroutines.append(rule_to_subroutine(rule))

    # For control distribution, the extra control wires trail the base wires in the rule's operands.
    ctrl_wires = (
        jnp.array(range(n_base_wires, n_base_wires + n_ctrl), dtype=int) if wrap_control else None
    )

    return build_rule_module(
        subroutines,
        device,
        call_args,
        call_kwargs,
        kwarg_names,
        wire_lens,
        ctrl_wires,
        name_to_resource_ids,
        target_id,
        hybrid_leaves,
    )


def build_rule_module(
    subroutines,
    device,
    call_args,
    call_kwargs,
    kwarg_names,
    wire_lens,
    ctrl_wires,
    name_to_resource_ids,
    target_id,
    hybrid_leaves=(),
) -> ir.Operation:
    """Trace ``subroutines`` into a module of standalone decomposition-rule functions.

    Args:
        subroutines (list): the rule bodies, as captured subroutines
        device (Device): the device to trace them on, sized for the operator's wires
        call_args (tuple): positional arguments to call each subroutine with
        call_kwargs (dict): keyword arguments to call each subroutine with
        kwarg_names (list[str]): the keyword argument names, in the order to flatten them
        wire_lens (dict[str, int]): wire names in declaration order and their lengths
        ctrl_wires: the control wires to append, or None when the rules are not controlled
        hybrid_leaves (list): flattened hybrid-argument leaves passed to :func:`rule_call_operands`
        name_to_resource_ids (dict): rule name to the graphOpId of each resource
        target_id (str): the graphOpId of the gate the rules decompose

    Returns:
        ir.Operation: the module holding the rules

    Raises:
        CompileError: if the rules could not be traced
    """

    operands = rule_call_operands(
        call_args, call_kwargs, kwarg_names, wire_lens, ctrl_wires, hybrid_leaves
    )

    @qp.qjit(target="mlir", capture=True, collect_decomp_rules=False)
    @qp.qnode(device=device)
    def circuit():
        for subroutine in subroutines:
            subroutine(*operands)

    module = circuit.mlir_module
    if module is None:
        raise CompileError(
            f"Failed to generate an MLIR module while compiling decomposition rules for {target_id}"
        )

    def update_funcop_attributes(op):
        """Update the decomposition rule attributes if op is a decomposition rule.

        For use with module.walk

        This function updates the following attributes:
            - Adds the `target_gate` attribute.
            - Adds the `resources` attribute.
            - Sets the visibility to public (so the inliner does not remove them)
        """
        if op.name == "func.func":
            rule_name = ir.StringAttr(op.attributes["sym_name"]).value.removesuffix("_" + target_id)
            if rule_name in name_to_resource_ids:
                op.attributes["resources"] = get_mlir_attribute_from_pyval(
                    {"operations": name_to_resource_ids[rule_name]}
                )
                op.attributes["target_gate"] = ir.StringAttr.get(target_id)
                op.attributes["sym_visibility"] = ir.StringAttr.get("public")
                op.attributes["frontend_name"] = ir.StringAttr.get(rule_name)

        return ir.WalkResult.ADVANCE

    with module.context, ir.Location.unknown():
        module.operation.walk(update_funcop_attributes)

    # Inline to avoid helper functions. We want all decomp rule functions to be standalone
    # Generic printing needed when parsing --quantum-opt string output back to jax IR ModuleOps
    # since Catalyst's python bindings never export the Catalyst dialects
    # Before inlining we need to remove the qnode function, since that has a call to the compiled
    # rule subroutine
    qnode_func_erasure_worklist = []

    def remove_qnode_func(op):
        if op.name == "catalyst.launch_kernel":
            qnode_func_erasure_worklist.append(op.parent)
            return ir.WalkResult.ADVANCE
        if op.name == "func.func" and "quantum.node" in op.attributes:
            qnode_func_erasure_worklist.append(op)
            return ir.WalkResult.ADVANCE
        return ir.WalkResult.ADVANCE

    with module.context, ir.Location.unknown():
        module.operation.walk(remove_qnode_func)

    for qnode_func in qnode_func_erasure_worklist:
        qnode_func.erase()

    inlined = _quantum_opt(
        "--inline=inlining-threshold=4294967295",  # Use uint max to indicate always inline
        "--mlir-print-op-generic",
        stdin=str(module),
    )

    inlined_module = ir.Operation.parse(inlined, context=module.context)

    def re_privatize_rules(op):
        if op.name == "func.func":
            rule_name = ir.StringAttr(op.attributes["sym_name"]).value.removesuffix("_" + target_id)
            if rule_name in name_to_resource_ids:
                op.attributes["sym_visibility"] = ir.StringAttr.get("private")
        return ir.WalkResult.ADVANCE

    with inlined_module.context, ir.Location.unknown():
        inlined_module.operation.walk(re_privatize_rules)

    return inlined_module


def collect_symbolic_resources(op_cls, op_name, kwargs, is_custom_op, *, kind, ctrl_wires=()):
    """Return resource data for the rules registered against ``Adjoint(op_name)``/``C(op_name)``.

    PennyLane names a controlled operator ``C(Op)`` whatever its control count, so the registry is
    always queried under that single name; ``ctrl_wires`` carries the count.

    Args:
        op_cls (type): the base operator's class
        op_name (str): the base operator's name
        kwargs (dict): the arguments to build the base operator with
        is_custom_op (bool): whether the operator lowers to ``qref.custom``
        kind (str): ``"adjoint"`` or ``"control"``
        ctrl_wires: the control wires, for ``kind="control"``

    Returns:
        list: the rules that apply to the probed operator
        dict: the arguments the rules were probed with
        dict: rule name to the resources it produces
        dict: rule name to the graphOpId of each resource
    """
    lookup_name = symbolic_op_name(op_name, kind)
    rules = list(qp.decomposition.list_decomps(lookup_name))
    if not rules:
        return [], {}, {}, {}

    # The base op is only a carrier of the dummy parameters and wires: the graph reasons about
    # resources in terms of abstract operators, matching `_get_kwargs` in PennyLane's graph.
    base_op = abstractify(build_base_op(op_cls, kwargs, is_custom_op))
    probe_args = symbolic_arguments(base_op, kind, ctrl_wires)

    name_to_resources = {}
    name_to_resource_ids = {}
    applicable_rules = []
    for rule in rules:
        if not _rule_is_applicable(op_name, rule, **probe_args):
            continue

        applicable_rules.append(rule)
        try:
            with qp.capture.toggle_ctx(True):
                resources = rule.compute_resources(**probe_args)
            name_to_resources[rule.name] = resources.gate_counts
            # The rule body names the ops it produces itself, so unlike the distribution pathway
            # these ids carry no added modifier; a resource that is itself symbolic is spelled the
            # way the compiler spells it (see :meth:`GraphOpID.peel_modifiers`).
            name_to_resource_ids[rule.name] = {
                GraphOpID(op).getGraphOpId(): count for op, count in resources.gate_counts.items()
            }
        except Exception as e:  # pylint: disable=broad-except
            warnings.warn(
                f"Failed to get resources for the {rule.name} decomposition rule: {e}",
                category=RuleLoweringWarning,
            )

    return applicable_rules, probe_args, name_to_resources, name_to_resource_ids


# pylint: disable=too-many-arguments
def compile_registered_symbolic_rules(
    op_name,
    target_id,
    dynamic_shape,
    wire_lens,
    static_data,
    extra_data=None,
    is_custom_op=False,
    op_cls=None,
    *,
    kind,
    n_ctrl=1,
    wrap_control=False,
) -> ir.Operation | None:
    """Return the module of rules registered against ``Adjoint(op_name)``/``C(op_name)`` that follow
    PennyLane's symbolic-argument convention, or None if there are none.

    These rules take a base operator instance rather than the base op's parameters, so the rule
    body rebuilds the operator from its own traced arguments.

    Args:
        op_name (str): the base operator's name
        target_id (str): the modified graphOpId the rules decompose
        dynamic_shape (dict): dynamic argument names to their MLIR types
        wire_lens (dict): wire argument names to their lengths
        static_data (dict): compiler-static argument names to their values
        extra_data (dict): argument values the graphOpId identifies by UID instead of spelling
        is_custom_op (bool): whether the operator lowers to ``qref.custom``
        op_cls (type[Operator2]): the base operator's class, required to rebuild it
        kind (str): ``"adjoint"`` or ``"control"``
        n_ctrl (int): the number of controls, for ``kind="control"`` or ``wrap_control``
        wrap_control (bool): with ``kind="adjoint"``, control each registered ``Adjoint(op)`` rule
            body to synthesize the composed modifier ``C(Adjoint(op))``. This reduces the adjoint
            under control (``C(Adjoint(RZ)) -> C(RZ)``), which the registered ``C(op)`` rules then
            terminate (``C(RZ) -> CRZ``) -- something plain distribution cannot reach, since a
            registered ``C(op)`` rule reads op-specific attributes and cannot take an adjoint base.

    Returns:
        ir.Operation or None: the module holding the rules, or None if there are none

    Raises:
        ValueError: if ``op_cls`` is not given
    """
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
    has_ctrl = kind == "control" or wrap_control
    n_ctrl = n_ctrl if has_ctrl else 0
    device = qp.device("null.qubit", wires=n_base_wires + n_ctrl)

    # A kind="control" rule takes the control wires as its own arguments; a wrap_control adjoint
    # rule instead has the control applied around its body so it is probed without control wires.
    collect_ctrl_wires = range(n_base_wires, n_base_wires + n_ctrl) if kind == "control" else ()
    rules, probe_args, name_to_resources, name_to_resource_ids = collect_symbolic_resources(
        op_cls,
        op_name,
        kwargs | static_and_extra,
        is_custom_op,
        kind=kind,
        ctrl_wires=collect_ctrl_wires,
    )
    if not rules:
        return None

    # Controlling an adjoint rule's body means every op it produces gains the control modifier.
    if wrap_control:
        ctrl_mod = _control_modifier(n_ctrl)
        name_to_resource_ids = {
            rule_name: {wrap_modifier_id(rid, ctrl_mod): count for rid, count in ids.items()}
            for rule_name, ids in name_to_resource_ids.items()
        }

    call_args, call_kwargs = split_call_args(kwargs, is_custom_op)
    kwarg_names = ordered_kwarg_names(call_kwargs, dynamic_shape)

    # Numeric hybrid args (e.g. a CDFHamiltonian) are passed as operands so their concrete values
    # do not bake into the rule body; the rest are closed over. The modified base op still carries
    # every hybrid arg, so a baked one would also make the gate op's params outnumber the rule's
    # inputs (crashing the signature analyzer during decompose-lowering).
    hybrid_specs, hybrid_leaves, closed_hybrid = flatten_hybrid_args(extra_data)

    def rule_to_subroutine(rule):
        def decomp_rule(*_operands):
            _args, _kwargs, _ctrl_wires = unpack_rule_operands(
                _operands, len(call_args), kwarg_names, wire_lens, has_ctrl, hybrid_specs
            )
            # The base is rebuilt from the traced arguments of the rule function, so the wires and
            # parameters the rule body reads off it are this function's own operands. Operand-passed
            # hybrid args are rebuilt into ``_kwargs``; static data and closed-over hybrid args are
            # supplied directly.
            with qp.capture.pause():
                base = op_cls(*_args, **_kwargs, **static_data, **closed_hybrid)
            # TODO: Call the rule itself instead of its _impl after merging
            # https://github.com/PennyLaneAI/pennylane/pull/10144
            if wrap_control:
                # Reduce the adjoint under control: C(Adjoint(op)) -> C(<adjoint decomposition>).
                qp.ctrl(
                    lambda: rule._impl(**symbolic_arguments(base, kind, None)),
                    control=list(_ctrl_wires),
                )()
            else:
                rule._impl(**symbolic_arguments(base, kind, _ctrl_wires))

        decomp_rule.__name__ = rule.name + "_" + target_id
        return qp.capture.subroutine(decomp_rule)

    subroutines = []
    for rule in rules:
        if rule.name not in name_to_resource_ids:
            continue
        subroutines.append(rule_to_subroutine(rule))

    if not subroutines:
        return None

    ctrl_wires = (
        jnp.array(range(n_base_wires, n_base_wires + n_ctrl), dtype=int) if has_ctrl else None
    )

    return build_rule_module(
        subroutines,
        device,
        call_args,
        call_kwargs,
        kwarg_names,
        wire_lens,
        ctrl_wires,
        name_to_resource_ids,
        target_id,
        hybrid_leaves,
    )


def registered_symbolic_rule_strings(op_name, target_id, kind, **kwargs) -> list[str]:
    """Return the rule strings from :func:`compile_registered_symbolic_rules`.

    A failure to lower them is reported as a warning and yields no rules.

    Args:
        op_name (str): the base operator's name
        target_id (str): the modified graphOpId the rules decompose
        kind (str): ``"adjoint"`` or ``"control"``
        **kwargs: forwarded to :func:`compile_registered_symbolic_rules`

    Returns:
        list[str]: the rules, as MLIR strings
    """
    try:
        module = compile_registered_symbolic_rules(op_name, target_id, kind=kind, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        warnings.warn(
            f"Failed to lower the registered {kind} decomposition rules for {target_id}: {e}",
            category=RuleLoweringWarning,
        )
        return []
    return get_rule_strings_from_module(module) if module is not None else []


def adjoint_variant_rule_strings(
    op_name,
    op_id,
    dynamic_shape,
    wire_lens,
    static_data,
    extra_data=None,
    is_custom_op=False,
    op_cls=None,
):
    """Return the rule strings whose ``target_gate`` is ``Adjoint(op_name)``.

    ``op_id`` is the *base* op's graphOpId (e.g. ``"S{...}"``). Two pathways contribute:
      1. rules registered directly against ``Adjoint(op_name)`` (``list_decomps("Adjoint(S)")``),
         which take the symbolic operator's arguments the way PennyLane writes them
         (``self_adjoint``, ``adjoint_rotation``, ...), and
      2. rules synthesized by distributing each base rule of ``op_name`` over adjoint
         (the ``wrap_adjoint`` pathway), dropping any whose body is non-invertible.

    Shared by the eager lowering-time closure (:func:`fetch_all_reachable_decomposition_rules_from_op`)
    and the compiler's on-demand loader (:func:`compile_decomposition_rules_wrapper`) so both build
    adjoint rules identically.

    Args:
        op_name (str): the base operator's name
        op_id (str): the base operator's graphOpId
        dynamic_shape (dict): dynamic argument names to their MLIR types
        wire_lens (dict): wire argument names to their lengths
        static_data (dict): compiler-static argument names to their values
        extra_data (dict): argument values the graphOpId identifies by UID instead of spelling
        is_custom_op (bool): whether the operator lowers to ``qref.custom``
        op_cls (type): the base operator's class; pathway 1 is skipped without it

    Returns:
        list[str]: the rules, as MLIR strings
    """
    out = []
    adj_name = f"Adjoint({op_name})"
    adj_id = name_wrap_adjoint(op_id)
    # (1) Rules registered directly against Adjoint(op_name). They take a base operator instance,
    # so they need the operator's class.
    # TODO: the on-demand decomp rules are skipped for now.
    if op_cls is not None:
        out.extend(
            registered_symbolic_rule_strings(
                op_name,
                adj_id,
                "adjoint",
                dynamic_shape=dynamic_shape,
                wire_lens=wire_lens,
                static_data=static_data,
                extra_data=extra_data,
                is_custom_op=is_custom_op,
                op_cls=op_cls,
            )
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
    op_cls=None,
):
    """Return the rule strings whose ``target_gate`` is ``<n>C(op_name)`` for each ``n`` in
    ``ctrl_counts``.

    The control analogue of :func:`adjoint_variant_rule_strings`. ``op_id`` is the *base* op's
    graphOpId (e.g. ``"RX{...}"``). For each control count ``n`` these pathways contribute:
      1. rules registered directly against ``C(op_name)``, which take the symbolic operator's
         arguments the way PennyLane writes them (``flip_zero_ctrl_values(...)``). PennyLane names
         a controlled operator ``C(Op)`` whatever its control count, so the registry is queried
         under that one name for every ``n``,
      2. rules synthesized by distributing each base rule of ``op_name`` over ``n`` controls
         (the ``wrap_control`` pathway), and
      3. rules for the nested modifier ``<n>C(Adjoint(op_name))`` synthesized by controlling each
         *adjointed* base rule (``wrap_adjoint`` + ``wrap_control``), so controlled-adjoint ops are
         reachable too.
    Distribution rules whose body is non-controllable are dropped.

    Args:
        op_name (str): the base operator's name
        op_id (str): the base operator's graphOpId
        ctrl_counts (list[int]): the control counts to build rules for
        dynamic_shape (dict): dynamic argument names to their MLIR types
        wire_lens (dict): wire argument names to their lengths
        static_data (dict): compiler-static argument names to their values
        extra_data (dict): argument values the graphOpId identifies by UID instead of spelling
        is_custom_op (bool): whether the operator lowers to ``qref.custom``
        op_cls (type): the base operator's class; pathway 1 is skipped without it

    Returns:
        list[str]: the rules, as MLIR strings
    """
    out = []
    for n in ctrl_counts:
        ctrl_mod = _control_modifier(n)
        ctrl_name = f"{ctrl_mod}({op_name})"
        ctrl_id = wrap_modifier_id(op_id, ctrl_mod)
        # (1) Rules registered directly against C(op_name). They take a base operator instance,
        # so they need the operator's class.
        # TODO: the on-demand decomp rules are skipped for now.
        if op_cls is not None:
            out.extend(
                registered_symbolic_rule_strings(
                    op_name,
                    ctrl_id,
                    "control",
                    n_ctrl=n,
                    dynamic_shape=dynamic_shape,
                    wire_lens=wire_lens,
                    static_data=static_data,
                    extra_data=extra_data,
                    is_custom_op=is_custom_op,
                    op_cls=op_cls,
                )
            )
            # (1b) <n>C(Adjoint(op_name)) by controlling each registered Adjoint(op_name) rule.
            # This reduces the adjoint under control (e.g. C(Adjoint(RZ)) -> C(RZ)), which the
            # registered C(op_name) rules of pathway 1 then terminate (C(RZ) -> CRZ). The
            # distribution of pathway 3 cannot reach this: it bottoms out at doubly-modified
            # primitives such as C(Adjoint(GlobalPhase)) that only registered rules can terminate.
            out.extend(
                registered_symbolic_rule_strings(
                    op_name,
                    wrap_modifier_id(name_wrap_adjoint(op_id), ctrl_mod),
                    "adjoint",
                    n_ctrl=n,
                    wrap_control=True,
                    dynamic_shape=dynamic_shape,
                    wire_lens=wire_lens,
                    static_data=static_data,
                    extra_data=extra_data,
                    is_custom_op=is_custom_op,
                    op_cls=op_cls,
                )
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
    """Return a string MLIR module containing the decomposition rules for an operator instance.

    Args:
        op_name (str): the operator's name, as PennyLane registers its rules
        op_id (str): the operator's graphOpId
        dynamic_shape (dict): dynamic argument names to their MLIR types
        wire_lens (dict): wire argument names to their lengths
        static_data (dict): compiler-static argument names to their values
        extra_data (dict): argument values the graphOpId identifies by UID instead of spelling
        is_custom_op (bool): whether the operator lowers to ``qref.custom``

    Returns:
        str: the module holding the rules
    """
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

    Args:
        op_name (str): the operator's *base* name
        op_id (str): the operator's full graphOpId, modifiers included
        dynamic_shape (dict): dynamic argument names to their MLIR types
        wire_lens (dict): wire argument names to their lengths
        static_data (dict): compiler-static argument names to their values
        extra_data (dict): argument values the graphOpId identifies by UID instead of spelling
        is_custom_op (bool): whether the operator lowers to ``qref.custom``

    Returns:
        str: a module holding the whole reachable rule closure
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
    op_cls=None,
):
    """Return every decomposition rule reachable from an operator, as MLIR strings.

    Starting from the given operator, this walks the resources its rules produce and captures the
    rules of each op it meets, together with their adjoint and controlled variants, until nothing
    new turns up.

    Args:
        op_name (str): the operator's name, as PennyLane registers its rules
        op_id (str): the operator's graphOpId
        dynamic_shape (dict): dynamic argument names to their MLIR types
        wire_lens (dict): wire argument names to their lengths
        static_data (dict): compiler-static argument names to their values
        extra_data (dict): argument values the graphOpId identifies by UID instead of spelling
        is_custom_op (bool): whether the operator lowers to ``qref.custom``
        n_ctrls (int): the number of controls on the operator instance being decomposed
        op_cls (type): the operator's class; classes of the ops met along the way are taken from
            the resources themselves

    Returns:
        list[str]: the rules, as MLIR strings
    """
    extra_data = extra_data or {}
    queue = deque()
    start = (op_name, dynamic_shape, wire_lens, static_data, extra_data, is_custom_op)
    queue.append(start)
    visited = {op_id}  # remember ops by their graph id

    op_classes = {op_name: op_cls} if op_cls is not None else {}

    # Control counts to synthesize `<n>C(...)` rules for. A single control is always captured
    # proactively; a multi-controlled instance (`n_ctrls > 1`) additionally needs its own count.
    ctrl_counts = [1] if n_ctrls <= 1 else [1, n_ctrls]

    def compile_variants(
        name, op_id, dynamic_shape, wire_lens, static_data, extra_data, is_custom_op, counts=None
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
                    op_cls=op_classes.get(name),
                )
            )
            out.extend(
                control_variant_rule_strings(
                    name,
                    op_id,
                    counts or ctrl_counts,
                    dynamic_shape,
                    wire_lens,
                    static_data,
                    extra_data=extra_data,
                    is_custom_op=is_custom_op,
                    op_cls=op_classes.get(name),
                )
            )
        return out

    rules = compile_variants(
        op_name, op_id, dynamic_shape, wire_lens, static_data, extra_data, is_custom_op
    )
    # Control counts already synthesized per op, so an op reached again under more controls only
    # pays for the <n>C(...) variants it is still missing.
    counts_done = {op_id: set(ctrl_counts)}

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
        all_kwargs = this_kwargs | this_static_data | this_extra_data

        # Explore the ops reachable through the rules of this op and of its adjoint. Keyed by
        # (explored op, rule name): the same rule name may be registered against both.
        resources = {
            (this_name, name): res
            for name, res in collect_resources_for_op(this_name, all_kwargs, this_is_custom_op)[
                0
            ].items()
        }
        if (
            not this_name.startswith("Adjoint(")
            and (this_op_cls := op_classes.get(this_name)) is not None
        ):
            resources |= {
                (f"Adjoint({this_name})", name): res
                for name, res in collect_symbolic_resources(
                    this_op_cls, this_name, all_kwargs, this_is_custom_op, kind="adjoint"
                )[2].items()
            }

        for (_, _rule_name), resource in resources.items():
            try:
                for op, _count in resource.items():
                    # A generic symbolic resource stands for its base under a modifier, and it is
                    # the base that owns the rules; compile_variants re-derives the modifier
                    # variants from there. Its control count has to be carried over, or the
                    # <n>C(...) node the resource names would be left without rules.
                    graph_op_id = GraphOpID(op)
                    op = graph_op_id.op
                    res_ctrls = graph_op_id.num_controls
                    probe_id = graph_op_id.getBaseGraphOpId()
                    probe = (
                        # The name and the id are paired to look up the rules registered for that
                        # id, so spell the name the way PennyLane's registry does rather than the
                        # way the graphOpId does: the two agree on Adjoint(...)/C(...), but
                        # a multi-controlled id reads <n>C(...), which PennyLane has no name for.
                        to_name(op),
                        graph_op_id.dynamic_shape,
                        graph_op_id.wire_lens,
                        graph_op_id.static_data,
                        graph_op_id.extra_data,
                        graph_op_id.is_custom_op,
                    )
                    # Remembered even for an op already visited: another op may reach it later and
                    # need its class to rebuild the base of a registered adjoint rule.
                    op_classes.setdefault(probe[0], type(op))

                    # The counts every op is captured under, plus the one this resource carries.
                    counts = set(ctrl_counts) | ({res_ctrls} if res_ctrls > 1 else set())
                    if probe_id not in visited:
                        visited.add(probe_id)
                        queue.append(probe)
                        counts_done[probe_id] = set(counts)
                        rules.extend(
                            compile_variants(probe[0], probe_id, *probe[1:], counts=sorted(counts))
                        )
                    elif missing := counts - counts_done.setdefault(probe_id, set(ctrl_counts)):
                        # Seen before, but under fewer controls: only the missing <n>C(...)
                        # variants are still owed, the rest are already in `rules`.
                        counts_done[probe_id] |= missing
                        rules.extend(
                            control_variant_rule_strings(
                                probe[0],
                                probe_id,
                                sorted(missing),
                                *probe[1:4],
                                extra_data=probe[4],
                                is_custom_op=probe[5],
                                op_cls=op_classes.get(probe[0]),
                            )
                        )
            except Exception as e:
                warnings.warn(
                    f"Failed to lower the {_rule_name} decomposition rule for {this_name}: {e}",
                    category=RuleLoweringWarning,
                )
            continue
    return rules
