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

"""Prepare dummy arguments and pack decomposition-rule operands.

Argument order matches the operation's parameter and wire operands. Hybrid argument
specifications rebuild nested values from those operands inside the rule trace.
"""

import itertools
from dataclasses import dataclass
from enum import Enum, auto

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from jax import Array
from pennylane.core.operator import Operator2
from pennylane.wires import Wires

from catalyst.decomposition.graph_op_id import CTRL_WORK_WIRES
from catalyst.decomposition.type_utils import get_dummy_values_for_dynamic_shape
from catalyst.utils.exceptions import CompileError


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
