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

"""Dataclass for handling operator modifiers related to graph-decomposition."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import wraps
from typing import ClassVar

import pennylane as qp
from pennylane.capture.primitives import ctrl_transform_prim
from pennylane.decomposition.resources import resolve_work_wire_type
from pennylane.ops.op_math.adjoint2 import Adjoint2
from pennylane.ops.op_math.controlled2 import ControlledOp2

from catalyst.utils.patching import Patcher


@dataclass(frozen=True)
class ModifierState:
    """Compose, inspect, and apply the modifiers carried by an operator.

    Current modifiers are adjoints and controls.
    Default values are False for `adjoint` and 0 for `control_count`.
    Controls may be accompanied by work wires and a work wire type. These control work wires
    are distinct from standard operator work wires.

    Attributes:
        adjoint: Whether the operator is adjointed.
        control_count: How many controls are applied to the operator.
        num_work_wires: How many work wires are available to the controls.
        work_wire_type: The type of those work wires, ``"zeroed"`` or ``"borrowed"``.

    Raises:
        ValueError: If `control_count` is negative, or there are work wires without any controls.
    """

    adjoint: bool = False
    control_count: int = 0
    num_work_wires: int = 0
    work_wire_type: str = "borrowed"

    def __post_init__(self):
        """Performs validation of the modifier state."""
        if not isinstance(self.control_count, int) or self.control_count < 0:
            raise ValueError(f"Modifier counts cannot be negative, got {self.control_count}")

        if not isinstance(self.adjoint, bool):
            raise ValueError(f"Adjoint can be True or False, got {self.adjoint}")
        if not isinstance(self.num_work_wires, int) or self.num_work_wires < 0:
            raise ValueError(f"Modifier counts cannot be negative, got {self.num_work_wires}")
        if self.num_work_wires and not self.control_count:
            raise ValueError(f"Control work wires require at least one control, got {self}")
        if not self.num_work_wires:
            object.__setattr__(self, "work_wire_type", "borrowed")

    def compose(self, inner: ModifierState) -> ModifierState:
        """Return the result of composing the current modifiers with `inner`."""
        work_wire_type = resolve_work_wire_type(
            range(self.num_work_wires),
            self.work_wire_type,
            range(inner.num_work_wires),
            inner.work_wire_type,
        )
        return ModifierState(
            self.adjoint != inner.adjoint,
            self.control_count + inner.control_count,
            self.num_work_wires + inner.num_work_wires,
            work_wire_type,
        )

    @property
    def is_controlled(self) -> bool:
        """Return True if this state is controlled."""
        return self.control_count > 0

    @property
    def is_modified(self) -> bool:
        """Return whether any adjoint or control modifier is applied."""
        return self.adjoint or self.is_controlled

    def without_adjoint(self) -> ModifierState:
        """Return the controls remaining after an adjoint rule handles the adjoint."""
        return replace(self, adjoint=False)

    def wrap_name(self, op_name: str) -> str:
        """Wrap an operator name with adjoint innermost and control outermost.

        This matches the compiler's ``wrapModifiers`` spelling. An adjoint cannot be added
        outside an already-controlled name, since that would produce a non-canonical id.
        Compose states before wrapping when the modifiers on an operator are already known.
        """
        if self.adjoint:
            op_name = self._wrap_modifier_name(op_name, "Adjoint")
        if self.is_controlled:
            op_name = self._wrap_modifier_name(op_name, self._control_modifier(self.control_count))
        return op_name

    def wrap_id(self, op_id: str) -> str:
        """Wrap a graphOpId's name, preserving its parameter, wire, static-data and UID suffix."""
        if not self.is_modified:
            return op_id
        assert "{" in op_id, f"Malformed op id for graph decomposition, got {op_id}"
        split = op_id.find("{")
        wrapped_id = self.wrap_name(op_id[:split]) + op_id[split:]
        if self.num_work_wires:
            # Imported lazily because GraphOpID also uses ModifierState.
            from catalyst.decomposition.graph_op_id import add_ctrl_work_wires_to_id

            wrapped_id = add_ctrl_work_wires_to_id(
                wrapped_id, self.num_work_wires, self.work_wire_type
            )
        return wrapped_id

    def wrap_callable(
        self, body: Callable, ctrl_wires=None, ctrl_values=None, work_wires=None
    ) -> Callable:
        """Apply adjoint and control regions to a rule body in canonical nesting order.

        ``ctrl_wires`` supplies the rule's control-wire operand when the state is controlled.
        ``ctrl_values`` supplies its control-value operand, defaulting to all ones.
        ``work_wires`` supplies the dynamic control work-wire operand.
        """
        if self.adjoint:
            body = qp.adjoint(body)
        if self.is_controlled:
            body = qp.ctrl(
                body,
                control=list(ctrl_wires),
                control_values=ctrl_values,
                work_wires=work_wires,
                work_wire_type=self.work_wire_type,
            )
            if self.num_work_wires:
                body = capture_control_work_wires(body)
        return body

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
    _MODIFIER_CANONICAL_ORDER: ClassVar[tuple[str, ...]] = ("C", "Adjoint")

    @classmethod
    def _wrap_modifier_name(cls, op_name: str, modifier: str) -> str:
        """Add one outer modifier, rejecting non-canonical nesting."""
        new_kind = cls._modifier_kind(modifier)
        inner_kind = cls._leading_modifier_kind(op_name)
        if inner_kind is not None:
            new_rank = cls._MODIFIER_CANONICAL_ORDER.index(new_kind)
            inner_rank = cls._MODIFIER_CANONICAL_ORDER.index(inner_kind)
            if new_rank > inner_rank:
                raise ValueError(
                    f"Non-canonical modifier order: cannot wrap {modifier!r} (canonically inner) "
                    f"around {op_name!r} whose outermost modifier {inner_kind!r} is canonically "
                    f"outer. Apply modifiers outermost-last in the order "
                    f"{cls._MODIFIER_CANONICAL_ORDER}."
                )
        return f"{modifier}({op_name})"

    @staticmethod
    def _modifier_kind(modifier: str) -> str:
        """Normalise a modifier token to its canonical form."""
        return "C" if modifier.endswith("C") else modifier

    @staticmethod
    def _control_modifier(n_ctrl: int) -> str:
        """Return the graphOpId control-modifier token for ``n_ctrl`` controls.

        A single control is written ``C`` and ``n > 1`` controls ``<n>C``, mirroring the compiler's
        ``wrapModifiers``.
        """
        assert n_ctrl >= 1, "control modifier requires at least one control"
        return "C" if n_ctrl == 1 else f"{n_ctrl}C"

    @staticmethod
    def _leading_modifier_kind(op_id: str) -> str | None:
        """Return the canonical kind of ``op_id``'s current outermost modifier, or None if bare."""
        if op_id.startswith("Adjoint("):
            return "Adjoint"

        if re.match(r"\d*C\(", op_id):
            return "C"

        return None

    @staticmethod
    def name_unwrap_adjoint(op_name: str, op_id: str) -> str:
        """Inverse of :meth:`wrap_id` with an adjoint state given the base ``op_name``.

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

    @staticmethod
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

    @staticmethod
    def peel_modifiers(op: qp.core.Operator2) -> tuple[qp.core.Operator2, ModifierState]:
        """Return the innermost base of ``op`` together with the modifiers wrapping it.

        A *generic* symbolic operator (``Adjoint2``/``ControlledOp2``, as opposed to a concrete
        class such as ``CH``) reports its wrapper's own arguments, e.g.
        ``Adjoint(S){}{}{}[1141126509488406748]``. That is not how the compiler spells a modified
        operator: ``wrapModifiers`` folds the modifier into the *base* operator's id. So the
        wrappers are peeled off here and counted, and :meth:`wrap_name` puts them back in
        canonical position (control outermost) rather than splicing them into a finished id.

        The walk terminates structurally: each step moves to ``op.base``, one wrapper shallower,
        and the chain ends at the first non-symbolic operator. The layers compose as
        :meth:`ModifierState.compose` does, so adjoints cancel pairwise, control and work-wire
        counts add, and work-wire types merge as they do in PennyLane.

        Args:
            op (Operator2): the operator to unwrap

        Returns:
            Operator2: the innermost base operator
            ModifierState: the modifiers wrapping the base
        """
        modifiers = ModifierState()
        while True:
            if isinstance(op, Adjoint2):
                layer = ModifierState(adjoint=True)
            elif isinstance(op, ControlledOp2):
                layer = ModifierState(
                    control_count=len(op.control_wires),
                    num_work_wires=len(op.work_wires),
                    work_wire_type=op.work_wire_type,
                )
            else:
                return op, modifiers
            modifiers = modifiers.compose(layer)
            op = op.base

    @classmethod
    def from_graph_op_id(cls, op_name: str, op_id: str) -> tuple[str, ModifierState]:
        """Recover the base id and modifiers supplied by the compiler's on-demand loader."""
        base_id = op_id
        modifiers = cls()
        if cls._leading_modifier_kind(op_id) == "C" and not op_name.startswith("Adjoint("):
            # Controls are outermost, including on a controlled adjoint.
            name = op_id.split("{", 1)[0]
            inner_name = (
                f"Adjoint({op_name})" if name.endswith(f"(Adjoint({op_name}))") else op_name
            )
            base_id, n_ctrls = cls.name_unwrap_control(inner_name, op_id)

            # Imported lazily because GraphOpID also uses ModifierState.
            from catalyst.decomposition.graph_op_id import remove_ctrl_work_wires_from_id

            base_id, n_work, work_wire_type = remove_ctrl_work_wires_from_id(base_id)
            modifiers = cls(
                control_count=n_ctrls, num_work_wires=n_work, work_wire_type=work_wire_type
            )

        if base_id.startswith("Adjoint(") and not op_name.startswith("Adjoint("):
            base_id = cls.name_unwrap_adjoint(op_name, base_id)
            modifiers = replace(modifiers, adjoint=True)

        return base_id, modifiers


def capture_control_work_wires(body: Callable) -> Callable:
    """Capture callable control work wires as operands while tracing a decomposition rule.

    PennyLane stores these wires in static primitive metadata. Rule bodies instead receive traced
    wire labels, including those forwarded by registered rules to nested callable controls.
    Move the labels into operands before binding, so JAX tracks them in the enclosing rule.
    The patch is scoped to rule tracing and also accepts already converted control primitives.
    """

    @wraps(body)
    def wrapper(*args, **kwargs):
        original_bind = ctrl_transform_prim.bind

        def bind(*invals, work_wires, **params):
            if work_wires:
                return original_bind(
                    *invals, *work_wires, work_wires=None, **(params | {"n_work": len(work_wires)})
                )
            return original_bind(*invals, work_wires=work_wires, **params)

        with Patcher((ctrl_transform_prim, "bind", bind)):
            return body(*args, **kwargs)

    return wrapper
