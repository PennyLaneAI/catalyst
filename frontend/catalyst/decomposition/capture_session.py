# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Managed state for the decomposition rule discovery and capture mechanism."""

from dataclasses import dataclass

import pennylane as qp
from pennylane.core.operator import abstractify

from catalyst.decomposition.graph_op_id import GraphOpID


@dataclass(frozen=True)
class RuleRequest:
    """Dataclass with information needed to discover decomposition rules for one operator (variant).

    Attributes:
        base_id: Canonical GraphOpID of the unmodified base operator. Used as the traversal key and
            as the starting point for adjoint/control target IDs.
        op_name: Frontend operator name used to query PennyLane's decomposition registry.
        op_cls: Concrete Operator2 class used to rebuild the base for registered symbolic rules.
            It is unavailable for an on-demand root supplied by the compiler, but recovered for
            descendants represented by Python operators.
        dynamic_shape: Non-hybrid dynamic argument names mapped to MLIR-style type spellings.
        wire_lens: Non-hybrid wire argument names mapped to their lengths.
        static_data: Compiler-known static/compilable arguments used by applicability, resource
            computation, and rule bodies.
        extra_data: Abstract hybrid-argument pytrees used to reconstruct rule arguments without
            retaining values or tracers from a specific program instance.
        is_custom_op: Whether the operator uses the positional, scalar-f64 signature of qref.custom.
        control_count: Controls carried by the encountered operation, including ambient control
            transform context. The base ID itself intentionally excludes modifiers.
    """

    base_id: str
    op_name: str
    op_cls: type | None
    dynamic_shape: dict
    wire_lens: dict
    static_data: dict
    extra_data: dict
    is_custom_op: bool
    control_count: int = 0

    @classmethod
    def from_operation(cls, op, ambient_control_count: int = 0) -> "RuleRequest":
        """Build a request from a settled captured Operator2 instance.

        A rule is a template keyed by the operator's identity, so only shapes, dtypes and pytree
        structure may reach it. The operator is abstractified first: that canonicalizes the
        properties by removing tracers (no leaks) and giving shapes/dtypes to Python constants.
        """
        with qp.capture.pause():
            op = abstractify(op)

        graph_op_id = GraphOpID(op)
        base = graph_op_id.op
        static_data = {
            name: getattr(base, name) for name in (*base.compilable_argnames, *base.static_argnames)
        }
        extra_data = {name: getattr(base, name) for name in base.hybrid_argnames}
        return cls(
            base_id=graph_op_id.getBaseGraphOpId(),
            op_name=graph_op_id.get_operator_name(),
            op_cls=type(base),
            dynamic_shape={
                name: types
                for name, types in graph_op_id.dynamic_shape.items()
                if name not in base.hybrid_argnames
            },
            wire_lens={
                name: length
                for name, length in graph_op_id.wire_lens.items()
                if name not in base.hybrid_argnames
            },
            static_data=static_data,
            extra_data=extra_data,
            is_custom_op=graph_op_id.is_custom_op,
            control_count=graph_op_id.num_controls + ambient_control_count,
        )
