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

"""Coordinate on-demand rule discovery and compilation for the compiler callback."""

import warnings

from pennylane.core.operator import get_op_class_by_name

from catalyst.decomposition.capture_session import DecompositionScope, OpDecompRequest
from catalyst.decomposition.graph_op_id import CTRL_WORK_WIRE_TYPE, CTRL_WORK_WIRES
from catalyst.decomposition.modifier_state import ModifierState
from catalyst.decomposition.precompiled_rule_loading import _RuleEntry, _validate_rule_entry
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.decomposition.type_utils import SerializedGraphOpID


def compile_on_demand_decomposition_rules(
    requests: list[dict], available_rules: list[_RuleEntry] | None = None
) -> str:
    """Return one MLIR module with the combined reachable closure of the requested operators.

    Resolve each class through PennyLane's OperatorRegistry and seed one DecompositionScope.
    Shared discovery and tracing deduplicate dependencies and combine modifier states for the
    same base identity.

    Args:
        requests (list[dict]): Each request supplies ``op_name``, the full ``op_id``,
            ``dynamic_shape``, ``wire_lens`` and ``static_data`` in operand order.
            Optional ``extra_data`` defaults to an empty dict, and ``is_custom_op`` to False.
        available_rules (list[dict]): Identities of rules actually loaded by the compiler,
            using the precompiled manifest's entry format. These rules are not retraced;
            discovery still follows their dependencies. Defaults to no available rules.

    Returns:
        str: a module holding the combined reachable rule closure
    """
    scope = DecompositionScope()
    for data in requests:
        op_name = data["op_name"]
        try:
            op_cls = get_op_class_by_name(op_name)
        except KeyError:
            # A standalone compiler may not have classes defined in the calling process.
            warnings.warn(
                f"Cannot load decomposition rules for {op_name}: no Operator2 class with this name "
                "is registered in the compiler's Python process.",
                category=RuleLoweringWarning,
            )
            continue

        base_id, modifiers = ModifierState.from_graph_op_id(
            op_name, SerializedGraphOpID(data["op_id"])
        )
        wire_lens = data["wire_lens"]
        static_data = data["static_data"]
        if modifiers.is_controlled:
            wire_lens = {name: size for name, size in wire_lens.items() if name != CTRL_WORK_WIRES}
            static_data = {
                name: value for name, value in static_data.items() if name != CTRL_WORK_WIRE_TYPE
            }

        scope.record_root(
            OpDecompRequest(
                base_id=base_id,
                op_name=op_name,
                op_cls=op_cls,
                dynamic_shape=data["dynamic_shape"],
                wire_lens=wire_lens,
                static_data=static_data,
                extra_data=data.get("extra_data") or {},
                is_custom_op=data.get("is_custom_op", False),
                modifier_state=modifiers,
            )
        )

    if not scope.roots:
        return "module {}"

    # Defer the interpreter import until the compiler invokes this callback.
    from catalyst.decomposition.rule_compilation import (  # pylint: disable=import-outside-toplevel
        _capture_rule_module,
    )

    identities = frozenset(_validate_rule_entry(entry) for entry in available_rules or ())
    return str(_capture_rule_module(scope, precompiled_rule_identities=identities))
