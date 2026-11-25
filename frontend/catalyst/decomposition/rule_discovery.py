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

"""Walk resource edges to discover the decomposition-rule batches a program needs.

The traversal propagates modifiers and prevents control-increasing cycles.
Preparation supplies each batch; capture or standalone compilation consumes it.
"""

from collections import deque

from pennylane.core.operator import Operator2

from catalyst.decomposition.capture_session import OpDecompRequest
from catalyst.decomposition.modifier_state import ModifierState
from catalyst.decomposition.rule_preparation import DecompTargetSpec, build_decomp_rule_variants
from catalyst.decomposition.type_utils import SerializedGraphOpID


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
