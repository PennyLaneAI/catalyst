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

from dataclasses import dataclass


@dataclass(frozen=True)
class ModifierState:
    """Represents the modifiers applied to an operator.

    Current modifiers are adjoints and controls.

    Attributes:
        adjoint: Whether the operator is adjointed.
        control_count: How many controls are applied to the operator.

    Raises:
        ValueError: If `control_count` is negative.
    """

    adjoint: bool = False
    control_count: int = 0

    def __post_init__(self):
        """Performs validation of the modifier state."""
        if self.control_count < 0:
            raise ValueError(f"Modifier counts cannot be negative, got {self}")

    def compose(self, inner: ModifierState) -> ModifierState:
        """Return the result of composing the current modifiers with `inner`."""
        return ModifierState(
            self.adjoint != inner.adjoint,
            self.control_count + inner.control_count,
        )
