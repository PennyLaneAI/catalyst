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


"""Utility functions for Catalyst's compiler passes."""

from collections.abc import Iterable
from pathlib import Path

from pennylane.decomposition.utils import to_name

from catalyst.utils.runtime_environment import BYTECODE_FILE_PATH


def prepare_decomposition_options(
    gate_set: Iterable[type | str] | dict[type | str, float],
    fixed_decomps: dict | None = None,
    alt_decomps: dict | None = None,
    _builtin_rule_path: Path = BYTECODE_FILE_PATH,
):
    """
    Prepares the options dictionary for Catalyst's graph decomposition pass.

    Args:
        gate_set: An iterable of gate types or a dictionary mapping gate types to their costs.
        fixed_decomps: An optional dictionary mapping gate types to specific decomposition rules.
        alt_decomps: An optional dictionary mapping gate types to lists of alternative decomposition
                     rules.
        _builtin_rule_path: The path to the precompiled decomposition rules bytecode file.

    Returns:
        A dictionary of options to be passed to the graph decomposition pass.
    """

    if not isinstance(gate_set, dict):
        gate_set = {to_name(op): 1.0 for op in gate_set}
    else:
        gate_set = {to_name(op): cost for op, cost in gate_set.items()}

    options: dict[str, dict | tuple | str] = {
        "gate_set": gate_set,
        "bytecode_rules": str(_builtin_rule_path),
    }

    if fixed_decomps:
        options |= {
            "fixed_decomps": {
                to_name(op): (rule if isinstance(rule, str) else rule.__name__)
                for op, rule in fixed_decomps.items()
            }
        }

    if alt_decomps:
        options |= {
            "alt_decomps": {
                to_name(op): tuple(
                    (rule if isinstance(rule, str) else rule.__name__) for rule in rules
                )
                for op, rules in alt_decomps.items()
            }
        }

    return options
