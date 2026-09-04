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

"""This module provides a warning category for grouping decomposition rule lowering warnings."""

import os
import warnings


class RuleLoweringWarning(Warning):
    pass


def _env_flag(name: str, default: str) -> bool:
    """Read a boolean environment variable, treating "0"/"false"/"" as false."""
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false")


# Default is "1".
SILENCE_RULE_LOWERING_WARNINGS = _env_flag("CATALYST_SILENCE_RULE_LOWERING_WARNINGS", "1")

if SILENCE_RULE_LOWERING_WARNINGS:  # pragma: no-cover
    warnings.filterwarnings("ignore", category=RuleLoweringWarning)
