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

"""Fixtures for testing individual prepared decomposition targets."""

from unittest.mock import patch

import pytest

from catalyst.decomposition.capture_session import DecompositionScope
from catalyst.decomposition.precompile_decomposition_rules import _capture_rule_module


@pytest.fixture
def capture_target():
    """Use the shared scope tracer while isolating the target under test from its descendants."""

    def capture(target_spec):
        if target_spec is None:
            return None
        with patch(
            "catalyst.from_plxpr.qfunc_interpreter.walk_reachable_decomp_rule_sets",
            return_value=[target_spec],
        ):
            return _capture_rule_module(DecompositionScope())

    return capture
