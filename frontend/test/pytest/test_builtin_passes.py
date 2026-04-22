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
"""Tests the passes found in 'builtin_passes.py'"""
import pytest
import pennylane as qp
import inspect

from catalyst.passes import builtin_passes

AVAILABLE_PASSES = {obj for _, obj in inspect.getmembers(builtin_passes) if isinstance(obj, qp.transforms.core.Transform)}

@pytest.mark.parametrize("pass_", AVAILABLE_PASSES)
def test_integration_with_compile_pipeline(pass_):
    """Tests that these passes can be fed into a 'qp.CompilePipeline' object."""

    pipeline = qp.CompilePipeline(pass_)
    assert len(pipeline) == 1
    assert pipeline[0].pass_name == pass_.pass_name
