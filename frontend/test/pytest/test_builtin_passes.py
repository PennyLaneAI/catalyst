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

import inspect
from typing import Any

import pytest
from pennylane.transforms.core import CompilePipeline, Transform

from catalyst.passes import builtin_passes


def assert_valid_transform(obj: Any) -> None:
    """Asserts that 'obj' satisfies basic 'Transform' object checks."""

    # Ensure the correct type.
    assert isinstance(obj, Transform)

    # Ensure 'pass_name' is set and non-empty.
    pass_name = getattr(obj, "pass_name", None)
    assert isinstance(pass_name, str) and pass_name

    # Ensure it has a docstring
    doc = inspect.getdoc(obj)
    assert doc and doc.strip()

    # Must be insertable into a qp.CompilePipeline
    try:
        pipeline = CompilePipeline(obj)
    except Exception as e:
        raise AssertionError(f"Cannot be inserted into a CompilePipeline: {e}")
    assert len(pipeline) == 1
    assert pipeline[0].pass_name == pass_name


@pytest.mark.parametrize("name", builtin_passes.__all__)
def test_passes_are_valid_transforms(name):
    """Tests that these passes are valid transform objects."""

    obj = getattr(builtin_passes, name)
    assert_valid_transform(obj)
