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

import pennylane as qp
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
        raise AssertionError(f"Cannot be inserted into a CompilePipeline: {e}") from e
    assert len(pipeline) == 1
    assert pipeline[0].pass_name == pass_name


_PASS_NAME_SPECIAL_CASES = {"diagonalize_measurements": "diagonalize-final-measurements"}


@pytest.mark.parametrize("name", builtin_passes.__all__)
def test_pass_name_matches_variable_name(name):
    """Tests that the variable name is just the pass name but snake case."""

    obj = getattr(builtin_passes, name)
    expected_pass_name = _PASS_NAME_SPECIAL_CASES.get(name, name.replace("_", "-"))
    assert obj.pass_name == expected_pass_name


@pytest.mark.parametrize("name", builtin_passes.__all__)
def test_passes_are_valid_transforms(name):
    """Tests that these passes are valid transform objects."""

    obj = getattr(builtin_passes, name)
    assert_valid_transform(obj)


@pytest.mark.parametrize("name", builtin_passes.__all__)
def test_pass_compiles_with_qjit(name):
    """Basic smoke test to ensure proper compilation. For example, if a pass name
    was incorrect this would fail."""

    obj = getattr(builtin_passes, name)

    @qp.qjit(target="mlir")
    @obj
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit():
        qp.H(0)
        return qp.expval(qp.Z(0))

    circuit()
