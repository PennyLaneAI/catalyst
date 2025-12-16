# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the Unified Compiler transform interpreter."""

import pytest

# pylint: disable=wrong-import-position
pytest.mark.xdsl
xdsl = pytest.importorskip("xdsl")

from xdsl.dialects import builtin, test, transform

from catalyst.python_interface.conversion import xdsl_from_qjit
from catalyst.python_interface.pass_api.transform_interpreter import (
    TransformFunctionsExt,
    TransformInterpreterPass,
    _create_schedule,
)


def _get_xdsl_attr_from_pyval(val):
    """Get an xDSL attribute corresponding to a Python value."""
    attr = None

    match val:
        case bool():
            attr = builtin.IntegerAttr(int(val), 1)
        case int():
            attr = builtin.IntegerAttr(val, 64)
        case float():
            attr = builtin.FloatAttr(val, 64)
        case str():
            attr = builtin.StringAttr(val)
        case list() | tuple():
            attr = builtin.ArrayAttr(tuple(_get_xdsl_attr_from_pyval(elt) for elt in val))
        case dict():
            attr = builtin.DictionaryAttr({k: _get_xdsl_attr_from_pyval(v) for k, v in val.items()})
        case _:
            raise ValueError(f"Invalid type {val}")

    return attr


def create_apply_registered_pass_op(pass_name, **options) -> transform.ApplyRegisteredPassOp:
    """Create an ApplyRegisteredPassOp using the provided pass name and
    pass options."""
    lowered_options = {}
    for arg, value in options.items():
        option_key = str(arg).replace("_", "-")
        lowered_options[option_key] = _get_xdsl_attr_from_pyval(value)

    dummy_operand = test.TestOp(result_types=(transform.TransformHandleType(),))

    pass_op = transform.ApplyRegisteredPassOp.create(
        properties={
            "pass_name": builtin.StringAttr(pass_name),
            "options": builtin.DictionaryAttr(lowered_options),
        },
        operands=(dummy_operand,),
        result_types=(transform.TransformHandleType(),),
    )
    return pass_op


class TestTransformInterpreterPass:
    """Unit tests for the TransformInterpreterPass."""
