# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module provides the implementation of Autograph primitives in terms of traceable Catalyst
functions. The purpose is to convert imperative style code to functional or graph-style code."""

from typing import Any, Callable, Tuple

# Use tensorflow implementations for handling function scopes and calls,
# as well as various utility objects.
import tensorflow.python.autograph.impl.api as tf_autograph_api
from tensorflow.python.autograph.core.converter import STANDARD_OPTIONS as STD
from tensorflow.python.autograph.core.converter import ConversionOptions
from tensorflow.python.autograph.core.function_wrappers import FunctionScope
from tensorflow.python.autograph.impl.api import AutoGraphError
from tensorflow.python.autograph.impl.api import converted_call as tf_converted_call
from tensorflow.python.autograph.operators.variables import (
    Undefined,
    UndefinedReturnValue,
)

import catalyst
from catalyst import cond
from catalyst.utils.patching import Patcher

__all__ = [
    "STD",
    "ConversionOptions",
    "AutoGraphError",
    "Undefined",
    "UndefinedReturnValue",
    "FunctionScope",
    "if_stmt",
    "converted_call",
]


def assert_results(results, var_names):
    """Assert that none of the results are undefined, i.e. have no value."""

    assert len(results) == len(var_names)

    for r, v in zip(results, var_names):
        if isinstance(r, Undefined):
            raise AutoGraphError(f"Some branches did not define a value for variable '{v}'")

    return results


# pylint: disable=too-many-arguments
def if_stmt(
    pred: bool,
    true_fn: Callable[[], Any],
    false_fn: Callable[[], Any],
    get_state: Callable[[], Tuple],
    set_state: Callable[[Tuple], None],
    symbol_names: Tuple[str],
    _num_results: int,
):
    """An implementation of the Autograph 'if' statement. The interface is defined by Autograph,
    here we merely provide an implementation of it in terms of Catalyst primitives."""

    # Cache the initial state of all modified variables. Required because we trace all branches,
    # and want to restore the initial state before entering each branch.
    init_state = get_state()

    @cond(pred)
    def functional_cond():
        set_state(init_state)
        true_fn()
        results = get_state()
        return assert_results(results, symbol_names)

    @functional_cond.otherwise
    def functional_cond():
        set_state(init_state)
        false_fn()
        results = get_state()
        return assert_results(results, symbol_names)

    set_state(functional_cond())


def converted_call(*args, **kwargs):
    """We want Autograph to use our own instance of the AST transformer when recursively
    transforming functions, but otherwise duplicate the same behaviour."""

    # pylint: disable=protected-access
    with Patcher((tf_autograph_api, "_TRANSPILER", catalyst.autograph._TRANSFORMER)):
        return tf_converted_call(*args, **kwargs)
