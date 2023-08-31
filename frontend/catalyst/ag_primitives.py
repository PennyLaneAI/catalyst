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

"""This module provides the implementation of AutoGraph primitives in terms of traceable Catalyst
functions. The purpose is to convert imperative style code to functional or graph-style code."""

import functools
from typing import Any, Callable, Tuple

# Use tensorflow implementations for handling function scopes and calls,
# as well as various utility objects.
import pennylane as qml
import tensorflow.python.autograph.impl.api as tf_autograph_api
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.core.converter import STANDARD_OPTIONS as STD
from tensorflow.python.autograph.core.converter import ConversionOptions
from tensorflow.python.autograph.core.function_wrappers import (
    FunctionScope,
    with_function_scope,
)
from tensorflow.python.autograph.impl.api import converted_call as tf_converted_call
from tensorflow.python.autograph.operators.variables import (
    Undefined,
    UndefinedReturnValue,
)

import catalyst
from catalyst.ag_utils import AutoGraphError
from catalyst.utils.patching import Patcher

__all__ = [
    "STD",
    "ConversionOptions",
    "Undefined",
    "UndefinedReturnValue",
    "FunctionScope",
    "with_function_scope",
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
    """An implementation of the AutoGraph 'if' statement. The interface is defined by AutoGraph,
    here we merely provide an implementation of it in terms of Catalyst primitives."""

    # Cache the initial state of all modified variables. Required because we trace all branches,
    # and want to restore the initial state before entering each branch.
    init_state = get_state()

    @catalyst.cond(pred)
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

    # Sometimes we unpack the results of nested tracing scopes so that the user doesn't have to
    # manipulate tuples when they don't expect it. Ensure set_state receives a tuple regardless.
    results = functional_cond()
    if not isinstance(results, tuple):
        results = (results,)
    set_state(results)


# Prevent autograph from converting PennyLane and Catalyst library code, this can lead to many
# issues such as always tracing through code that should only be executed conditionally. We might
# have to be even more restrictive in the future to prevent issues if necessary.
module_allowlist = (
    config.DoNotConvert("pennylane"),
    config.DoNotConvert("catalyst"),
) + config.CONVERSION_RULES


def converted_call(fn, *args, **kwargs):
    """We want AutoGraph to use our own instance of the AST transformer when recursively
    transforming functions, but otherwise duplicate the same behaviour."""

    with Patcher(
        (tf_autograph_api, "_TRANSPILER", catalyst.autograph.TRANSFORMER),
        (config, "CONVERSION_RULES", module_allowlist),
    ):
        # We need to unpack nested QNode and QJIT calls as autograph will have trouble handling
        # them. Ideally, we only want the wrapped function to be transformed by autograph, rather
        # than the QNode or QJIT call method.

        # For nested QJIT calls, the class already forwards to the wrapped function, bypassing any
        # class functionality. We just do the same here:
        if isinstance(fn, catalyst.QJIT):
            fn = fn.user_function

        # For QNode calls, we employ a wrapper to correctly forward the quantum function call to
        # autograph, while still invoking the QNode call method in the surrounding tracing context.
        if isinstance(fn, qml.QNode):

            @functools.wraps(fn.func)
            def qnode_call_wrapper():
                return tf_converted_call(fn.func, *args, **kwargs)

            new_qnode = qml.QNode(qnode_call_wrapper, device=fn.device, diff_method=fn.diff_method)
            return new_qnode()

        return tf_converted_call(fn, *args, **kwargs)
