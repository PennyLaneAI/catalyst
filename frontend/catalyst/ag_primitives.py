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
from typing import Any, Callable, Tuple, SupportsIndex, Iterator

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
import jax.numpy as jnp

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


def _call_catalyst_for(start, stop, step, body_fn, get_state, opts, array_iterable=None):
    """Dispatch to a Catalyst implementation of for loops."""

    # Do not allow unpacking in Catalyst for loops.
    assert len(opts["iterate_names"].split(",")) == 1

    @catalyst.for_loop(start, stop, step)
    def functional_for(i):
        if array_iterable is None:
            body_fn(i)
        else:
            body_fn(array_iterable[i])

        return get_state()

    return functional_for()


def _call_python_for(body_fn, get_state, non_array_iterable):
    """Fallback to a Python implementation of for loops."""
    for elem in non_array_iterable:
        body_fn(elem)

    return get_state()


# pylint: disable=too-many-arguments
def for_stmt(
    iteration_target: Any,
    _extra_test: Callable[[], bool] | None,
    body_fn: Callable[[int], None],
    get_state: Callable[[], Tuple],
    set_state: Callable[[Tuple], None],
    _symbol_names: Tuple[str],
    opts: dict,
):
    """An implementation of the AutoGraph 'for .. in ..' statement. The interface is defined by
    AutoGraph, here we merely provide an implementation of it in terms of Catalyst primitives."""

    assert _extra_test is None

    if isinstance(iteration_target, CRange):
        # Ideally we iterate over a simple range.
        start, stop, step = iteration_target.get_raw_range()
        results = _call_catalyst_for(start, stop, step, body_fn, get_state, opts)
    else:
        # Otherwise we can attempt to convert the iteration target to an array.
        # If this fails, we must fall back to Python.
        try:
            iteration_array = jnp.asarray(iteration_target)
        except:
            iteration_array = None

        if iteration_array is not None:
            start, stop, step = 0, len(iteration_target), 1
            results = _call_catalyst_for(
                start, stop, step, body_fn, get_state, opts, iteration_array
            )
        else:
            results = _call_python_for(body_fn, get_state, iteration_target)

    # Sometimes we unpack the results of nested tracing scopes so that the user doesn't have to
    # manipulate tuples when they don't expect it. Ensure set_state receives a tuple regardless.
    if not isinstance(results, tuple):
        results = (results,)
    set_state(results)


# Prevent autograph from converting PennyLane and Catalyst library code, this can lead to many
# issues such as always tracing through code that should only be executed conditionally. We might
# have to be even more restrictive in the future to prevent issues if necessary.
module_allowlist = (
    config.DoNotConvert("pennylane"),
    config.DoNotConvert("catalyst"),
    config.DoNotConvert("jax"),
) + config.CONVERSION_RULES


def converted_call(fn, args, kwargs, caller_fn_scope=None, options=None):
    """We want AutoGraph to use our own instance of the AST transformer when recursively
    transforming functions, but otherwise duplicate the same behaviour."""

    with Patcher(
        (tf_autograph_api, "_TRANSPILER", catalyst.autograph.TRANSFORMER),
        (config, "CONVERSION_RULES", module_allowlist),
    ):
        # Dispatch range calls to a custom range class that enables constructs like
        # `for .. in range(..)` to be converted natively to `for_loop` calls. This is beneficial
        # since the Python range function does not allow tracers as arguments.
        if fn is range:
            assert len(args) in (1, 3)
            return CRange(*args)

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
                return tf_converted_call(fn.func, args, kwargs, caller_fn_scope, options)

            new_qnode = qml.QNode(qnode_call_wrapper, device=fn.device, diff_method=fn.diff_method)
            return new_qnode()

        return tf_converted_call(fn, args, kwargs, caller_fn_scope, options)


class CRange:
    """Catalyst range object.

    Can be passed to a Python for loop for native conversion to a for_loop call.
    Otherwise this class behaves exactly like the Python range class.

    Without this native conversion, all iteration targets in a Python for loop must be convertible
    to arrays. For all other inputs the loop will be treated as a regular Python loop.
    """

    def __init__(self, start_stop, stop=None, step=None):
        self._py_range = None
        self._start = start_stop if stop is not None else 0
        self._stop = stop if stop is not None else start_stop
        self._step = step if step is not None else 1

    def get_raw_range(self):
        return self._start, self._stop, self._step

    @property
    def py_range(self):
        if self._py_range is None:
            self._py_range = range(self._start, self._stop, self._step)
        return self._py_range

    # Interface of the Python range class.
    @property
    def start(self) -> int:
        return self.py_range.start

    @property
    def stop(self) -> int:
        return self.py_range.stop

    @property
    def step(self) -> int:
        return self.py_range.step

    def count(self, __value: int) -> int:
        return self.py_range.count(__value)

    def index(self, __value: int) -> int:
        return self.py_range.index(__value)

    def __len__(self) -> int:
        return self.py_range.__len__()

    def __eq__(self, __value: object) -> bool:
        return self.py_range.__eq__(__value)

    def __hash__(self) -> int:
        return self.py_range.__hash__()

    def __contains__(self, __key: object) -> bool:
        return self.py_range.__contains__(__key)

    def __iter__(self) -> Iterator[int]:
        return self.py_range.__iter__()

    def __getitem__(self, __key: SupportsIndex | slice) -> int | range:
        self.py_range.__getitem__(__key)

    def __reversed__(self) -> Iterator[int]:
        self.py_range.__reversed__()
