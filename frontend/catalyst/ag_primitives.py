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
import warnings
from typing import Any, Callable, Iterator, SupportsIndex, Tuple

import jax.numpy as jnp

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


def _call_catalyst_for(start, stop, step, body_fn, get_state, enum_start=None, array_iterable=None):
    """Dispatch to a Catalyst implementation of for loops."""

    @catalyst.for_loop(start, stop, step)
    def functional_for(i):
        if enum_start is None and array_iterable is None:
            # for i in range(..)
            body_fn(i)
        elif enum_start is None:
            # for x in array
            body_fn(array_iterable[i])
        else:
            # for (i, x) in enumerate(array)
            body_fn((i + enum_start, array_iterable[i]))

        return get_state()

    return functional_for()


def _call_python_for(body_fn, get_state, non_array_iterable):
    """Fallback to a Python implementation of for loops."""

    for elem in non_array_iterable:
        body_fn(elem)

    return get_state()


def for_stmt(
    iteration_target: Any,
    _extra_test: Callable[[], bool] | None,
    body_fn: Callable[[int], None],
    get_state: Callable[[], Tuple],
    set_state: Callable[[Tuple], None],
    _symbol_names: Tuple[str],
    _opts: dict,
):
    """An implementation of the AutoGraph 'for .. in ..' statement. The interface is defined by
    AutoGraph, here we merely provide an implementation of it in terms of Catalyst primitives."""

    assert _extra_test is None

    # The general approach is to convert as much code as possible into a graph-based form:
    # - For loops over iterables will attempt a conversion of the iterable to array, and fall back
    #   to Python otherwise.
    # - For loops over a Python range will be converted to a native Catalyst for loop. However,
    #   since the now dynamic iteration variable can cause issues in downstream user code, any
    #   errors raised during the tracing of the loop body will restart the tracing process using
    #   a Python loop instead.
    # - For loops over a Python enumeration use a combination of the above, providing a dynamic
    #   iteration variable and conversion of the iterable to array. If either fails, a fallback to
    #   Python is used.
    # The fallback mechanism for tracing errors will raise a warning as the user may need to be
    # aware that the graph conversion failed, for instance for lack of converting lists into arrays,
    # but the conversion from iterable to array will fall back to Python silently.
    fallback = False

    if isinstance(iteration_target, CRange):
        start, stop, step = iteration_target.get_raw_range()
        enum_start = None
        iteration_array = None
    elif isinstance(iteration_target, CEnumerate):
        start, stop, step = 0, len(iteration_target.iteration_target), 1
        enum_start = iteration_target.start_idx
        try:
            iteration_array = jnp.asarray(iteration_target.iteration_target)
        except:  # pylint: disable=bare-except
            iteration_array = None
            fallback = True
    else:
        start, stop, step = 0, len(iteration_target), 1
        enum_start = None
        try:
            iteration_array = jnp.asarray(iteration_target)
        except:  # pylint: disable=bare-except
            iteration_array = None
            fallback = True

    # Attempt to trace the Catalyst for loop.
    if not fallback:
        try:
            results = _call_catalyst_for(
                start, stop, step, body_fn, get_state, enum_start, iteration_array
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            fallback = True

            warnings.warn(
                f"Tracing of an AutoGraph converted for loop failed with the following exception:\n"
                f"{type(e).__name__}: {e}\n"
                f"\n"
                f"If you intended for the conversion to happen, make sure that the (now dynamic) "
                f"loop variable is not used in tracing-incompatible ways, for instance by indexing "
                f"a Python list with it. In that case, the list should be wrapped into an array.\n"
                f"To understand different types of JAX tracing errors, please refer to the guide "
                f"at: https://jax.readthedocs.io/en/latest/errors.html\n"
                f"If you did not intend for the conversion to happen, you may safely ignore this "
                f"warning."
            )

    # If anything goes wrong, we fall back to Python.
    if fallback:
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
            return CRange(*args, **(kwargs if kwargs is not None else {}))
        elif fn is enumerate:
            return CEnumerate(*args, **(kwargs if kwargs is not None else {}))

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
        """Get the raw values defining this range: start, stop, step."""
        return self._start, self._stop, self._step

    @property
    def py_range(self):
        """Access the underlying Python range object. If it doesn't exist, create one."""
        if self._py_range is None:
            self._py_range = range(self._start, self._stop, self._step)
        return self._py_range

    # Interface of the Python range class.
    # pylint: disable=missing-function-docstring

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


class CEnumerate(enumerate):
    """Catalyst enumeration object.

    Can be passed to a Python for loop for conversion into a for_loop call. The loop index, as well
    as the iterable element will be provided to the loop body.
    Otherwise this class behaves exactly like the Python enumerate class.

    Note that the iterable must be convertible to an array, otherwise the loop will be treated as a
    regular Python loop.
    """

    def __init__(self, iterable, start=0):
        self.iteration_target = iterable
        self.start_idx = start
