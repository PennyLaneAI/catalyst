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

"""Utility functions for Catalyst's AutoGraph module. This module can be safely imported without
a TensorFlow installation."""

import inspect

import pennylane as qml

# pylint: disable=import-outside-toplevel


class AutoGraphError(Exception):
    """Errors related to Catalyst's AutoGraph module."""


def _test_ag_import():
    """Reusable function for attempting to import Catalyst's AutoGraph module, which requires
    TensorFlow to be installed."""

    try:
        import catalyst.autograph  # pylint: disable=unused-import
    except ImportError as e:
        raise ImportError(
            "The AutoGraph feature in Catalyst requires TensorFlow. "
            "Please install TensorFlow (https://www.tensorflow.org/install) and ensure it is "
            "available in the current environment."
        ) from e


def autograph_source(fn):
    """Utility function to retrieve the source code of a function converted by AutoGraph.

    Args:
        fn (Callable): the original function object that was converted

    Returns:
        str: the source code of the converted function

    Raises:
        AutoGraphError: If the given function was not converted by AutoGraph, an error will be
                        raised.
        ImportError: If TensorFlow is not installed, an error will be raised.

    **Example**

    .. code-block:: python

        def decide(x):
            if x < 5:
                y = 15
            else:
                y = 1
            return y

        @qjit(autograph=True)
        def func(x):
            y = decide(x)
            return y ** 2

        print(autograph_source(decide))
    """
    _test_ag_import()
    from catalyst import QJIT
    from catalyst.ag_primitives import STD as STD_OPTIONS
    from catalyst.autograph import TOPLEVEL_OPTIONS, TRANSFORMER

    # Handle directly converted objects.
    if hasattr(fn, "ag_unconverted"):
        return inspect.getsource(fn)

    # Unwrap known objects to get the function actually transformed by autograph.
    if isinstance(fn, QJIT):
        fn = fn.original_function
    if isinstance(fn, qml.QNode):
        fn = fn.func

    if TRANSFORMER.has_cache(fn, STD_OPTIONS):
        new_fn = TRANSFORMER.get_cached_function(fn, STD_OPTIONS)
        return inspect.getsource(new_fn)
    elif TRANSFORMER.has_cache(fn, TOPLEVEL_OPTIONS):
        new_fn = TRANSFORMER.get_cached_function(fn, TOPLEVEL_OPTIONS)
        return inspect.getsource(new_fn)

    raise AutoGraphError(
        "The given function was not converted by AutoGraph. If you expect the"
        "given function to be converted, please submit a bug report."
    )


def print_code(fn):
    """Convenience function for testing to print the transformed code."""

    print(autograph_source(fn))  # pragma: nocover


def check_cache(fn):
    """Convenience function for testing to check the TRANSFORMER cache."""
    _test_ag_import()
    from catalyst.ag_primitives import STD as STD_OPTIONS
    from catalyst.autograph import TOPLEVEL_OPTIONS, TRANSFORMER

    return TRANSFORMER.has_cache(fn, STD_OPTIONS) or TRANSFORMER.has_cache(fn, TOPLEVEL_OPTIONS)


def run_autograph(fn):
    """Safe wrapper around the AutoGraph decorator from the catalyst.autograph module."""
    _test_ag_import()
    from catalyst.autograph import autograph

    return autograph(fn)
