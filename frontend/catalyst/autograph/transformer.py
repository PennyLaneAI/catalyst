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

"""
AutoGraph is a source-to-source transformation system for converting imperative code into
traceable code for compute graph generation. The system is implemented in the Diastatic-Malt
package (originally from TensorFlow).
Here, we integrate AutoGraph into Catalyst to improve the UX and allow programmers to use built-in
Python control flow and other imperative expressions rather than the functional equivalents provided
by Catalyst.
"""
import copy
import functools

from malt.core import config
from pennylane.capture import autograph as pl_autograph
from pennylane.capture.autograph.transformer import (
    PennyLaneTransformer,
)

import catalyst
from catalyst.autograph import ag_primitives
from catalyst.passes.pass_api import PassPipelineWrapper, QNodeWrapper
from catalyst.utils.patching import Patcher


class CatalystTransformer(PennyLaneTransformer):
    """A source-to-source transformer that extends the PennyLane transformer
    to handle Catalyst-specific objects like QNodeWrapper."""

    def transform(self, obj, user_context):
        """Launch the transformation process, with special handling for
        Catalyst's QNodeWrapper and PassPipelineWrapper."""

        fn = obj
        if isinstance(obj, QNodeWrapper):
            fn = obj
            data = []
            while isinstance(fn, QNodeWrapper):
                data.append((fn.pass_name_or_pipeline, fn.flags, fn.valued_options))
                fn = fn.qnode
            fn = obj.original_qnode.func

        else:
            new_obj, module, source_map = super().transform(obj, user_context)

            if isinstance(obj, PassPipelineWrapper):
                new_qnode = copy.copy(obj.original_qnode)
                new_qnode.func = new_obj
                data.reverse()
                for _pass, flags, kwopts in data:
                    new_qnode = PassPipelineWrapper(new_qnode, _pass, *flags, **kwopts)
                new_obj = new_qnode

            return new_obj, module, source_map

        new_fn, module, source_map = self.transform_function(fn, user_context)

        if isinstance(obj, QNodeWrapper):
            new_qnode = copy.copy(obj.original_qnode)
            new_qnode.func = new_fn
            data.reverse()
            for _pass, flags, kwopts in data:
                new_qnode = PassPipelineWrapper(new_qnode, _pass, *flags, **kwopts)
            return new_qnode, module, source_map

        return new_fn, module, source_map


def run_autograph(fn, *modules):
    """Decorator that converts the given function into graph form."""

    new_fn = pl_autograph.run_autograph(fn)

    allowed_modules = tuple(config.Convert(module) for module in modules)
    allowed_modules += ag_primitives.module_allowlist

    @functools.wraps(new_fn)
    def wrapper(*args, **kwargs):
        with Patcher(
            (ag_primitives, "module_allowlist", allowed_modules),
        ):
            return new_fn(*args, **kwargs)

    return wrapper


def autograph_source(fn):
    """Utility function to retrieve the source code of a function converted by AutoGraph.

    .. warning::

        Nested functions (those not directly decorated with ``@qjit``) are only lazily converted by
        AutoGraph. Make sure that the function has been traced at least once before accessing its
        transformed source code, for example by specifying the signature of the compiled program
        or by running it at least once.

    Args:
        fn (Callable): the original function object that was converted

    Returns:
        str: the source code of the converted function

    Raises:
        AutoGraphError: If the given function was not converted by AutoGraph, an error will be
                        raised.

    **Example**

    .. code-block:: python

        def decide(x):
            if x < 5:
                y = 15
            else:
                y = 1
            return y

        @qjit(autograph=True)
        def func(x: int):
            y = decide(x)
            return y ** 2

    >>> print(autograph_source(decide))
    def decide_1(x):
        with ag__.FunctionScope('decide', 'fscope', ag__.STD) as fscope:
            def get_state():
                return (y,)
            def set_state(vars_):
                nonlocal y
                (y,) = vars_
            def if_body():
                nonlocal y
                y = 15
            def else_body():
                nonlocal y
                y = 1
            y = ag__.Undefined('y')
            ag__.if_stmt(x < 5, if_body, else_body, get_state, set_state, ('y',), 1)
            return y
    """

    # Unwrap known objects to get the function actually transformed by autograph.
    if isinstance(fn, catalyst.QJIT):
        fn = fn.original_function
    if isinstance(fn, QNodeWrapper):
        fn = fn.original_qnode

    return pl_autograph.autograph_source(fn)


# Keep a global instance of the transformer to benefit from caching.
TRANSFORMER = CatalystTransformer()
