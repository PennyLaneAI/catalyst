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

"""AutoGraph is a source-to-source transformation system for converting imperative code into
traceable code for compute graph generation. The system is implemented in the tensorflow project.
Here, we integrate AutoGraph into Catalyst to improve the UX and allow programmers to use built-in
Python control flow and other imperative expressions rather than the functional equivalents provided
by Catalyst."""

import inspect

import pennylane as qml
from tensorflow.python.autograph.converters import (
    call_trees,
    control_flow,
    functions,
    logical_expressions,
)
from tensorflow.python.autograph.core import converter, unsupported_features_checker
from tensorflow.python.autograph.pyct import transpiler

from catalyst import ag_primitives
from catalyst.ag_utils import AutoGraphError


class CFTransformer(transpiler.PyToPy):
    """A source-to-source transformer to convert imperative style control flow into a function style
    suitable for tracing."""

    def transform(self, obj, user_context):
        """Launch the transformation process. Typically this only works on function objects.
        Here we also allow QNodes to be transformed."""

        # By default AutoGraph will only convert function or method objects, not arbitrary classes
        # such as QNode objects. Here we handle them explicitly, but we might need a more general
        # way to handle these in the future.
        # We may also need to check how this interacts with other common function decorators.
        fn = obj
        if isinstance(obj, qml.QNode):
            fn = obj.func

        if not (inspect.isfunction(fn) or inspect.ismethod(fn)):
            raise AutoGraphError(f"Unsupported object for transformation: {type(fn)}")

        new_fn, module, source_map = self.transform_function(fn, user_context)
        new_obj = new_fn

        if isinstance(obj, qml.QNode):
            new_obj = qml.QNode(new_fn, device=obj.device, diff_method=obj.diff_method)

        return new_obj, module, source_map

    def transform_ast(self, node, ctx):
        """This method must be overwritten to run all desired transformations. AutoGraph provides
        several existing transforms, but we can all also provide our own in the future."""

        # Check some unsupported Python code ahead of time.
        unsupported_features_checker.verify(node)

        # First transform the top-level function to avoid infinite recursion.
        node = functions.transform(node, ctx)

        # Convert function calls. This allows us to convert these called functions as well.
        node = call_trees.transform(node, ctx)

        # Convert Python control flow to custom 'ag__.if_stmt' ... functions.
        node = control_flow.transform(node, ctx)

        # Convert logical expressions
        node = logical_expressions.transform(node, ctx)

        return node

    def get_extra_locals(self):
        """Here we can provide any extra names that the converted function should have access to.
        At a minimum we need to provide the module with definitions for AutoGraph primitives."""

        return {"ag__": ag_primitives}

    def get_caching_key(self, user_context):
        """AutoGraph automatically caches transformed functions, the caching key is a combination of
        the function source as well as a custom key provided by us here. Changing AutoGraph options
        should trigger the function transform again, rather than getting it from cache."""

        return user_context.options

    def has_cache(self, fn, cache_key):
        """Check for the presence of the given function in the cache. Functions to be converted are
        cached by the function object itself as well as the conversion options."""

        return self._cache.has(fn, cache_key)

    def get_cached_function(self, fn, cache_key):
        """Retrieve a Python function object for a previously converted function.
        Note that repeatedly calling this function with the same arguments will result in new
        function objects every time, however their source code should be identical with the
        exception of auto-generated names."""

        # Converted functions are cached as a _PythonFnFactory object.
        cached_factory = self._cached_factory(fn, cache_key)

        # Convert to a Python function object before returning (e.g. to obtain its source code).
        new_fn = cached_factory.instantiate(
            fn.__globals__,
            fn.__closure__ or (),
            defaults=fn.__defaults__,
            kwdefaults=getattr(fn, "__kwdefaults__", None),
        )

        return new_fn


def autograph(fn):
    """Decorator that converts the given function into graph form."""

    user_context = converter.ProgramContext(TOPLEVEL_OPTIONS)

    new_fn, module, source_map = TRANSFORMER.transform(fn, user_context)
    new_fn.ag_module = module
    new_fn.ag_source_map = source_map
    new_fn.ag_unconverted = fn

    return new_fn


TOPLEVEL_OPTIONS = converter.ConversionOptions(
    recursive=True,
    user_requested=True,
    internal_convert_user_code=True,
    optional_features=None,
)


# Keep a global instance of the transformer to benefit from caching.
TRANSFORMER = CFTransformer()
