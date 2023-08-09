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

"""Autograph is a source-to-source transformation system for converting imperative code into
traceable code for compute graph generation. The system is implemented in the tensorflow project.
Here, we integrate Autograph into Catalyst to improve the UX and allow programmers to use built-in
Python control flow and other imperative expressions rather than the functional equivalents provided
by Catalyst."""

from tensorflow.python.autograph.converters import control_flow, functions
from tensorflow.python.autograph.core import converter, unsupported_features_checker
from tensorflow.python.autograph.pyct import cfg, transpiler

from catalyst import ag_primitives
from catalyst.ag_primitives import AutographError  # pylint: disable=unused-import


class CFTransformer(transpiler.PyToPy):
    """A source-to-source transformer to convert imperative style control flow into a function style
    suitable for tracing."""

    def transform_ast(self, node, user_context):
        """This method must be overwritten to run all desired transformations. Autograph provides
        several existing transforms, but we can all also provide our own in the future."""

        # Check some unsupported Python code ahead of time.
        unsupported_features_checker.verify(node)

        # First transform the top-level function to avoid infinite recursion.
        node = functions.transform(node, user_context)

        # Convert Python control flow to custom 'ag__.if_stmt' ... functions.
        node = control_flow.transform(node, user_context)

        return node

    def get_extra_locals(self):
        """Here we can provide any extra names that the converted function should have access to.
        At a minimum we need to provide the module with definitions for Autograph primitives."""

        return {"ag__": ag_primitives}

    def get_caching_key(self, user_context):
        """Autograph automatically caches transformed functions, the caching key is a combination of
        the function source as well as a custom key provided by us here. Changing Autograph options
        should trigger the function transform again, rather than getting it from cache."""

        return user_context.options


def autograph(fn):
    """Control flow conversion decorator used for testing."""

    user_context = converter.ProgramContext(converter.STANDARD_OPTIONS)
    new_fn, module, source_map = CFTransformer().transform(fn, user_context)

    return new_fn
