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
"""xDSL universe for containing all dialects and passes."""
from functools import partial

from xdsl.passes import ModulePass
from xdsl.universe import Universe

from catalyst.python_interface import dialects, transforms

shared_dialects = ("stablehlo",)

# Create a map from dialect names to dialect classes. Dialects that are already
# provided by xDSL cannot be loaded into the multiverse, so we don't add them to
# our universe.
names_to_dialects = {}
for name in dialects.__all__:
    if (d := getattr(dialects, name)).name not in shared_dialects:

        def dialect_accessor(dialect):  # pylint: disable=missing-function-docstring
            return dialect

        # We use partial so that the correct value of `d` is returned by the function.
        # Without it, the function created by each loop iteration returns the `d` from
        # the last iteration of the loop
        names_to_dialects[d.name] = partial(dialect_accessor, d)

# Create a map from pass names to their respective ModulePass. The transforms module
# contains CompilerTransform instances as well as ModulePasses. We only want to collect
# the ModulePasses. We cannot use issubclass with instances, which is why we first
# check if isinstance(transform, type).
names_to_passes = {}
for name in transforms.__all__:
    if isinstance((t := getattr(transforms, name)), type) and issubclass(t, ModulePass):

        def transform_accessor(transform):  # pylint: disable=missing-function-docstring
            return transform

        # We use partial so that the correct value of `t` is returned by the function.
        # Without it, the function created by each loop iteration returns the `t` from
        # the last iteration of the loop
        names_to_passes[t.name] = partial(transform_accessor, t)

# The Universe is used to expose custom dialects and transforms to xDSL. It is
# specified as an entry point in PennyLane's pyproject.toml file, which makes
# it available to look up by xDSL for tools such as xdsl-opt, xdsl-gui, etc.
CATALYST_XDSL_UNIVERSE = Universe(all_dialects=names_to_dialects, all_passes=names_to_passes)
