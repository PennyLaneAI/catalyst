# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone Plugin interface."""

import platform
from pathlib import Path

import pennylane as qml

from catalyst.passes import PassPlugin


def getStandalonePluginAbsolutePath():
    """Returns the absolute path to the standalone plugin"""

    ext = "so" if platform.system() == "Linux" else "dylib"
    return Path(Path(__file__).parent.absolute(), f"lib/StandalonePlugin.{ext}")


def name2pass(_name):
    """Example entry point for standalone plugin"""

    return getStandalonePluginAbsolutePath(), "standalone-switch-bar-foo"


def SwitchBarToFoo(*flags, **valued_options):
    """Applies the "standalone-switch-bar-foo" pass"""

    def add_pass_to_pipeline(**kwargs):
        pass_pipeline = kwargs.get("pass_pipeline", [])
        pass_pipeline.append(
            PassPlugin(
                getStandalonePluginAbsolutePath(),
                "standalone-switch-bar-foo",
                *flags,
                **valued_options,
            )
        )
        return pass_pipeline

    def decorator(qnode):
        if not isinstance(qnode, qml.QNode):
            # Technically, this apply pass is general enough that it can apply to
            # classical functions too. However, since we lack the current infrastructure
            # to denote a function, let's limit it to qnodes
            raise TypeError(f"A QNode is expected, got the classical function {qnode}")

        def qnode_call(*args, **kwargs):
            kwargs["pass_pipeline"] = add_pass_to_pipeline(**kwargs)
            return qnode(*args, **kwargs)

        return qnode_call

    # When the decorator is used without ()
    if len(flags) == 1 and isinstance(flags[0], qml.QNode):
        qnode = flags[0]

        def qnode_call(*args, **kwargs):
            kwargs["pass_pipeline"] = add_pass_to_pipeline(**kwargs)
            return qnode(*args, **kwargs)

        return qnode_call

    # When the decorator is used with ()
    return decorator
