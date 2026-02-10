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

"""XDSL Plugin Interface"""

# This file contains what looks like a plugin
# but in reality it is just not a "real" MLIR plugin.
# It just follows the same convention to be able to add passes
# that are implemented in xDSL

import importlib.util
from pathlib import Path


def getXDSLPluginAbsolutePath():
    """Returns a fake path"""
    try:
        # pylint: disable-next=import-outside-toplevel,unused-import
        import xdsl
    except ImportError as e:
        # pragma: nocover
        msg = "The xdsl plugin requires the xdsl package to be installed"
        raise ImportError(msg) from e

    return Path("xdsl-does-not-use-a-real-path")


def name2pass(name):
    """Identity function

    This function is useful if we want a map from user-facing pass name
    to mlir-facing pass names.
    """
    return getXDSLPluginAbsolutePath(), name
