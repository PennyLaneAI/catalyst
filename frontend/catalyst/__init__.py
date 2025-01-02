# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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
This package contains the Catalyst Python interface.
"""

# pylint: disable=wrong-import-position

import sys
import types
from os.path import dirname

import jaxlib as _jaxlib

_jaxlib_version = "0.4.28"
if _jaxlib.__version__ != _jaxlib_version:
    import warnings

    warnings.warn(
        "Catalyst detected a version mismatch for the installed 'jaxlib' package. Please make sure "
        "to install the exact version required by Catalyst to avoid undefined behaviour.\n"
        f"Expected: {_jaxlib_version} Found: {_jaxlib.__version__}",
    )


from catalyst._configuration import INSTALLED
from catalyst._version import __version__

try:
    if INSTALLED:
        from catalyst._revision import __revision__  # pragma: no cover
    else:
        from subprocess import check_output

        __revision__ = (
            check_output(["/usr/bin/env", "git", "rev-parse", "HEAD"], cwd=dirname(__file__))
            .decode()
            .strip()
        )
except Exception:  # pylint: disable=broad-exception-caught  # pragma: no cover
    # Revision was not determined
    __revision__ = None

if not INSTALLED:
    import os

    default_bindings_path = os.path.join(
        os.path.dirname(__file__), "../../mlir/build/python_packages/quantum"
    )
    if os.path.exists(default_bindings_path):  # pragma: no cover
        sys.path.insert(0, default_bindings_path)

# Patch certain modules to integrate our MLIR bindings with JAX. This needs to happen before any
# part of 'mlir_quantum' is imported.
# Note that '__import__' does not return the specific submodule, only the parent package.
# pylint: disable=protected-access
sys.modules["mlir_quantum.ir"] = __import__("jaxlib.mlir.ir").mlir.ir
sys.modules["mlir_quantum._mlir_libs"] = __import__("jaxlib.mlir._mlir_libs").mlir._mlir_libs
# C++ extensions to the dialects are mocked out.
sys.modules["mlir_quantum._mlir_libs._quantumDialects.gradient"] = types.ModuleType(
    "mlir_quantum._mlir_libs._quantumDialects.gradient"
)
sys.modules["mlir_quantum._mlir_libs._quantumDialects.quantum"] = types.ModuleType(
    "mlir_quantum._mlir_libs._quantumDialects.quantum"
)
sys.modules["mlir_quantum._mlir_libs._quantumDialects.catalyst"] = types.ModuleType(
    "mlir_quantum._mlir_libs._quantumDialects.catalyst"
)
sys.modules["mlir_quantum._mlir_libs._quantumDialects.mitigation"] = types.ModuleType(
    "mlir_quantum._mlir_libs._quantumDialects.mitigation"
)

from catalyst import debug, logging
from catalyst.api_extensions import *
from catalyst.api_extensions import __all__ as _api_extension_list
from catalyst.autograph import *
from catalyst.autograph import __all__ as _autograph_functions
from catalyst.compiler import CompileOptions
from catalyst.debug.assertion import debug_assert
from catalyst.jit import QJIT, qjit
from catalyst.passes import Pass, PassPlugin, pipeline
from catalyst.utils.exceptions import (
    AutoGraphError,
    CompileError,
    DifferentiableCompileError,
)

autograph_ignore_fallbacks = False
"""bool: Specify whether AutoGraph should avoid raising
warnings when conversion fails and control flow instead falls back
to being interpreted by Python at compile-time.

**Example**

In certain cases, AutoGraph will fail to convert control flow (for example,
when an object that can not be converted to a JAX array is indexed in a
loop), and will raise a warning informing of the failure.

>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f():
...     x = ["0.1", "0.2", "0.3"]
...     for i in range(3):
...         qml.RX(float(x[i]), wires=i)
...     return qml.expval(qml.PauliZ(0))
Warning: Tracing of an AutoGraph converted for loop failed with an exception:
...
If you intended for the conversion to happen, make sure that the (now dynamic)
loop variable is not used in tracing-incompatible ways, for instance by indexing a
Python list with it. In that case, the list should be wrapped into an array.

Setting this variable to ``True`` will suppress warning messages:

>>> catalyst.autograph_strict_conversion = False
>>> catalyst.autograph_ignore_fallbacks = True
>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f():
...     x = ["0.1", "0.2", "0.3"]
...     for i in range(3):
...         qml.RX(float(x[i]), wires=i)
...     return qml.expval(qml.PauliZ(0))
>>> f()
array(0.99500417)
"""

autograph_strict_conversion = False
"""bool: Specify whether AutoGraph should raise exceptions
when conversion fails, rather than falling back to interpreting
control flow by Python at compile-time.

**Example**

In certain cases, AutoGraph will fail to convert control flow (for example,
when an object that cannot be converted to a JAX array is indexed in a
loop), and will automatically fallback to interpreting the control flow
logic at compile-time via Python:

>>> dev = qml.device("lightning.qubit", wires=1)
>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f():
...     params = ["0", "1", "2"]
...     for x in params:
...         qml.RY(int(x) * jnp.pi / 4, wires=0)
...     return qml.expval(qml.PauliZ(0))
>>> f()
array(-0.70710678)

Setting this variable to ``True`` will cause AutoGraph
to error rather than fallback when conversion fails:

>>> catalyst.autograph_strict_conversion = True
>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f():
...     params = ["0", "1", "2"]
...     for x in params:
...         qml.RY(int(x) * jnp.pi / 4, wires=0)
...     return qml.expval(qml.PauliZ(0))
AutoGraphError: Could not convert the iteration target ['0', '1', '2'] to array
while processing the following with AutoGraph:
  File "<ipython-input-44-dbae11e6d745>", line 7, in f
    for x in params:
"""


__all__ = (
    "qjit",
    "QJIT",
    "autograph_ignore_fallbacks",
    "autograph_strict_conversion",
    "AutoGraphError",
    "CompileError",
    "DifferentiableCompileError",
    "debug_assert",
    "CompileOptions",
    "debug",
    "pipeline",
    "Pass",
    "PassPlugin",
    *_api_extension_list,
    *_autograph_functions,
)
