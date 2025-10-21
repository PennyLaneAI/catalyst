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

import importlib
import re
import glob
import sys
import types
from os.path import dirname

import jaxlib as _jaxlib

_jaxlib_version = "0.6.2"
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

from jaxlib.mlir._mlir_libs import _mlir as _ods_cext
from catalyst.jax_extras.patches import patch_primitives
from catalyst.jax_extras.patches import mock_attribute

patch_primitives()
_ods_cext.globals = mock_attribute(
    _ods_cext.globals, "register_traceback_file_exclusion", lambda x: None
)

# Disable JAX's Shardy partitioner for JAX 0.7+ compatibility
# Shardy adds 'sdy' dialect attributes that Catalyst doesn't support yet
import jax

jax.config.update("jax_use_shardy_partitioner", False)

from catalyst import debug, logging, passes
from catalyst.api_extensions import *
from catalyst.api_extensions import __all__ as _api_extension_list
from catalyst.autograph import *
from catalyst.autograph import __all__ as _autograph_functions
from catalyst.compiler import CompileOptions
from catalyst.debug.assertion import debug_assert
from catalyst.jit import QJIT, qjit
from catalyst.passes.pass_api import pipeline
from catalyst.utils.exceptions import (
    AutoGraphError,
    CompileError,
    DifferentiableCompileError,
    PlxprCaptureCFCompatibilityError,
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
    "PlxprCaptureCFCompatibilityError",
    "CompileError",
    "DifferentiableCompileError",
    "debug_assert",
    "CompileOptions",
    "debug",
    "passes",
    "pipeline",
    *_api_extension_list,
    *_autograph_functions,
)


def find_package_root(package_name):
    """Find a package path in a package."""
    # resolve package name to package
    package = importlib.import_module(package_name)
    return os.path.dirname(package.__file__)


pennylane_root = find_package_root("pennylane")
jax_root = find_package_root("jax")

patch_setup = {
    "pennylane_jax07_0": {
        "file_path": pennylane_root + "/workflow/_capture_qnode.py",
        "pattern": r"DynamicJaxprTracer\(jaxpr_trace, o\)",
        "replacement": "DynamicJaxprTracer(jaxpr_trace, o, None)",
    },
    "pennylane_jax07_1": {
        "file_path": pennylane_root + "/workflow/_capture_qnode.py",
        "pattern": (
            r"out_tracers = \[pe\.DynamicJaxprTracer\("
            r"jaxpr_trace, o, None\) for o in new_shapes\]\s+"
            r"eqn = jax\.core\.new_jaxpr_eqn\(\s+"
            r"invars,\s+"
            r"\[jaxpr_trace\.makevar\(o\) "
            r"for o in out_tracers\],\s+"
            r"qnode_prim,\s+"
            r"params,\s+"
            r"jax\.core\.no_effects,\s+"
            r"source_info=source_info,?\s+"
            r"\)\s+"
            r"jaxpr_trace\.frame\.add_eqn\(eqn\)\s+"
            r"return out_tracers"
        ),
        "replacement": (
            "eqn, out_tracers = jaxpr_trace.make_eqn("
            "tracers, new_shapes, qnode_prim, params, [], source_info)\n"
            "    jaxpr_trace.frame.add_eqn(eqn)\n"
            "    return out_tracers"
        ),
    },
    "pennylane_jax07_2": {
        "file_path": pennylane_root + "/capture/dynamic_shapes.py",
        "pattern": (
            r"invars = \[jaxpr_trace\.getvar\(x\) "
            r"for x in tracers\]\s+"
            r"eqn = jax\.core\.new_jaxpr_eqn\(\s+"
            r"invars,\s+"
            r"returned_vars,\s+"
            r"primitive,\s+"
            r"params,\s+"
            r"jax\.core\.no_effects,\s+"
            r"\)\s+"
            r"jaxpr_trace\.frame\.add_eqn\(eqn\)\s+"
            r"return out_tracers"
        ),
        "replacement": (
            "out_avals = [t.aval for t in out_tracers]\n"
            "        eqn, out_tracers = jaxpr_trace.make_eqn("
            "tracers, out_avals, primitive, params, [], source_info)\n"
            "        jaxpr_trace.frame.add_eqn(eqn)\n"
            "        return out_tracers"
        ),
    },
    "jax_07_0": {
        "file_path": jax_root + "/_src/pjit.py",
        "pattern": r"arg\.var",
        "replacement": "arg.val",
    },
    "jax_07_1": {
        "file_path": jax_root + "/_src/pjit.py",
        "pattern": r"outvars = map\(trace\.frame\.newvar, _out_type\(jaxpr\)\)",
        "replacement": "outvars = list(map(trace.frame.newvar, _out_type(jaxpr)))",
    },
    "jax_07_2": {
        "file_path": jax_root + "/_src/pjit.py",
        "pattern": (
            r"eqn = core\.new_jaxpr_eqn\(\s*"
            r"\[arg\.val for arg in args\], "
            r"outvars, jit_p, params,\s*"
            r"jaxpr\.effects, source_info\)\s*"
            r"trace\.frame\.add_eqn\(eqn\)\s*"
            r"out_tracers = \[pe\.DynamicJaxprTracer\("
            r"trace, v\.aval, v, source_info\)\s*"
            r"for v in outvars\]"
        ),
        "replacement": (
            "out_avals = [v.aval for v in outvars]\n"
            "    out_tracers = [pe.DynamicJaxprTracer("
            "trace, aval, v, source_info) "
            "for aval, v in zip(out_avals, outvars)]\n"
            "    eqn, out_tracers = trace.make_eqn("
            "args, out_avals, jit_p, params, "
            "jaxpr.effects, source_info, out_tracers=out_tracers)\n"
            "    trace.frame.add_eqn(eqn)"
        ),
    },
}


for patch_name, patch_info in patch_setup.items():
    with open(patch_info["file_path"], "r", encoding="utf-8") as f:
        content = f.read()
    if re.search(patch_info["pattern"], content):
        with open(patch_info["file_path"], "w", encoding="utf-8") as f:
            f.write(re.sub(patch_info["pattern"], patch_info["replacement"], content))
    else:
        print(f"No match found for {patch_name} in {patch_info['file_path']}")
