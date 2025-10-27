"""Patch JAX installation to fix compatibility issues with Catalyst."""

import importlib
import os
import re


def find_package_root(package_name):
    """Find a package path in a package."""
    # resolve package name to package
    package = importlib.import_module(package_name)
    return os.path.dirname(package.__file__)


jax_root = find_package_root("jax")

patch_setup = {
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
