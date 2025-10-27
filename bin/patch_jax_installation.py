import importlib
import os
import re


def find_package_root(package_name):
    """Find a package path in a package."""
    # resolve package name to package
    package = importlib.import_module(package_name)
    return os.path.dirname(package.__file__)


pennylane_root = find_package_root("pennylane")
jax_root = find_package_root("jax")

patch_setup = {
    # "pennylane_jax07_0": {
    #     "file_path": pennylane_root + "/workflow/_capture_qnode.py",
    #     "pattern": r"DynamicJaxprTracer\(jaxpr_trace, o\)",
    #     "replacement": "DynamicJaxprTracer(jaxpr_trace, o, None)",
    # },
    # "pennylane_jax07_1": {
    #     "file_path": pennylane_root + "/workflow/_capture_qnode.py",
    #     "pattern": (
    #         r"out_tracers = \[pe\.DynamicJaxprTracer\("
    #         r"jaxpr_trace, o, None\) for o in new_shapes\]\s+"
    #         r"eqn = jax\.core\.new_jaxpr_eqn\(\s+"
    #         r"invars,\s+"
    #         r"\[jaxpr_trace\.makevar\(o\) "
    #         r"for o in out_tracers\],\s+"
    #         r"qnode_prim,\s+"
    #         r"params,\s+"
    #         r"jax\.core\.no_effects,\s+"
    #         r"source_info=source_info,?\s+"
    #         r"\)\s+"
    #         r"jaxpr_trace\.frame\.add_eqn\(eqn\)\s+"
    #         r"return out_tracers"
    #     ),
    #     "replacement": (
    #         "eqn, out_tracers = jaxpr_trace.make_eqn("
    #         "tracers, new_shapes, qnode_prim, params, [], source_info)\n"
    #         "    jaxpr_trace.frame.add_eqn(eqn)\n"
    #         "    return out_tracers"
    #     ),
    # },
    # "pennylane_jax07_2": {
    #     "file_path": pennylane_root + "/capture/dynamic_shapes.py",
    #     "pattern": (
    #         r"invars = \[jaxpr_trace\.getvar\(x\) "
    #         r"for x in tracers\]\s+"
    #         r"eqn = jax\.core\.new_jaxpr_eqn\(\s+"
    #         r"invars,\s+"
    #         r"returned_vars,\s+"
    #         r"primitive,\s+"
    #         r"params,\s+"
    #         r"jax\.core\.no_effects,\s+"
    #         r"\)\s+"
    #         r"jaxpr_trace\.frame\.add_eqn\(eqn\)\s+"
    #         r"return out_tracers"
    #     ),
    #     "replacement": (
    #         "out_avals = [t.aval for t in out_tracers]\n"
    #         "        eqn, out_tracers = jaxpr_trace.make_eqn("
    #         "tracers, out_avals, primitive, params, [], source_info)\n"
    #         "        jaxpr_trace.frame.add_eqn(eqn)\n"
    #         "        return out_tracers"
    #     ),
    # },
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
