# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serialize a PennyLane ``Placement`` into the ``catalyst.backline`` module attribute and
build the pipelines that lower it. Backend hints in a node's ``init_args`` are forwarded verbatim,
so a new backend needs no change here.

Note: "node" here is a backline participant (a controller or coprocessor), distinct from
:class:`catalyst.Executor`, which deploys the ``catalyst-executor`` process a node may run on.
"""

import os
from pathlib import Path

from catalyst.utils.runtime_environment import get_lib_path

# Backend hints forwarded verbatim from a node's ``init_args`` to the attribute node.
_INIT_KEYS = ("backend_lib", "config", "data_path", "in_bytes", "out_bytes")

# A transport backend library is named for its backend and role:
#     libcatalyst_transport_<backend>_<role>.<ext>
# A ``make -C runtime`` build passes ``CMAKE_LIBRARY_OUTPUT_DIRECTORY``, so in-tree backends land
# flat in ``<RUNTIME_LIB_DIR>``; a bare ``cmake`` build instead mirrors the source tree, nesting them
# under ``transport/<backend>/``. Both are searched. Out-of-tree backends build elsewhere entirely,
# so ``CATALYST_TRANSPORT_PATH`` (``:``-separated, like ``PATH``) lists extra directories to search
# first; each is searched directly, assuming no layout within it.
_BACKEND_PATH_ENV = "CATALYST_TRANSPORT_PATH"
_BACKEND_SUBDIR = "transport"
_BACKEND_LIB_EXTS = ("so", "dylib")


def _backend_search_dirs(backend: str) -> list[Path]:
    """Directories to search for ``backend``'s libraries, most specific first."""
    override = os.environ.get(_BACKEND_PATH_ENV, "")
    dirs = [Path(p) for p in override.split(os.pathsep) if p]
    lib_dir = Path(get_lib_path("runtime", "RUNTIME_LIB_DIR"))
    dirs.append(lib_dir)
    dirs.append(lib_dir / _BACKEND_SUBDIR / backend)
    return dirs


def _resolve_backend_lib(backend: str, role: str, remote: bool) -> str:
    """Resolve a backend name and node role to the transport library the node should load.

    Args:
        backend: Backend name as given on the node, e.g. ``"cpu_verbs"``.
        role: ``"controller"`` or ``"coprocessor"``. Each backend ships one library per role.
        remote: Whether the node runs on another machine. A remote node loads the library from the
            bundle deployed alongside it, so it is named by filename only; a local node is given the
            full path into this installation.

    Returns:
        str: The library path for a local node, or its bare filename for a remote one.

    Raises:
        ValueError: If no library matches on a local node, naming every directory searched. A
            backend shipping only one role (``gpu_verbs`` has no controller library) fails here
            rather than later at ``dlopen``.
    """
    names = [f"libcatalyst_transport_{backend}_{role}.{ext}" for ext in _BACKEND_LIB_EXTS]
    if remote:
        return names[0]
    search = _backend_search_dirs(backend)
    for directory in search:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return str(candidate)
    searched = ", ".join(str(d) for d in search)
    raise ValueError(
        f"no transport backend library for backend={backend!r} role={role!r}: looked for "
        f"{names[0]} in {searched}. Transport backends are built with -DENABLE_TRANSPORT=ON "
        f"and are not shipped in the wheel; set {_BACKEND_PATH_ENV} to point at an "
        "out-of-tree build."
    )


def _node_dict(node, role: str) -> dict:
    """Map a backline node to a ``catalyst.backline`` node dict. Reads ``node.executor`` as-is.

    ``comm_host``/``oob_port`` are coprocessor-only; a controller
    carries no connection endpoint of its own.

    Args:
        node: A backline node.
        role: The node's role, used to resolve its ``backend`` to a library. An explicit
            ``init_args["backend_lib"]`` path takes precedence over ``backend``.
    """
    d: dict = {"remote": bool(node.remote)}
    if node.label is not None:
        d["name"] = node.label
    comm_host = getattr(node, "comm_host", None)
    if comm_host is not None:
        d["peer"] = comm_host
    oob_port = getattr(node, "oob_port", None)
    if oob_port is not None:
        d["oob_port"] = oob_port
    backend = getattr(node, "backend", None)
    if backend is not None and not (node.init_args or {}).get("backend_lib"):
        d["backend_lib"] = _resolve_backend_lib(backend, role, bool(node.remote))
    executor = getattr(node, "executor", None)
    triple = getattr(executor, "triple", None) if executor is not None else None
    if triple is not None:
        d["triple"] = triple
    if executor is not None and getattr(executor, "address", None):
        d["address"] = executor.address
    init = node.init_args or {}
    d.update({k: init[k] for k in _INIT_KEYS if k in init})
    return d


def _load_coprocessor_fn_libs(placement) -> None:
    """Load the library providing each in-process (local) coprocessor's CoprocessorFn."""
    import ctypes  # pylint: disable=import-outside-toplevel

    for coproc in placement.coprocessors:
        if coproc.remote:
            continue
        lib_path = getattr(coproc.coprocessor_fn, "lib_path", None)
        if lib_path:
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)


def realize_executors(placement) -> None:
    """Prepare ``placement`` for execution: launch the executors it asked for, and load the
    CoprocessorFn libraries its in-process coprocessors need. Idempotent."""
    _realize_executor(placement.controller)
    for coproc in placement.coprocessors:
        _realize_executor(coproc)
    _load_coprocessor_fn_libs(placement)


def add_transport_passes(stages):
    """Insert the transport passes into ``stages`` (in place) at the stages where their inputs exist.

    Applies to any base pipeline (e.g. ``default_pipeline()`` or a QEC one), so backline stays
    independent of the base.
    """
    from catalyst.pipelines import insert_pass_before

    for name, passes in stages:
        if name == "QuantumCompilationStage":
            passes.append("inject-transport-session")
        elif name == "BufferizationStage":
            passes.append("lower-decode-to-transport")
        elif name == "MLIRToLLVMDialectConversion":
            insert_pass_before(
                passes, ref_pass="convert-catalyst-to-llvm", new_pass="convert-transport-to-llvm"
            )
    return stages


def backline_pipeline():
    """Default Catalyst pipeline plus the transport passes."""
    from catalyst.pipelines import default_pipeline

    return add_transport_passes(default_pipeline())


def serialize_backline(placement) -> dict:
    """Serialize a ``Placement`` into the ``catalyst.backline`` attribute dict."""
    result = {
        "transport": placement.transport.name,
        "controller": _node_dict(placement.controller, "controller"),
    }
    nodes = []
    for coproc in placement.coprocessors:
        node = _node_dict(coproc, "coprocessor")
        # ``symbol`` names the decode function; the library providing it is loaded as an executor
        # plugin, so it must not be written into ``backend_lib``, which is the transport backend.
        node["symbol"] = coproc.coprocessor_fn.symbol_name
        nodes.append(node)
    if nodes:
        result["coprocessors"] = nodes
    return result


def _literal(value) -> str:
    """One node field as MLIR attribute syntax."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value} : i64"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _node_text(node: dict) -> str:
    """One node dict as ``#transport.node<...>``."""
    fields = ", ".join(f"{k} = {_literal(v)}" for k, v in node.items())
    return f"#transport.node<{fields}>"


def backline_attr_text(placement) -> str:
    """Render a ``Placement`` as ``#transport.backline<...>`` attribute syntax."""
    d = serialize_backline(placement)
    parts = [
        f"transport = {_literal(d.get('transport') or '')}",
        f"controller = {_node_text(d['controller'])}",
    ]
    if coprocs := d.get("coprocessors"):
        parts.append("coprocessors = [" + ", ".join(_node_text(c) for c in coprocs) + "]")
    return "#transport.backline<" + ", ".join(parts) + ">"


def _launch_executor(label, options):
    """Build and launch a ``catalyst.Executor`` from a node's executor options.

    The executor determines its own target triple, detecting it on the target host when the options
    do not name one.
    """
    from catalyst.executor import Executor

    options.setdefault("name", label or "executor")
    return Executor(**options).launch()


def _realize_executor(node):
    """Return the node's launched executor, building it on first use."""
    from pennylane.backline import ExecutorSpec

    executor = getattr(node, "executor", None)
    if executor is not None and not isinstance(executor, ExecutorSpec):
        return executor  # already launched
    options = (
        executor.options
        if isinstance(executor, ExecutorSpec)
        else getattr(node, "executor_options", None)
    )
    if options is None:
        return None
    executor = _launch_executor(getattr(node, "label", None), dict(options))
    object.__setattr__(node, "executor", executor)  # cache on the (frozen) node
    return executor
