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
build the pipelines that lower it. A node's hardware and the placement transport select the
compiled transport backend.

A placement adds three passes to the compiler's stages: ``inject-transport-session`` emits the
session's bring-up and teardown, ``lower-decode-to-transport`` turns a decode into a transport round
over that session, and ``convert-transport-to-llvm`` lowers the result to runtime calls. A placement
naming a ``qec_code`` also encodes the circuit, which registers the encoding passes and adds
``convert-qecp-to-llvm``. See ``_TRANSPORT_PASSES`` and ``_QEC_LOWERING_PASSES`` for where each is
inserted.

Note: "node" here is a backline participant (a controller or coprocessor), distinct from
:class:`catalyst.Executor`, which deploys the ``catalyst-executor`` process a node may run on.
"""

import ctypes
import os
from pathlib import Path
from typing import NamedTuple

import jax
import pennylane as qp
from jax.interpreters.mlir import ir
from pennylane.backline import Node, Placement
from pennylane.devices import Device

from catalyst.executor import Executor
from catalyst.pipelines import insert_pass_before
from catalyst.utils.exceptions import CompileError
from catalyst.utils.runtime_environment import get_lib_path

# Backend hints forwarded verbatim from a node's ``init_args`` to the attribute node.
_INIT_KEYS = ("backend_lib", "config")

# Concrete runtime backends are a Catalyst implementation detail. PennyLane describes only the
# transport protocol and the hardware on which each node runs.
_BACKENDS = {
    ("rdma", "cpu"): "cpu_verbs",
    ("rdma", "gpu"): "gpu_verbs",
    ("rdma", "fpga"): "hwhs",
    ("memcpy", "cpu"): "memcpy",
    ("memcpy", "gpu"): "memcpy_gpu",
}

# A transport backend library is named for its backend and role:
#     libcatalyst_transport_<backend>_<role>.<ext>
# A ``make -C runtime`` build passes ``CMAKE_LIBRARY_OUTPUT_DIRECTORY``, so in-tree backends land
# flat in ``<RUNTIME_LIB_DIR>``. A bare ``cmake`` build instead mirrors the source tree under
# ``transport/<transport>/[<backend>/]``. Both are searched. Out-of-tree backends build elsewhere,
# so ``CATALYST_TRANSPORT_PATH`` (``:``-separated, like ``PATH``) lists extra directories to search
# first; each is searched directly, assuming no layout within it.
_BACKEND_PATH_ENV = "CATALYST_TRANSPORT_PATH"
_BACKEND_SUBDIR = "transport"
_BACKEND_LIB_EXTS = ("so", "dylib")

# Every dispatched node needs these in its executor: the compiled program calls
# ``__catalyst__transport__*`` and ``__catalyst__rt__*`` directly.
_EXECUTOR_RUNTIME_PLUGINS = ("librt_transport.so", "librt_capi.so")


def _resolve_backend(transport: str, hardware: str) -> str:
    """Return Catalyst's concrete backend for a transport and hardware pair."""
    try:
        return _BACKENDS[(transport, hardware)]
    except KeyError:
        raise ValueError(
            f"unsupported backline transport/hardware combination: "
            f"transport={transport!r}, hardware={hardware!r}"
        ) from None


def _backend_search_dirs(transport: str, backend: str) -> list[Path]:
    """Directories to search for a transport backend's libraries, most specific first."""
    override = os.environ.get(_BACKEND_PATH_ENV, "")
    dirs = [Path(p) for p in override.split(os.pathsep) if p]
    lib_dir = Path(get_lib_path("runtime", "RUNTIME_LIB_DIR"))
    dirs.append(lib_dir)
    transport_dir = lib_dir / _BACKEND_SUBDIR / transport
    dirs.append(transport_dir)
    dirs.append(transport_dir / backend)
    return dirs


def _resolve_backend_lib(transport: str, hardware: str, role: str, remote: bool) -> str:
    """Resolve transport, hardware, and node role to the backend library the node should load.

    Args:
        transport: Placement transport name, such as ``"rdma"`` or ``"memcpy"``.
        hardware: Node hardware name: ``"cpu"``, ``"gpu"``, or ``"fpga"``.
        role: ``"controller"`` or ``"coprocessor"``. Each backend ships one library per role.
        remote: Whether the node runs on another machine. A remote node loads the library from the
            bundle deployed alongside it, so it is named by filename only, and always with a ``.so``
            extension: a node on another machine has to be Linux. A local node is given the full
            path into this installation, which may be either ``.so`` or ``.dylib``.

    Returns:
        str: The library path for a local node, or its bare filename for a remote one.

    Raises:
        ValueError: If the transport/hardware pair is unsupported or no library matches on a local
            node, naming every directory searched.
    """
    backend = _resolve_backend(transport, hardware)
    names = [f"libcatalyst_transport_{backend}_{role}.{ext}" for ext in _BACKEND_LIB_EXTS]
    if remote:
        return names[0]
    search = _backend_search_dirs(transport, backend)
    for directory in search:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return str(candidate)
    searched = ", ".join(str(d) for d in search)
    raise ValueError(
        f"no transport backend library for transport={transport!r} hardware={hardware!r} "
        f"backend={backend!r} role={role!r}: looked for {names[0]} in {searched}. "
        "Transport backends are built with -DENABLE_TRANSPORT=ON and are not shipped in the "
        f"wheel; set {_BACKEND_PATH_ENV} to point at an out-of-tree build."
    )


def _out_of_process(node: Node) -> bool:
    """Whether the node's code is dispatched to an executor rather than run in this process."""
    return node.executor_options is not None or node.executor is not None


def _node_dict(node: Node, role: str, transport: str) -> dict:
    """Map a backline node to a ``catalyst.backline`` node dict. Reads ``node.executor`` as-is.

    Coprocessor connection information is carried by ``endpoint`` in current PennyLane Backline;
    a controller carries no connection endpoint of its own.

    Args:
        node: A backline node.
        role: The node's role, used when resolving the backend library.
        transport: The placement's transport name. Together with ``node.hardware``, this selects
            the concrete backend. An explicit ``init_args["backend_lib"]`` path takes precedence.
    """
    d: dict = {"out_of_process": bool(_out_of_process(node))}
    if node.name is not None:
        d["name"] = node.name
    if role == "controller":
        d["in_bytes"] = node.in_bytes
        d["out_bytes"] = node.out_bytes

    endpoint = getattr(node, "endpoint", None)
    if endpoint is not None:
        d["peer"] = endpoint.host
        if endpoint.port is not None:
            d["oob_port"] = endpoint.port

    init = node.init_args or {}
    if unknown := sorted(set(init) - set(_INIT_KEYS)):
        raise CompileError(
            f"backline node has unrecognized init_args {unknown}; recognized: {list(_INIT_KEYS)}."
        )
    hardware = getattr(node, "hardware", None)
    if hardware is not None and not init.get("backend_lib"):
        d["backend_lib"] = _resolve_backend_lib(transport, hardware, role, bool(node.remote))
    # A node may be handed any object as its executor, so its fields are read rather than assumed.
    executor = node.executor
    if executor is not None:
        triple = getattr(executor, "triple", None)
        if triple is not None:
            d["triple"] = triple
        try:
            address = executor.address
        except (AttributeError, RuntimeError) as e:
            # Either it has no ``address`` at all, or it has neither launched nor settled on one.
            # The cause says which; both leave the compiled program with nowhere to dispatch.
            who = f"{role} {node.name!r}" if node.name is not None else f"unnamed {role}"
            raise CompileError(
                f"backline {who} has an executor ({type(executor).__name__}) that cannot say "
                f"where it serves, so the compiled program would have nowhere to dispatch it: "
                f"{e}. Pass executor_options= and let the compiler settle the address, or launch "
                f"the executor before compiling."
            ) from e
        if address:  # also rejects "", which would serialize as an empty, unusable address
            d["address"] = address
    d.update({k: init[k] for k in _INIT_KEYS if k in init})
    return d


def _load_coprocessor_fn_libs(placement: Placement) -> None:
    """Load the library providing each in-process coprocessor's CoprocessorFn.

    Keyed on the coprocessor being in this process rather than on ``remote``: one dispatched to an
    executor loads the library itself, and its own installation is where the path resolves -- even
    when that executor is a subprocess of this one on the same machine.
    """
    for coproc in placement.coprocessors:
        if _out_of_process(coproc):
            continue
        lib_path = getattr(coproc.coprocessor_fn, "lib_path", None)
        if lib_path:
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)


def launch_executors(placement: Placement | None) -> None:
    """Deploy the placement's executors and load the CoprocessorFn libraries its in-process
    coprocessors need.
    """
    if placement is None:
        return
    for node in (placement.controller, *placement.coprocessors):
        executor = _realize_executor(node)
        if executor is not None:
            executor.launch()
    _load_coprocessor_fn_libs(placement)


def serialize_backline(placement: Placement) -> dict:
    """Serialize a ``Placement`` into the ``catalyst.backline`` attribute dict."""
    transport = placement.transport.name
    result = {
        "transport": transport,
        "controller": _node_dict(placement.controller, "controller", transport),
    }
    nodes = []
    for coproc in placement.coprocessors:
        node = _node_dict(coproc, "coprocessor", transport)
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


def backline_attr_text(placement: Placement) -> str:
    """Render a ``Placement`` as ``#transport.backline<...>`` attribute syntax."""
    d = serialize_backline(placement)
    parts = [
        f"transport = {_literal(d['transport'])}",
        f"controller = {_node_text(d['controller'])}",
    ]
    if coprocs := d.get("coprocessors"):
        parts.append("coprocessors = [" + ", ".join(_node_text(c) for c in coprocs) + "]")
    return "#transport.backline<" + ", ".join(parts) + ">"


def _check_machine_agrees(node: Node, host, address=None, preset: bool = False) -> None:
    """Reject an executor whose location contradicts the node's ``remote``.

    An ``address`` alone is not checked, since it may name either machine.

    Args:
        node: The backline node the executor belongs to.
        host: The host the executor ssh's to, if any.
        address: The address an attached executor serves on, if any.
        preset: Whether the executor was attached directly rather than built from options.

    Raises:
        CompileError: If ``remote`` and the executor's location disagree.
    """
    fix = "drop its executor" if preset else "drop 'host' from its executor_options"
    if host and not node.remote:
        raise CompileError(
            f"backline node is not remote but its executor is deployed to host={host!r} over ssh, "
            f"which puts it on another machine -- so the node's libraries would be looked for in "
            f"this installation and loaded from that one. Set remote=True on the node, or {fix} to "
            f"run it here as a subprocess."
        )
    if node.remote and not (host or address or preset):
        raise CompileError(
            "backline node is remote but its executor_options name neither a 'host' to deploy it "
            "to nor an 'address' to attach to, which asks for a subprocess of this process on this "
            "machine. Pass a 'host', or leave remote unset to run the node here."
        )


def _coprocessor_fn_lib(node: Node) -> Path | None:
    """The library providing a coprocessor's CoprocessorFn, or ``None`` if it names none."""
    lib_path = getattr(getattr(node, "coprocessor_fn", None), "lib_path", None)
    return Path(lib_path) if lib_path else None


def _executor_plugins(node: Node, given) -> list[str]:
    """The plugins an executor needs: the runtime libraries, those ``given``, then a coprocessor's
    decode function and a controller's device runtime, each appended only if not already listed."""
    # Deferred: qjit_device imports the device stack, which imports this module.
    from catalyst.device.qjit_device import (  # pylint: disable=import-outside-toplevel
        extract_backend_info,
    )

    remote = bool(node.remote)
    # The executor resolves a bare plugin filename against its workspace and takes an absolute path
    # as given, so a node running here must name these by path: this installation's runtime is not
    # in the loader's search path, and a failed dlopen is only reported by the executor's log.
    rt_lib = None if remote else Path(get_lib_path("runtime", "RUNTIME_LIB_DIR"))
    plugins = [
        lib if remote else str(rt_lib / lib)
        for lib in _EXECUTOR_RUNTIME_PLUGINS
        if not any(Path(p).name == lib for p in given)
    ]
    plugins += list(given)
    implied = []
    fn_lib = _coprocessor_fn_lib(node)
    if fn_lib is not None:
        implied.append((fn_lib, fn_lib.name))
    device = getattr(node, "device", None)
    if device is not None:
        # Last: plugins open RTLD_GLOBAL and the first definition of a symbol wins, and the device
        # runtime carries its own copy of the runtime's exception types.
        lib = Path(extract_backend_info(device).lpath)
        implied.append((lib, lib.with_suffix(f".{_BACKEND_LIB_EXTS[0]}").name))
    for lib, remote_name in implied:
        # A node on another machine resolves a library by filename, the deployed bundle supplying
        # the file; one running here resolves it by path into this installation.
        name = remote_name if remote else lib.name
        if not any(Path(p).name == name for p in plugins):
            plugins.append(name if remote else str(lib))
    return plugins


def _realize_executor(node: Node) -> Executor | None:
    """The node's executor, built from its ``executor_options`` on first use and cached on it.

    ``None`` when the node requested none, which leaves it running in this process.

    Raises:
        CompileError: If ``remote`` and the executor disagree about which machine the node is on, if
            a remote node has no executor to reach it by, or if a ``host`` is named without a
            ``port``.
    """
    executor = node.executor
    options = node.executor_options
    if node.remote and executor is None and options is None:
        raise CompileError(
            "backline node is remote but was given no executor to reach it by. A node on another "
            "machine is reached by dispatching its code to an executor deployed there, so pass "
            "executor_options with a 'host'. To run the node on this machine, leave remote unset."
        )
    if executor is not None:
        missing = [m for m in ("address", "launch") if not hasattr(type(executor), m)]
        if missing:
            raise CompileError(
                f"backline node was given {type(executor).__name__} as its executor, which is "
                f"missing {missing}. A node's executor has to say where it serves and be "
                f"launchable, since the compiled program carries that address and deploys it "
                f"before dispatching. Pass executor_options= and let the compiler build one."
            )
        _check_machine_agrees(node, getattr(executor, "host", None), preset=True)
        return executor
    if options is None:
        return None
    options = dict(options)

    options.setdefault("name", node.name or "executor")
    options["plugins"] = _executor_plugins(node, options.get("plugins") or ())
    fn_lib = _coprocessor_fn_lib(node)
    if fn_lib is not None and node.remote:
        # Named by filename among the plugins above, which resolves against the workspace -- so on
        # another machine the file has to travel there alongside whatever else is deployed.
        options["deploy"] = [*options.get("deploy", []), str(fn_lib)]
    _check_machine_agrees(node, options.get("host"), address=options.get("address"))
    if options.get("host") and options.get("port") is None:
        raise CompileError(
            f"backline node's executor_options name host={options['host']!r} without a port. Its "
            f"address would then settle only on deployment, so compiling would have to ssh to that "
            f"host to learn it. Pin 'port' to keep the address predictable and let the deployment "
            f"wait until execution."
        )
    executor = Executor(**options)
    if executor.resolve() is None:
        # Only a local executor reaches here: it searches for a free port, so its address is
        # knowable only once it is up. That search stays on loopback and costs no network.
        executor.launch()
    object.__setattr__(node, "executor", executor)  # cache on the (frozen) node
    return executor


def attach_backline_attr(mlir_module, placement: Placement) -> None:
    """Serialize ``placement`` onto ``mlir_module`` as the ``catalyst.backline`` attribute."""
    with mlir_module.context:
        mlir_module.operation.attributes["catalyst.backline"] = ir.Attribute.parse(
            backline_attr_text(placement)
        )


def module_attributes(device: Device) -> dict[str, str]:
    """The MLIR attributes ``device`` puts on the module of a QNode that runs on it.

    Values are plain Python, ready for ``get_mlir_attribute_from_pyval``.

    Args:
        device: A PennyLane device that may carry a backline placement.

    Returns:
        dict: Attribute name to value, empty for a device that implies no attributes.
    """
    # Only a controller is tagged here, because a QNode is always the controller. The coprocessor
    # runs a precompiled function, so it is not traced from Python and nothing on this side captures
    # its code. ``inject-transport-session`` creates the coprocessor's module itself and sets its
    # role there.
    # A controller running in this process needs no module of its own either,
    # so only a dispatched one is tagged.
    controller = getattr(getattr(device, "placement", None), "controller", None)
    if controller is not None and controller.remote:
        return {"catalyst.backline_role": "controller"}
    return {}


# ========================================= the pipelines =========================================


def _insert_passes(stages, specs):
    """Insert each ``(stage, pass, ref_pass)`` of ``specs`` into ``stages`` in place, returning it.

    A pass already in its stage, or a spec whose stage is absent, is skipped.
    """
    for stage, new_pass, ref_pass in specs:
        for name, passes in stages:
            if name != stage or new_pass in passes:
                continue
            if ref_pass is None:
                passes.append(new_pass)
            else:
                insert_pass_before(passes, ref_pass=ref_pass, new_pass=new_pass)
    return stages


# --- per QNode, at trace: the QEC encoding that rewrites a circuit into the code ---


class _QecLowering(NamedTuple):
    """How a QEC code is lowered: the logical-qubit count and the encoder's own parameters."""

    k: int
    qec_code: str
    number_errors: int


# Lowering parameters per QEC code, keyed by the name a placement declares.
_QEC_CODES = {
    "steane": _QecLowering(k=1, qec_code="Steane", number_errors=1),
}


def _validated_qec_code(code):
    """``code`` if it has a known lowering, or ``None`` if there is none to apply.

    Raises:
        ValueError: If a code is named but has no lowering here.
    """
    if code is not None and code not in _QEC_CODES:
        raise ValueError(f"no lowering for QEC code {code!r}; known codes: {sorted(_QEC_CODES)}.")
    return code


def _qec_pass_specs(code):
    """The encoding chain for ``code`` as ``(pass_name, kwargs)`` pairs, in application order.

    Names are literal rather than read off the transforms, which cannot be imported mid-trace; a
    unit test pins them against the transforms themselves.
    """
    params = _QEC_CODES[code]
    return (
        ("convert-quantum-to-qecl", {"k": params.k}),
        ("symbol-dce", {}),
        ("inject-noise-to-qecl", {}),
        (
            "convert-qecl-to-qecp",
            {"qec_code": params.qec_code, "number_errors": params.number_errors},
        ),
        ("convert-qecp-to-quantum", {}),
    )


def device_pass_pipeline(device: Device) -> tuple:
    """The passes ``device`` requires of any QNode that runs on it, in application order.

    A device naming a ``qec_code`` asks for its circuits to run encoded in that code; this turns
    that into the pass chain. A device requiring nothing answers the same question with ``()``.

    Returns:
        tuple: ``BoundTransform``s to append to the QNode's own pass pipeline.
    """
    code = _validated_qec_code(getattr(device, "qec_code", None))
    if code is None:
        return ()
    return tuple(
        qp.transforms.core.BoundTransform(qp.transform(pass_name=name), kwargs=kwargs)
        for name, kwargs in _qec_pass_specs(code)
    )


# --- per program, at compile: passes inserted into the compiler's own stages ---


# Each entry is ``(stage, pass, pass to insert before)``; a ``None`` reference appends to the stage.
_TRANSPORT_PASSES = (
    ("QuantumCompilationStage", "inject-transport-session", None),
    ("BufferizationStage", "lower-decode-to-transport", None),
    ("MLIRToLLVMDialectConversion", "convert-transport-to-llvm", "convert-catalyst-to-llvm"),
)

# Added on top of the above only for a placement naming a ``qec_code``.
_QEC_LOWERING_PASSES = (
    ("MLIRToLLVMDialectConversion", "convert-qecp-to-llvm", "convert-quantum-to-llvm"),
)


def placement_pipeline(placement: Placement, stages: list) -> list:
    """``stages`` with the passes ``placement`` needs to be lowered, ready to compile with.

    Args:
        placement: The :class:`~pennylane.backline.Placement` to compile for.
        stages: The base pipeline, as ``(stage_name, passes)`` pairs.

    Returns:
        list: ``stages``, edited in place and returned.
    """
    stages = _insert_passes(stages, _TRANSPORT_PASSES)
    if _validated_qec_code(placement.qec_code) is not None:
        # Importing these registers the encoding passes.
        # pylint: disable=import-outside-toplevel,unused-import
        from catalyst.python_interface.transforms import qecl, qecp  # noqa: F401

        stages = _insert_passes(stages, _QEC_LOWERING_PASSES)
    return stages


# ------------------------------------ finding the placement -------------------------------------


def _placement_of(obj, is_device=False):
    """The backline placement ``obj`` runs over, or ``None``.

    ``obj`` carries a device, unless ``is_device``, in which case it is one.
    """
    device = obj if is_device else getattr(obj, "device", None)
    return getattr(device, "placement", None)


def _traced_placements(jaxpr):
    """Yield the placement each QNode in ``jaxpr`` runs over, in encounter order and with repeats.

    Recurses through nested jaxprs, so a QNode inside a control-flow region counts.
    """
    for eqn in jaxpr.eqns:
        # A qnode equation carries its device on the qnode; a device equation is the device.
        placement = _placement_of(eqn.params.get("qnode")) or _placement_of(
            eqn.params.get("device"), is_device=True
        )
        if placement is not None:
            yield placement
        for inner in jax.core.jaxprs_in_params(eqn.params):
            yield from _traced_placements(inner)


def find_placement(callable_, jaxpr=None, name: str = "this program") -> Placement | None:
    """The placement a compiled program runs over, or ``None`` if it declares none.

    Args:
        callable_: The callable being compiled.
        jaxpr: Its traced form, searched when the callable carries no device itself. ``None`` skips
            that search, for a caller that has not traced yet.
        name: Program name, used in the error below.

    Returns:
        The placement, or ``None``.

    Raises:
        CompileError: If the program's QNodes run over more than one placement. It is serialized
            onto the root module, of which there is one, so a program can only carry one.
    """
    placement = _placement_of(callable_)
    if placement is not None:
        return placement
    found = (
        list({id(p): p for p in _traced_placements(jaxpr)}.values()) if jaxpr is not None else []
    )
    if len(found) > 1:
        names = ", ".join(p.controller.name or "<unnamed>" for p in found)
        raise CompileError(
            f"{name!r} runs QNodes over {len(found)} different backline placements (controllers: "
            f"{names}), but a compiled program carries one. Split them across separate "
            f"qjit-compiled functions."
        )
    return found[0] if found else None


def settle_executors(placement: Placement) -> None:
    """Give every node in ``placement`` an executor with a known address.

    A compiled program carries those addresses, so they are settled before it is built. Deployment
    waits until :func:`launch_executors`, so this costs no ssh.
    """
    for node in (placement.controller, *placement.coprocessors):
        _realize_executor(node)
