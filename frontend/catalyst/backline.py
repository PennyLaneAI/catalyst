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

"""Serialize a PennyLane ``Backline`` placement into the ``catalyst.backline`` module attribute and
build the pipelines that lower it. Backend hints in a node's ``init_args`` are forwarded verbatim,
so a new backend needs no change here.

Note: "node" here is a backline participant (a controller or coprocessor), distinct from
:class:`catalyst.Executor`, which deploys the ``catalyst-executor`` process a node may run on.
"""

# Backend hints forwarded verbatim from a node's ``init_args`` to the attribute node.
_INIT_KEYS = ("backend_lib", "config", "data_path", "in_bytes", "out_bytes")


def _node_dict(node) -> dict:
    """Map one backline node (controller/coprocessor) to a ``catalyst.backline`` node dict."""
    d = {"remote": bool(node.remote)}
    if node.name is not None:
        d["name"] = node.name
    if node.addr is not None:
        d["peer"] = node.addr
    if node.port is not None:
        d["oob_port"] = int(node.port)
    executor = _realize_executor(node)
    triple = (getattr(executor, "triple", None) if executor is not None else None) or node.triple
    if triple is not None:
        d["triple"] = triple
    if executor is not None and getattr(executor, "address", None):
        d["address"] = executor.address
    init = node.init_args or {}
    d.update({k: init[k] for k in _INIT_KEYS if k in init})
    return d


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


def serialize_backline(backline) -> dict:
    """Serialize a ``Backline`` into the ``catalyst.backline`` attribute dict."""
    result = {
        "transport": backline.transport.name,
        "controller": _node_dict(backline.controller),
    }
    nodes = []
    for coproc in backline.coprocessors:
        node = _node_dict(coproc)
        fn = coproc.coprocessor_fn
        node["symbol"] = fn.symbol_name
        if fn.lib_path is not None:
            node.setdefault("backend_lib", fn.lib_path)
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


def backline_attr_text(backline) -> str:
    """Render a ``Backline`` as ``#transport.backline<...>`` attribute syntax."""
    d = serialize_backline(backline)
    parts = [
        f"transport = {_literal(d.get('transport') or '')}",
        f"controller = {_node_text(d['controller'])}",
    ]
    if coprocs := d.get("coprocessors"):
        parts.append("coprocessors = [" + ", ".join(_node_text(c) for c in coprocs) + "]")
    return "#transport.backline<" + ", ".join(parts) + ">"


def _launch_executor(name, triple, options):
    """Build and launch a ``catalyst.Executor`` from a node's executor options.
    """
    from catalyst.executor import Executor

    options.setdefault("name", name or "executor")
    if triple is not None:
        options.setdefault("triple", triple)
    return Executor(**options).launch()


def _realize_executor(node):
    """Return the node's launched executor, building it from an 
    :class:`~pennylane.backline.ExecutorSpec` on first use.
    """
    from pennylane.backline import ExecutorSpec

    executor = getattr(node, "executor", None)
    if isinstance(executor, ExecutorSpec):
        executor = _launch_executor(node.name, node.triple, dict(executor.options))
        object.__setattr__(node, "executor", executor)  # cache on the (frozen) node
    return executor
