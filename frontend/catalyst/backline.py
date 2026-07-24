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
build the pipelines that lower it. Backend hints in ``Executor.init_args`` are forwarded verbatim,
so a new backend needs no change here.
"""

# Backend hints forwarded verbatim from ``Executor.init_args`` to the attribute node.
_INIT_KEYS = ("backend_lib", "config", "data_path", "in_bytes", "out_bytes")


def _executor_dict(executor) -> dict:
    """Map one ``Executor`` to a ``catalyst.backline`` node dict (addr/port -> peer/oob_port)."""
    node = {"remote": bool(executor.remote)}
    if executor.name is not None:
        node["name"] = executor.name
    if executor.addr is not None:
        node["peer"] = executor.addr
    if executor.port is not None:
        node["oob_port"] = int(executor.port)
    if executor.triple is not None:
        node["triple"] = executor.triple
    init = executor.init_args or {}
    node.update({k: init[k] for k in _INIT_KEYS if k in init})
    return node


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
        "controller": _executor_dict(backline.controller),
    }
    nodes = []
    for coproc in backline.coprocessors:
        node = _executor_dict(coproc)
        fn = coproc.coprocessor_fn
        node["symbol"] = fn.symbol_name
        if fn.lib_path is not None:
            node.setdefault("backend_lib", fn.lib_path)
        nodes.append(node)
    if nodes:
        result["coprocessors"] = nodes
    return result
