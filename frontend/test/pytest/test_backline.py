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
"""Unit tests for the backline frontend: serialize_backline and the pipeline helpers."""

import pennylane as qp
import pytest

from catalyst import qjit
from catalyst.backline import add_transport_passes, backline_pipeline, serialize_backline

pytestmark = pytest.mark.skipif(
    not hasattr(qp, "backline"), reason="pennylane.backline UI not available"
)

if hasattr(qp, "backline"):
    from pennylane.backline import Transport, register_transport

    register_transport("net")(lambda: Transport("net"))


def _controller(**kw):
    init = {
        "backend_lib": "backend.so",
        "config": "cfg",
        "data_path": "cpu_verbs",
        "in_bytes": 3,
        "out_bytes": 8,
    }
    return qp.Controller(
        qp.device("null.qubit", wires=2),
        name="ctrl",
        addr="127.0.0.1",
        port="18590",
        remote=False,
        init_args=init,
        **kw
    )


def _coproc(name, port="18590", fn="coproc_fn"):
    return qp.Coprocessor(
        name=name,
        addr="127.0.0.1",
        port=port,
        remote=False,
        coprocessor_fn=fn,
        init_args={"backend_lib": "backend.so", "config": "cfg", "data_path": "cpu_verbs"},
    )


def test_executor_node_mapping():
    """addr/port -> peer/oob_port; init_args hints forwarded; name kept."""
    d = serialize_backline(qp.backline(_controller(), transport="net").backline)
    assert d["transport"] == "net"
    ctrl = d["controller"]
    assert ctrl["peer"] == "127.0.0.1" and ctrl["oob_port"] == 18590
    assert ctrl["backend_lib"] == "backend.so" and ctrl["config"] == "cfg"
    assert ctrl["data_path"] == "cpu_verbs" and ctrl["in_bytes"] == 3 and ctrl["out_bytes"] == 8
    assert ctrl["name"] == "ctrl"


def test_controller_only_has_no_coprocessors():
    d = serialize_backline(qp.backline(_controller(), transport="net").backline)
    assert "coprocessors" not in d


def test_single_coprocessor():
    dev = qp.backline(_controller(), _coproc("cop0"), transport="net")
    d = serialize_backline(dev.backline)
    assert len(d["coprocessors"]) == 1
    assert d["coprocessors"][0]["name"] == "cop0"
    assert d["coprocessors"][0]["symbol"] == "coproc_fn"


def test_multiple_coprocessors_all_serialized():
    """All coprocessors are serialized as a list, in order."""
    dev = qp.backline(
        _controller(), _coproc("cop0", "18590"), _coproc("cop1", "18591"), transport="net"
    )
    d = serialize_backline(dev.backline)
    assert [c["name"] for c in d["coprocessors"]] == ["cop0", "cop1"]
    assert [c["oob_port"] for c in d["coprocessors"]] == [18590, 18591]


def test_transport_object_serializes_to_name():
    transport = Transport("net")
    d = serialize_backline(qp.backline(_controller(), transport=transport).backline)
    assert d["transport"] == "net"


def test_add_transport_passes_places_each_pass():
    """add_transport_passes inserts each pass into its stage; transport lowers before catalyst."""
    stages = add_transport_passes(
        [
            ("QuantumCompilationStage", []),
            ("BufferizationStage", []),
            ("MLIRToLLVMDialectConversion", ["convert-catalyst-to-llvm"]),
        ]
    )
    d = dict(stages)
    assert d["QuantumCompilationStage"] == ["inject-transport-session"]
    assert d["BufferizationStage"] == ["lower-decode-to-transport"]
    assert d["MLIRToLLVMDialectConversion"] == [
        "convert-transport-to-llvm",
        "convert-catalyst-to-llvm",
    ]


def test_backline_pipeline_carries_transport_passes():
    """backline_pipeline is the default pipeline with the transport passes wired in."""
    stages = dict(backline_pipeline())
    assert "inject-transport-session" in stages["QuantumCompilationStage"]
    assert "lower-decode-to-transport" in stages["BufferizationStage"]
    assert "convert-transport-to-llvm" in stages["MLIRToLLVMDialectConversion"]


def test_backline_qnode_capture_path():
    """A backline qnode compiles to MLIR carrying the catalyst.backline attribute."""
    qp.capture.enable()
    try:
        dev = qp.backline(_controller(), _coproc("cop0"), transport="net")

        @qjit(target="mlir", capture=True)
        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            return qp.probs()

        ir = circuit.mlir
        assert "catalyst.backline" in ir
        assert 'transport = "net"' in ir
    finally:
        qp.capture.disable()


def test_remote_controller_module_tagged_with_role():
    """A remote controller's module carries catalyst.backline_role.

    The transport passes locate it by role rather than by matching the triple/address that
    catalyst.target and catalyst.dispatch copy from the node.
    """
    qp.capture.enable()
    try:
        ctrl = qp.Controller(
            qp.device("null.qubit", wires=2),
            name="ctrl",
            addr="127.0.0.1",
            port="18590",
            remote=True,
            triple="aarch64-unknown-linux-gnu",
            init_args={"backend_lib": "backend.so", "config": "cfg", "data_path": "cpu_verbs"},
        )
        dev = qp.backline(ctrl, transport="net")

        @qjit(target="mlir", capture=True)
        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(0)
            return qp.probs()

        assert 'catalyst.backline_role = "controller"' in circuit.mlir
    finally:
        qp.capture.disable()
