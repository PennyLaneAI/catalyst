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
from catalyst.backline import (
    _resolve_backend_lib,
    add_transport_passes,
    backline_pipeline,
    serialize_backline,
)

pytestmark = pytest.mark.skipif(
    not hasattr(qp, "backline"), reason="pennylane.backline UI not available"
)

if hasattr(qp, "backline"):
    from pennylane.backline import Transport


@pytest.fixture(autouse=True)
def _net_transport():
    """Register a test-only ``"net"`` transport per test; unregister on teardown."""
    from pennylane import backline as _bl
    from pennylane.backline import register_transport

    register_transport("net")(lambda: Transport("net"))
    try:
        yield
    finally:
        getattr(_bl, "_transports", {}).pop("net", None)


def _controller(**kw):
    init = {
        "backend_lib": "backend.so",
        "config": "cfg",
        "data_path": "cpu_verbs",
        "in_bytes": 3,
        "out_bytes": 8,
    }
    return qp.Controller(
        device=qp.device("null.qubit", wires=2), label="ctrl", remote=False, init_args=init, **kw
    )


def _coproc(label, oob_port=18590, fn="coproc_fn", **kw):
    return qp.Coprocessor(
        label=label,
        comm_host="127.0.0.1",
        oob_port=oob_port,
        remote=False,
        coprocessor_fn=fn,
        init_args={"backend_lib": "backend.so", "config": "cfg", "data_path": "cpu_verbs"},
        **kw,
    )


def test_controller_node_mapping():
    """label -> name; init_args hints forwarded. A controller carries no endpoint of its own."""
    d = serialize_backline(qp.backline(controller=_controller(), transport="net").placement)
    assert d["transport"] == "net"
    ctrl = d["controller"]
    assert ctrl["name"] == "ctrl"
    assert ctrl["backend_lib"] == "backend.so" and ctrl["config"] == "cfg"
    assert ctrl["data_path"] == "cpu_verbs" and ctrl["in_bytes"] == 3 and ctrl["out_bytes"] == 8
    assert "peer" not in ctrl and "oob_port" not in ctrl


def test_coprocessor_endpoint_mapping():
    """comm_host/oob_port -> peer/oob_port, and oob_port stays an int."""
    dev = qp.backline(controller=_controller(), coprocessors=[_coproc("cop0")], transport="net")
    cop = serialize_backline(dev.placement)["coprocessors"][0]
    assert cop["peer"] == "127.0.0.1"
    assert cop["oob_port"] == 18590 and isinstance(cop["oob_port"], int)


def test_controller_only_has_no_coprocessors():
    d = serialize_backline(qp.backline(controller=_controller(), transport="net").placement)
    assert "coprocessors" not in d


def test_single_coprocessor():
    dev = qp.backline(controller=_controller(), coprocessors=[_coproc("cop0")], transport="net")
    d = serialize_backline(dev.placement)
    assert len(d["coprocessors"]) == 1
    assert d["coprocessors"][0]["name"] == "cop0"
    assert d["coprocessors"][0]["symbol"] == "coproc_fn"


def test_multiple_coprocessors_all_serialized():
    """All coprocessors are serialized as a list, in order."""
    dev = qp.backline(
        controller=_controller(),
        coprocessors=[_coproc("cop0", 18590), _coproc("cop1", 18591)],
        transport="net",
    )
    d = serialize_backline(dev.placement)
    assert [c["name"] for c in d["coprocessors"]] == ["cop0", "cop1"]
    assert [c["oob_port"] for c in d["coprocessors"]] == [18590, 18591]


def test_transport_object_serializes_to_name():
    transport = Transport("net")
    d = serialize_backline(qp.backline(controller=_controller(), transport=transport).placement)
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


def test_backline_qnode_capture_path(use_capture):
    """A backline qnode compiles to MLIR carrying the catalyst.backline attribute."""
    dev = qp.backline(controller=_controller(), coprocessors=[_coproc("cop0")], transport="net")

    @qjit(target="mlir", capture=True)
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(0)
        qp.CNOT([0, 1])
        return qp.probs()

    ir = circuit.mlir
    assert "catalyst.backline" in ir
    assert 'transport = "net"' in ir


def test_remote_controller_module_tagged_with_role(use_capture):
    """A remote controller's module carries catalyst.backline_role.

    The transport passes locate it by role rather than by matching a triple or address, which now
    come only from the node's executor.
    """
    ctrl = qp.Controller(
        device=qp.device("null.qubit", wires=2),
        label="ctrl",
        remote=True,
        init_args={"backend_lib": "backend.so", "config": "cfg", "data_path": "cpu_verbs"},
    )
    dev = qp.backline(controller=ctrl, transport="net")

    @qjit(target="mlir", capture=True)
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(0)
        return qp.probs()

    assert 'catalyst.backline_role = "controller"' in circuit.mlir


@pytest.fixture
def fake_lib_dir(tmp_path, monkeypatch):
    """Stand in for the built runtime lib dir, mirroring the real layout.

    The build puts each backend under ``<RUNTIME_LIB_DIR>/transport/<backend>/``, so entries are
    given as ``"<backend>/<libname>"``.
    """
    monkeypatch.setattr("catalyst.backline.get_lib_path", lambda *_: str(tmp_path))

    def make(*entries):
        for e in entries:
            path = tmp_path / "transport" / e
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        return tmp_path

    return make


class TestBackendResolution:
    """``node.backend`` names a backend; the compiler resolves it to a library per role."""

    def test_name_is_backend_and_role(self, fake_lib_dir):
        """The library is named for its backend and role: no stem guessing, no glob."""
        fake_lib_dir(
            "cpu_verbs/libcatalyst_transport_cpu_verbs_controller.so",
            "cpu_verbs/libcatalyst_transport_cpu_verbs_coprocessor.so",
        )
        assert _resolve_backend_lib("cpu_verbs", "controller", False).endswith(
            "transport/cpu_verbs/libcatalyst_transport_cpu_verbs_controller.so"
        )
        assert _resolve_backend_lib("cpu_verbs", "coprocessor", False).endswith(
            "transport/cpu_verbs/libcatalyst_transport_cpu_verbs_coprocessor.so"
        )

    def test_remote_node_gets_the_bare_filename(self, fake_lib_dir):
        """A remote node loads from its deployed bundle, so it is given a name, not a local path."""
        fake_lib_dir("cpu_verbs/libcatalyst_transport_cpu_verbs_coprocessor.so")
        assert (
            _resolve_backend_lib("cpu_verbs", "coprocessor", True)
            == "libcatalyst_transport_cpu_verbs_coprocessor.so"
        )

    def test_out_of_tree_backend_via_search_path(self, tmp_path, monkeypatch):
        """CATALYST_TRANSPORT_PATH lets a backend built outside the tree resolve.

        Entries are searched directly, so an out-of-tree build needs no particular directory layout.
        """
        outside = tmp_path / "rdma_dev-build"
        outside.mkdir()
        (outside / "libcatalyst_transport_hwhs_controller.so").write_bytes(b"")
        monkeypatch.setattr("catalyst.backline.get_lib_path", lambda *_: str(tmp_path / "empty"))
        monkeypatch.setenv("CATALYST_TRANSPORT_PATH", str(outside))
        assert _resolve_backend_lib("hwhs", "controller", False) == str(
            outside / "libcatalyst_transport_hwhs_controller.so"
        )

    def test_search_path_takes_precedence_over_in_tree(self, tmp_path, monkeypatch, fake_lib_dir):
        """An override entry wins, so a local build can shadow an installed backend."""
        fake_lib_dir("cpu_verbs/libcatalyst_transport_cpu_verbs_controller.so")
        outside = tmp_path / "override"
        outside.mkdir()
        pinned = outside / "libcatalyst_transport_cpu_verbs_controller.so"
        pinned.write_bytes(b"")
        monkeypatch.setenv("CATALYST_TRANSPORT_PATH", str(outside))
        assert _resolve_backend_lib("cpu_verbs", "controller", False) == str(pinned)

    def test_missing_backend_names_every_directory_searched(self, fake_lib_dir, monkeypatch):
        """The error is actionable: what was looked for, where, and how to fix it."""
        fake_lib_dir("cpu_verbs/libcatalyst_transport_cpu_verbs_coprocessor.so")
        monkeypatch.setenv("CATALYST_TRANSPORT_PATH", "/opt/nowhere")
        with pytest.raises(ValueError, match="no transport backend library") as e:
            _resolve_backend_lib("nope_verbs", "coprocessor", False)
        msg = str(e.value)
        assert "libcatalyst_transport_nope_verbs_coprocessor.so" in msg
        assert "/opt/nowhere" in msg
        assert "ENABLE_TRANSPORT=ON" in msg
        assert "CATALYST_TRANSPORT_PATH" in msg

    def test_role_mismatch_fails_before_dlopen(self, fake_lib_dir):
        """gpu_verbs ships no controller library, so a controller lookup fails here."""
        fake_lib_dir("gpu_verbs/libcatalyst_transport_gpu_verbs_coprocessor.so")
        with pytest.raises(ValueError, match="role='controller'"):
            _resolve_backend_lib("gpu_verbs", "controller", False)

    def test_backend_populates_backend_lib_per_role(self, fake_lib_dir):
        """Each node's backend resolves against its own role.

        The shared fixtures pin an explicit ``backend_lib``, which would take precedence, so these
        nodes are built without one.
        """
        fake_lib_dir(
            "cpu_verbs/libcatalyst_transport_cpu_verbs_controller.so",
            "gpu_verbs/libcatalyst_transport_gpu_verbs_coprocessor.so",
        )
        ctrl = qp.Controller(
            device=qp.device("null.qubit", wires=2), label="ctrl", backend="cpu_verbs"
        )
        cop = qp.Coprocessor(
            label="cop0", comm_host="127.0.0.1", coprocessor_fn="coproc_fn", backend="gpu_verbs"
        )
        d = serialize_backline(
            qp.backline(controller=ctrl, coprocessors=[cop], transport="net").placement
        )
        assert d["controller"]["backend_lib"].endswith("_cpu_verbs_controller.so")
        assert d["coprocessors"][0]["backend_lib"].endswith("_gpu_verbs_coprocessor.so")

    def test_explicit_backend_lib_wins_over_backend(self, fake_lib_dir):
        """An explicit ``init_args["backend_lib"]`` path is not overridden by ``backend``."""
        fake_lib_dir("cpu_verbs/libcatalyst_transport_cpu_verbs_controller.so")
        ctrl = qp.Controller(
            device=qp.device("null.qubit", wires=2),
            label="ctrl",
            backend="cpu_verbs",
            init_args={"backend_lib": "/opt/explicit.so"},
        )
        d = serialize_backline(qp.backline(controller=ctrl, transport="net").placement)
        assert d["controller"]["backend_lib"] == "/opt/explicit.so"

    def test_no_backend_leaves_backend_lib_unset(self):
        """Omitting ``backend`` leaves the field to ``init_args`` or the compiler default."""
        ctrl = qp.Controller(device=qp.device("null.qubit", wires=2), label="ctrl")
        d = serialize_backline(qp.backline(controller=ctrl, transport="net").placement)
        assert "backend_lib" not in d["controller"]
