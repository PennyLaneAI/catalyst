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
    _realize_executor,
    _resolve_backend_lib,
    add_transport_passes,
    backline_pipeline,
    realize_executors,
    serialize_backline,
)

pytestmark = pytest.mark.skipif(
    not hasattr(qp, "backline"), reason="pennylane.backline UI not available"
)

if hasattr(qp, "backline"):
    from pennylane.backline import Transport



def _controller(**kw):
    init = {
        "backend_lib": "backend.so",
        "config": "cfg",
        "in_bytes": 3,
        "out_bytes": 8,
    }
    kw.setdefault("remote", False)
    kw.setdefault("init_args", init)
    return qp.Controller(device=qp.device("null.qubit", wires=2), label="ctrl", **kw)


def _coproc(label, oob_port=18590, fn="coproc_fn", **kw):
    kw.setdefault("remote", False)
    kw.setdefault("init_args", {"backend_lib": "backend.so", "config": "cfg"})
    return qp.Coprocessor(
        label=label, comm_host="127.0.0.1", oob_port=oob_port, coprocessor_fn=fn, **kw
    )


def test_controller_node_mapping():
    """label -> name; init_args hints forwarded. A controller carries no endpoint of its own."""
    d = serialize_backline(qp.backline(controller=_controller(), transport="rdma").placement)
    assert d["transport"] == "rdma"
    ctrl = d["controller"]
    assert ctrl["name"] == "ctrl"
    assert ctrl["backend_lib"] == "backend.so" and ctrl["config"] == "cfg"
    assert ctrl["in_bytes"] == 3 and ctrl["out_bytes"] == 8
    assert "peer" not in ctrl and "oob_port" not in ctrl


def test_coprocessor_endpoint_mapping():
    """comm_host/oob_port -> peer/oob_port, and oob_port stays an int."""
    dev = qp.backline(controller=_controller(), coprocessors=[_coproc("cop0")], transport="rdma")
    cop = serialize_backline(dev.placement)["coprocessors"][0]
    assert cop["peer"] == "127.0.0.1"
    assert cop["oob_port"] == 18590 and isinstance(cop["oob_port"], int)


def test_controller_only_has_no_coprocessors():
    d = serialize_backline(qp.backline(controller=_controller(), transport="rdma").placement)
    assert "coprocessors" not in d


def test_single_coprocessor():
    dev = qp.backline(controller=_controller(), coprocessors=[_coproc("cop0")], transport="rdma")
    d = serialize_backline(dev.placement)
    assert len(d["coprocessors"]) == 1
    assert d["coprocessors"][0]["name"] == "cop0"
    assert d["coprocessors"][0]["symbol"] == "coproc_fn"


def test_in_process_coprocessor_fn_lib_is_loaded(monkeypatch):
    """An in-process coprocessor's CoprocessorFn library is loaded, so its symbol can resolve.

    Nothing else loads it: there is no executor to have taken it as a ``--plugin``, and for
    ``cpu_verbs`` the decoder is a library of its own rather than part of the backend.
    """
    loaded = []
    monkeypatch.setattr(
        "ctypes.CDLL", lambda path, mode=None: loaded.append((path, mode)) or object()
    )
    import ctypes  # pylint: disable=import-outside-toplevel

    fn = qp.CoprocessorFunction("decode_fn", lib_path="/opt/libdecode.so")
    dev = qp.backline(
        controller=_controller(), coprocessors=[_coproc("cop0", fn=fn)], transport="rdma"
    )
    realize_executors(dev.placement)
    assert loaded == [("/opt/libdecode.so", ctypes.RTLD_GLOBAL)]


def test_remote_coprocessor_fn_lib_is_not_loaded_here(monkeypatch):
    """A remote coprocessor's library belongs on its own machine; its executor loads it there."""
    loaded = []
    monkeypatch.setattr("ctypes.CDLL", lambda path, mode=None: loaded.append(path) or object())
    fn = qp.CoprocessorFunction("decode_fn", lib_path="/opt/libdecode.so")
    dev = qp.backline(
        controller=_controller(),
        coprocessors=[_coproc("cop0", fn=fn, remote=True)],
        transport="rdma",
    )
    realize_executors(dev.placement)
    assert loaded == []


def test_coprocessor_fn_without_lib_path_loads_nothing(monkeypatch):
    """No ``lib_path`` means resolve from what is already loaded, so nothing is opened."""
    loaded = []
    monkeypatch.setattr("ctypes.CDLL", lambda path, mode=None: loaded.append(path) or object())
    dev = qp.backline(controller=_controller(), coprocessors=[_coproc("cop0")], transport="rdma")
    realize_executors(dev.placement)
    assert loaded == []


def test_multiple_coprocessors_all_serialized():
    """All coprocessors are serialized as a list, in order."""
    dev = qp.backline(
        controller=_controller(),
        coprocessors=[_coproc("cop0", 18590), _coproc("cop1", 18591)],
        transport="rdma",
    )
    d = serialize_backline(dev.placement)
    assert [c["name"] for c in d["coprocessors"]] == ["cop0", "cop1"]
    assert [c["oob_port"] for c in d["coprocessors"]] == [18590, 18591]


def test_transport_object_serializes_to_name():
    transport = Transport("rdma")
    d = serialize_backline(qp.backline(controller=_controller(), transport=transport).placement)
    assert d["transport"] == "rdma"


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
    dev = qp.backline(controller=_controller(), coprocessors=[_coproc("cop0")], transport="rdma")

    @qjit(target="mlir", capture=True)
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(0)
        qp.CNOT([0, 1])
        return qp.probs()

    ir = circuit.mlir
    assert "catalyst.backline" in ir
    assert 'transport = "rdma"' in ir


def test_remote_controller_module_tagged_with_role(use_capture):
    """A remote controller's module carries catalyst.backline_role.

    The transport passes locate it by role rather than by matching a triple or address, which now
    come only from the node's executor.
    """
    ctrl = qp.Controller(
        device=qp.device("null.qubit", wires=2),
        label="ctrl",
        remote=True,
        init_args={"backend_lib": "backend.so", "config": "cfg"},
    )
    dev = qp.backline(controller=ctrl, transport="rdma")

    @qjit(target="mlir", capture=True)
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(0)
        return qp.probs()

    assert 'catalyst.backline_role = "controller"' in circuit.mlir


@pytest.fixture
def fake_lib_dir(tmp_path, monkeypatch):
    """Stand in for the built runtime lib dir, laid out as a bare ``cmake`` build.

    That build mirrors the source tree, nesting each backend under
    ``<RUNTIME_LIB_DIR>/transport/<backend>/``, so entries are given as ``"<backend>/<libname>"``.
    See :func:`flat_lib_dir` for the layout ``make -C runtime`` produces.
    """
    monkeypatch.setattr("catalyst.backline.get_lib_path", lambda *_: str(tmp_path))

    def make(*entries):
        for e in entries:
            path = tmp_path / "transport" / e
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        return tmp_path

    return make


@pytest.fixture
def flat_lib_dir(tmp_path, monkeypatch):
    """Stand in for the built runtime lib dir as ``make -C runtime`` lays it out.

    That build passes ``CMAKE_LIBRARY_OUTPUT_DIRECTORY``, which puts every library flat in
    ``<RUNTIME_LIB_DIR>``, so entries are bare library names.
    """
    monkeypatch.setattr("catalyst.backline.get_lib_path", lambda *_: str(tmp_path))

    def make(*entries):
        for e in entries:
            (tmp_path / e).write_bytes(b"")
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

    def test_flat_lib_dir_is_searched(self, flat_lib_dir):
        """``make -C runtime`` passes ``CMAKE_LIBRARY_OUTPUT_DIRECTORY``, flattening the lib dir.

        That is the layout a released or ``make``-built tree has, so it must resolve without the
        ``transport/<backend>/`` nesting a bare ``cmake`` build produces.
        """
        root = flat_lib_dir("libcatalyst_transport_cpu_verbs_controller.so")
        assert _resolve_backend_lib("cpu_verbs", "controller", False) == str(
            root / "libcatalyst_transport_cpu_verbs_controller.so"
        )

    def test_remote_node_gets_the_bare_filename(self, fake_lib_dir):
        """A remote node loads from its deployed bundle, so it is given a name, not a local path."""
        fake_lib_dir("cpu_verbs/libcatalyst_transport_cpu_verbs_coprocessor.so")
        assert (
            _resolve_backend_lib("cpu_verbs", "coprocessor", True)
            == "libcatalyst_transport_cpu_verbs_coprocessor.so"
        )

    def test_remote_node_does_not_probe_this_machine(self, fake_lib_dir):
        """A remote backend need not exist here: the bundle it loads from is on the other machine.

        The FPGA's ``hwhs`` library is built for aarch64 and lives only on the board, so probing
        the host would test the wrong filesystem and reject a valid placement.
        """
        fake_lib_dir()  # nothing built locally
        assert (
            _resolve_backend_lib("hwhs", "controller", True)
            == "libcatalyst_transport_hwhs_controller.so"
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
            qp.backline(controller=ctrl, coprocessors=[cop], transport="rdma").placement
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
        d = serialize_backline(qp.backline(controller=ctrl, transport="rdma").placement)
        assert d["controller"]["backend_lib"] == "/opt/explicit.so"

    def test_no_backend_leaves_backend_lib_unset(self):
        """Omitting ``backend`` leaves the field to ``init_args`` or the compiler default."""
        ctrl = qp.Controller(device=qp.device("null.qubit", wires=2), label="ctrl")
        d = serialize_backline(qp.backline(controller=ctrl, transport="rdma").placement)
        assert "backend_lib" not in d["controller"]


class TestExecutorRealization:
    """``executor_options`` on a node becomes a launched ``catalyst.Executor``.

    These use an attach-only executor (an ``address`` with neither ``local`` nor ``host``), whose
    ``launch()`` short-circuits without spawning a process — so the whole path is exercised without
    needing the ``catalyst-executor`` binary.
    """

    def test_no_options_launches_nothing(self):
        """``executor_options=None`` means no executor was requested."""
        node = _controller()
        assert node.executor_options is None
        assert _realize_executor(node) is None
        assert node.executor is None

    def test_options_produce_a_launched_executor_cached_on_the_node(self):
        """The executor is built, launched, and cached back onto the frozen node."""
        node = _controller(
            executor_options={"address": "10.0.0.9:1373", "triple": "aarch64-unknown-linux-gnu"}
        )
        ex = _realize_executor(node)
        assert ex is not None
        assert ex.address == "10.0.0.9:1373"
        assert ex.triple == "aarch64-unknown-linux-gnu"
        assert node.executor is ex

    def test_realization_is_idempotent(self):
        """A second call returns the same executor rather than launching another."""
        node = _controller(executor_options={"address": "10.0.0.9:1373"})
        assert _realize_executor(node) is _realize_executor(node)

    def test_label_seeds_the_executor_name(self):
        """The node's label names the executor, which uses it for its logs."""
        node = _controller(executor_options={"address": "10.0.0.9:1373"})
        assert _realize_executor(node).name == "ctrl"

    def test_a_preset_executor_is_returned_untouched(self):
        """Setting ``executor`` directly attaches an already-launched one; options are ignored."""

        class _Ex:
            address = "attached:1"
            triple = "x86_64-unknown-linux-gnu"

        ex = _Ex()
        node = _controller(executor=ex, executor_options={"address": "ignored:2"})
        assert _realize_executor(node) is ex

    def test_realize_executors_walks_every_node(self):
        """``realize_executors`` covers the controller and each coprocessor."""
        ctrl = _controller(executor_options={"address": "ctrl:1"})
        cop = _coproc("cop0", executor_options={"address": "cop:2"})
        dev = qp.backline(controller=ctrl, coprocessors=[cop], transport="rdma")
        realize_executors(dev.placement)
        assert ctrl.executor.address == "ctrl:1"
        assert cop.executor.address == "cop:2"

    def test_executor_address_and_triple_reach_the_serialized_node(self):
        """The launched executor supplies the node's dispatch address and target triple."""
        ctrl = _controller(
            remote=True,
            executor_options={"address": "10.0.0.9:1373", "triple": "aarch64-unknown-linux-gnu"},
        )
        dev = qp.backline(controller=ctrl, transport="rdma")
        realize_executors(dev.placement)
        node = serialize_backline(dev.placement)["controller"]
        assert node["address"] == "10.0.0.9:1373"
        assert node["triple"] == "aarch64-unknown-linux-gnu"

    def test_a_high_oob_port_survives_to_the_ir(self, use_capture):
        """A port above 32767 appears as itself, not as a negative number."""
        cop = _coproc("cop0", oob_port=40000)
        dev = qp.backline(controller=_controller(), coprocessors=[cop], transport="rdma")

        @qjit(target="mlir", capture=True)
        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(0)
            return qp.probs()

        assert "oob_port = 40000" in circuit.mlir
        assert "-25536" not in circuit.mlir
