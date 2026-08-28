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

import os
import platform
from pathlib import Path
from unittest import mock

import numpy as np
import pennylane as qp
import pytest
from pennylane.backline import Transport

from catalyst import Executor, qjit
from catalyst.backline import (
    _EXECUTOR_RUNTIME_PLUGINS,
    _TRANSPORT_PASSES,
    _insert_passes,
    _qec_pass_specs,
    _realize_executor,
    _resolve_backend,
    _resolve_backend_lib,
    device_pass_pipeline,
    launch_executors,
    placement_pipeline,
    remote_device_lib,
    serialize_backline,
)
from catalyst.device.qjit_device import BackendInfo, extract_backend_info
from catalyst.from_plxpr.from_plxpr import _get_device_kwargs
from catalyst.utils.exceptions import CompileError
from catalyst.utils.runtime_environment import get_lib_path

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


class _Attached:
    """An executor already launched elsewhere: it says where it serves and can be launched."""

    address = None
    triple = None

    def launch(self):
        return self


def _controller(**kw):
    init = {
        "backend_lib": "backend.so",
        "config": "cfg",
    }
    kw.setdefault("init_args", init)
    return qp.Controller(device=qp.device("null.qubit", wires=2), name="ctrl", **kw)


def _coproc(name, oob_port=18590, fn="coproc_fn", **kw):
    kw.setdefault("init_args", {"backend_lib": "backend.so", "config": "cfg"})
    return qp.Coprocessor(
        name=name, endpoint=qp.Endpoint("127.0.0.1", oob_port), coprocessor_fn=fn, **kw
    )


def test_controller_node_mapping():
    """The node name and init arguments are forwarded; controllers carry no endpoint."""
    d = serialize_backline(qp.Backline(controller=_controller(), transport="rdma").placement)
    assert d["transport"] == "rdma"
    ctrl = d["controller"]
    assert ctrl["name"] == "ctrl"
    assert ctrl["backend_lib"] == "backend.so" and ctrl["config"] == "cfg"
    assert "peer" not in ctrl and "oob_port" not in ctrl
    assert "coprocessors" not in d  # omitted, not an empty list


def test_default_message_sizes_are_serialized():
    """PennyLane's message-size defaults are explicit in the transport configuration."""
    controller = qp.Controller(init_args={"backend_lib": "backend.so"})
    node = serialize_backline(qp.Backline(controller=controller, transport="rdma").placement)[
        "controller"
    ]

    assert node["in_bytes"] == 8
    assert node["out_bytes"] == 8


def test_coprocessor_endpoint_mapping():
    """comm_host/oob_port -> peer/oob_port, and oob_port stays an int."""
    dev = qp.Backline(controller=_controller(), coprocessors=[_coproc("cop0")], transport="rdma")
    cop = serialize_backline(dev.placement)["coprocessors"][0]
    assert cop["peer"] == "127.0.0.1"
    assert cop["oob_port"] == 18590 and isinstance(cop["oob_port"], int)


def test_controller_only_has_no_coprocessors():
    d = serialize_backline(qp.Backline(controller=_controller(), transport="rdma").placement)
    assert "coprocessors" not in d


def test_single_coprocessor():
    dev = qp.Backline(controller=_controller(), coprocessors=[_coproc("cop0")], transport="rdma")
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
    dev = qp.Backline(
        controller=_controller(), coprocessors=[_coproc("cop0", fn=fn)], transport="rdma"
    )
    launch_executors(dev.placement)
    assert loaded == [("/opt/libdecode.so", ctypes.RTLD_GLOBAL)]


def test_out_of_process_coprocessor_fn_lib_is_not_loaded_here(monkeypatch):
    """A dispatched coprocessor's executor loads the library itself, in the process that needs it."""
    loaded = []
    monkeypatch.setattr("ctypes.CDLL", lambda path, mode=None: loaded.append(path) or object())
    fn = qp.CoprocessorFunction("decode_fn", lib_path="/opt/libdecode.so")
    cop = _coproc("cop0", fn=fn, remote=True, executor_options={"host": "192.0.2.11", "port": 7813})
    dev = qp.Backline(controller=_controller(), coprocessors=[cop], transport="rdma")
    with mock.patch.object(Executor, "launch", autospec=True):
        launch_executors(dev.placement)
    assert loaded == []


def test_coprocessor_fn_without_lib_path_loads_nothing(monkeypatch):
    """No ``lib_path`` means resolve from what is already loaded, so nothing is opened."""
    loaded = []
    monkeypatch.setattr("ctypes.CDLL", lambda path, mode=None: loaded.append(path) or object())
    dev = qp.Backline(controller=_controller(), coprocessors=[_coproc("cop0")], transport="rdma")
    launch_executors(dev.placement)
    assert loaded == []


def test_unlaunched_executor_names_the_node_it_came_from():
    """An executor with no address fails with the node named."""
    from catalyst import Executor  # pylint: disable=import-outside-toplevel

    ctrl = _controller(executor=Executor(host="10.0.0.9", user="me"))
    dev = qp.Backline(controller=ctrl, transport="rdma")
    with pytest.raises(CompileError, match="cannot say where it serves.*not launched"):
        serialize_backline(dev.placement)


def test_unrecognized_init_args_are_rejected():
    """An init_args key the attribute has no parameter for fails here, naming the ones it does."""
    ctrl = _controller(init_args={"config": "cfg", "in_byte": 8})
    dev = qp.Backline(controller=ctrl, transport="rdma")
    with pytest.raises(CompileError, match=r"unrecognized init_args \['in_byte'\]"):
        serialize_backline(dev.placement)


def test_multiple_coprocessors_all_serialized():
    """All coprocessors are serialized as a list, in order."""
    dev = qp.Backline(
        controller=_controller(),
        coprocessors=[_coproc("cop0", 18590), _coproc("cop1", 18591)],
        transport="rdma",
    )
    d = serialize_backline(dev.placement)
    assert [c["name"] for c in d["coprocessors"]] == ["cop0", "cop1"]
    assert [c["oob_port"] for c in d["coprocessors"]] == [18590, 18591]


def test_transport_object_serializes_to_name():
    transport = Transport("rdma")
    d = serialize_backline(qp.Backline(controller=_controller(), transport=transport).placement)
    assert d["transport"] == "rdma"


def test_transport_memcpy_object_serializes_to_name():
    """The in-process ``memcpy`` transport carries through the Transport enum verbatim."""
    transport = Transport("memcpy")
    d = serialize_backline(qp.Backline(controller=_controller(), transport=transport).placement)
    assert d["transport"] == "memcpy"


def test_transport_passes_are_placed_in_each_stage():
    """Each transport pass lands in its own stage, and transport lowers before catalyst."""
    stages = _insert_passes(
        [
            ("QuantumCompilationStage", []),
            ("BufferizationStage", []),
            ("MLIRToLLVMDialectConversion", ["convert-catalyst-to-llvm"]),
        ],
        _TRANSPORT_PASSES,
    )
    d = dict(stages)
    assert d["QuantumCompilationStage"] == ["inject-transport-session"]
    assert d["BufferizationStage"] == ["lower-decode-to-transport"]
    assert d["MLIRToLLVMDialectConversion"] == [
        "convert-transport-to-llvm",
        "convert-catalyst-to-llvm",
    ]


def test_device_pass_pipeline_is_empty_for_a_device_that_needs_nothing():
    """A device that declares no encoding contributes no passes."""
    assert device_pass_pipeline(qp.device("null.qubit", wires=2)) == ()
    assert device_pass_pipeline(qp.Backline(controller=_controller(), transport="rdma")) == ()


def test_device_pass_pipeline_requests_the_encoding_chain():
    """A placement naming a code asks for the whole chain, in application order."""
    dev = qp.Backline(controller=_controller(), transport="rdma", qec_code="steane")
    assert [t.pass_name for t in device_pass_pipeline(dev)] == [
        name for name, _ in _qec_pass_specs("steane")
    ]


@pytest.mark.parametrize(
    "qec_code, wants_qec", [(None, False), ("steane", True)], ids=["unencoded", "steane"]
)
def test_placement_pipeline_returns_the_stages(qec_code, wants_qec):
    """``configure`` adds the transport passes always and the QEC lowering only when asked."""
    dev = qp.Backline(controller=_controller(), transport="rdma", qec_code=qec_code)
    # Both reference passes are present, since each insertion anchors to one of them.
    stages = placement_pipeline(
        dev.placement,
        [
            ("QuantumCompilationStage", []),
            ("BufferizationStage", []),
            (
                "MLIRToLLVMDialectConversion",
                ["convert-quantum-to-llvm", "convert-catalyst-to-llvm"],
            ),
        ],
    )
    assert stages is not None
    d = dict(stages)
    assert "inject-transport-session" in d["QuantumCompilationStage"]
    assert ("convert-qecp-to-llvm" in d["MLIRToLLVMDialectConversion"]) is wants_qec


def test_backline_qnode_capture_path(use_capture):
    """A backline qnode compiles to MLIR carrying the catalyst.backline attribute."""
    dev = qp.Backline(controller=_controller(), coprocessors=[_coproc("cop0")], transport="rdma")

    @qjit(target="mlir", capture=True)
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(0)
        qp.CNOT([0, 1])
        return qp.probs()

    ir = circuit.mlir
    assert "catalyst.backline" in ir
    assert 'transport = "rdma"' in ir


def test_backline_qnode_capture_path_memcpy(use_capture):
    """A backline qnode with the in-process ``memcpy`` transport compiles to MLIR carrying
    ``transport = "memcpy"``.
    """
    dev = qp.Backline(controller=_controller(), coprocessors=[_coproc("cop0")], transport="memcpy")

    @qjit(target="mlir", capture=True)
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(0)
        qp.CNOT([0, 1])
        return qp.probs()

    ir = circuit.mlir
    assert "catalyst.backline" in ir
    assert 'transport = "memcpy"' in ir


@pytest.mark.skipif(
    "CATALYST_TRANSPORT_PATH" not in os.environ,
    reason=(
        "Backline demo integration tests need the transport backend libraries built with "
        "-DENABLE_TRANSPORT=ON, which the wheel does not ship. Add that build directory to "
        "CATALYST_TRANSPORT_PATH to run them; the check-transport-gpu workflow does this on a "
        "GPU runner."
    ),
)
class TestBacklineDemoIntegration:
    """Mirrors ``backline/demos`` end-to-end, adapted to run in the unit-test environment.

    Each test mirrors a ``backline/demos`` script and JIT-compiles the whole placement. The ones
    that call the QNode also execute it, so they need the transport backend libraries and the
    decoder library reachable to the dynamic loader.
    """

    def test_local_cpu_to_local_cpu_memcpy(self, use_capture):
        """Demo 1: local CPU ↔ local CPU over ``memcpy``, JIT-compiled and executed.

        Mirrors ``demos/demo_1_local_cpu_to_local_cpu_memcpy.py``: both roles in this process,
        exchanging the decode round through a shared buffer. Executes the QNode end-to-end.
        """
        steane_lib = str(
            Path(get_lib_path("runtime", "RUNTIME_LIB_DIR")) / "libsteane_coprocessor_cpu.so"
        )
        ctrl = qp.Controller(
            name="cpu-controller",
            device=qp.device("null.qubit", wires=3),
            init_args={"backend_lib": "libcatalyst_transport_memcpy_controller.so"},
        )
        coproc = qp.Coprocessor(
            name="cpu-coproc",
            coprocessor_fn=qp.CoprocessorFunction("steane_coprocessor", lib_path=steane_lib),
            init_args={"backend_lib": "libcatalyst_transport_memcpy_coprocessor.so"},
        )
        dev = qp.Backline(
            controller=ctrl, coprocessors=[coproc], transport="memcpy", qec_code="steane"
        )

        @qjit(capture=True)
        @qp.set_shots(1)
        @qp.qnode(dev, mcm_method="one-shot")
        def ghz():
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            qp.CNOT([1, 2])
            return qp.sample([qp.measure(0), qp.measure(1), qp.measure(2)])

        ir = ghz.mlir
        assert "catalyst.backline" in ir
        assert 'transport = "memcpy"' in ir
        assert 'symbol = "steane_coprocessor"' in ir
        for pass_name, _ in _qec_pass_specs("steane"):
            assert pass_name in ir, f"{pass_name} missing from the scheduled pipeline"

        samples = np.asarray(ghz())
        # qp.sample keeps the shots axis explicit: shots=1 with 3 measurements -> (1, 3).
        assert samples.shape == (1, 3), f"expected shape (1, 3), got {samples.shape}"

    @pytest.mark.parametrize(
        "executor_options, oob_port",
        [
            pytest.param(None, 18590, id="in_process"),
            pytest.param({}, 18591, id="out_of_process"),
        ],
    )
    def test_local_cpu_to_local_cpu_rdma_loopback(
        self, use_capture, stop_node_executors, executor_options, oob_port
    ):
        """Demo 1a: local CPU to local CPU over RDMA loopback, in-process and out-of-process.

        Both roles reach each other through the ``rxe0`` soft-RoCE device installed by the
        ``setup-soft-roce`` composite action, so the ibverbs path is real while the fabric
        underneath it is software. The cases differ only in where the coprocessor runs, which
        ``pennylane.backline.Node.remote`` documents as a property of ``executor_options`` with
        ``remote`` left at ``False``:

        * ``in_process`` -- ``executor_options=None``, the default: no executor is built, so both
          roles stay in this process. A direct mirror of
          ``demos/demo_1a_local_cpu_to_local_cpu_rdma.py``.
        * ``out_of_process`` -- ``executor_options={}``, naming neither a ``host`` nor an
          ``address``: a ``catalyst-executor`` subprocess on this machine, its libraries still
          resolving from this installation. This is how the roles actually deploy, one queue pair
          per process -- an earlier in-process version of this test SIGSEGV'd inside ``ghz()`` on
          GH-hosted ubuntu's ``rdma_rxe`` with two queue pairs in one process. It also covers the
          executor plugin path: ``_realize_executor`` derives ``_executor_plugins`` only on the
          ``executor_options`` branch, since a preset ``executor=Executor()`` is taken at the
          node's word, so the subprocess gets the runtime libraries and the decode library
          preloaded by absolute path. Needs a runtime built with ``ENABLE_EXECUTOR=ON``, which is
          what puts ``catalyst-executor`` where ``default_executor_bin`` looks for it.

        A port per case, so both run in one session without meeting each other on the
        coprocessor's out-of-band port. The config strings follow the demo's ``LOCAL_CFG`` shape -
        ``dev=<rxe device>;gid=<idx>``, which
        ``runtime/lib/transport/rdma/common/BackendConfig.hpp`` requires. ``backend_lib`` is left
        off ``init_args`` so the compiler resolves the transport backend via the (``transport``,
        ``hardware``) mapping in ``backline._resolve_backend_lib`` to an absolute path under
        ``RUNTIME_LIB_DIR`` - avoiding a runtime ``dlopen`` of a bare filename that the loader
        would search for on ``LD_LIBRARY_PATH``.
        """
        steane_lib = str(
            Path(get_lib_path("runtime", "RUNTIME_LIB_DIR")) / "libsteane_coprocessor_cpu.so"
        )
        ctrl = qp.Controller(
            name="cpu-controller",
            device=qp.device("null.qubit", wires=3),
            init_args={"config": "dev=rxe0;gid=1"},
        )
        coproc = qp.Coprocessor(
            name="cpu-coproc",
            coprocessor_fn=qp.CoprocessorFunction("steane_coprocessor", lib_path=steane_lib),
            endpoint=qp.Endpoint("127.0.0.1", oob_port),
            executor_options=executor_options,
            init_args={"config": "dev=rxe0;gid=1"},
        )
        # Registered before compiling, because ``_realize_executor`` launches the subprocess
        # during compilation: a failure after that point still has to release the port. A no-op
        # for the in-process case, whose node never gets an executor.
        stop_node_executors(coproc)
        dev = qp.Backline(
            controller=ctrl, coprocessors=[coproc], transport="rdma", qec_code="steane"
        )

        @qjit(capture=True)
        @qp.set_shots(1)
        @qp.qnode(dev, mcm_method="one-shot")
        def ghz():
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            qp.CNOT([1, 2])
            return qp.sample([qp.measure(0), qp.measure(1), qp.measure(2)])

        ir = ghz.mlir
        assert "catalyst.backline" in ir
        assert 'transport = "rdma"' in ir
        assert 'symbol = "steane_coprocessor"' in ir
        assert 'peer = "127.0.0.1"' in ir
        assert "libcatalyst_transport_cpu_verbs_controller.so" in ir
        assert "libcatalyst_transport_cpu_verbs_coprocessor.so" in ir
        for pass_name, _ in _qec_pass_specs("steane"):
            assert pass_name in ir, f"{pass_name} missing from the scheduled pipeline"

        # Where the coprocessor ended up, asserted rather than inferred from a passing execution.
        executor = coproc.executor
        if executor_options is None:
            assert executor is None, "the default asks for no executor, so the node stays here"
        else:
            assert executor is not None, "executor_options={} did not produce an executor"
            # ``address`` raises unless the executor is up, and only a local one serves on
            # loopback, so this pins the mode as well as the liveness.
            assert executor.address.startswith(
                "127.0.0.1:"
            ), f"a local executor serves on loopback, got {executor.address}"
            plugins = executor._cfg.plugins  # pylint: disable=protected-access
            plugin_names = [Path(plugin).name for plugin in plugins]
            for required in (*_EXECUTOR_RUNTIME_PLUGINS, "libsteane_coprocessor_cpu.so"):
                assert required in plugin_names, f"{required} missing from the executor's plugins"
            assert all(plugin.startswith("/") for plugin in plugins), (
                "a node on this machine needs absolute plugin paths: "
                "the runtime is not on the loader's search path"
            )

        samples = np.asarray(ghz())
        # qp.sample keeps the shots axis explicit: shots=1 with 3 measurements -> (1, 3).
        assert samples.shape == (1, 3), f"expected shape (1, 3), got {samples.shape}"

    def test_cpu_controller_to_gpu_coproc_memcpy_manual_qec(
        self, use_capture, gpu_triton_platform, gpu_transport_backend
    ):
        """CPU controller ↔ GPU coprocessor over ``memcpy`` with a manually-scheduled QEC cycle.

        Ports demo 2's CSS BP decoder + manual QEC pattern to local memcpy transport, executed
        on whichever GPU is attached (NVIDIA via ``cuda:*`` or AMD via ``hip:*`` — the platform
        is picked by ``gpu_triton_platform``). The QNode extracts syndromes, calls the
        coprocessor's Triton-built ``hgp_bp_osd_decoder`` via ``qp.backline.decode`` for each of
        the 13 data qubits, and applies corrections in the same shot. The elaborate encoded /
        logical-op / iterated-error-injection structure follows demo 2 and is what distinguishes
        this test from the shorter ``*_triton_css_bp`` variant above.
        """
        del gpu_transport_backend  # gate only: the HIP-built backend library must exist
        pytest.importorskip("triton")

        n_data = 13
        aux = n_data
        checks = np.array([[1] * 7 + [0] * 6, [0] * 6 + [1] * 7])
        logical_support = list(range(n_data))
        swap_pairs = [(0, 1), (2, 3)]

        decoder = qp.backline.css_bp_decoder(
            checks, checks, postprocess="osd", num_iters=5, prob=0.1, platform=gpu_triton_platform
        )

        ctrl = qp.Controller(
            name="cpu-controller",
            device=qp.device("null.qubit", wires=n_data + 1),
        )
        coproc = qp.Coprocessor(
            name="gpu-coproc",
            coprocessor_fn=decoder,
            hardware="gpu",
        )
        dev = qp.Backline(controller=ctrl, coprocessors=[coproc], transport="memcpy")

        def encode_logical_zero():
            encoder = {
                0: [6, 9, 11],
                1: [7, 9, 10, 11, 12],
                2: [8, 10, 12],
                3: [6, 11],
                4: [7, 11, 12],
                5: [8, 12],
            }
            for pivot, targets in encoder.items():
                qp.Hadamard(wires=pivot)
                for t in targets:
                    qp.CNOT(wires=[pivot, t])

        def logical_circuit():
            for w in logical_support:
                qp.X(wires=w)
            for w in range(n_data):
                qp.Hadamard(wires=w)
            for a, b in swap_pairs:
                qp.SWAP(wires=[a, b])
            for w in logical_support:
                qp.Z(wires=w)
            for w in range(n_data):
                qp.Hadamard(wires=w)
            for a, b in swap_pairs:
                qp.SWAP(wires=[a, b])

        def add_error(error_qubit, error_kind):
            for q in range(n_data):
                is_target = error_qubit == q
                qp.cond(is_target & (error_kind == 1), qp.X)(wires=q)
                qp.cond(is_target & (error_kind == 2), qp.Y)(wires=q)
                qp.cond(is_target & (error_kind == 3), qp.Z)(wires=q)

        def extract_syndromes():
            z_syndrome = []
            for row in checks:
                for q in np.flatnonzero(row):
                    qp.CNOT(wires=[int(q), aux])
                z_syndrome.append(qp.measure(aux, reset=True))
            x_syndrome = []
            for row in checks:
                qp.Hadamard(wires=aux)
                for q in np.flatnonzero(row):
                    qp.CNOT(wires=[aux, int(q)])
                qp.Hadamard(wires=aux)
                x_syndrome.append(qp.measure(aux, reset=True))
            return z_syndrome, x_syndrome

        def apply_correction(correction, pauli):
            for q in range(n_data):
                qp.cond(correction[q], pauli)(wires=q)

        def mean_stabilizer(rows, pauli):
            return qp.dot(
                [1 / len(rows)] * len(rows),
                [qp.prod(*(pauli(wires=int(q)) for q in np.flatnonzero(row))) for row in rows],
            )

        @qjit(capture=True)
        @qp.set_shots(1)
        @qp.qnode(dev, mcm_method="one-shot")
        def encoded_decoded_circuit(error_kind: int):
            encode_logical_zero()
            logical_circuit()
            for error_qubit in range(n_data):
                add_error(error_qubit, error_kind)
                z_syndrome, x_syndrome = extract_syndromes()
                correction_z = qp.backline.decode(x_syndrome, decoder_id=0)
                correction_x = qp.backline.decode(z_syndrome, decoder_id=1)
                apply_correction(correction_x, qp.X)
                apply_correction(correction_z, qp.Z)
            return (
                qp.expval(mean_stabilizer(checks, qp.Z)),
                qp.expval(mean_stabilizer(checks, qp.X)),
            )

        ir = encoded_decoded_circuit.mlir
        assert "catalyst.backline" in ir
        assert 'transport = "memcpy"' in ir
        assert decoder.symbol_name in ir
        # Only the symbol reaches the IR: backline dlopens an in-process coprocessor's library
        # itself (RTLD_GLOBAL, in _load_coprocessor_fn_libs) and backend_lib names the transport
        # plugin instead. Assert the Triton build produced a library, not that its path is in the
        # module.
        assert decoder.lib_path is not None and Path(decoder.lib_path).exists()
        assert "libcatalyst_transport_memcpy_controller.so" in ir
        assert "libcatalyst_transport_memcpy_gpu_coprocessor.so" in ir

        result = encoded_decoded_circuit(0)  # error_kind=0: identity, no injected error
        assert len(result) == 2

    def test_cpu_controller_to_gpu_coproc_memcpy_precompiled(
        self, use_capture, gpu_triton_platform, gpu_transport_backend
    ):
        """Precompiled GPU decoder (``gpu_steane_launcher``) reached over local memcpy, executed.

        Adapts demo 4's steane-on-GPU pattern to local memcpy transport - remote SSH
        executors from the demo are stripped so both roles run in this process on the GPU
        runner. The ``gpu_steane_launcher`` symbol lives inside
        ``libcatalyst_transport_memcpy_gpu_coprocessor.so`` (via the static
        ``catalyst_transport_coproc_gpu`` lib), so ``CoprocessorFunction`` carries no
        ``lib_path``; the runtime resolves the symbol after dlopen of the backend .so.

        Skipped by ``gpu_triton_platform`` on runners without a GPU. On the GPU workflow this
        executes end-to-end once HIP-on-CUDA is installed and the runtime CMake builds the
        transport GPU coproc lib with the launcher symbol.
        """
        # Both are gates only: one for GPU + Triton driver presence, one for the HIP-built
        # backend library. Neither value is consumed here.
        del gpu_triton_platform, gpu_transport_backend
        ctrl = qp.Controller(
            name="cpu-controller",
            device=qp.device("null.qubit", wires=3),
        )
        coproc = qp.Coprocessor(
            name="gpu-coproc",
            coprocessor_fn=qp.CoprocessorFunction("gpu_steane_launcher"),
            hardware="gpu",
        )
        dev = qp.Backline(
            controller=ctrl, coprocessors=[coproc], transport="memcpy", qec_code="steane"
        )

        @qjit(capture=True)
        @qp.set_shots(1)
        @qp.qnode(dev, mcm_method="one-shot")
        def ghz():
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            qp.CNOT([1, 2])
            return qp.sample([qp.measure(0), qp.measure(1), qp.measure(2)])

        ir = ghz.mlir
        assert "catalyst.backline" in ir
        assert 'transport = "memcpy"' in ir
        assert 'symbol = "gpu_steane_launcher"' in ir
        assert "libcatalyst_transport_memcpy_gpu_coprocessor.so" in ir
        for pass_name, _ in _qec_pass_specs("steane"):
            assert pass_name in ir, f"{pass_name} missing from the scheduled pipeline"

        samples = np.asarray(ghz())
        # qp.sample keeps the shots axis explicit: shots=1 with 3 measurements -> (1, 3).
        assert samples.shape == (1, 3), f"expected shape (1, 3), got {samples.shape}"

    def test_cpu_controller_to_gpu_coproc_triton_css_bp(
        self, use_capture, gpu_triton_platform, gpu_transport_backend
    ):
        """CSS BP decoder built via Triton, adapted from demo 2 to a local memcpy placement.

        Mirrors ``demos/demo_2_remote_cpu_to_remote_gpu_triton.py``: X- and Z-type parity checks
        of the [[13,1,3]] hypergraph-product code fed to ``qp.backline.css_bp_decoder`` to
        produce a Triton-compiled coprocessor library, wired into the QNode. Remote SSH
        executors from the demo are replaced with a local memcpy backend so both roles run in
        this process on the GPU runner, and the QNode is executed.

        The decoder is Triton-generated rather than built into the runtime, which is what this
        case covers that its siblings do not: codegen for the detected ``cuda:<cc>:<warp>``
        platform, the [[13,1,3]] hypergraph-product lowering, and the GPU transport backend
        loading the generated ``.so`` to run the decode round.
        """
        pytest.importorskip("triton")
        platform = gpu_triton_platform
        del gpu_transport_backend  # gate only: the built backend library must exist to execute

        n_data = 13
        aux = n_data
        Hx = np.array(
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            ]
        )
        Hz = np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            ]
        )

        try:
            decoder = qp.backline.css_bp_decoder(
                Hx, Hz, postprocess="osd", num_iters=5, prob=0.1, platform=platform
            )
        except (ImportError, RuntimeError, OSError) as exc:
            pytest.skip(f"Triton css_bp_decoder build unavailable: {exc}")

        ctrl = qp.Controller(
            name="cpu-controller",
            device=qp.device("null.qubit", wires=n_data + 1),
        )
        coproc = qp.Coprocessor(
            name="gpu-coproc",
            coprocessor_fn=decoder,
            hardware="gpu",
        )
        dev = qp.Backline(controller=ctrl, coprocessors=[coproc], transport="memcpy")

        @qjit(capture=True)
        @qp.set_shots(1)
        @qp.qnode(dev, mcm_method="one-shot")
        def circuit():
            qp.Hadamard(0)
            z_syndrome = []
            for row in Hz:
                for q in np.flatnonzero(row):
                    qp.CNOT(wires=[int(q), aux])
                z_syndrome.append(qp.measure(aux, reset=True))
            x_syndrome = []
            for row in Hx:
                qp.Hadamard(wires=aux)
                for q in np.flatnonzero(row):
                    qp.CNOT(wires=[aux, int(q)])
                qp.Hadamard(wires=aux)
                x_syndrome.append(qp.measure(aux, reset=True))
            correction_z = qp.backline.decode(x_syndrome, decoder_id=0)
            correction_x = qp.backline.decode(z_syndrome, decoder_id=1)
            for q in range(n_data):
                qp.cond(correction_x[q], qp.X)(wires=q)
                qp.cond(correction_z[q], qp.Z)(wires=q)
            return qp.sample([qp.measure(q) for q in range(n_data)])

        ir = circuit.mlir
        assert "catalyst.backline" in ir
        assert 'transport = "memcpy"' in ir
        assert "libcatalyst_transport_memcpy_controller.so" in ir
        assert "libcatalyst_transport_memcpy_gpu_coprocessor.so" in ir
        assert decoder.symbol_name in ir
        # Only the symbol: backline loads an in-process coprocessor's library itself, with
        # ctypes RTLD_GLOBAL in _load_coprocessor_fn_libs, so the transport backend resolves the
        # decode function out of the global namespace. Writing the path into backend_lib would
        # displace the transport plugin, which _node_dict says in as many words. So assert the
        # build produced a library, not that its path reached the IR.
        assert decoder.lib_path is not None and Path(decoder.lib_path).exists()

        samples = np.asarray(circuit())
        # qp.sample keeps the shots axis explicit: shots=1 with 13 measurements -> (1, 13).
        assert samples.shape == (1, n_data), f"expected shape (1, {n_data}), got {samples.shape}"


def test_placement_behind_a_wrapper_is_found(use_capture):
    """A placement reaches the module even when qjit was applied to a wrapper, not the QNode.
    The transport passes locate it by role rather than by matching a triple or address, which now
    come only from the node's executor."""
    ctrl = qp.Controller(
        device=qp.device("null.qubit", wires=2),
        name="ctrl",
        executor=_Attached(),
        init_args={"backend_lib": "backend.so", "config": "cfg"},
    )
    dev = qp.Backline(controller=ctrl, coprocessors=[_coproc("cop0")], transport="rdma")

    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(0)
        return qp.probs()

    @qjit(target="mlir", capture=True)
    def workflow():
        return 2.0 * circuit()

    ir = workflow.mlir
    assert "module @workflow" in ir  # the wrapper is the root module
    assert "catalyst.backline" in ir
    assert 'transport = "rdma"' in ir
    assert 'symbol = "coproc_fn"' in ir  # the whole placement, coprocessor included


def test_qec_pass_names_match_the_transforms_that_provide_them():
    """The scheduled names must match the passes' own, since they are written out literally."""
    from catalyst.python_interface.transforms.qecl import (
        convert_quantum_to_qecl_pass,
        inject_noise_to_qecl_pass,
    )
    from catalyst.python_interface.transforms.qecp import (
        convert_qecl_to_qecp_pass,
        convert_qecp_to_quantum_pass,
    )

    scheduled = [name for name, _ in _qec_pass_specs("steane")]
    assert scheduled == [
        convert_quantum_to_qecl_pass.pass_name,
        "symbol-dce",  # a built-in MLIR pass, with no transform of its own
        inject_noise_to_qecl_pass.pass_name,
        convert_qecl_to_qecp_pass.pass_name,
        convert_qecp_to_quantum_pass.pass_name,
    ]


def test_qec_encoding_reaches_a_qnode_behind_a_wrapper(use_capture):
    """A placement's qec_code encodes the QNode even when qjit was applied to a wrapper."""
    dev = qp.Backline(
        controller=_controller(),
        coprocessors=[_coproc("cop0")],
        transport="rdma",
        qec_code="steane",
    )

    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(0)
        return qp.probs()

    @qjit(target="mlir", capture=True)
    def workflow():
        return 2.0 * circuit()

    ir = workflow.mlir
    for name, _ in _qec_pass_specs("steane"):
        assert name in ir, f"{name} missing from the scheduled pipeline"


def test_remote_controller_behind_a_wrapper_is_still_tagged(use_capture):
    """A remote controller called through a wrapper still gets its module tagged by role."""
    ctrl = qp.Controller(
        device=qp.device("null.qubit", wires=2),
        name="ctrl",
        remote=True,
        executor_options={"address": "ctrl:1"},
        init_args={"backend_lib": "backend.so", "config": "cfg"},
    )
    dev = qp.Backline(controller=ctrl, transport="rdma")

    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(0)
        return qp.probs()

    @qjit(target="mlir", capture=True)
    def workflow():
        return 2.0 * circuit()

    assert 'catalyst.backline_role = "controller"' in workflow.mlir


def test_two_qnodes_over_one_placement_are_accepted(use_capture):
    """Two QNodes sharing a device carry one placement between them, not one each."""
    dev = qp.Backline(controller=_controller(), coprocessors=[_coproc("cop0")], transport="rdma")

    @qp.qnode(dev)
    def circuit_a():
        qp.Hadamard(0)
        return qp.probs()

    @qp.qnode(dev)
    def circuit_b():
        qp.CNOT([0, 1])
        return qp.probs()

    @qjit(target="mlir", capture=True)
    def workflow():
        return circuit_a() + circuit_b()

    ir = workflow.mlir
    assert "module @workflow" in ir
    assert ir.count("catalyst.backline = #transport.backline") == 1


def test_two_placements_in_one_program_are_rejected(use_capture):
    """A compiled program carries one placement, so two distinct ones cannot be expressed."""
    dev_a = qp.Backline(controller=_controller(), transport="rdma")
    dev_b = qp.Backline(controller=_controller(), transport="rdma")

    @qp.qnode(dev_a)
    def circuit_a():
        qp.Hadamard(0)
        return qp.probs()

    @qp.qnode(dev_b)
    def circuit_b():
        qp.Hadamard(0)
        return qp.probs()

    with pytest.raises(CompileError, match="2 different backline placements"):

        @qjit(target="mlir", capture=True)
        def workflow():
            return circuit_a() + circuit_b()


@pytest.fixture
def fake_lib_dir(tmp_path, monkeypatch):
    """Stand in for the built runtime lib dir, laid out as a bare ``cmake`` build.

    That build mirrors the source tree under ``<RUNTIME_LIB_DIR>/transport/<transport>/``.
    Entries are paths relative to that ``transport`` directory. See :func:`flat_lib_dir` for the
    layout ``make -C runtime`` produces.
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
    """Transport and node hardware select a Catalyst backend library."""

    @pytest.mark.parametrize(
        ("transport", "hardware", "expected"),
        [
            ("rdma", "cpu", "cpu_verbs"),
            ("rdma", "gpu", "gpu_verbs"),
            ("rdma", "fpga", "hwhs"),
            ("memcpy", "cpu", "memcpy"),
            ("memcpy", "gpu", "memcpy_gpu"),
        ],
    )
    def test_transport_and_hardware_select_backend(self, transport, hardware, expected):
        """Concrete backend names remain a Catalyst implementation detail."""
        assert _resolve_backend(transport, hardware) == expected

    def test_unsupported_transport_hardware_pair_is_rejected(self):
        """A transport must have an implementation for the requested hardware."""
        with pytest.raises(ValueError, match="transport='memcpy', hardware='fpga'"):
            _resolve_backend("memcpy", "fpga")

    def test_nested_cmake_backend_is_found(self, fake_lib_dir):
        """A bare CMake build nests RDMA backends below the transport directory."""
        fake_lib_dir(
            "rdma/cpu_verbs/libcatalyst_transport_cpu_verbs_controller.so",
            "rdma/cpu_verbs/libcatalyst_transport_cpu_verbs_coprocessor.so",
        )
        assert _resolve_backend_lib("rdma", "cpu", "controller", False).endswith(
            "transport/rdma/cpu_verbs/libcatalyst_transport_cpu_verbs_controller.so"
        )
        assert _resolve_backend_lib("rdma", "cpu", "coprocessor", False).endswith(
            "transport/rdma/cpu_verbs/libcatalyst_transport_cpu_verbs_coprocessor.so"
        )

    def test_memcpy_backends_share_transport_directory(self, fake_lib_dir):
        """CPU and GPU memcpy libraries are emitted from the same CMake directory."""
        fake_lib_dir(
            "memcpy/libcatalyst_transport_memcpy_controller.so",
            "memcpy/libcatalyst_transport_memcpy_gpu_coprocessor.so",
        )
        assert _resolve_backend_lib("memcpy", "cpu", "controller", False).endswith(
            "transport/memcpy/libcatalyst_transport_memcpy_controller.so"
        )
        assert _resolve_backend_lib("memcpy", "gpu", "coprocessor", False).endswith(
            "transport/memcpy/libcatalyst_transport_memcpy_gpu_coprocessor.so"
        )

    def test_flat_lib_dir_is_searched(self, flat_lib_dir):
        """``make -C runtime`` passes ``CMAKE_LIBRARY_OUTPUT_DIRECTORY``, flattening the lib dir.

        That is the layout a released or ``make``-built tree has, so it must resolve without the
        ``transport/<transport>/[<backend>/]`` nesting a bare ``cmake`` build produces.
        """
        root = flat_lib_dir("libcatalyst_transport_cpu_verbs_controller.so")
        assert _resolve_backend_lib("rdma", "cpu", "controller", False) == str(
            root / "libcatalyst_transport_cpu_verbs_controller.so"
        )

    def test_remote_node_gets_the_bare_filename(self, fake_lib_dir):
        """A remote node loads from its deployed bundle, so it is given a name, not a local path."""
        fake_lib_dir("rdma/cpu_verbs/libcatalyst_transport_cpu_verbs_coprocessor.so")
        assert (
            _resolve_backend_lib("rdma", "cpu", "coprocessor", True)
            == "libcatalyst_transport_cpu_verbs_coprocessor.so"
        )

    def test_remote_node_does_not_probe_this_machine(self, fake_lib_dir):
        """A remote backend need not exist here: the bundle it loads from is on the other machine.

        The FPGA's ``hwhs`` library is built for aarch64 and lives only on the board, so probing
        the host would test the wrong filesystem and reject a valid placement.
        """
        fake_lib_dir()  # nothing built locally
        assert (
            _resolve_backend_lib("rdma", "fpga", "controller", True)
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
        assert _resolve_backend_lib("rdma", "fpga", "controller", False) == str(
            outside / "libcatalyst_transport_hwhs_controller.so"
        )

    def test_search_path_takes_precedence_over_in_tree(self, tmp_path, monkeypatch, fake_lib_dir):
        """An override entry wins, so a local build can shadow an installed backend."""
        fake_lib_dir("rdma/cpu_verbs/libcatalyst_transport_cpu_verbs_controller.so")
        outside = tmp_path / "override"
        outside.mkdir()
        pinned = outside / "libcatalyst_transport_cpu_verbs_controller.so"
        pinned.write_bytes(b"")
        monkeypatch.setenv("CATALYST_TRANSPORT_PATH", str(outside))
        assert _resolve_backend_lib("rdma", "cpu", "controller", False) == str(pinned)

    def test_missing_backend_names_every_directory_searched(self, fake_lib_dir, monkeypatch):
        """The error is actionable: what was looked for, where, and how to fix it."""
        fake_lib_dir("rdma/cpu_verbs/libcatalyst_transport_cpu_verbs_coprocessor.so")
        monkeypatch.setenv("CATALYST_TRANSPORT_PATH", "/opt/nowhere")
        with pytest.raises(ValueError, match="no transport backend library") as e:
            _resolve_backend_lib("rdma", "gpu", "coprocessor", False)
        msg = str(e.value)
        assert "libcatalyst_transport_gpu_verbs_coprocessor.so" in msg
        assert "/opt/nowhere" in msg
        assert "ENABLE_TRANSPORT=ON" in msg
        assert "CATALYST_TRANSPORT_PATH" in msg

    def test_role_mismatch_fails_before_dlopen(self, fake_lib_dir):
        """gpu_verbs ships no controller library, so a controller lookup fails here."""
        fake_lib_dir("rdma/gpu_verbs/libcatalyst_transport_gpu_verbs_coprocessor.so")
        with pytest.raises(ValueError, match="role='controller'"):
            _resolve_backend_lib("rdma", "gpu", "controller", False)

    def test_hardware_populates_backend_lib_per_role(self, fake_lib_dir):
        """Each node's hardware resolves with the placement transport and its own role.

        The shared fixtures pin an explicit ``backend_lib``, which would take precedence, so these
        nodes are built without one.
        """
        fake_lib_dir(
            "rdma/cpu_verbs/libcatalyst_transport_cpu_verbs_controller.so",
            "rdma/gpu_verbs/libcatalyst_transport_gpu_verbs_coprocessor.so",
        )
        ctrl = qp.Controller(device=qp.device("null.qubit", wires=2), name="ctrl", hardware="cpu")
        cop = qp.Coprocessor(
            name="cop0",
            endpoint=qp.Endpoint("127.0.0.1", 18590),
            coprocessor_fn="coproc_fn",
            hardware="gpu",
        )
        d = serialize_backline(
            qp.Backline(controller=ctrl, coprocessors=[cop], transport="rdma").placement
        )
        assert d["controller"]["backend_lib"].endswith("_cpu_verbs_controller.so")
        assert d["coprocessors"][0]["backend_lib"].endswith("_gpu_verbs_coprocessor.so")

    def test_memcpy_cpu_to_cpu_backend_libs(self, fake_lib_dir):
        """A CPU controller paired with a CPU coprocessor over ``memcpy``."""
        fake_lib_dir(
            "memcpy/libcatalyst_transport_memcpy_controller.so",
            "memcpy/libcatalyst_transport_memcpy_coprocessor.so",
        )
        ctrl = qp.Controller(device=qp.device("null.qubit", wires=2), name="ctrl", hardware="cpu")
        cop = qp.Coprocessor(
            name="cop0",
            endpoint=qp.Endpoint("127.0.0.1", 18590),
            coprocessor_fn="coproc_fn",
            hardware="cpu",
        )
        d = serialize_backline(
            qp.Backline(controller=ctrl, coprocessors=[cop], transport="memcpy").placement
        )
        assert d["transport"] == "memcpy"
        assert d["controller"]["backend_lib"].endswith("_memcpy_controller.so")
        assert d["coprocessors"][0]["backend_lib"].endswith("_memcpy_coprocessor.so")

    def test_memcpy_cpu_to_gpu_backend_libs(self, fake_lib_dir):
        """A CPU controller paired with a GPU coprocessor over ``memcpy``."""
        fake_lib_dir(
            "memcpy/libcatalyst_transport_memcpy_controller.so",
            "memcpy/libcatalyst_transport_memcpy_gpu_coprocessor.so",
        )
        ctrl = qp.Controller(device=qp.device("null.qubit", wires=2), name="ctrl", hardware="cpu")
        cop = qp.Coprocessor(
            name="cop0",
            endpoint=qp.Endpoint("127.0.0.1", 18590),
            coprocessor_fn="coproc_fn",
            hardware="gpu",
        )
        d = serialize_backline(
            qp.Backline(controller=ctrl, coprocessors=[cop], transport="memcpy").placement
        )
        assert d["transport"] == "memcpy"
        assert d["controller"]["backend_lib"].endswith("_memcpy_controller.so")
        assert d["coprocessors"][0]["backend_lib"].endswith("_memcpy_gpu_coprocessor.so")

    def test_explicit_backend_lib_bypasses_builtin_mapping(self):
        """An explicit backend library supports out-of-tree transport/hardware combinations."""
        ctrl = qp.Controller(
            device=qp.device("null.qubit", wires=2),
            name="ctrl",
            hardware="fpga",
            init_args={"backend_lib": "/opt/explicit.so"},
        )
        d = serialize_backline(
            qp.Backline(controller=ctrl, transport=Transport("custom")).placement
        )
        assert d["controller"]["backend_lib"] == "/opt/explicit.so"

    def test_default_cpu_hardware_selects_backend(self, fake_lib_dir):
        """Omitting hardware selects the CPU backend."""
        fake_lib_dir("rdma/cpu_verbs/libcatalyst_transport_cpu_verbs_controller.so")
        ctrl = qp.Controller(device=qp.device("null.qubit", wires=2), name="ctrl")
        d = serialize_backline(qp.Backline(controller=ctrl, transport="rdma").placement)
        assert d["controller"]["backend_lib"].endswith("_cpu_verbs_controller.so")


@pytest.fixture
def no_launch():
    """Keep ``_realize_executor`` from deploying."""
    with mock.patch.object(Executor, "launch", autospec=True) as launch:
        yield launch


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

    def test_node_name_seeds_the_executor_name(self):
        """The node's name names the executor."""

    @pytest.mark.parametrize("options", [{}, {"triple": "aarch64-unknown-linux-gnu"}])
    def test_options_naming_no_location_run_on_this_machine(self, options):
        """Options naming neither a host nor an address ask for a subprocess here."""
        node = _controller(executor_options=options)
        with mock.patch.object(Executor, "launch", autospec=True) as launch:
            ex = _realize_executor(node)
        assert node.remote is False
        assert ex.host is None
        launch.assert_called_once()  # a free-port search settles its address only once it is up

    def test_a_remote_node_without_an_executor_is_rejected(self):
        """Another machine is reached by dispatching to an executor deployed there."""
        node = _controller(remote=True)
        with pytest.raises(CompileError, match="remote but was given no executor"):
            _realize_executor(node)

    def test_an_ssh_executor_on_a_node_that_is_not_remote_is_rejected(self):
        """``host`` deploys over ssh, so the node's libraries come from that machine, not this one."""
        node = _controller(executor_options={"host": "192.0.2.10", "port": 7810})
        with pytest.raises(CompileError, match="not remote but its executor is deployed to host"):
            _realize_executor(node)

    def test_a_remote_node_whose_executor_names_no_machine_is_rejected(self):
        """Options naming neither host nor address ask for a subprocess of this process."""
        node = _controller(remote=True, executor_options={"port": 7810})
        with pytest.raises(CompileError, match="neither a 'host' .* nor an 'address'"):
            _realize_executor(node)

    def test_a_controllers_device_runtime_is_implied(self):
        """A controller dispatches a QNode, so it needs its device's runtime, which the compiled
        program would otherwise name by the compiling host's own absolute path. A remote node names
        it by filename, the deployed bundle supplying the file."""
        node = _controller(remote=True, executor_options={"host": "192.0.2.10", "port": 7810})
        _realize_executor(node)
        plugins = node.executor._cfg.plugins  # pylint: disable=protected-access
        assert plugins[-1] == "librtd_null_qubit.so"

    def test_a_remote_nodes_device_runtime_drops_this_platforms_extension(self, no_launch):
        """A node on another machine is Linux and loads the runtime from its deployed bundle, so a
        macOS host's ``.dylib`` would not name the file there."""
        node = _controller(remote=True, executor_options={"host": "h", "port": 1})
        info = extract_backend_info(node.device)
        macos = BackendInfo(
            info.device_name, info.c_interface_name, "/opt/lib/librtd_null_qubit.dylib", info.kwargs
        )
        with mock.patch("catalyst.device.qjit_device.extract_backend_info", return_value=macos):
            _realize_executor(node)
        plugins = node.executor._cfg.plugins  # pylint: disable=protected-access
        assert plugins[-1] == "librtd_null_qubit.so"

    def test_a_dispatched_controller_names_its_runtime_by_filename(self):
        """A program carrying this installation's path opens it on the far machine and fails."""
        dev = qp.Backline(controller=_controller(remote=True), transport="rdma")
        info = extract_backend_info(dev)
        assert remote_device_lib(dev, info.lpath) == "librtd_null_qubit.so"
        assert _get_device_kwargs(dev)["rtd_lib"] == "librtd_null_qubit.so"

    def test_a_controller_in_this_process_keeps_its_full_path(self):
        """Nothing was deployed, so the path into this installation is what resolves."""
        dev = qp.Backline(controller=_controller(), transport="rdma")
        info = extract_backend_info(dev)
        assert remote_device_lib(dev, info.lpath) is None
        assert _get_device_kwargs(dev)["rtd_lib"] == info.lpath

    def test_a_device_with_no_placement_is_left_alone(self):
        """Only a backline placement implies a remote node, so every other device is untouched."""
        plain = qp.device("null.qubit", wires=1)
        assert remote_device_lib(plain, "/opt/lib/librtd_null_qubit.so") is None

    def test_a_dispatched_runtime_is_named_with_a_linux_extension(self):
        """A node on another machine is Linux, so a macOS host's .dylib is not what it asks for."""
        dev = qp.Backline(controller=_controller(remote=True), transport="rdma")
        assert remote_device_lib(dev, "/opt/lib/librtd_null_qubit.dylib") == "librtd_null_qubit.so"

    def test_a_remote_controllers_device_runtime_is_not_carried(self, no_launch):
        """A node of another architecture needs the copy cross-built for its own bundle.

        Sending this host's would land the wrong architecture in its workspace under that filename,
        displacing the right one.
        """
        node = _controller(remote=True, executor_options={"host": "192.0.2.10", "port": 7810})
        _realize_executor(node)
        cfg = node.executor._cfg  # pylint: disable=protected-access
        assert cfg.plugins[-1] == "librtd_null_qubit.so"
        assert not cfg.deploy

    def test_a_dispatched_coprocessor_carries_its_function_library(self, tmp_path):
        """A decoder built for this run is not in the target's bundle, so it travels and is opened."""
        lib = tmp_path / "libdecoder.so"
        lib.write_bytes(b"\x7fELF")
        fn = qp.CoprocessorFunction("_persistent_decoder_kernel_abc", lib_path=str(lib))
        node = _coproc(
            "cop0", fn=fn, remote=True, executor_options={"host": "192.0.2.11", "port": 7813}
        )
        with mock.patch.object(Executor, "launch", autospec=True):
            _realize_executor(node)
        cfg = node.executor._cfg  # pylint: disable=protected-access
        assert cfg.plugins[-1] == "libdecoder.so"  # by filename: resolves against the workspace
        assert cfg.deploy == [str(lib)]  # and the file is carried there

    def test_a_coprocessor_on_this_machine_deploys_nothing(self, no_launch):
        """Its library is already reachable, so the full path is enough and nothing travels."""
        fn = qp.CoprocessorFunction("fn", lib_path="/opt/libdecode.so")
        node = _coproc("cop0", fn=fn, executor_options={})
        _realize_executor(node)
        cfg = node.executor._cfg  # pylint: disable=protected-access
        assert cfg.plugins[-1] == "/opt/libdecode.so"
        assert not cfg.deploy

    def test_a_coprocessor_gets_no_device_runtime(self, no_launch):
        """Only a controller dispatches a QNode, so only it needs a device runtime."""
        node = _coproc("cop0", remote=True, executor_options={"host": "h", "port": 7812})
        _realize_executor(node)
        plugins = node.executor._cfg.plugins  # pylint: disable=protected-access
        assert plugins == list(_EXECUTOR_RUNTIME_PLUGINS)

    def test_plugins_the_node_named_keep_their_place(self, no_launch):
        """A runtime library the node names itself keeps the position it gave it rather than being
        added a second time, and the device runtime still lands last."""
        node = _controller(
            remote=True,
            executor_options={"host": "h", "port": 1, "plugins": ["libsteane.so", "librt_capi.so"]},
        )
        _realize_executor(node)
        plugins = node.executor._cfg.plugins  # pylint: disable=protected-access
        assert plugins.index("libsteane.so") < plugins.index("librt_capi.so")  # order kept
        assert plugins.count("librt_capi.so") == 1
        assert plugins[-1] == "librtd_null_qubit.so"

    def test_a_node_on_this_machine_gets_full_library_paths(self, no_launch):
        """Its libraries come from this installation, so a filename alone would not resolve, and the
        device runtime keeps this platform's extension rather than a remote node's ``.so``."""
        ext = "dylib" if platform.system() == "Darwin" else "so"
        node = _controller(executor_options={})
        _realize_executor(node)
        plugins = node.executor._cfg.plugins  # pylint: disable=protected-access
        assert all(p.startswith("/") for p in plugins)
        assert [p.rsplit("/", 1)[-1] for p in plugins][-1] == f"librtd_null_qubit.{ext}"

    def test_an_attached_executor_is_taken_at_the_nodes_word(self):
        """An ``address`` may name either machine, so ``remote`` is not second-guessed for it."""
        for remote in (True, False):
            node = _controller(remote=remote, executor_options={"address": "10.0.0.9:1373"})
            assert _realize_executor(node).address == "10.0.0.9:1373"

    def test_a_host_without_a_port_is_rejected(self):
        """An ssh deployment must pin its port, so compiling never has to reach the host."""
        node = _controller(remote=True, executor_options={"host": "192.0.2.10", "user": "me"})
        with pytest.raises(CompileError, match=r"host='192\.0\.2\.10' without a port"):
            _realize_executor(node)

    def test_a_host_with_a_port_defers_its_deployment(self):
        """A pinned port makes the address predictable, so compiling costs no ssh."""
        node = _controller(
            remote=True, executor_options={"host": "192.0.2.10", "user": "me", "port": 7810}
        )
        with mock.patch.object(Executor, "launch", autospec=True) as launch:
            ex = _realize_executor(node)
        assert ex.address == "127.0.0.1:7810"  # the local end of the ssh tunnel
        launch.assert_not_called()

    def test_name_seeds_the_executor_name(self):
        """The node's name names the executor, which uses it for its logs."""
        node = _controller(executor_options={"address": "10.0.0.9:1373"})
        assert _realize_executor(node).name == "ctrl"

    def test_a_preset_executor_is_returned_untouched(self):
        """Setting ``executor`` directly attaches an already-launched one; options are ignored."""

        class _Ex:
            address = "attached:1"
            triple = "x86_64-unknown-linux-gnu"

            def launch(self):
                return self

        ex = _Ex()
        node = _controller(executor=ex, executor_options={"address": "ignored:2"})
        assert _realize_executor(node) is ex

    def test_an_executor_missing_the_shape_is_rejected(self):
        """An object that cannot say where it serves, or be launched, is refused while compiling."""

        class _NoLaunch:
            address = "attached:1"

        node = _controller(executor=_NoLaunch())
        with pytest.raises(CompileError, match=r"missing \['launch'\]"):
            _realize_executor(node)

    def test_launch_executors_walks_every_node(self):
        """``launch_executors`` covers the controller and each coprocessor."""
        ctrl = _controller(executor_options={"address": "ctrl:1"})
        cop = _coproc("cop0", executor_options={"address": "cop:2"})
        dev = qp.Backline(controller=ctrl, coprocessors=[cop], transport="rdma")
        launch_executors(dev.placement)
        assert ctrl.executor.address == "ctrl:1"
        assert cop.executor.address == "cop:2"

    def test_executor_address_and_triple_reach_the_serialized_node(self):
        """The launched executor supplies the node's dispatch address and target triple."""
        ctrl = _controller(
            executor_options={"address": "10.0.0.9:1373", "triple": "aarch64-unknown-linux-gnu"},
        )
        dev = qp.Backline(controller=ctrl, transport="rdma")
        launch_executors(dev.placement)
        node = serialize_backline(dev.placement)["controller"]
        assert node["address"] == "10.0.0.9:1373"
        assert node["triple"] == "aarch64-unknown-linux-gnu"

    def test_a_high_oob_port_survives_to_the_ir(self, use_capture):
        """A port above 32767 appears as itself, not as a negative number."""
        cop = _coproc("cop0", oob_port=40000)
        dev = qp.Backline(controller=_controller(), coprocessors=[cop], transport="rdma")

        @qjit(target="mlir", capture=True)
        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(0)
            return qp.probs()

        assert "oob_port = 40000" in circuit.mlir
        assert "-25536" not in circuit.mlir
