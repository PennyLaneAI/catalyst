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

"""Unit tests for :mod:`catalyst.executor.manager` — the public :class:`Executor`: config
defaults, mode dispatch, attach-mode launch, workspace lifecycle (``setup_workspace`` /
``remove_workspace``), and the ``deploy`` sources placed in the workspace. Subprocess-facing
calls (``SCP.deploy``, ``RemoteOps.rmdir``) are mocked."""

from unittest.mock import patch

import pytest

from catalyst.executor.manager import (
    Executor,
    ExecutorConfig,
    _Mode,
    _SessionRegistry,
    _sessions,
    _start_on_free_port,
)
from catalyst.executor.ssh import SCP
from catalyst.executor.utils import ExecutorPaths


class TestExecutorConfigDefaults:
    """Default values of :class:`ExecutorConfig`."""

    def test_defaults(self):
        """Fresh :class:`ExecutorConfig` uses documented defaults."""
        c = ExecutorConfig()
        assert c.user == ""
        assert c.port is None
        assert c.deploy is None
        assert c.sudo is False  # root is opt-in: the executor runs arbitrary compiled objects
        assert c.ready_timeout == 60.0
        assert c.verbose == 1


class TestExecutorConstruction:
    """Attribute wiring in :meth:`Executor.__init__` for each mode."""

    def test_carry_mode_defaults(self):
        """Attach-only mode (an address, no host) — inert state and default name."""
        ex = Executor("1.2.3.4:5")
        assert ex.host is None
        assert ex._mode is _Mode.ATTACHED
        assert ex.name == "executor"
        assert not ex._launched

    def test_subprocess_mode_is_inferred(self):
        """Naming neither a host nor an address leaves the subprocess mode."""
        ex = Executor()
        assert ex._mode is _Mode.LOCAL

    def test_host_mode_stored(self):
        """``host`` and ``user`` land on ``.host`` and the config."""
        ex = Executor(host="10.0.0.9", user="me")
        assert ex.host == "10.0.0.9"
        assert ex._cfg.user == "me"

    def test_no_address_default(self):
        """``address`` has no default: nothing is assumed about where an executor is serving."""
        assert Executor()._address is None


class TestModeSelection:
    """What is given picks the mode, so there is no combination that names none."""

    def test_construction_stays_inert(self):
        """Choosing a mode never deploys anything — construction cannot raise."""
        ex = Executor()  # must not raise
        assert not ex._launched

    def test_each_mode_is_reachable(self):
        """An address attaches, a host ssh's, and neither spawns a subprocess."""
        assert Executor("1.2.3.4:5").launch()._launched
        assert Executor(host="h").host == "h"
        assert Executor()._mode is _Mode.LOCAL  # launch() would spawn; the mode is enough


class TestExecutorAddress:
    """The :attr:`Executor.address` property guard."""

    def test_unlaunched_raises(self):
        """Reading ``address`` before ``launch()`` raises :class:`RuntimeError`."""
        ex = Executor("1.2.3.4:5")
        with pytest.raises(RuntimeError, match="not launched"):
            _ = ex.address

    def test_attach_mode_launch_exposes_address(self):
        """Attach-only mode: after ``launch()``, ``.address`` returns the given endpoint."""
        # Attach-only mode (neither local nor host) short-circuits launch(), no subprocess.
        ex = Executor("1.2.3.4:5").launch()
        assert ex.address == "1.2.3.4:5"


class TestResolve:
    """Address commitment ahead of deployment via :meth:`Executor.resolve`."""

    def test_attached_address_resolves(self):
        """An attached executor already has its address, so ``.address`` reads before launching."""
        ex = Executor("1.2.3.4:5")
        assert ex.resolve() is ex
        assert ex.address == "1.2.3.4:5"

    def test_unpinned_subprocess_does_not_resolve(self):
        """A subprocess searches for a free port, so its address is settled only once it is up."""
        ex = Executor()
        assert ex.resolve() is None
        with pytest.raises(RuntimeError, match="not launched"):
            _ = ex.address

    @pytest.mark.parametrize("kwargs", [{}, {"host": "h"}])
    def test_pinned_port_resolves_to_loopback(self, kwargs):
        """A pinned port fixes the address: loopback for a local run, the tunnel end for a remote."""
        ex = Executor(port=9123, **kwargs)
        assert ex.resolve() is ex
        assert ex.address == "127.0.0.1:9123"

    def test_unpinned_port_does_not_resolve(self):
        """Without a pinned port the address is only settled by the free-port search at launch."""
        assert Executor(host="h").resolve() is None


class TestSetupWorkspace:
    """Preconditions and behavior of :meth:`Executor.setup_workspace`."""

    def test_missing_host_raises(self):
        """Requires ``host=`` — otherwise :class:`ValueError`."""
        ex = Executor(workspace="~/ws", deploy=["/tmp/b"])
        with pytest.raises(ValueError, match="host="):
            ex.setup_workspace()

    def test_missing_workspace_raises(self):
        """Requires a pinned ``workspace=`` — otherwise :class:`ValueError`."""
        ex = Executor(host="h", deploy=["/tmp/b"])
        with pytest.raises(ValueError, match="workspace="):
            ex.setup_workspace()

    def test_missing_deploy_raises(self):
        """Requires ``deploy=`` — otherwise :class:`ValueError`."""
        ex = Executor(host="h", workspace="~/ws")
        with pytest.raises(ValueError, match="deploy="):
            ex.setup_workspace()

    def test_calls_deploy_and_clears_it(self, tmp_path):
        """Places the sources and clears ``deploy`` so a later ``launch()`` won't copy again."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        ex = Executor(host="h", user="me", workspace="~/ws", deploy=[str(bundle)])
        with patch.object(Executor, "_deploy_sources") as deploy:
            ex.setup_workspace()
        deploy.assert_called_once()
        assert ex._cfg.deploy == []


class TestRemoveWorkspace:
    """Preconditions and behavior of :meth:`Executor.remove_workspace`."""

    def test_missing_host_raises(self):
        """Requires ``host=`` — otherwise :class:`ValueError`."""
        ex = Executor(workspace="~/ws")
        with pytest.raises(ValueError, match="host="):
            ex.remove_workspace()

    def test_missing_workspace_raises(self):
        """Requires a pinned ``workspace=`` — otherwise :class:`ValueError`."""
        ex = Executor(host="h")
        with pytest.raises(ValueError, match="workspace="):
            ex.remove_workspace()

    def test_delegates_to_remote_rmdir(self):
        """Delegates to :meth:`RemoteOps.rmdir` and threads through ``force=``."""
        ex = Executor(host="h", user="me", workspace="~/ws")
        with patch("catalyst.executor.manager.RemoteOps.rmdir") as rmdir:
            ex.remove_workspace(force=True)
        rmdir.assert_called_once()
        assert rmdir.call_args.kwargs.get("force") is True


class TestDeploySources:
    """What :meth:`Executor._deploy_sources` places in the workspace."""

    def test_noop_when_nothing_named(self):
        """An executor already present on the target is used as it stands, so nothing is copied."""
        ex = Executor(host="h")
        with patch("catalyst.executor.manager.SCP.deploy") as scp_deploy:
            ex._deploy_sources("me", "h", "ws")
        scp_deploy.assert_not_called()

    def test_passes_sources_through_as_paths(self, tmp_path):
        """Sources reach :meth:`SCP.deploy` as :class:`Path`, unmodified.

        Cross-building is the caller's job. This only copies.
        """
        bundle = tmp_path / "b"
        bundle.mkdir()
        extra = tmp_path / "libdecoder.so"
        extra.write_bytes(b"\x7fELF")
        ex = Executor(host="h", user="me", deploy=[str(bundle), str(extra)])
        with patch("catalyst.executor.manager.SCP.deploy") as scp_deploy:
            ex._deploy_sources("me", "h", "ws")
        scp_deploy.assert_called_once_with("me", "h", [bundle, extra], "ws")

    def test_a_directory_contributes_its_files_and_a_file_itself(self, tmp_path):
        """The workspace is flat, so a directory is flattened into it and a file lands beside."""
        bundle = tmp_path / "b"
        bundle.mkdir()
        (bundle / "librt.so").write_bytes(b"a")
        (bundle / "README.md").write_text("not an artifact")
        extra = tmp_path / "libdecoder.so"
        extra.write_bytes(b"b")
        with patch("catalyst.executor.ssh.SCP.copy") as copy, patch(
            "catalyst.executor.ssh.RemoteOps.mkdir"
        ):
            SCP.deploy("me", "h", [bundle, extra], "ws")
        copied = sorted(f.name for f in copy.call_args[0][2])
        assert copied == ["libdecoder.so", "librt.so"]  # README.md is documentation, not payload

    def test_a_missing_source_is_reported(self, tmp_path):
        """Naming something absent fails here rather than as a dlopen error on the target."""
        with pytest.raises(RuntimeError, match="nothing to deploy"):
            SCP.deploy("me", "h", [tmp_path / "absent"], "ws")


class TestAttachModeLifecycle:
    """Attach mode launches nothing, so its lifecycle is pure bookkeeping."""

    def test_launch_short_circuits(self):
        """No process is spawned and the given address is preserved."""
        ex = Executor("1.2.3.4:5").launch()
        assert ex._launched and ex._proc is None
        assert ex.address == "1.2.3.4:5"

    @pytest.mark.parametrize(
        "calls", [("launch",), ("launch", "launch"), ("stop",), ("launch", "stop", "stop")]
    )
    def test_launch_and_stop_are_idempotent(self, calls):
        """Any order and repetition of launch/stop is safe, and never spawns."""
        ex = Executor("1.2.3.4:5")
        for c in calls:
            getattr(ex, c)()
        assert ex._launched is (calls[-1] == "launch")
        assert ex._proc is None


class TestContextManager:
    """``with Executor(...) as ex`` lifecycle."""

    def test_enter_launches_exit_stops(self):
        """``__enter__`` launches; ``__exit__`` stops."""
        with Executor("1.2.3.4:5") as ex:
            assert ex._launched is True
        assert ex._launched is False


class TestRepr:
    """The :meth:`Executor.__repr__` shape."""

    def test_shape(self):
        """``repr()`` includes name, host, and launched state."""
        r = repr(Executor("1.2.3.4:5", host="h"))
        assert "name='executor'" in r
        assert "host='h'" in r
        assert "launched=False" in r


class _FakeProc:
    """Stand-in for a :class:`_ExecutorProcess`, recording teardown and optionally failing."""

    def __init__(self, port=0, fail_ports=()):
        self.addr = f"127.0.0.1:{port}"
        self._port = port
        self._fail_ports = fail_ports
        self.stopped = False
        self.workspace_torn_down = False
        self.port_conflict = False

    def start(self):
        if self._port in self._fail_ports:
            self.port_conflict = True
            raise RuntimeError(f"port {self._port} is already in use")
        return self

    def stop(self):
        self.stopped = True

    def teardown_workspace(self):
        self.workspace_torn_down = True

    def _log_message(self, msg, level=1):
        pass


class TestStartOnFreePort:
    """The port-retry loop behind :meth:`Executor.launch`."""

    def test_returns_the_first_process_that_binds(self):
        """No retry when the first attempt succeeds."""
        made = []

        def make(port):
            made.append(port)
            return _FakeProc(port)

        proc = _start_on_free_port(make, 9000)
        assert proc.addr == "127.0.0.1:9000"
        assert made == [9000], "should not have tried a second port"

    def test_retries_on_a_busy_port(self):
        """A busy port moves on to the next candidate, and the ports differ."""
        made = []

        def make(port):
            made.append(port)
            return _FakeProc(port, fail_ports={9000})

        proc = _start_on_free_port(make, 9000)
        assert made[0] == 9000, "the pinned port is tried first"
        assert len(made) == 2 and made[1] != 9000, "the retry uses a different port"
        assert proc.addr == f"127.0.0.1:{made[1]}"

    def test_exhaustion_raises_with_the_last_error(self):
        """Every candidate busy: reports how many were tried and how to pin one."""

        def make(port):
            return _FakeProc(port, fail_ports=range(70000))  # everything fails

        with pytest.raises(RuntimeError) as exc:
            _start_on_free_port(make, 9000, max_tries=2)
        assert "no free executor port after 3 tries" in str(exc.value)
        assert "port=" in str(exc.value), "should say how to pin one"

    def test_strict_tries_the_pinned_port_only(self):
        """``strict``: a busy pinned port is an error, since the address is already published."""
        procs = []

        def make(port):
            procs.append(_FakeProc(port, fail_ports={9000}))
            return procs[-1]

        with pytest.raises(RuntimeError, match="pinned executor port 9000 is already in use"):
            _start_on_free_port(make, 9000, strict=True)
        assert len(procs) == 1, "must not fall back to another port"
        assert procs[0].workspace_torn_down, "a launch that never bound leaves nothing behind"

    def test_a_non_port_failure_is_not_retried(self):
        """Only a reported port conflict is retryable: another port fails the same way."""
        made = []

        def exited():
            raise RuntimeError("executor exited")

        def make(port):
            made.append(port)
            proc = _FakeProc(port)
            proc.start = exited
            return proc

        with pytest.raises(RuntimeError, match="executor exited"):
            _start_on_free_port(make, 9000)
        assert made == [9000], "a broken executor must not be tried on six more ports"

    def test_an_interrupted_launch_tears_down_its_process(self):
        """A ^C partway through a launch must not leave the executor or its workspace behind."""
        proc = _FakeProc(9000)

        def interrupted():
            raise KeyboardInterrupt

        proc.start = interrupted
        with pytest.raises(KeyboardInterrupt):
            _start_on_free_port(lambda port: proc, 9000)
        assert proc.stopped and proc.workspace_torn_down


class TestSessionRegistry:
    """Bookkeeping behind the ``atexit`` shutdown hook."""

    def test_shutdown_stops_everything_registered(self):
        """Both processes are stopped and torn down, despite sharing a name."""
        reg = _SessionRegistry()
        a, b = _FakeProc(), _FakeProc()
        reg.register(a)
        reg.register(b)
        reg._shutdown_all()
        assert (a.stopped, b.stopped) == (True, True)
        assert (a.workspace_torn_down, b.workspace_torn_down) == (True, True)

    def test_unregister_removes_only_that_process(self):
        """Unregistering one leaves the other under the hook."""
        reg = _SessionRegistry()
        a, b = _FakeProc(), _FakeProc()
        reg.register(a)
        reg.register(b)
        reg.unregister(a)
        reg._shutdown_all()
        assert not a.stopped
        assert b.stopped, "the second executor escaped the atexit hook"

    def test_unregister_tolerates_untracked(self):
        """Called for something never registered, or already cleared, it is a no-op."""
        reg = _SessionRegistry()
        proc = _FakeProc()
        reg.unregister(proc)
        reg.register(proc)
        reg._shutdown_all()
        reg.unregister(proc)  # the hook already cleared the list


class TestDetectTriple:
    """Target-triple resolution in :meth:`Executor._detect_triple`."""

    def test_explicit_triple_short_circuits(self):
        """An explicit ``triple=`` is used as-is, with no probe."""
        with patch("catalyst.executor.manager.RemoteOps.capture") as cap:
            assert Executor(host="h", triple="aarch64-unknown-linux-gnu").triple == (
                "aarch64-unknown-linux-gnu"
            )
        cap.assert_not_called()

    def test_local_uses_this_machine(self):
        """A subprocess reads the host platform rather than going over SSH."""
        with patch("catalyst.executor.manager.platform.system", return_value="Linux"), patch(
            "catalyst.executor.manager.platform.machine", return_value="x86_64"
        ):
            assert Executor().triple == "x86_64-unknown-linux-gnu"

    def test_remote_probes_over_ssh(self):
        """``host=`` maps the remote's ``uname -sm`` onto a triple."""
        with patch("catalyst.executor.manager.RemoteOps.capture", return_value="Linux aarch64"):
            assert Executor(host="h", user="me").triple == "aarch64-unknown-linux-gnu"

    def test_failed_probe_is_none(self):
        """An unreachable host yields ``None``, and the compiler falls back to the host triple."""
        with patch("catalyst.executor.manager.RemoteOps.capture", return_value=None):
            assert Executor(host="h").triple is None

    def test_attach_mode_has_no_triple(self):
        """Nothing to probe when the executor is managed elsewhere."""
        assert Executor("1.2.3.4:5").triple is None


class TestMakers:
    """The per-attempt process factories behind :meth:`Executor.launch`."""

    def test_local_maker_builds_a_local_process(self):
        """The subprocess mode produces a loopback process carrying the configured plugins and env."""
        ex = Executor(plugins=["libx.so"], env={"K": "V"}, executor_bin="/bin/exec")
        proc = ex._local_maker()(9000)
        assert proc.addr == "127.0.0.1:9000"
        assert proc._plugins == ["libx.so"]
        assert proc._env == {"K": "V"}

    def test_remote_maker_builds_a_remote_process(self):
        """``host=`` produces a tunnelled process. The one-time sudo resolve runs during setup."""
        ex = Executor(host="10.0.0.9", user="me", sudo=True, plugins=["libx.so"])
        with patch(
            "catalyst.executor.manager.RemoteOps.resolve_sudo", return_value="pw"
        ) as resolve, patch("catalyst.executor.manager.RemoteOps.mkdir"):
            make = ex._remote_maker()
        proc = make(9000)
        assert (proc.host, proc.user) == ("10.0.0.9", "me")
        assert proc.sudo_password == "pw"
        assert proc.addr == "127.0.0.1:9000"

    def test_remote_maker_reuses_one_auth_context_across_retries(self):
        """Retries must not re-prompt for sudo, re-scp the bundle, or re-create the workspace."""
        ex = Executor(host="h", user="me", sudo=True)
        with patch(
            "catalyst.executor.manager.RemoteOps.resolve_sudo", return_value="pw"
        ) as r, patch("catalyst.executor.manager.RemoteOps.mkdir") as mkdir:
            make = ex._remote_maker()
            make(9000)
            make(9001)
        r.assert_called_once()
        mkdir.assert_called_once()

    def test_remote_maker_creates_the_workspace_when_nothing_is_copied(self):
        """A launch that copies nothing still needs the workspace: the launch command cd's into it."""
        ex = Executor(host="h", user="me")
        with patch("catalyst.executor.manager.RemoteOps.mkdir") as mkdir:
            ex._remote_maker()
        mkdir.assert_called_once()

    def test_remote_maker_runs_a_pinned_workspace_binary_from_the_workspace(self):
        """A pinned workspace holds the binary from an earlier ``setup_workspace``, so run it as ./."""
        ex = Executor(host="h", user="me", workspace="~/cat-ws")
        with patch("catalyst.executor.manager.RemoteOps.mkdir"):
            proc = ex._remote_maker()(9000)
        assert proc.executor_bin == f"./{ExecutorPaths.EXECUTOR_BIN}"


class TestLaunchAndStop:
    """The deploy path of :meth:`Executor.launch` and the teardown in :meth:`Executor.stop`."""

    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        """``launch()`` appends to a module-global and restore it so a failure cannot leak."""
        saved = list(_sessions._procs)
        yield
        _sessions._procs[:] = saved

    def _launched(self, **kw):
        ex = Executor(**kw)
        proc = _FakeProc(9000)
        with patch("catalyst.executor.manager._start_on_free_port", return_value=proc):
            ex.launch()
        return ex, proc

    def test_launch_adopts_the_bound_address(self):
        """``address`` comes from the process that actually bound, not from the request."""
        ex, _ = self._launched()
        assert ex.address == "127.0.0.1:9000"
        assert ex._launched

    def test_launch_registers_for_atexit(self):
        """A launched executor is tracked so it is torn down even without an explicit stop()."""
        ex, proc = self._launched()
        assert proc in _sessions._procs

    @pytest.mark.parametrize("resolved", [True, False])
    def test_launch_pins_the_port_only_when_resolved(self, resolved):
        """A resolved address is already published, so its port is binding; unresolved, the
        free-port search stays in play."""
        ex = Executor(port=9123)
        if resolved:
            ex.resolve()
        with patch("catalyst.executor.manager._start_on_free_port") as start:
            start.return_value = _FakeProc(9123)
            ex.launch()
        assert start.call_args.kwargs["strict"] is resolved

    def test_launch_is_idempotent(self):
        """A second launch() does not spawn again."""
        ex, _ = self._launched()
        with patch("catalyst.executor.manager._start_on_free_port") as start:
            ex.launch()
        start.assert_not_called()

    def test_stop_tears_down_and_deregisters(self):
        """stop() stops the process, removes the workspace, and leaves the registry clean."""
        ex, proc = self._launched()
        ex.stop()
        assert proc.stopped and proc.workspace_torn_down
        assert proc not in _sessions._procs
        assert ex._proc is None and not ex._launched

    def test_stop_is_idempotent(self):
        """A second stop() is a no-op rather than an error."""
        ex, _ = self._launched()
        ex.stop()
        ex.stop()
        assert ex._proc is None
