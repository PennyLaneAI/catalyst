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

"""Unit tests for :mod:`catalyst.executor.process` — the ``catalyst-executor`` subprocess
lifecycle (spawn, output pump, ready/port-conflict detection, teardown) for the base class and
its local / remote subclasses. ``subprocess.Popen`` is mocked throughout."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from catalyst.executor.process import _ExecutorProcess, _LocalProcess, _RemoteProcess
from catalyst.executor.utils import Paths, Patterns, PortInUse


def _mk_base_proc(**overrides):
    """Construct a bare _ExecutorProcess (base class) with sensible defaults for tests."""
    defaults = dict(
        name="test-executor",
        addr="127.0.0.1:9999",
        bind_port=9999,
        ready_timeout=0.5,
        log_path=None,
    )
    defaults.update(overrides)
    return _ExecutorProcess(**defaults)


class TestExecutorProcessConstruction:
    """Constructor defaults and class-level constants of :class:`_ExecutorProcess`."""

    def test_defaults(self):
        """Constructor stores its arguments and leaves lifecycle flags unset."""
        p = _mk_base_proc()
        assert p.name == "test-executor"
        assert p.addr == "127.0.0.1:9999"
        assert p.ready_timeout == 0.5
        assert p.proc is None
        assert not p._ready.is_set()
        assert not p._port_conflict.is_set()

    def test_localhost_constant(self):
        """``LOCALHOST`` class attribute is the loopback IPv4 address."""
        assert _ExecutorProcess.LOCALHOST == "127.0.0.1"


class TestExecutorProcessLog:
    """Log file open/write behavior of :meth:`_open_log` and :meth:`_log_tee`."""

    def test_log_tee_noop_when_no_fh(self):
        """:meth:`_log_tee` is a no-op when no log file handle is attached."""
        p = _mk_base_proc()
        p._log_tee("anything")  # no-op, no crash

    def test_log_tee_writes(self, tmp_path):
        """:meth:`_log_tee` writes the given line through the opened log handle."""
        log = tmp_path / "x.log"
        p = _mk_base_proc(log_path=str(log))
        p._open_log()
        p._log_tee("hello")
        assert p._log_fh is not None
        p._log_fh.close()
        content = log.read_text()
        assert "hello" in content

    def test_open_log_swallows_oserror(self, tmp_path):
        """:meth:`_open_log` swallows OSError and leaves ``_log_fh`` as None."""
        # Point at a directory (not a file) — open() raises IsADirectoryError.
        p = _mk_base_proc(log_path=str(tmp_path))
        p._open_log()  # must not raise
        assert p._log_fh is None


class TestExecutorProcessPumpFlags:
    """Pump-side flag transitions driven by :class:`Patterns` matches on output lines."""

    def test_ready_line_sets_flag(self):
        """A ready-pattern match sets the ``_ready`` event."""
        p = _mk_base_proc()
        # Simulate the pump identifying a ready line.
        line = "Listening on 127.0.0.1:9999"
        if Patterns.is_ready(line):
            p._ready.set()
        assert p._ready.is_set()

    def test_port_conflict_line_sets_flag(self):
        """A port-conflict-pattern match sets the ``_port_conflict`` event."""
        p = _mk_base_proc()
        line = "Address already in use"
        if Patterns.is_port_conflict(line):
            p._port_conflict.set()
        assert p._port_conflict.is_set()


class TestExecutorProcessCheckPortConflict:
    """Behavior of :meth:`_check_port_conflict` under the port-conflict flag."""

    def test_raises_when_flag_set(self):
        """Raises :class:`PortInUse` when the port-conflict flag is set."""
        p = _mk_base_proc()
        p._port_conflict.set()
        with pytest.raises(PortInUse):
            p._check_port_conflict()

    def test_noop_when_flag_unset(self):
        """No-op when the port-conflict flag is clear."""
        _mk_base_proc()._check_port_conflict()  # no raise


class TestExecutorProcessCheckEarlyExit:
    """Behavior of :meth:`_check_early_exit` across process-state combinations."""

    def test_running_process_noop(self):
        """No-op while the child process is still running."""
        p = _mk_base_proc()
        p.proc = MagicMock()
        p.proc.poll.return_value = None
        p._check_early_exit()  # no raise

    def test_dead_process_raises_systemexit(self):
        """Raises :class:`SystemExit` when the child exited without a port conflict."""
        p = _mk_base_proc()
        p.proc = MagicMock()
        p.proc.poll.return_value = 1
        p.proc.returncode = 1
        with pytest.raises(SystemExit, match="exited"):
            p._check_early_exit()

    def test_dead_with_port_conflict_raises_portinuse(self):
        """:class:`PortInUse` takes precedence when both dead-process and port-conflict apply."""
        p = _mk_base_proc()
        p.proc = MagicMock()
        p.proc.poll.return_value = 1
        p._port_conflict.set()
        with pytest.raises(PortInUse):
            p._check_early_exit()


class TestExecutorProcessShutdown:
    """Signal escalation performed by :meth:`_shutdown` across child states."""

    def test_idempotent_no_proc(self):
        """:meth:`_shutdown` is a no-op when no child process is attached."""
        _mk_base_proc()._shutdown()  # no raise, no proc

    def test_dead_proc_no_signal(self):
        """Sends no signal when the child has already exited."""
        p = _mk_base_proc()
        p.proc = MagicMock()
        p.proc.poll.return_value = 0  # already exited
        p._shutdown()
        p.proc.terminate.assert_not_called()

    def test_alive_proc_terminated(self):
        """Sends SIGTERM once to a running child."""
        p = _mk_base_proc()
        p.proc = MagicMock()
        p.proc.poll.return_value = None
        p.proc.wait.return_value = 0
        p._shutdown()
        p.proc.terminate.assert_called_once()

    def test_hanging_proc_killed_after_timeout(self):
        """Escalates to SIGKILL when the child does not exit within ``wait_time``."""
        p = _mk_base_proc()
        p.proc = MagicMock()
        p.proc.poll.return_value = None
        p.proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=1)
        p._shutdown(wait_time=0.01)
        p.proc.kill.assert_called_once()


class TestExecutorProcessWaitForReady:
    """Polling loop in :meth:`_wait_for_ready` and its exit conditions."""

    def test_returns_when_ready(self):
        """Returns ``self`` immediately once the ready flag is set."""
        p = _mk_base_proc(ready_timeout=1.0)
        p.proc = MagicMock()
        p.proc.poll.return_value = None
        p._ready.set()  # pre-flag ready so the very first wait() returns immediately
        result = p._wait_for_ready(poll_interval=0.05)
        assert result is p

    def test_timeout_raises(self):
        """Raises :class:`SystemExit` when readiness is not signaled within ``ready_timeout``."""
        p = _mk_base_proc(ready_timeout=0.05)
        p.proc = MagicMock()
        p.proc.poll.return_value = None
        with pytest.raises(SystemExit, match="did not become ready"):
            p._wait_for_ready(poll_interval=0.02)

    def test_port_conflict_raises(self):
        """Raises :class:`PortInUse` when the port-conflict flag is set during polling."""
        p = _mk_base_proc(ready_timeout=0.5)
        p.proc = MagicMock()
        p.proc.poll.return_value = None
        p._port_conflict.set()
        with pytest.raises(PortInUse):
            p._wait_for_ready(poll_interval=0.02)


class TestLocalProcessConstruction:
    """Constructor wiring for :class:`_LocalProcess`."""

    def test_address_composition(self):
        """Composes ``addr`` as ``127.0.0.1:<port>`` from the given port."""
        p = _LocalProcess(port=1373, executor_bin="/bin/exec")
        assert p.addr == "127.0.0.1:1373"
        assert p._bind_port == 1373

    def test_defaults(self):
        """Applies default plugin list, env mapping, and process name when not supplied."""
        p = _LocalProcess(port=1373, executor_bin="/bin/exec")
        assert p._plugins == []
        assert p._env == {}
        assert p.name == "executor"


class TestLocalProcessSpawn:
    """Argv construction and environment handling of :meth:`_LocalProcess._spawn`."""

    def test_argv_shape(self):
        """Builds argv with binary, ``--bind``, ``--plugin`` entries and forwards env additions."""
        p = _LocalProcess(
            port=1373,
            executor_bin="/tmp/catalyst-executor",
            plugins=["/opt/libx.so"],
            env={"K": "V"},
        )
        # Capture the argv passed to _popen without actually spawning.
        captured = {}

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            captured["env"] = kwargs.get("env")
            return MagicMock()

        with patch.object(_LocalProcess, "_popen", side_effect=fake_popen):
            p._spawn()
        assert captured["argv"][0] == "/tmp/catalyst-executor"
        assert "--bind=127.0.0.1:1373" in captured["argv"]
        assert "--plugin=/opt/libx.so" in captured["argv"]
        # env extends os.environ, custom key present:
        assert captured["env"]["K"] == "V"

    def test_expands_home_in_binary(self):
        """Expands a leading ``~`` in the executor binary path before spawning."""
        p = _LocalProcess(port=1373, executor_bin="~/bin/exec")
        captured = {}
        with patch.object(
            _LocalProcess,
            "_popen",
            side_effect=lambda argv, **kw: (captured.setdefault("argv", argv), MagicMock())[1],
        ):
            p._spawn()
        # ~ should have been expanded.
        assert "~" not in captured["argv"][0]

    def test_env_var_expansion(self, monkeypatch):
        """Expands ``$VAR`` references inside env values against the current environment."""
        monkeypatch.setenv("MY_LIB", "/opt/mylib")
        p = _LocalProcess(port=1373, executor_bin="/bin/exec", env={"LIB": "$MY_LIB/x"})
        captured = {}
        with patch.object(
            _LocalProcess,
            "_popen",
            side_effect=lambda argv, **kw: (captured.setdefault("env", kw.get("env")), MagicMock())[1],
        ):
            p._spawn()
        assert captured["env"]["LIB"] == "/opt/mylib/x"


class TestRemoteProcessConstruction:
    """Constructor wiring for :class:`_RemoteProcess`."""

    def test_local_port_defaults_to_remote_port(self):
        """``local_port`` defaults to the remote ``port`` when not attached."""
        p = _RemoteProcess(host="h", user="me", port=1373, workspace="~/ws")
        assert p.local_port == 1373
        assert p.addr == "127.0.0.1:1373"

    def test_local_port_explicit(self):
        """An attached ``local_port`` takes precedence over the remote port for ``addr``."""
        p = _RemoteProcess(host="h", user="me", port=1373, local_port=5000, workspace="~/ws")
        assert p.local_port == 5000
        assert p.addr == "127.0.0.1:5000"

    def test_defaults(self):
        """Applies default sudo, workspace-cleanup, executor-binary, and ready-tracking values."""
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/ws")
        assert p.sudo is True
        assert p.cleanup_ws is False
        assert p.executor_bin == f"./{Paths.EXECUTOR_BIN}"
        assert not p._ready_reached


class TestRemoteProcessLogHeader:
    """Formatting of the remote-process log-file header string."""

    def test_contains_metadata(self):
        """Header string includes name, host:port, workspace, and comma-joined plugins."""
        p = _RemoteProcess(
            host="h",
            user="me",
            port=1373,
            workspace="~/ws",
            plugins=["libx.so", "liby.so"],
            name="worker",
        )
        header = p._log_header()
        assert "worker" in header
        assert "h:1373" in header
        assert "~/ws" in header
        assert "libx.so, liby.so" in header


class TestRemoteProcessScanLine:
    """Auth-prompt detection performed by :meth:`_RemoteProcess._scan_line`."""

    def test_ssh_prompt_sets_ssh_kind(self):
        """Sets the auth flag with ``ssh`` kind on an SSH password prompt."""
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/ws")
        p._scan_line("me@h's password:")
        assert p._auth_prompt.is_set()
        assert p._auth_kind == "ssh"

    def test_sudo_fail_sets_sudo_kind(self):
        """Sets the auth flag with ``sudo`` kind on a sudo authentication failure."""
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/ws")
        p._scan_line("sudo: incorrect password")
        assert p._auth_prompt.is_set()
        assert p._auth_kind == "sudo"

    def test_benign_line_no_flag(self):
        """Leaves the auth flag clear on a benign non-prompt line."""
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/ws")
        p._scan_line("Welcome!")
        assert not p._auth_prompt.is_set()


class TestRemoteProcessCheckFailure:
    """Behavior of :meth:`_check_failure` when the auth-prompt flag is set."""

    def test_no_flag_noop(self):
        """No-op when the auth-prompt flag is clear."""
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/ws")
        p._check_failure()  # no raise

    def test_flag_raises_with_help(self):
        """Raises :class:`SystemExit` with kind-specific help text when the auth flag is set."""
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/ws")
        p._auth_kind = "ssh"
        p._auth_prompt.set()
        with pytest.raises(SystemExit, match="ssh-copy-id"):
            p._check_failure()


class TestRemoteProcessAuthHelp:
    """Kind-specific help text produced by :meth:`_auth_help`."""

    def test_ssh_help(self):
        """SSH help text suggests ``ssh-copy-id`` with the configured user and host."""
        p = _RemoteProcess(host="hostx", user="alice", port=1, workspace="~/ws")
        p._auth_kind = "ssh"
        msg = p._auth_help()
        assert "ssh-copy-id alice@hostx" in msg

    def test_sudo_help(self):
        """Sudo help text mentions the :attr:`sudo_password` option."""
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/ws")
        p._auth_kind = "sudo"
        msg = p._auth_help()
        assert "sudo_password=" in msg


class TestRemoteProcessOnReady:
    """State transition triggered by :meth:`_on_ready`."""

    def test_sets_ready_reached(self):
        """Sets the ``_ready_reached`` flag when the ready hook fires."""
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/ws")
        assert not p._ready_reached
        p._on_ready()
        assert p._ready_reached


class TestRemoteProcessTeardownExtra:
    """Remote ``pkill`` invocation from :meth:`_teardown_extra`."""

    def test_skips_when_not_ready(self):
        """No-op before the ready hook has fired: nothing to pkill remotely."""
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/ws")
        with patch("catalyst.executor.process.SSH.pkill") as pkill:
            p._teardown_extra()
        pkill.assert_not_called()

    def test_runs_pkill_when_ready(self):
        """Runs a port-scoped pkill once the remote executor has been ready."""
        p = _RemoteProcess(host="h", user="me", port=1373, workspace="~/ws")
        p._ready_reached = True
        with patch("catalyst.executor.process.SSH.pkill") as pkill:
            p._teardown_extra()
        pkill.assert_called_once()
        # Pattern should be port-scoped so we can't kill someone else's process.
        pat = pkill.call_args.args[2]
        assert "0.0.0.0:1373" in pat


class TestRemoteProcessTeardownWorkspace:
    """Guards on :meth:`teardown_workspace` that gate remote directory removal."""

    def test_pinned_dir_left_intact(self):
        """Pinned workspace (``cleanup_ws=False``) is never removed."""
        # cleanup_ws=False (pinned) — never remove.
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/mydir", cleanup_ws=False)
        with patch("catalyst.executor.process.SSH.rmdir") as rmdir:
            p.teardown_workspace()
        rmdir.assert_not_called()

    def test_prefix_guard_left_intact(self):
        """Directories without the safe workspace prefix are not removed even when cleanup is enabled."""
        # cleanup_ws=True but the basename lacks the safe prefix.
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/mydir", cleanup_ws=True)
        with patch("catalyst.executor.process.SSH.rmdir") as rmdir:
            p.teardown_workspace()
        rmdir.assert_not_called()

    def test_auto_dir_with_prefix_removed(self):
        """Auto-generated workspace matching the safe prefix is removed when cleanup is enabled."""
        ws = f"~/{Paths.WORKSPACE_PREFIX}me-2026-01-01_00-00-00-abc"
        p = _RemoteProcess(host="h", user="me", port=1, workspace=ws, cleanup_ws=True)
        with patch("catalyst.executor.process.SSH.rmdir") as rmdir:
            p.teardown_workspace()
        rmdir.assert_called_once()


class TestRemoteProcessPipeSudoPassword:
    """Stdin handling of :meth:`_pipe_sudo_password` for the remote sudo prompt."""

    def test_noop_when_no_password(self):
        """No-op when no :attr:`sudo_password` is attached."""
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/ws")
        p.proc = MagicMock()
        p._pipe_sudo_password()
        p.proc.stdin.write.assert_not_called()

    def test_writes_password(self):
        """Writes the attached password with a trailing newline and flushes stdin."""
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/ws", sudo_password="pw")
        p.proc = MagicMock()
        p._pipe_sudo_password()
        p.proc.stdin.write.assert_called_once_with("pw\n")
        p.proc.stdin.flush.assert_called_once()

    def test_swallows_broken_pipe(self):
        """Swallows :class:`BrokenPipeError` when the child stdin has already closed."""
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/ws", sudo_password="pw")
        p.proc = MagicMock()
        p.proc.stdin.write.side_effect = BrokenPipeError()
        # Must not raise.
        p._pipe_sudo_password()
