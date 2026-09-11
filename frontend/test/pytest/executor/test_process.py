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
lifecycle (spawn, output watching, ready/port-conflict detection, teardown) for the base class and
its local / remote subclasses. ``subprocess.Popen`` is mocked throughout."""

import re
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from catalyst.executor.process import _ExecutorProcess, _LocalProcess, _RemoteProcess
from catalyst.executor.ssh import RemoteLauncher
from catalyst.executor.utils import ExecutorFlags, ExecutorPaths, OutputPatterns


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


class TestExecutorProcessCheckPortConflict:
    """Behavior of :meth:`_check_port_conflict` under the port-conflict flag."""

    def test_raises_when_flag_set(self):
        """Raises when the port-conflict flag is set."""
        p = _mk_base_proc()
        p._port_conflict.set()
        assert p.port_conflict
        with pytest.raises(RuntimeError, match="already in use"):
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

    def test_dead_process_raises_runtime_error(self):
        """Raises :class:`RuntimeError` when the child exited without a port conflict."""
        p = _mk_base_proc()
        p.proc = MagicMock()
        p.proc.poll.return_value = 1
        p.proc.returncode = 1
        with pytest.raises(RuntimeError, match="exited"):
            p._check_early_exit()

    def test_dead_with_port_conflict_reports_the_conflict(self):
        """A dead process that hit a port conflict reports the conflict, not the exit."""
        p = _mk_base_proc()
        p.proc = MagicMock()
        p.proc.poll.return_value = 1
        p._port_conflict.set()
        with pytest.raises(RuntimeError, match="already in use"):
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
        """Raises :class:`RuntimeError` when readiness is not signaled within ``ready_timeout``."""
        p = _mk_base_proc(ready_timeout=0.05)
        p.proc = MagicMock()
        p.proc.poll.return_value = None
        with pytest.raises(RuntimeError, match="did not become ready"):
            p._wait_for_ready(poll_interval=0.02)

    def test_port_conflict_raises(self):
        """Raises when the port-conflict flag is set during polling."""
        p = _mk_base_proc(ready_timeout=0.5)
        p.proc = MagicMock()
        p.proc.poll.return_value = None
        p._port_conflict.set()
        with pytest.raises(RuntimeError, match="already in use"):
            p._wait_for_ready(poll_interval=0.02)


class TestLocalProcessConstruction:
    """Constructor wiring for :class:`_LocalProcess`."""

    def test_address_composition(self):
        """Composes ``addr`` as ``127.0.0.1:<port>`` from the given port."""
        p = _LocalProcess(port=9000, executor_bin="/bin/exec")
        assert p.addr == "127.0.0.1:9000"
        assert p._bind_port == 9000

    def test_defaults(self):
        """Applies default plugin list, env mapping, and process name when not supplied."""
        p = _LocalProcess(port=9000, executor_bin="/bin/exec")
        assert p._plugins == []
        assert p._env == {}
        assert p.name == "executor"


class TestLocalProcessSpawn:
    """Argv construction and environment handling of :meth:`_LocalProcess._spawn`."""

    def test_argv_shape(self):
        """Builds argv with binary, ``--bind``, ``--plugin`` entries and forwards env additions."""
        p = _LocalProcess(
            port=9000,
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
        assert "--bind=127.0.0.1:9000" in captured["argv"]
        assert "--plugin=/opt/libx.so" in captured["argv"]
        # env extends os.environ, custom key present:
        assert captured["env"]["K"] == "V"

    def test_expands_home_in_binary(self):
        """Expands a leading ``~`` in the executor binary path before spawning."""
        p = _LocalProcess(port=9000, executor_bin="~/bin/exec")
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
        p = _LocalProcess(port=9000, executor_bin="/bin/exec", env={"LIB": "$MY_LIB/x"})
        captured = {}
        with patch.object(
            _LocalProcess,
            "_popen",
            side_effect=lambda argv, **kw: (captured.setdefault("env", kw.get("env")), MagicMock())[
                1
            ],
        ):
            p._spawn()
        assert captured["env"]["LIB"] == "/opt/mylib/x"


class TestRemoteProcessConstruction:
    """Constructor wiring for :class:`_RemoteProcess`."""

    def test_addr_is_the_local_tunnel_endpoint(self):
        """``addr`` is loopback on the same port the executor binds remotely.

        Both ends of the SSH forward use one port number, so there is nothing to reconcile
        between the tunnel endpoint and the remote bind.
        """
        p = _RemoteProcess(host="h", user="me", port=9000, workspace="~/ws")
        assert p.addr == "127.0.0.1:9000"
        assert p._bind_port == 9000

    def test_defaults(self):
        """Applies default sudo, workspace-cleanup and executor-binary values."""
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/ws")
        assert p.sudo is False
        assert p.cleanup_ws is False
        assert p.executor_bin == f"./{ExecutorPaths.EXECUTOR_BIN}"
        assert p.proc is None


class TestRemoteProcessLogHeader:
    """Formatting of the remote-process log-file header string."""

    def test_contains_metadata(self):
        """Header string includes name, host:port, workspace, and comma-joined plugins."""
        p = _RemoteProcess(
            host="h",
            user="me",
            port=9000,
            workspace="~/ws",
            plugins=["libx.so", "liby.so"],
            name="worker",
        )
        header = p._log_header()
        assert "worker" in header
        assert "h:9000" in header
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
        """Raises :class:`RuntimeError` with kind-specific help text when the auth flag is set."""
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/ws")
        p._auth_kind = "ssh"
        p._auth_prompt.set()
        with pytest.raises(RuntimeError, match="ssh-copy-id"):
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


class TestRemoteProcessTeardownExtra:
    """Remote ``pkill`` invocation from :meth:`_teardown_extra`."""

    def test_skips_when_nothing_was_spawned(self):
        """No-op with no process of ours on the far end: there is nothing to pkill."""
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/ws")
        with patch("catalyst.executor.process.RemoteOps.pkill") as pkill:
            p._teardown_extra()
        pkill.assert_not_called()

    def test_skips_when_the_port_was_someone_elses(self):
        """No-op after a port conflict: the executor answering there belongs to another launch."""
        p = _RemoteProcess(host="h", user="me", port=9000, workspace="~/ws")
        p.proc = MagicMock()
        p._port_conflict.set()
        with patch("catalyst.executor.process.RemoteOps.pkill") as pkill:
            p._teardown_extra()
        pkill.assert_not_called()

    def test_runs_pkill_for_a_process_that_never_reported_ready(self):
        """Runs a port-scoped pkill for anything spawned, ready or not: a ^C partway through a
        deploy lands in that window and leaves an executor holding the port."""
        p = _RemoteProcess(host="h", user="me", port=9000, workspace="~/ws")
        p.proc = MagicMock()
        with patch("catalyst.executor.process.RemoteOps.pkill") as pkill:
            p._teardown_extra()
        pkill.assert_called_once()
        # Pattern should be port-scoped so we can't kill someone else's process, and must match
        # the loopback bind address the remote launch command actually uses.
        pat = pkill.call_args.args[2]
        assert f"{ExecutorFlags.BIND_HOST}:9000" in pat
        assert "0.0.0.0" not in pat

    def test_pkill_pattern_matches_the_real_launch_command(self):
        """The teardown pattern is a regex run against the live process list, so it has to match
        the command :meth:`RemoteLauncher._remote_cmd` actually emits.
        """
        p = _RemoteProcess(host="h", user="me", port=9000, workspace="~/ws")
        p.proc = MagicMock()
        with patch("catalyst.executor.process.RemoteOps.pkill") as pkill:
            p._teardown_extra()
        pat = pkill.call_args.args[2]

        launch_cmd = RemoteLauncher._remote_cmd(
            "~/ws",
            9000,
            plugins=[],
            env={},
            sudo=False,
            use_password=False,
            executor_bin=f"./{ExecutorPaths.EXECUTOR_BIN}",
        )
        assert re.search(pat, launch_cmd), f"{pat!r} does not match {launch_cmd!r}"


class TestRemoteProcessTeardownWorkspace:
    """Guards on :meth:`teardown_workspace` that gate remote directory removal."""

    def test_pinned_dir_left_intact(self):
        """Pinned workspace (``cleanup_ws=False``) is never removed."""
        # cleanup_ws=False (pinned) — never remove.
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/mydir", cleanup_ws=False)
        with patch("catalyst.executor.process.RemoteOps.rmdir") as rmdir:
            p.teardown_workspace()
        rmdir.assert_not_called()

    def test_prefix_guard_left_intact(self):
        """Directories without the safe workspace prefix are not removed even when cleanup is enabled."""
        # cleanup_ws=True but the basename lacks the safe prefix.
        p = _RemoteProcess(host="h", user="me", port=1, workspace="~/mydir", cleanup_ws=True)
        with patch("catalyst.executor.process.RemoteOps.rmdir") as rmdir:
            p.teardown_workspace()
        rmdir.assert_not_called()

    def test_auto_dir_with_prefix_removed(self):
        """Auto-generated workspace matching the safe prefix is removed when cleanup is enabled."""
        ws = f"~/{ExecutorPaths.WORKSPACE_PREFIX}me-2026-01-01_00-00-00-abc"
        p = _RemoteProcess(host="h", user="me", port=1, workspace=ws, cleanup_ws=True)
        with patch("catalyst.executor.process.RemoteOps.rmdir") as rmdir:
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


class TestLaunchFailuresAreOrdinaryExceptions:
    """Launch failures must be catchable by ``except Exception``.

    ``SystemExit`` would slip past it and terminate the interpreter instead of reporting a
    failed launch.
    """

    def test_caught_by_except_exception(self):
        """A ready-timeout is caught by a plain ``except Exception``."""
        p = _mk_base_proc(ready_timeout=0.01)
        p.proc = MagicMock()
        p.proc.poll.return_value = None
        try:
            p._wait_for_ready(poll_interval=0.005)
        except Exception as e:  # pylint: disable=broad-except
            assert "did not become ready" in str(e)
        else:
            pytest.fail("expected a launch failure")


class _FakeStdout:
    """Line iterator standing in for ``Popen.stdout``."""

    def __init__(self, lines):
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)


class TestWatchOutput:
    """The reader thread, which turns executor output into readiness and conflict flags."""

    def _pump(self, cls, lines, **kw):
        p = cls(**kw)
        p.proc = MagicMock()
        p.proc.stdout = _FakeStdout(lines)
        p._watch_output()
        return p

    def test_ready_line_releases_the_wait(self):
        """A ready line sets the flag ``_wait_for_ready`` blocks on."""
        p = self._pump(
            _LocalProcess, ["Listening on 127.0.0.1:9000\n"], port=9000, executor_bin="/bin/exec"
        )
        assert p._ready.is_set()
        assert not p._port_conflict.is_set()

    def test_port_conflict_line_is_flagged(self):
        """A bind failure sets the flag that makes the launch retry on another port."""
        p = self._pump(
            _LocalProcess, ["Address already in use\n"], port=9000, executor_bin="/bin/exec"
        )
        assert p._port_conflict.is_set()

    def test_lines_are_teed_to_the_log(self, tmp_path):
        """Executor output lands in the per-launch log, not only on stderr."""
        log = tmp_path / "x.log"
        p = _LocalProcess(port=9000, executor_bin="/bin/exec", log_path=str(log))
        p._open_log()
        p.proc = MagicMock()
        p.proc.stdout = _FakeStdout(["hello from the executor\n"])
        p._watch_output()
        p._log_fh.close()
        assert "hello from the executor" in log.read_text()

    def test_scan_line_hook_runs_for_remote(self):
        """Every line passes through the remote subclass's auth scan."""
        p = self._pump(
            _RemoteProcess,
            ["me@h's password:\n"],
            host="h",
            user="me",
            port=9000,
            workspace="~/ws",
        )
        assert p._auth_prompt.is_set()
        assert p._auth_kind == "ssh"


class TestStart:
    """Spawn, pump, wait, and clean up on failure, in :meth:`_ExecutorProcess.start`."""

    def test_returns_once_ready(self):
        """Returns the process once it announces its bind."""
        p = _LocalProcess(port=9000, executor_bin="/bin/exec", ready_timeout=2.0)

        def fake_spawn():
            p.proc = MagicMock()
            p.proc.poll.return_value = None
            p.proc.stdout = _FakeStdout(["Listening on 127.0.0.1:9000\n"])

        with patch.object(_LocalProcess, "_spawn", side_effect=fake_spawn):
            assert p.start() is p
        assert p._ready.is_set()

    def test_failure_shuts_the_child_down_before_propagating(self):
        """A launch that never becomes ready must not leave the subprocess running."""
        p = _LocalProcess(port=9000, executor_bin="/bin/exec", ready_timeout=0.05)

        def fake_spawn():
            p.proc = MagicMock()
            p.proc.poll.return_value = None
            p.proc.stdout = _FakeStdout([])  # silent: never becomes ready

        with patch.object(_LocalProcess, "_spawn", side_effect=fake_spawn):
            with pytest.raises(RuntimeError, match="did not become ready"):
                p.start()
        p.proc.terminate.assert_called_once(), "the child was left running"


class TestRemoteSpawn:
    """The ssh argv and the sudo-password stdin path built by :meth:`_RemoteProcess._spawn`."""

    def _spawn(self, **kw):
        p = _RemoteProcess(host="h", user="me", port=9000, workspace="~/ws", **kw)
        captured = {}

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            captured["stdin"] = kwargs.get("stdin")
            return MagicMock()

        with patch.object(_RemoteProcess, "_popen", side_effect=fake_popen):
            p._spawn()
        return p, captured

    def test_builds_the_tunnelled_ssh_command(self):
        """argv is an ssh invocation carrying the forward and the remote launch command."""
        _, cap = self._spawn(plugins=["libx.so"])
        assert cap["argv"][0] == "ssh"
        assert "me@h" in cap["argv"]
        assert "9000:localhost:9000" in cap["argv"]
        assert cap["argv"][-1].startswith("cd ")

    def test_no_password_closes_stdin(self):
        """Without a sudo password the child gets no stdin at all."""
        _, cap = self._spawn()
        assert cap["stdin"] == subprocess.DEVNULL

    def test_password_is_piped_not_argv(self):
        """A sudo password goes over stdin, never into argv where ps or the log would show it."""
        p, cap = self._spawn(sudo=True, sudo_password="hunter2")
        assert cap["stdin"] == subprocess.PIPE
        assert not any("hunter2" in a for a in cap["argv"]), "password leaked into argv"
        p.proc.stdin.write.assert_called_once_with("hunter2\n")


class TestRemoteProcessSetenvRefusal:
    """A sudo SETENV refusal aborts the launch with its own remedy, not the password advice."""

    def _proc(self):
        return _RemoteProcess(host="h", user="me", port=9000, workspace="~/ws", sudo=True)

    def test_flagged_as_its_own_kind(self):
        """Kept apart from ``sudo``: that help suggests sudo_password=, which cannot fix a policy."""
        p = self._proc()
        p._scan_line("sudo: sorry, you are not allowed to preserve the environment")
        assert p._auth_prompt.is_set()
        assert p._auth_kind == "setenv"

    def test_aborts_with_both_remedies(self):
        """Bails immediately rather than waiting out ready_timeout, naming both ways out."""
        p = self._proc()
        p._scan_line("sudo: sorry, you are not allowed to preserve the environment")
        with pytest.raises(RuntimeError) as exc:
            p._check_failure()
        assert "NOPASSWD:SETENV:" in str(exc.value)
        assert "sudo=False" in str(exc.value)


class TestProcessLogAndTeardown:
    """Log-file handling and the teardown hook, which only run during a real launch."""

    def test_remote_log_gets_a_header(self, tmp_path):
        """A fresh remote log opens with the host/port/workspace banner."""
        log = tmp_path / "x.log"
        p = _RemoteProcess(
            host="h",
            user="me",
            port=9000,
            workspace="~/ws",
            plugins=["libx.so"],
            log_path=str(log),
            name="worker",
        )
        p._open_log()
        p._log_fh.close()
        text = log.read_text()
        assert "worker" in text and "h:9000" in text and "libx.so" in text

    def test_shutdown_closes_the_log(self, tmp_path):
        """The log handle is released even when there is no child to signal."""
        p = _mk_base_proc(log_path=str(tmp_path / "x.log"))
        p._open_log()
        assert p._log_fh is not None
        p._shutdown()
        assert p._log_fh is None, "log handle leaked"

    def test_stop_runs_the_subclass_teardown(self):
        """Base ``stop`` is shutdown plus the subclass hook."""
        p = _mk_base_proc()
        with patch.object(_ExecutorProcess, "_teardown_extra") as extra:
            p.stop()
        extra.assert_called_once()

    def test_remote_stop_still_terminates_the_child(self):
        """The remote override must not forget ``super().stop()``."""
        p = _RemoteProcess(host="h", user="me", port=9000, workspace="~/ws")
        p.proc = MagicMock()
        p.proc.poll.return_value = None
        p.proc.wait.return_value = 0
        p.stop()
        p.proc.terminate.assert_called_once()

    def test_popen_merges_stderr_into_stdout(self):
        """The pump reads a single stream, so the child's stderr has to arrive on stdout."""
        # Popen as a context manager closes the pipes and reaps the child on exit, so nothing
        # is left behind even if the assertion fails.
        with _ExecutorProcess._popen(["sh", "-c", "echo out; echo err >&2"]) as proc:
            lines = [line.strip() for line in proc.stdout]
        assert lines == ["out", "err"]
