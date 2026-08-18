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

"""Unit tests for :mod:`catalyst.executor.ssh` — the ``ssh``/``scp`` command builders and the
remote ``catalyst-executor`` argv assembler. Subprocess is mocked throughout; these tests do
not open real SSH connections."""

import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from catalyst.executor.ssh import SCP, RemoteLauncher, RemoteOps, SSHArgv
from catalyst.executor.utils import ExecutorFlags, ExecutorPaths, set_verbose

# ---------------------------------------------------------------------------
# SSHArgv — control socket, argv construction
# ---------------------------------------------------------------------------


class TestSSHConstants:
    """Sanity checks on :class:`SSHArgv` class-level constants."""

    def test_base_cmd_tuple(self):
        """``BASE_CMD`` starts with ``ssh`` and includes the keepalive option."""
        assert SSHArgv.BASE_CMD[0] == "ssh"
        assert "ServerAliveInterval=15" in SSHArgv.BASE_CMD

    def test_probe_opts(self):
        """``PROBE_OPTS`` enables ``BatchMode`` to avoid interactive prompts."""
        assert "BatchMode=yes" in SSHArgv.PROBE_OPTS

    def test_control_persist_positive(self):
        """``CONTROL_PERSIST`` is a positive duration so the multiplex socket lingers."""
        assert SSHArgv.CONTROL_PERSIST > 0


class TestSSHCtlOpts:
    """Verifies the shape of :meth:`SSHArgv.ctl_opts` control-master argv fragment."""

    def test_shape(self):
        """``ctl_opts`` emits ``-o`` pairs for ``ControlMaster``, ``ControlPath``, ``ControlPersist``."""
        opts = SSHArgv._ctl_flags(Path("/tmp/cm"))
        assert opts[0] == "-o"
        assert opts[1] == "ControlMaster=auto"
        assert opts[2] == "-o"
        assert opts[3] == "ControlPath=/tmp/cm/%C"
        assert opts[4] == "-o"
        assert opts[5] == f"ControlPersist={SSHArgv.CONTROL_PERSIST}"

    def test_disabled_when_path_too_long(self):
        """No flags at all when the control path would overflow ``sun_path``."""
        assert SSHArgv._ctl_flags(Path("/tmp") / ("x" * 200)) == []


class TestSSHCtlDir:
    """The per-user control-socket directory :meth:`SSHArgv._private_dir`."""

    def test_is_created_and_private(self, tmp_path):
        """Created if absent, and closed to group and other.

        Reaching a control socket is enough to open sessions on its remote host, the master having
        authenticated already, so the directory is the whole of what keeps other users out.
        """
        d = SSHArgv._private_dir(tmp_path)
        assert d == tmp_path / SSHArgv.CONTROL_DIR
        assert d.stat().st_mode & 0o077 == 0, oct(d.stat().st_mode)

    def test_none_when_it_cannot_be_made(self, tmp_path):
        """A home that cannot hold the directory yields ``None``, not an exception.

        Blocked with a file standing where the directory would go, rather than by permissions,
        which root would be free to ignore.
        """
        blocked = tmp_path / "home"
        blocked.write_text("")
        assert SSHArgv._private_dir(blocked) is None


class TestSSHBaseCmd:
    """Argv assembly for :meth:`SSHArgv.base`."""

    def test_starts_with_ssh_binary(self):
        """First token is the ``ssh`` executable name."""
        cmd = SSHArgv.base("me", "h")
        assert cmd[0] == "ssh"

    def test_target_appended_last(self):
        """``user@host`` is the final positional argument."""
        cmd = SSHArgv.base("me", "h")
        assert cmd[-1] == "me@h"

    def test_multiplex_true_inserts_the_control_opts(self):
        """``multiplex=True`` splices :meth:`ctl_opts` in, whatever this machine's dirs allow."""
        with_mux = SSHArgv.base("me", "h", multiplex=True)
        without = SSHArgv.base("me", "h", multiplex=False)
        n = len(SSHArgv.BASE_CMD)
        assert with_mux == without[:n] + SSHArgv.ctl_opts() + without[n:]

    def test_multiplex_false_omits_control_opts(self):
        """``multiplex=False`` omits the ``ControlMaster`` option."""
        cmd = SSHArgv.base("me", "h", multiplex=False)
        assert "ControlMaster=auto" not in cmd

    def test_extra_opts_between_base_and_target(self):
        """Caller-supplied ``opts`` land between the base options and the target."""
        cmd = SSHArgv.base("me", "h", opts=["-L", "1:localhost:2"])
        idx_L = cmd.index("-L")
        idx_target = cmd.index("me@h")
        assert idx_L < idx_target


# ---------------------------------------------------------------------------
# RemoteOps — command execution
# ---------------------------------------------------------------------------


class TestSSHCapture:
    """Behavior of :meth:`RemoteOps.capture` around subprocess success and failure."""

    def test_success_returns_stripped_stdout(self):
        """Returns remote stdout with trailing whitespace stripped."""
        with patch("subprocess.check_output", return_value="Linux x86_64\n"):
            assert RemoteOps.capture("me", "h", "uname -sm") == "Linux x86_64"

    def test_failure_returns_none(self):
        """Returns ``None`` when the underlying subprocess raises."""
        with patch("subprocess.check_output", side_effect=subprocess.TimeoutExpired("ssh", 15)):
            assert RemoteOps.capture("me", "h", "uname -sm") is None


class TestSSHRun:
    """Return-code and exception plumbing for :meth:`RemoteOps.run`."""

    def test_quiet_returns_rc(self):
        """Default ``quiet=True`` returns the return code and pipes stdout to ``DEVNULL``."""
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as m:
            rc = RemoteOps.run("me", "h", "true")
        assert rc == 0
        # quiet=True should redirect stdout/stderr to DEVNULL
        assert m.call_args.kwargs.get("stdout") == subprocess.DEVNULL

    def test_quiet_swallows_exceptions_as_neg_one(self):
        """Quiet mode maps subprocess exceptions to ``-1``."""
        with patch("subprocess.run", side_effect=OSError("boom")):
            assert RemoteOps.run("me", "h", "true") == -1

    def test_nonquiet_propagates_exceptions(self):
        """``quiet=False`` re-raises subprocess exceptions to the caller."""
        with patch("subprocess.run", side_effect=RuntimeError("no")):
            with pytest.raises(RuntimeError):
                RemoteOps.run("me", "h", "true", quiet=False)

    def test_error_kw_raises_on_nonzero(self):
        """``error=`` raises :class:`RuntimeError` including the message and return code."""
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=2)):
            with pytest.raises(RuntimeError, match="oh no.*rc=2"):
                RemoteOps.run("me", "h", "true", error="oh no")

    def test_error_kw_ok_on_zero(self):
        """``error=`` is silent on ``rc=0`` and returns the code."""
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)):
            # No raise expected.
            assert RemoteOps.run("me", "h", "true", error="oh no") == 0


class TestSSHMkdir:
    """Success and failure paths for :meth:`RemoteOps.mkdir`."""

    def test_success_no_raise(self):
        """Returns silently when the remote ``mkdir`` succeeds."""
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)):
            RemoteOps.mkdir("me", "h", "~/ws")  # should not raise

    def test_failure_raises(self):
        """Raises :class:`RuntimeError` when the remote ``mkdir`` returns nonzero."""
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=1)):
            with pytest.raises(RuntimeError, match="failed to create remote directory"):
                RemoteOps.mkdir("me", "h", "~/ws")


class TestSSHRmdir:
    """Safety and force semantics of :meth:`RemoteOps.rmdir`."""

    def test_safety_refusal_raises_valueerror(self):
        """``rc=3`` from the remote guard raises :class:`ValueError` to refuse the removal."""
        # rc=3 signals resolved to / or $HOME.
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=3)):
            with pytest.raises(ValueError, match="refusing"):
                RemoteOps.rmdir("me", "h", "~")

    def test_force_reraises_on_nonzero(self):
        """``force=True`` promotes any nonzero return code to :class:`RuntimeError`."""
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=1)):
            with pytest.raises(RuntimeError, match="failed to remove"):
                RemoteOps.rmdir("me", "h", "~/ws", force=True)

    def test_default_swallows_nonzero(self):
        """Default (non-force) mode silently ignores a nonzero return code."""
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=1)):
            # No raise.
            RemoteOps.rmdir("me", "h", "~/ws")

    def test_missing_dir_ok(self):
        """``rc=0`` is the happy path when the directory does not exist."""
        # rc=0 is the happy path (cd fails inside the remote, script `exit 0`s).
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)):
            RemoteOps.rmdir("me", "h", "~/ws")


class TestSSHPkill:
    """Sudo flag composition in :meth:`RemoteOps.pkill`."""

    def test_no_sudo(self):
        """No ``sudo`` prefix appears in the remote command when ``sudo=False``."""
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as m:
            RemoteOps.pkill("me", "h", "catalyst-exec")
        # Verify no sudo wrapper in the remote cmd.
        assert "sudo" not in m.call_args.args[0][-1]

    def test_sudo_with_password_uses_stdin(self):
        """``sudo=True`` with a password uses ``sudo -S`` and pipes the password on stdin."""
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as m:
            RemoteOps.pkill("me", "h", "x", sudo=True, sudo_password="pw")
        assert m.call_args.kwargs.get("input") == "pw\n"
        assert "sudo -S" in m.call_args.args[0][-1]

    def test_sudo_no_password_uses_np(self):
        """``sudo=True`` without a password uses non-interactive ``sudo -n``."""
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as m:
            RemoteOps.pkill("me", "h", "x", sudo=True)
        assert "sudo -n" in m.call_args.args[0][-1]


# ---------------------------------------------------------------------------
# RemoteOps — auth probing
# ---------------------------------------------------------------------------


class TestSSHNeedsSudoPassword:
    """Probing sudo policy via :meth:`RemoteOps.needs_sudo_password`."""

    def test_rc_zero_means_no_password_needed(self):
        """``rc=0`` from the sudo probe returns ``False`` (NOPASSWD path)."""
        # NOPASSWD path: rc=0 → returns False (no password needed).
        with patch("catalyst.executor.ssh.RemoteOps.run", return_value=0):
            assert RemoteOps.needs_sudo_password("me", "h") is False

    def test_rc_nonzero_means_password_needed(self):
        """Nonzero (non-255) return codes indicate a password is required."""
        # sudo -n rejected → returns True (password required).
        with patch("catalyst.executor.ssh.RemoteOps.run", return_value=1):
            assert RemoteOps.needs_sudo_password("me", "h") is True

    def test_rc_255_raises_ssh_error(self):
        """``rc=255`` (ssh transport failure) raises :class:`RuntimeError` pointing at ``ssh-copy-id``."""
        with patch("catalyst.executor.ssh.RemoteOps.run", return_value=255):
            with pytest.raises(RuntimeError, match="ssh-copy-id"):
                RemoteOps.needs_sudo_password("me", "h")


class TestSSHResolveSudo:
    """Password resolution logic in :meth:`RemoteOps.resolve_sudo`."""

    def test_nopasswd_returns_none(self):
        """Returns ``None`` when the remote does not require a sudo password."""
        with patch("catalyst.executor.ssh.RemoteOps.needs_sudo_password", return_value=False):
            assert RemoteOps.resolve_sudo("me", "h") is None

    def test_explicit_password_used_when_needed(self):
        """An attached password is returned verbatim when a password is required."""
        with patch("catalyst.executor.ssh.RemoteOps.needs_sudo_password", return_value=True):
            assert RemoteOps.resolve_sudo("me", "h", "secret") == "secret"

    def test_interactive_prompt_when_no_password(self):
        """Prompts via :func:`getpass` when a password is required but none is attached."""
        with patch("catalyst.executor.ssh.RemoteOps.needs_sudo_password", return_value=True), patch(
            "catalyst.executor.ssh.getpass.getpass", return_value="typed"
        ):
            assert RemoteOps.resolve_sudo("me", "h") == "typed"

    def test_aborted_prompt_raises(self):
        """A ``KeyboardInterrupt`` at the prompt is converted to :class:`RuntimeError`."""
        with patch("catalyst.executor.ssh.RemoteOps.needs_sudo_password", return_value=True), patch(
            "catalyst.executor.ssh.getpass.getpass", side_effect=KeyboardInterrupt
        ):
            with pytest.raises(RuntimeError, match="no sudo password"):
                RemoteOps.resolve_sudo("me", "h")


# ---------------------------------------------------------------------------
# SCP
# ---------------------------------------------------------------------------


class TestSCPRun:
    """Retry and error semantics of :meth:`SCP.copy`."""

    def _mock_run(self, rcs):
        """Yield SimpleNamespace(returncode=rc) once per rc, in order."""
        iterator = iter(rcs)
        return lambda *a, **kw: SimpleNamespace(returncode=next(iterator))

    def test_success_on_first_try_no_retry(self):
        """A first-attempt success completes without a retry."""
        with patch("subprocess.run", side_effect=self._mock_run([0])) as m:
            SCP.copy("me", "h", [Path("/a")], "ws")
        assert m.call_count == 1

    def test_retries_with_legacy_o_flag(self):
        """First attempt without ``-O`` failing triggers a retry that includes ``-O``."""
        # First (modern) fails, second (legacy) succeeds.
        calls = []

        def fake_run(*args, **kwargs):
            calls.append(args[0])
            return SimpleNamespace(returncode=(1 if len(calls) == 1 else 0))

        with patch("subprocess.run", side_effect=fake_run):
            SCP.copy("me", "h", [Path("/a")], "ws")
        assert len(calls) == 2
        assert "-O" not in calls[0]
        assert "-O" in calls[1]

    def test_both_attempts_fail_raises(self):
        """Both attempts failing raises :class:`RuntimeError`."""
        with patch("subprocess.run", side_effect=self._mock_run([1, 1])):
            with pytest.raises(RuntimeError, match="scp to"):
                SCP.copy("me", "h", [Path("/a")], "ws")


class TestSCPRunVerbosity:
    """Verbosity flag routing in :meth:`SCP.copy`."""

    def test_verbose_flag_at_level_2(self):
        """Verbosity level 2 threads ``-v`` into the scp argv."""
        set_verbose(2)
        try:
            captured = []
            with patch(
                "subprocess.run",
                side_effect=lambda cmd, *a, **kw: (
                    captured.append(cmd),
                    SimpleNamespace(returncode=0),
                )[1],
            ):
                SCP.copy("me", "h", [Path("/a")], "ws")
            # -v should appear (not -q).
            assert "-v" in captured[0]
        finally:
            set_verbose(1)

    def test_quiet_flag_at_level_1(self):
        """Verbosity level 1 threads ``-q`` into the scp argv."""
        set_verbose(1)
        captured = []
        with patch(
            "subprocess.run",
            side_effect=lambda cmd, *a, **kw: (
                captured.append(cmd),
                SimpleNamespace(returncode=0),
            )[1],
        ):
            SCP.copy("me", "h", [Path("/a")], "ws")
        assert "-q" in captured[0]


class TestSCPDeploy:
    """Bundle discovery and orchestration in :meth:`SCP.deploy`."""

    def test_empty_bundle_raises(self, tmp_path):
        """An empty bundle directory raises :class:`RuntimeError`."""
        # Empty directory, no artifacts.
        with pytest.raises(RuntimeError, match="no artifacts"):
            SCP.deploy("me", "h", [tmp_path], "ws")

    def test_readme_only_is_still_empty(self, tmp_path):
        """A bundle containing only ``README.md`` is treated as empty and raises."""
        # README.md is explicitly filtered out.
        (tmp_path / "README.md").write_text("ignore me")
        with pytest.raises(RuntimeError, match="no artifacts"):
            SCP.deploy("me", "h", [tmp_path], "ws")

    def test_deploys_via_ssh_mkdir_then_scp_run(self, tmp_path):
        """Creates the remote workspace then invokes :meth:`SCP.copy` with the filtered file list."""
        (tmp_path / "catalyst-executor").write_text("bin")
        (tmp_path / "libfoo.so").write_text("lib")
        with patch("catalyst.executor.ssh.RemoteOps.mkdir") as mkdir, patch(
            "catalyst.executor.ssh.SCP.copy"
        ) as scprun:
            SCP.deploy("me", "h", [tmp_path], "ws")
        mkdir.assert_called_once_with("me", "h", "ws")
        scprun.assert_called_once()
        # files argument (positional index 2) sorted, README excluded
        files = scprun.call_args.args[2]
        names = sorted(f.name for f in files)
        assert names == ["catalyst-executor", "libfoo.so"]


# ---------------------------------------------------------------------------
# RemoteLauncher — remote-executor argv builders
# ---------------------------------------------------------------------------


class TestRemoteLauncherHelpers:
    """Argv-builder primitives on :class:`RemoteLauncher`."""

    def test_env_prefix_quotes_bare_strings(self):
        """Bare string env values are single-quoted for shell safety."""
        assert RemoteLauncher._env_prefix({"FOO": "bar baz"}) == "FOO='bar baz'"

    def test_plugin_args_bare_name_pins_to_pwd(self):
        """A bare plugin filename is pinned to ``$PWD`` on the remote."""
        out = RemoteLauncher._plugin_args(["libx.so"])
        assert out == "--plugin=$PWD/libx.so"

    def test_plugin_args_absolute_path(self):
        """An absolute plugin path is preserved verbatim."""
        out = RemoteLauncher._plugin_args(["/opt/lib/x.so"])
        assert "--plugin=/opt/lib/x.so" in out

    def test_chmod_prefix_only_for_local_binary(self):
        """``chmod +x`` prefix is emitted only for the workspace-local executor binary."""
        assert RemoteLauncher._chmod_prefix(f"./{ExecutorPaths.EXECUTOR_BIN}").startswith(
            "chmod +x"
        )
        assert RemoteLauncher._chmod_prefix("/opt/bin/catalyst-executor") == ""

    def test_exec_prefix_no_sudo(self):
        """Without sudo the exec prefix is a bare ``exec``."""
        assert RemoteLauncher._exec_prefix(sudo=False, use_password=False) == "exec"

    def test_exec_prefix_sudo_no_password(self):
        """``sudo=True, use_password=False`` emits ``exec sudo -E`` to preserve the environment."""
        assert RemoteLauncher._exec_prefix(sudo=True, use_password=False) == "exec sudo -E"

    def test_exec_prefix_sudo_with_password_stdin(self):
        """``sudo=True, use_password=True`` includes both ``-S`` (stdin) and ``-E`` (env)."""
        out = RemoteLauncher._exec_prefix(sudo=True, use_password=True)
        assert "-S" in out and "-E" in out


class TestRemoteLauncherRemoteCmd:
    """End-to-end composition performed by :meth:`RemoteLauncher._remote_cmd`."""

    def test_composes_cd_env_exec(self):
        """Emits ``cd``, env prefix, plugin args, bind port, and ``chmod`` for a local binary."""
        cmd = RemoteLauncher._remote_cmd(
            "~/ws",
            9000,
            plugins=["libx.so"],
            env={"FOO": "bar"},
            sudo=False,
            use_password=False,
            executor_bin=f"./{ExecutorPaths.EXECUTOR_BIN}",
        )
        assert "cd " in cmd
        assert "FOO=bar" in cmd
        assert "--bind=127.0.0.1:9000" in cmd
        assert "--plugin=$PWD/libx.so" in cmd
        # chmod prefix should be present because we're using the workspace-local binary.
        assert "chmod +x" in cmd


class TestRemoteLauncherSecurityDefaults:
    """The remote executor must not be network-exposed, nor root, unless asked for."""

    def _cmd(self, **overrides):
        kwargs = dict(
            workspace="~/ws",
            remote_port=9000,
            plugins=[],
            env={},
            sudo=False,
            use_password=False,
            executor_bin=f"./{ExecutorPaths.EXECUTOR_BIN}",
        )
        kwargs.update(overrides)
        return RemoteLauncher._remote_cmd(**kwargs)

    def test_binds_loopback_never_wildcard(self):
        """The executor runs arbitrary compiled objects, so it must bind loopback only."""
        cmd = self._cmd()
        assert "0.0.0.0" not in cmd
        assert f"--bind={ExecutorFlags.BIND_HOST}:9000" in cmd

    def test_no_sudo_by_default(self):
        """``ssh_argv`` runs the executor as the connecting user unless ``sudo=True``."""
        argv = RemoteLauncher.ssh_argv("me", "h", "~/ws", 9000, [], {})
        assert "sudo" not in argv[-1]

    def test_sudo_is_opt_in(self):
        """``sudo=True`` is still honored for backends needing privileged memory registration."""
        assert "exec sudo -E" in self._cmd(sudo=True)


class TestRemoteLauncherSshOpts:
    """SSH option assembly for the launcher's transport-level flags."""

    def test_local_forward_uses_one_port_at_both_ends(self):
        """Emits ``-L <port>:localhost:<port>`` — the tunnel and the remote bind share a port."""
        opts = RemoteLauncher._ssh_opts(port=9000, use_password=False)
        assert "-L" in opts
        assert "9000:localhost:9000" in opts

    def test_exit_on_forward_failure(self):
        """Sets ``ExitOnForwardFailure=yes`` so a forward failure fails the ssh session."""
        opts = RemoteLauncher._ssh_opts(5, use_password=False)
        assert "ExitOnForwardFailure=yes" in opts

    def test_pseudo_terminal_without_password(self):
        """NOPASSWD path uses ``-tt`` so ssh close signals SIGHUP to the executor."""
        # NOPASSWD path uses -tt so SSH close SIGHUPs the executor.
        opts = RemoteLauncher._ssh_opts(5, use_password=False)
        assert "-tt" in opts

    def test_no_pseudo_terminal_with_password(self):
        """Password-sudo path omits ``-tt`` so ``sudo -S`` reads the password from stdin."""
        # sudo -S needs an unechoed pipe on stdin, so we omit -tt.
        opts = RemoteLauncher._ssh_opts(5, use_password=True)
        assert "-tt" not in opts


class TestRemoteLauncherSshArgv:
    """Top-level argv shape produced by :meth:`RemoteLauncher.ssh_argv`."""

    def test_full_argv_shape(self):
        """Argv starts with ``ssh``, contains ``user@host`` and the port forward, ends with the remote command."""
        argv = RemoteLauncher.ssh_argv(
            "me",
            "h",
            "~/ws",
            9000,
            plugins=["libx.so"],
            env={"FOO": "bar"},
            sudo=False,
        )
        assert argv[0] == "ssh"
        assert argv[-1].startswith("cd ")  # last arg is the remote command
        assert "me@h" in argv
        assert "9000:localhost:9000" in argv
