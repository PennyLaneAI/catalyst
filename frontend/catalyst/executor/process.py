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

"""The launched ``catalyst-executor`` process objects: a base owning the shared subprocess lifecycle
(spawn, stream output, wait for bind, tear down) and the local / remote subclasses. The public
:class:`~catalyst.executor.Executor` drives these; :mod:`.ssh` provides the remote command plumbing."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
from typing import Self

from .utils import (
    Log,
    Paths,
    Patterns,
    pdeathsig,
    PortInUse,
)
from .ssh import ExecutorCli, RemoteLauncher, SSH


class _ExecutorProcess:
    """A launched ``catalyst-executor`` process — owns the subprocess, streams output, waits for
    bind, tears it down. Subclasses override :meth:`_spawn` and the ``_*`` hooks (auth, cleanup).
    ``.addr`` is the client-facing endpoint; ``.name`` tags streamed output as ``[<name>]``."""

    LOCALHOST = "127.0.0.1"  # loopback: local bind, and the client end of the remote SSH tunnel

    def __init__(
        self, *, name: str, addr: str, bind_port: int, ready_timeout: float, log_path: str | None
    ):
        self.name = name
        self.addr = addr
        self._bind_port = bind_port
        self.ready_timeout = ready_timeout
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self._log_fh = None
        self._ready = threading.Event()
        self._port_conflict = threading.Event()

    # --- subclass hooks ---------------------------------------------------------------------------
    def _spawn(self) -> None:
        """Build the command and set ``self.proc`` (a started ``subprocess.Popen``)."""
        raise NotImplementedError

    def _log_header(self) -> str:
        """Optional header string written once at the top of a fresh log file (remote: run banner)."""
        return ""

    def _scan_line(self, line: str) -> None:
        """Inspect an output line for extra conditions (remote: auth prompts)."""

    def _check_failure(self) -> None:
        """Raise if a non-port failure was detected while waiting for readiness (remote: auth)."""

    def _on_ready(self) -> None:
        """Called once the executor is bound."""

    def _teardown_extra(self) -> None:
        """Extra teardown after the local process is stopped (remote: backstop pkill)."""

    def teardown_workspace(self) -> None:
        """Best-effort removal of an auto-generated remote workspace (remote only)."""

    # --- shared lifecycle -------------------------------------------------------------------------
    @staticmethod
    def _popen(
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        stdin: int | None = None,
    ) -> subprocess.Popen:
        """Spawn a subprocess with the shared pump-friendly settings: stdout piped, stderr
        merged into stdout, line-buffered text mode, ``PR_SET_PDEATHSIG`` so the child dies with
        the parent. Subclass ``_spawn`` methods should use this instead of calling
        :class:`subprocess.Popen` directly."""
        return subprocess.Popen(
            argv,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            preexec_fn=pdeathsig,
        )

    def _say(self, msg: str, level: int = 1) -> None:
        """Log a launcher narrative line — prefixed with ``<name>:`` when non-default, teed to
        the per-launch log. Executor stdout streams separately via :meth:`_pump_output`."""
        line = msg if self.name == "executor" else f"{self.name}: {msg}"
        Log.info(line, level)
        self._log_tee(f"# [launcher] {line}")

    def _log_tee(self, text: str) -> None:
        """Append ``text`` to the per-launch log file if one is open. Best-effort — write errors
        are swallowed so a log-tee failure never aborts the launch."""
        if self._log_fh is not None:
            with contextlib.suppress(Exception):
                self._log_fh.write(text + "\n")
                self._log_fh.flush()

    def _open_log(self) -> None:
        """Open the per-launch log file (if any) in append mode and write the subclass header.
        Failing to open is non-fatal — the launch continues without teeing."""
        if not self.log_path:
            return
        try:
            self._log_fh = open(self.log_path, "a")
        except OSError as e:
            self._say(f"could not open log {self.log_path}: {e} (continuing without it)")
            return
        header = self._log_header()
        if header:
            self._log_fh.write(header)
            self._log_fh.flush()
        self._say(f"teeing output -> {self.log_path}")

    def _pump_output(self) -> None:
        """Echo executor stdout live (tagged ``[<name>]``), tee to the log, and flag readiness /
        port-conflict on matching lines. Runs in a daemon thread."""
        assert self.proc and self.proc.stdout
        for raw in self.proc.stdout:
            line = raw.rstrip("\n")
            print(f"[{self.name}] {line}", file=sys.stderr, flush=True)
            self._log_tee(line)
            if Patterns.is_port_conflict(line):
                self._port_conflict.set()
            if Patterns.is_ready(line):
                self._ready.set()
            self._scan_line(line)

    def start(self) -> Self:
        """Spawn the executor and block until it binds. Raises :class:`PortInUse` on a port
        collision (retryable) or :class:`SystemExit` on other failures; :meth:`_shutdown` runs
        before propagating."""
        self._open_log()
        self._spawn()
        assert self.proc is not None, "_spawn() must set self.proc"
        threading.Thread(target=self._pump_output, daemon=True).start()
        try:
            return self._wait_for_ready()
        except BaseException:
            self._shutdown()
            raise

    def _wait_for_ready(self, poll_interval: float = 0.25) -> Self:
        """Block up to ``ready_timeout`` for a readiness or failure signal, polling every
        ``poll_interval`` seconds. Raises :class:`PortInUse` on port collision or
        :class:`SystemExit` on early exit / deadline. Cleanup is the caller's job."""
        assert self.proc is not None
        t0 = time.monotonic()
        deadline = t0 + self.ready_timeout
        while time.monotonic() < deadline:
            if self._ready.wait(timeout=poll_interval):
                self._on_ready()
                self._say(f"ready in {time.monotonic() - t0:.1f}s — address {self.addr}")
                return self
            self._check_port_conflict()
            self._check_failure()
            self._check_early_exit()
        raise SystemExit(
            f"executor did not become ready within {self.ready_timeout:.0f}s — see the "
            f"[{self.name}] log above (raise ready_timeout= if the host is slow)."
        )

    def _check_port_conflict(self) -> None:
        """Raise :class:`PortInUse` if the pump thread has flagged a port bind failure."""
        if self._port_conflict.is_set():
            raise PortInUse(self._bind_port)

    def _check_early_exit(self) -> None:
        """Raise if the executor died before becoming ready — :class:`PortInUse` if the pump
        thread flagged the death as a port collision (checked one more time in case the flag
        landed late), else :class:`SystemExit`. No-op if the child is still running."""
        assert self.proc is not None
        if self.proc.poll() is None:
            return
        self._check_port_conflict()
        raise SystemExit(
            f"executor exited (code {self.proc.returncode}) before becoming ready — "
            f"see the [{self.name}] log above."
        )

    def _shutdown(self, wait_time: float = 10) -> None:
        """Close the log file and terminate the subprocess; escalate to SIGKILL if it doesn't
        exit within ``wait_time`` seconds. Idempotent."""
        fh, self._log_fh = self._log_fh, None  # stop the pump teeing, then close
        if fh is not None:
            with contextlib.suppress(Exception):
                fh.close()
        if not self.proc or self.proc.poll() is not None:
            return  # no live child to signal
        with contextlib.suppress(ProcessLookupError):
            self.proc.terminate()
        try:
            self.proc.wait(timeout=wait_time)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                self.proc.kill()

    def stop(self) -> None:
        """Terminate the executor and run subclass-specific teardown (remote: backstop
        ``pkill``). Idempotent."""
        self._shutdown()
        self._teardown_extra()


class _LocalProcess(_ExecutorProcess):
    """A ``catalyst-executor`` running as a local subprocess on ``127.0.0.1`` (no SSH, no tunnel)."""

    def __init__(
        self,
        *,
        port: int,
        executor_bin: str,
        plugins: list[str] | None = None,
        env: dict[str, str] | None = None,
        ready_timeout: float = 60.0,
        name: str = "executor",
        log_path: str | None = None,
    ):
        super().__init__(
            name=name,
            addr=f"{self.LOCALHOST}:{port}",
            bind_port=port,
            ready_timeout=ready_timeout,
            log_path=log_path,
        )
        self._executor_bin = executor_bin
        self._plugins = plugins or []
        self._env = dict(env or {})

    def _spawn(self) -> None:
        """Start ``catalyst-executor`` as a local subprocess bound to ``127.0.0.1:<port>``, with
        each ``plugins`` entry passed as ``--plugin=<path>``. ``env`` extends the parent
        environment. ``PR_SET_PDEATHSIG`` ensures the child dies with the parent."""
        exe = os.path.expanduser(os.path.expandvars(self._executor_bin))
        argv = [exe, f"{ExecutorCli.BIND_FLAG}{self.LOCALHOST}:{self._bind_port}"]
        argv += [
            f"{ExecutorCli.PLUGIN_FLAG}{os.path.expanduser(os.path.expandvars(p))}"
            for p in self._plugins
        ]
        Log.cmd(argv)
        proc_env = dict(os.environ)
        for key, value in self._env.items():
            proc_env[key] = os.path.expandvars(value)
        self._say(f"starting local executor on {self.addr}")
        self.proc = self._popen(argv, env=proc_env)


class _RemoteProcess(_ExecutorProcess):
    """A ``catalyst-executor`` started on a remote host over a port-forwarded SSH. ``.addr`` is the
    local tunnel endpoint ``127.0.0.1:<local_port>``; closing the SSH connection stops the executor,
    with a port-scoped ``pkill`` backstop on teardown."""

    def __init__(
        self,
        *,
        host: str,
        user: str,
        port: int,
        local_port: int | None = None,
        workspace: str,
        plugins: list[str] | None = None,
        env: dict[str, str] | None = None,
        sudo: bool = True,
        sudo_password: str | None = None,
        executor_bin: str = f"./{Paths.EXECUTOR_BIN}",
        cleanup_ws: bool = False,
        ready_timeout: float = 60.0,
        name: str = "executor",
        log_path: str | None = None,
    ):
        local_port = local_port or port
        super().__init__(
            name=name,
            addr=f"{self.LOCALHOST}:{local_port}",
            bind_port=port,
            ready_timeout=ready_timeout,
            log_path=log_path,
        )
        self.host = host
        self.user = user
        self.local_port = local_port
        self.workspace = workspace
        self.cleanup_ws = cleanup_ws
        self._plugins = plugins or []
        self._env = dict(env or {})
        self.sudo = sudo
        self.sudo_password = sudo_password
        self.executor_bin = executor_bin
        self._auth_prompt = threading.Event()
        self._auth_kind = ""  # "ssh" or "sudo" — picks the help text
        self._ready_reached = False  # gates the teardown pkill (don't kill others' ports)

    def _log_header(self) -> str:
        """Header block written once at the top of a fresh log file, recording host, port,
        workspace, timestamp, and plugins so appended runs are separable."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        return (
            f"\n# ==== {self.name} @ {self.host}:{self._bind_port} | ws={self.workspace} | "
            f"{ts} ====\n# plugins: {', '.join(self._plugins)}\n"
        )

    def _scan_line(self, line: str) -> None:
        """Watch executor output for SSH password / sudo-failure prompts and flag them so
        :meth:`_check_failure` can bail with a helpful message rather than hanging on stdin."""
        if Patterns.is_ssh_prompt(line):
            self._auth_kind = "ssh"
            self._auth_prompt.set()
        elif Patterns.is_sudo_fail(line):
            self._auth_kind = "sudo"
            self._auth_prompt.set()

    def _check_failure(self) -> None:
        """Abort the launch with a helpful message if :meth:`_scan_line` saw an auth prompt we
        can't satisfy (missing SSH key, or wrong/absent sudo password)."""
        if self._auth_prompt.is_set():
            self._shutdown()
            raise SystemExit(self._auth_help())

    def _on_ready(self) -> None:
        """Record that the remote executor actually bound its port. Gates the teardown
        ``pkill`` so a port collision (where the process there is someone else's) never wrongly
        kills it."""
        self._ready_reached = True

    def _spawn(self) -> None:
        """Open the SSH port-forward and start ``catalyst-executor`` on the remote host.

        Builds the remote shell command via :meth:`RemoteLauncher.build` (cd + env + exec, wrapped
        in ``sudo`` when requested) and the local ``ssh -L <local>:localhost:<remote>`` tunnel.
        With ``sudo_password``, pipes the password into ``sudo -S`` on stdin (no PTY);
        NOPASSWD mode uses ``-tt`` so closing SSH SIGHUPs the executor. Sets ``self.proc`` to
        the started ``subprocess.Popen``."""
        use_pw = self.sudo_password is not None
        remote_cmd = RemoteLauncher.build(
            self.workspace,
            self._bind_port,
            self._plugins,
            self._env,
            use_password=use_pw,
            sudo=self.sudo,
            executor_bin=self.executor_bin,
        )
        # -L: the port-forward the client connects through. ExitOnForwardFailure: fail loudly if the
        # local port is taken. multiplex=False: a dedicated connection. Password mode pipes into
        # `sudo -S` so NO PTY (a PTY would echo it and break the stdin pipe); NOPASSWD keeps -tt so
        # closing ssh SIGHUPs the executor.
        opts = [
            "-o",
            "ExitOnForwardFailure=yes",
            "-L",
            f"{self.local_port}:localhost:{self._bind_port}",
        ]
        if not use_pw:
            opts = ["-tt"] + opts
        ssh = SSH.base(self.user, self.host, opts, multiplex=False) + [remote_cmd]
        self._say(
            f"starting executor on {self.host}:{self._bind_port} "
            f"(tunnel {self.addr} -> remote:{self._bind_port})"
        )
        self._say(f"remote: {remote_cmd}", level=2)
        Log.cmd(ssh)
        self.proc = self._popen(
            ssh, stdin=(subprocess.PIPE if use_pw else subprocess.DEVNULL)
        )
        if use_pw:  # feed the sudo password straight into `sudo -S` on stdin
            assert self.proc.stdin is not None and self.sudo_password is not None
            with contextlib.suppress(BrokenPipeError, OSError):
                self.proc.stdin.write(self.sudo_password + "\n")
                self.proc.stdin.flush()

    def _auth_help(self) -> str:
        """A human-readable hint for the user to fix the detected auth failure — install an SSH
        key, or supply/verify the sudo password. Called by :meth:`_check_failure`."""
        if self._auth_kind == "ssh":
            return (
                "SSH wants a password — install your key once (only adds your key, does not\n"
                f"affect other users of the {self.user} account):\n"
                f"    ssh-copy-id {self.user}@{self.host}"
            )
        return (
            "sudo on the remote host rejected the password (or none was available).\n"
            "  Provide it via sudo_password=, or run interactively so it can prompt.\n"
            f"  Check it by hand:  ssh {self.user}@{self.host} sudo -v"
        )

    def _teardown_extra(self) -> None:
        """Backstop ``pkill`` for OUR executor on the remote host, in case closing the SSH
        connection didn't SIGHUP it. Skipped when we never bound the port (the process there
        would be someone else's, and a port-scoped ``pkill`` must not wrongly kill it)."""
        # Backstop kill of OUR executor (closing the -tt ssh already SIGHUPs it). Only when we
        # actually bound this port — on a port collision the process there is someone else's, and a
        # port-scoped pkill would wrongly kill it.
        if not self._ready_reached:
            return
        pat = f"{Paths.EXECUTOR_BIN}.*{ExecutorCli.BIND_FLAG}0.0.0.0:{self._bind_port}"
        SSH.pkill(
            self.user, self.host, pat,
            sudo=self.sudo, sudo_password=self.sudo_password,
        )

    def teardown_workspace(self) -> None:
        """Remove the auto-generated remote workspace. Guarded by the ``catalyst-exec-`` prefix so it
        can never wipe a user-pinned dir; a no-op for a pinned workspace."""
        basename = self.workspace.rsplit("/", 1)[-1]
        if not self.cleanup_ws or not basename.startswith(Paths.WORKSPACE_PREFIX):
            return
        self._say(f"removing remote workspace {self.workspace}", level=2)
        SSH.rm_rf(self.user, self.host, self.workspace)

    def stop(self) -> None:
        """Stop the remote executor and close the SSH tunnel. Idempotent."""
        self._say("stopping executor + closing tunnel")
        super().stop()
