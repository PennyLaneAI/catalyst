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

"""``catalyst-executor`` process objects: a base owning the subprocess lifecycle, plus local and
remote subclasses. Driven by :class:`~catalyst.executor.Executor`; remote command plumbing lives
in :mod:`.ssh`."""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Self, TextIO

from .ssh import RemoteLauncher, RemoteOps
from .utils import (
    ExecutorFlags,
    ExecutorPaths,
    OutputPatterns,
    log_cmd,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ExecutorProcess:
    """Base ``catalyst-executor`` process: spawn, stream output, wait for bind, tear down.

    Subclasses override :meth:`_spawn` and the ``_*`` hooks. ``.addr`` is the client-facing
    endpoint; ``.name`` tags streamed output as ``[<name>]``.
    """

    LOCALHOST = "127.0.0.1"  # local bind, and the client end of the remote SSH tunnel

    def __init__(
        self, *, name: str, addr: str, bind_port: int, ready_timeout: float, log_path: str | None
    ):
        self.name = name
        self.addr = addr
        self._bind_port = bind_port
        self.ready_timeout = ready_timeout
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self._log_fh: TextIO | None = None
        self._ready = threading.Event()
        self._port_conflict = threading.Event()

    def _spawn(self) -> None:
        """Build the command and set ``self.proc``."""
        raise NotImplementedError

    def _log_header(self) -> str:
        """Optional banner written once at the top of a fresh log file."""
        return ""

    def _scan_line(self, line: str) -> None:
        """Inspect an output line for extra conditions (remote: auth prompts)."""

    def _check_failure(self) -> None:
        """Raise if a non-port failure was detected while waiting for readiness."""

    def _on_ready(self) -> None:
        """Called once the executor is bound."""

    def _teardown_extra(self) -> None:
        """Extra teardown after the local process is stopped."""

    def teardown_workspace(self) -> None:
        """Best-effort removal of an auto-generated remote workspace."""

    # --- shared lifecycle ------------------------------------------------------------------------
    @staticmethod
    def _popen(
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        stdin: int | None = None,
    ) -> subprocess.Popen:
        """Spawn a subprocess with line-buffered merged output for :meth:`_watch_output` to read.

        Given a session of its own so a ^C at the terminal is not delivered to it. An ssh client
        killed by that signal drops its port-forward without closing the remote side, leaving the
        executor running; this way the interrupt raises here, where teardown happens.
        """
        return subprocess.Popen(
            argv,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )

    def _log_message(self, msg: str, level: int = 1) -> None:
        """Log a launcher narrative line, prefixed by ``<name>:`` when non-default; teed to the log."""
        line = msg if self.name == "executor" else f"{self.name}: {msg}"
        (logger.debug if level >= 2 else logger.info)(line)
        self._log_tee(f"# [launcher] {line}")

    def _log_tee(self, text: str) -> None:
        """Append ``text`` to the per-launch log. Write errors are swallowed."""
        if self._log_fh is not None:
            with contextlib.suppress(Exception):
                self._log_fh.write(text + "\n")
                self._log_fh.flush()

    def _open_log(self) -> None:
        """Open the per-launch log in append mode and write the subclass header. Failure is non-fatal."""
        if not self.log_path:
            return
        try:
            self._log_fh = open(self.log_path, "a")
        except OSError as e:
            self._log_message(f"could not open log {self.log_path}: {e} (continuing without it)")
            return
        header = self._log_header()
        if header:
            self._log_fh.write(header)
            self._log_fh.flush()
        self._log_message(f"teeing output -> {self.log_path}")

    def _watch_output(self) -> None:
        """Echo executor stdout live, tee to the log, and flag readiness / port conflicts.
        Runs in a daemon thread."""
        assert self.proc and self.proc.stdout
        for raw in self.proc.stdout:
            line = raw.rstrip("\n")
            print(f"[{self.name}] {line}", file=sys.stderr, flush=True)
            self._log_tee(line)
            if OutputPatterns.is_port_conflict(line):
                self._port_conflict.set()
            if OutputPatterns.is_ready(line):
                self._ready.set()
            self._scan_line(line)

    def start(self) -> Self:
        """Spawn the executor and block until it binds.

        Raises:
            RuntimeError: On any launch failure. :attr:`port_conflict` tells a retryable port
                collision from the rest.
        """
        self._open_log()
        self._spawn()
        assert self.proc is not None, "_spawn() must set self.proc"
        threading.Thread(target=self._watch_output, daemon=True).start()
        try:
            return self._wait_for_ready()
        except BaseException:
            self._shutdown()
            raise

    def _wait_for_ready(self, poll_interval: float = 0.25) -> Self:
        """Block up to ``ready_timeout`` for readiness or a failure signal.

        Raises:
            RuntimeError: On port collision, early exit, or timeout. Cleanup is the caller's job.
        """
        assert self.proc is not None
        t0 = time.monotonic()
        deadline = t0 + self.ready_timeout
        while time.monotonic() < deadline:
            if self._ready.wait(timeout=poll_interval):
                self._on_ready()
                self._log_message(f"ready in {time.monotonic() - t0:.1f}s — address {self.addr}")
                return self
            self._check_port_conflict()
            self._check_failure()
            self._check_early_exit()
        raise RuntimeError(
            f"executor did not become ready within {self.ready_timeout:.0f}s — see the "
            f"[{self.name}] log above (raise ready_timeout= if the host is slow)."
        )

    @property
    def port_conflict(self) -> bool:
        """Whether the executor reported its port already bound — the one retryable failure."""
        return self._port_conflict.is_set()

    def _check_port_conflict(self) -> None:
        """Raise if the watcher thread flagged a port bind failure."""
        if self.port_conflict:
            raise RuntimeError(f"port {self._bind_port} is already in use")

    def _check_early_exit(self) -> None:
        """Raise if the executor died before becoming ready.

        Re-checks the port-conflict flag first, in case it landed late. No-op while the child is
        still running.
        """
        assert self.proc is not None
        if self.proc.poll() is None:
            return
        self._check_port_conflict()
        raise RuntimeError(
            f"executor exited (code {self.proc.returncode}) before becoming ready — "
            f"see the [{self.name}] log above."
        )

    def _shutdown(self, wait_time: float = 10) -> None:
        """Close the log and terminate the subprocess; SIGKILL if it doesn't exit within
        ``wait_time`` seconds. Idempotent."""
        fh, self._log_fh = self._log_fh, None  # stop the watcher teeing, then close
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
        """Terminate the executor and run subclass teardown. Idempotent."""
        self._shutdown()
        self._teardown_extra()


class _LocalProcess(_ExecutorProcess):
    """``catalyst-executor`` running as a local subprocess on ``127.0.0.1`` (no SSH)."""

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
        """Start ``catalyst-executor`` bound to ``127.0.0.1:<port>``. ``env`` extends the parent
        environment; ``plugins`` become ``--plugin=<path>`` args."""
        exe = os.path.expanduser(os.path.expandvars(self._executor_bin))
        argv = [exe, f"{ExecutorFlags.BIND_FLAG}{self.LOCALHOST}:{self._bind_port}"]
        argv += [
            f"{ExecutorFlags.PLUGIN_FLAG}{os.path.expanduser(os.path.expandvars(p))}"
            for p in self._plugins
        ]
        log_cmd(argv)
        proc_env = dict(os.environ)
        for key, value in self._env.items():
            proc_env[key] = os.path.expandvars(value)
        self._log_message(f"starting local executor on {self.addr}")
        self.proc = self._popen(argv, env=proc_env)


class _RemoteProcess(_ExecutorProcess):
    """``catalyst-executor`` running remotely over a port-forwarded SSH.

    ``.addr`` is the local tunnel endpoint. Both ends of the forward use the same port number,
    so ``.addr`` is ``127.0.0.1:<port>``. Closing SSH stops the executor; a port-scoped ``pkill``
    runs as a teardown backstop.
    """

    def __init__(
        self,
        *,
        host: str,
        user: str,
        port: int,
        workspace: str,
        plugins: list[str] | None = None,
        env: dict[str, str] | None = None,
        sudo: bool = False,
        sudo_password: str | None = None,
        executor_bin: str = f"./{ExecutorPaths.EXECUTOR_BIN}",
        cleanup_ws: bool = False,
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
        self.host = host
        self.user = user
        self.workspace = workspace
        self.cleanup_ws = cleanup_ws
        self._plugins = plugins or []
        self._env = dict(env or {})
        self.sudo = sudo
        self.sudo_password = sudo_password
        self.executor_bin = executor_bin
        self._auth_prompt = threading.Event()
        self._auth_kind = ""  # "ssh", "setenv" or "sudo"; picks the help text

    def _log_header(self) -> str:
        """Log-file banner (host, port, workspace, timestamp, plugins)."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        return (
            f"\n# ==== {self.name} @ {self.host}:{self._bind_port} | ws={self.workspace} | "
            f"{ts} ====\n# plugins: {', '.join(self._plugins)}\n"
        )

    def _scan_line(self, line: str) -> None:
        """Flag SSH-password prompts and sudo refusals for :meth:`_check_failure` to bail on."""
        if OutputPatterns.is_ssh_prompt(line):
            kind = "ssh"
        elif OutputPatterns.is_sudo_setenv_refusal(line):
            kind = "setenv"
        elif OutputPatterns.is_sudo_fail(line):
            kind = "sudo"
        else:
            return
        self._auth_kind = kind
        self._auth_prompt.set()

    def _check_failure(self) -> None:
        """Abort the launch with :meth:`_auth_help`'s hint if an auth prompt was seen."""
        if self._auth_prompt.is_set():
            self._shutdown()
            raise RuntimeError(self._auth_help())

    def _spawn(self) -> None:
        """Open the SSH port-forward and start the remote executor via :meth:`RemoteLauncher.ssh_argv`.
        Pipes the sudo password on stdin when given."""
        use_pw = self.sudo_password is not None
        ssh = RemoteLauncher.ssh_argv(
            self.user,
            self.host,
            self.workspace,
            self._bind_port,
            self._plugins,
            self._env,
            sudo=self.sudo,
            sudo_password=self.sudo_password,
            executor_bin=self.executor_bin,
        )
        self._log_message(
            f"starting executor on {self.host}:{self._bind_port} "
            f"(tunnel {self.addr} -> remote:{self._bind_port})"
        )
        log_cmd(ssh)
        self.proc = self._popen(ssh, stdin=(subprocess.PIPE if use_pw else subprocess.DEVNULL))
        self._pipe_sudo_password()

    def _pipe_sudo_password(self) -> None:
        """Feed :attr:`sudo_password` into the child's stdin. No-op without one; write errors are
        swallowed (the watcher thread surfaces auth failures separately)."""
        if self.sudo_password is None:
            return
        assert self.proc is not None and self.proc.stdin is not None
        with contextlib.suppress(BrokenPipeError, OSError):
            self.proc.stdin.write(self.sudo_password + "\n")
            self.proc.stdin.flush()

    def _auth_help(self) -> str:
        """Fix hint for the detected failure."""
        if self._auth_kind == "ssh":
            return (
                f"SSH needs a password. Install your key:\n"
                f"    ssh-copy-id {self.user}@{self.host}"
            )
        if self._auth_kind == "setenv":
            return (
                f"Remote sudo refused to preserve env=. Grant SETENV in sudoers, or pass "
                f"sudo=False:\n"
                f"    {self.user} ALL=(ALL) NOPASSWD:SETENV: {self.executor_bin}"
            )
        return (
            f"Remote sudo rejected the password. Pass sudo_password= or run interactively.\n"
            f"    ssh {self.user}@{self.host} sudo -v"
        )

    def _teardown_extra(self) -> None:
        """Port-scoped ``pkill`` backstop for our executor.

        Skipped when nothing was spawned, and when the port turned out to be taken -- the executor
        answering there is someone else's. Anything we did spawn is fair game even if it never
        reported ready, which is the window a ^C during a deploy falls into.
        """
        if self.proc is None or self.port_conflict:
            return
        # The port is followed by a non-digit rather than end-of-string, since --plugin= args come
        # after --bind=; without it, port 2000 also matches an executor serving on 20001.
        pat = (
            f"{ExecutorPaths.EXECUTOR_BIN}.*"
            f"{ExecutorFlags.BIND_FLAG}{ExecutorFlags.BIND_HOST}:{self._bind_port}([^0-9]|$)"
        )
        RemoteOps.pkill(
            self.user,
            self.host,
            pat,
            sudo=self.sudo,
            sudo_password=self.sudo_password,
        )

    def teardown_workspace(self) -> None:
        """Remove the auto-generated remote workspace. Guarded by the ``catalyst-exec-`` prefix
        so a user-pinned dir is never wiped."""
        basename = self.workspace.rsplit("/", 1)[-1]
        if not self.cleanup_ws or not basename.startswith(ExecutorPaths.WORKSPACE_PREFIX):
            return
        self._log_message(f"removing remote workspace {self.workspace}", level=2)
        RemoteOps.rmdir(self.user, self.host, self.workspace)  # force=False: silent teardown

    def stop(self) -> None:
        """Stop the remote executor and close the SSH tunnel. Idempotent."""
        self._log_message("stopping executor + closing tunnel")
        super().stop()
