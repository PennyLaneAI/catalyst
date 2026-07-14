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

"""The :class:`Executor` — deploy a ``catalyst-executor`` and give its address to ``target(...)``.

An executor is an out-of-process ``catalyst-executor`` server this talks to over TCP: a QNode
compiled for a ``target(address=...)`` device, or a ``kernel.declare(remote=...)`` library call, is
dispatched to it. Configure it with explicit Python arguments (no environment variables).
Construction is inert; :meth:`Executor.launch` deploys, :meth:`Executor.stop` tears down, and it is a
context manager::

    from catalyst import Executor

    # local: run catalyst-executor as a subprocess on 127.0.0.1 (no SSH):
    with Executor(local=True, plugins=[...]) as ex:
        dev = target(qml.device(...), address=ex.address)
        ...

    # remote: run it on another host over a forwarded SSH:
    ex = Executor(host="10.0.0.9", user="me", plugins=[...]).launch()
    dev = target(qml.device(...), address=ex.address)   # ... ex.stop() (also at process exit)

    # or attach to one already running/tunnelled (neither local nor host):
    ex = Executor("127.0.0.1:1234").launch()

    # persistent remote workspace: deploy the bundle once, reuse across runs, remove explicitly:
    Executor(host="10.0.0.9", workspace="~/cat-ws", bundle="...").setup_workspace()  # deploy once
    Executor(host="10.0.0.9", workspace="~/cat-ws").launch()                         # reuse each run
    Executor(host="10.0.0.9", workspace="~/cat-ws").remove_workspace()               # remove when done
"""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import faulthandler
import getpass
import os
import platform
import random
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from catalyst.utils.runtime_environment import get_lib_path

# A segfault in a loaded plugin (e.g. a device lib reacting to a remote abort) otherwise core-dumps
# silently — this prints a C-level traceback instead.
with contextlib.suppress(Exception):
    faulthandler.enable()

# Executor stderr lines that mean "bound and accepting". The first launch prints "Listening on
# <h>:<p>"; "executor ready, ..." only recurs after a client disconnects.
_READY_RE = re.compile(r"Listening on \S+:\d+|executor ready, waiting for next connection")
# ssh login still wanting a password/passphrase means key auth isn't set up — we can't feed it.
_SSH_PW_RE = re.compile(r"'s password:|Enter passphrase for key")
# sudo telling us the password we fed (via sudo -S) was wrong.
_SUDO_FAIL_RE = re.compile(
    r"Sorry, try again|incorrect password|authentication failure|sudo: \d+ incorrect"
)
# A port collision — the remote bind or the local -L forward is already taken.
_PORT_RE = re.compile(r"Address already in use|Could not request local forwarding")

# Random bind port so concurrent launches on a shared host don't collide; retried on conflict.
_PORT_TRIES = 6


class PortInUse(Exception):
    """The chosen executor port was already taken (likely another user on the host)."""


def _random_port() -> int:
    return random.randint(20000, 59999)


def _timestamp() -> str:
    # Filesystem-safe but clearly separated (no bare run-together digits): 2026-06-30_04-48-15
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def _default_workspace() -> str:
    """Per-run remote dir so concurrent launches on a shared account don't clobber each other. The
    ``catalyst-exec-`` prefix also gates the ``rm -rf`` cleanup (safety)."""
    return f"~/catalyst-exec-{getpass.getuser()}-{_timestamp()}-{random.randint(0, 0xfff):03x}"


def _default_executor_bin() -> str:
    """Locate the shipped/built ``catalyst-executor`` binary (for ``local=True``).

    In an installed wheel it sits in the packaged lib dir; in a source build it is under the runtime
    build's ``remote/`` subdir. Falls back to the name on ``PATH``."""
    rt_lib = Path(get_lib_path("runtime", "RUNTIME_LIB_DIR"))
    for candidate in (rt_lib / "catalyst-executor", rt_lib / "remote" / "catalyst-executor"):
        if candidate.exists():
            return str(candidate)
    return "catalyst-executor"


def _triple_from_uname(system: str, machine: str) -> str | None:
    """Map ``uname -s`` / ``uname -m`` to an LLVM target triple for the common cases."""
    arch = {"aarch64": "aarch64", "arm64": "aarch64", "x86_64": "x86_64", "amd64": "x86_64"}.get(
        machine.strip().lower()
    )
    if arch is None:
        return None
    system = system.strip().lower()
    if system == "linux":
        return f"{arch}-unknown-linux-gnu"
    if system == "darwin":
        return f"{'arm64' if arch == 'aarch64' else arch}-apple-darwin"
    return None


# Verbosity: 0 quiet (errors only) · 1 phases + executor stream (default) · 2 full ssh/scp commands
# + scp -v + timings · 3+ adds `ssh -v`. The executor's output always streams regardless.
_VERBOSITY = 1


def _set_verbosity(level: int) -> None:
    """Set the launcher's output verbosity (also settable per launch via ``Executor(verbose=)``)."""
    global _VERBOSITY
    _VERBOSITY = level


def _log(msg: str, level: int = 1) -> None:
    if level <= _VERBOSITY:
        print(f"[remote-exec] {msg}", file=sys.stderr, flush=True)


def _logcmd(cmd: list[str]) -> None:
    """Echo a command we're about to run (verbosity >= 2)."""
    _log("$ " + " ".join(shlex.quote(c) for c in cmd), level=2)


def _resolve_log_path(
    host: str, explicit: str | None = None, disabled: bool = False, name: str = "executor"
) -> str | None:
    """Host-side log file for an executor's output — one per launch in the cwd, named by the executor
    (so several executors each get their own file). ``explicit`` pins a path; ``disabled`` turns it
    off."""
    if disabled:
        return None
    if explicit:
        return explicit
    tag = "" if name == "executor" else f"-{name}"
    return f"catalyst-executor{tag}-{host}-{_timestamp()}.log"


# --------------------------------------------------------------------------------------------------
# SSH helpers (remote launch)
# --------------------------------------------------------------------------------------------------


# Connection multiplexing: the first short op opens a master, the rest reuse it (no re-handshake),
# and it self-expires. Applied to the chatty control ops (probe/mkdir/scp/pkill), NOT the long-lived
# executor session — that keeps one clean connection whose close SIGHUPs the executor.
def _ctl_opts() -> list[str]:
    return [
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPath=~/.ssh/catalyst-cm-%r@%h:%p",
        "-o",
        "ControlPersist=30",
    ]


def _set_pdeathsig() -> None:
    """preexec_fn: ask the kernel to SIGTERM this child when the parent (python) dies — so a host
    crash (segfault/SIGKILL, which skip atexit) doesn't leak the ssh tunnel + executor."""
    with contextlib.suppress(Exception):
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG=1


# preexec_fn only exists on POSIX; harmless no-op reference elsewhere.
_PDEATHSIG = _set_pdeathsig if hasattr(os, "fork") else None


def _ssh_base(
    user: str, host: str, opts: list[str] | None = None, multiplex: bool = True
) -> list[str]:
    cmd = ["ssh", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
    if multiplex:
        cmd += _ctl_opts()
    cmd += ["-v"] * max(0, _VERBOSITY - 2)  # ssh protocol debug at -vvv (verbosity 3+)
    if opts:
        cmd += opts
    cmd.append(f"{user}@{host}")
    return cmd


def _copy_bundle(bundle: Path, user: str, host: str, workspace: str) -> None:
    """mkdir the remote workspace and scp every artifact in ``bundle`` into it."""
    files = sorted(p for p in bundle.iterdir() if p.is_file() and p.name != "README.md")
    if not files:
        raise SystemExit(
            f"no artifacts in {bundle} — pass build=<recipe> to cross-compile the executor + "
            "runtime libs for the target, or point bundle= at a prebuilt directory."
        )
    total = sum(f.stat().st_size for f in files)
    _log(f"copying {len(files)} artifact(s), {total/1e6:.1f} MB -> {user}@{host}:{workspace}/")
    for f in files:
        _log(f"  - {f.name}  ({f.stat().st_size/1e6:.2f} MB)", level=2)
    mkdir = _ssh_base(user, host) + [f"mkdir -p {workspace}"]
    _logcmd(mkdir)
    if subprocess.call(mkdir) != 0:
        raise SystemExit("failed to create remote workspace")
    scp = [
        "scp",
        *_ctl_opts(),
        "-v" if _VERBOSITY >= 2 else "-q",
        *[str(f) for f in files],
        f"{user}@{host}:{workspace}/",
    ]
    _logcmd(scp)
    t0 = time.monotonic()
    if subprocess.call(scp) != 0:
        raise SystemExit("scp of bundle failed")
    _log(f"copied in {time.monotonic() - t0:.1f}s", level=2)


def _remote_path_expr(path: str) -> str:
    """Shell expression for ``path`` that expands a leading ``~`` via ``$HOME`` and quotes the rest,
    so it survives ``cd``/``rm`` without tilde-in-quotes breakage or word-splitting."""
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def _remove_remote_dir(user: str, host: str, workspace: str) -> None:
    """``rm -rf`` a remote workspace, guarded so it can never delete ``/`` or the home directory.

    Resolves ``workspace`` (including a leading ``~``) to a canonical path on the remote and refuses
    if it is empty, ``/``, or ``$HOME`` itself. A missing directory is a no-op."""
    remote = (
        f"ws={_remote_path_expr(workspace)}; "
        'd=$(cd "$ws" 2>/dev/null && pwd) || exit 0; '
        'if [ -z "$d" ] || [ "$d" = "/" ] || [ "$d" = "$HOME" ]; then exit 3; fi; '
        'rm -rf "$d"'
    )
    cmd = _ssh_base(user, host) + [remote]
    _logcmd(cmd)
    rc = subprocess.call(cmd, timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc == 3:
        raise ValueError(
            f"refusing to remove workspace {workspace!r}: it resolves to '/' or the home directory"
        )
    if rc != 0:
        raise RuntimeError(f"failed to remove remote workspace {workspace!r} (ssh rc={rc})")


def _probe_auth(user: str, host: str) -> tuple[bool, bool]:
    """Non-interactively check (key-based SSH works, sudo is passwordless) on the remote host.

    Returns ``(ssh_ok, sudo_nopasswd)``. BatchMode means a missing key fails fast (rc 255) instead of
    prompting; ``sudo -n true`` returns 0 only when sudo needs no password."""
    cmd = _ssh_base(user, host, ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]) + [
        "sudo -n true 2>/dev/null"
    ]
    _logcmd(cmd)
    rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc == 255:  # ssh itself failed (no usable key, host unreachable, …)
        return (False, False)
    return (True, rc == 0)


def _resolve_sudo_password(user: str, host: str, sudo_password: str | None = None) -> str | None:
    """The remote sudo password, when NOPASSWD isn't set: the explicit ``sudo_password`` if given,
    else a one-time getpass prompt. Returns None if sudo needs no password (nothing to do)."""
    ssh_ok, nopasswd = _probe_auth(user, host)
    if not ssh_ok:
        raise SystemExit(
            f"can't SSH to {user}@{host} without a password — install your key once:\n"
            f"    ssh-copy-id {user}@{host}\n"
            "  (this only ADDS your key; it does not affect other users of the account.)"
        )
    if nopasswd:
        return None
    if sudo_password is not None:
        return sudo_password
    _log("remote sudo needs a password (no NOPASSWD) — prompting once", level=1)
    try:
        return getpass.getpass(f"[remote] sudo password for {user}@{host}: ")
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\nno sudo password provided — pass sudo_password= or run interactively")


def _build_remote_cmd(
    workspace: str,
    remote_port: int,
    plugins: list[str],
    env: dict[str, str],
    use_password: bool = False,
    sudo: bool = True,
    executor_bin: str = "./catalyst-executor",
) -> str:
    """The shell command run on the remote host: cd, export env, exec the executor.

    ``sudo`` wraps the executor in sudo (some devices need root): with ``use_password`` use ``sudo -S``
    (reads the password piped to stdin) with an empty prompt, otherwise plain ``sudo -E`` under a PTY
    (NOPASSWD). ``sudo=False`` runs it as the login user. ``executor_bin`` and any plugin path that is
    absolute / ``$VAR`` / ``~`` is used as-is; a bare plugin name resolves against the workspace (a
    scp'd bundle). ``env`` values are left unquoted so ``$VAR`` in them expands on the remote."""
    env_prefix = " ".join(f"{k}={v}" for k, v in env.items())

    def _plugin_arg(p: str) -> str:
        return f"--plugin={p}" if p[:1] in ("/", "$", "~") else f"--plugin=$PWD/{shlex.quote(p)}"

    plugin_args = " ".join(_plugin_arg(p) for p in plugins)
    # scp drops the +x bit; only relevant for a workspace-local executor binary.
    chmod = (
        "chmod +x ./catalyst-executor 2>/dev/null; "
        if executor_bin == "./catalyst-executor"
        else ""
    )
    if sudo:
        launcher = "exec sudo -S -E -p ''" if use_password else "exec sudo -E"
    else:
        launcher = "exec"
    return (
        f"cd {workspace} && {chmod}{env_prefix} "
        f"{launcher} {executor_bin} --bind=0.0.0.0:{remote_port} {plugin_args}"
    )


# --------------------------------------------------------------------------------------------------
# Executor processes: a base owning the shared subprocess lifecycle, and local / remote subclasses
# --------------------------------------------------------------------------------------------------


class _ExecutorProcess:
    """A launched ``catalyst-executor`` process: owns the subprocess, streams its output, waits for
    it to bind, and tears it down. Subclasses supply only what differs — :meth:`_spawn` builds and
    starts the process, and the ``_*`` hooks add remote-only behaviour (auth handling, cleanup).

    ``.addr`` is what a client connects to; ``.name`` labels the streamed output as ``[<name>]``."""

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
    def _say(self, msg: str, level: int = 1) -> None:
        line = msg if self.name == "executor" else f"{self.name}: {msg}"
        _log(line, level)
        # Tee the launcher's own narrative (launch cmd, readiness, teardown) into the log file too,
        # so the file is self-contained rather than only the executor's stdout/stderr.
        if self._log_fh is not None:
            with contextlib.suppress(Exception):
                self._log_fh.write(f"# [launcher] {line}\n")
                self._log_fh.flush()

    def _open_log(self) -> None:
        if not self.log_path:
            return
        try:
            self._log_fh = open(self.log_path, "a")
            header = self._log_header()
            if header:
                self._log_fh.write(header)
                self._log_fh.flush()
            self._say(f"teeing output -> {self.log_path}")
        except OSError as e:
            self._say(f"could not open log {self.log_path}: {e} (continuing without it)")
            self._log_fh = None

    def _pump_output(self) -> None:
        """Echo the executor's stdout+stderr live, tee it to the log, and flag readiness. Each line
        is tagged ``[<name>]`` so several executors in one terminal stay distinguishable."""
        assert self.proc and self.proc.stdout
        for raw in self.proc.stdout:
            line = raw.rstrip("\n")
            print(f"[{self.name}] {line}", file=sys.stderr, flush=True)
            if self._log_fh is not None:
                with contextlib.suppress(Exception):
                    self._log_fh.write(line + "\n")
                    self._log_fh.flush()
            if _PORT_RE.search(line):
                self._port_conflict.set()
            if _READY_RE.search(line):
                self._ready.set()
            self._scan_line(line)

    def start(self) -> "_ExecutorProcess":
        """Spawn the executor and block until it is listening. Raises :class:`PortInUse` on a port
        collision (so the caller can retry another port), or ``SystemExit`` on other failures."""
        self._open_log()
        self._spawn()
        assert self.proc is not None, "_spawn() must set self.proc"
        threading.Thread(target=self._pump_output, daemon=True).start()
        t0 = time.monotonic()
        deadline = t0 + self.ready_timeout
        while time.monotonic() < deadline:
            if self._ready.wait(timeout=0.25):
                self._on_ready()
                self._say(f"ready in {time.monotonic() - t0:.1f}s — address {self.addr}")
                return self
            if self._port_conflict.is_set():
                self._shutdown()
                raise PortInUse(self._bind_port)
            self._check_failure()
            if self.proc.poll() is not None:
                returncode = self.proc.returncode
                self._shutdown()
                if self._port_conflict.is_set():
                    raise PortInUse(self._bind_port)
                raise SystemExit(
                    f"executor exited (code {returncode}) before becoming ready — "
                    f"see the [{self.name}] log above."
                )
        self._shutdown()
        raise SystemExit(
            f"executor did not become ready within {self.ready_timeout:.0f}s — see the "
            f"[{self.name}] log above (raise ready_timeout= if the host is slow)."
        )

    def _shutdown(self) -> None:
        fh, self._log_fh = self._log_fh, None  # stop the pump teeing, then close
        if fh is not None:
            with contextlib.suppress(Exception):
                fh.close()
        if self.proc and self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    self.proc.kill()

    def stop(self) -> None:
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
            addr=f"127.0.0.1:{port}",
            bind_port=port,
            ready_timeout=ready_timeout,
            log_path=log_path,
        )
        self._executor_bin = executor_bin
        self._plugins = plugins or []
        self._env = dict(env or {})

    def _spawn(self) -> None:
        exe = os.path.expanduser(os.path.expandvars(self._executor_bin))
        argv = [exe, f"--bind=127.0.0.1:{self._bind_port}"]
        argv += [f"--plugin={os.path.expanduser(os.path.expandvars(p))}" for p in self._plugins]
        proc_env = dict(os.environ)
        for key, value in self._env.items():
            proc_env[key] = os.path.expandvars(value)
        self._say(f"starting local executor on {self.addr}")
        _logcmd(argv)
        self.proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=proc_env,
            preexec_fn=_PDEATHSIG,
        )


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
        executor_bin: str = "./catalyst-executor",
        cleanup_ws: bool = False,
        ready_timeout: float = 60.0,
        name: str = "executor",
        log_path: str | None = None,
    ):
        local_port = local_port or port
        super().__init__(
            name=name,
            addr=f"127.0.0.1:{local_port}",
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
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        return (
            f"\n# ==== {self.name} @ {self.host}:{self._bind_port} | ws={self.workspace} | "
            f"{ts} ====\n# plugins: {', '.join(self._plugins)}\n"
        )

    def _scan_line(self, line: str) -> None:
        if _SSH_PW_RE.search(line):  # ssh login prompt — key auth isn't set up
            self._auth_kind = "ssh"
            self._auth_prompt.set()
        elif _SUDO_FAIL_RE.search(line):  # sudo rejected the password we fed
            self._auth_kind = "sudo"
            self._auth_prompt.set()

    def _check_failure(self) -> None:
        if self._auth_prompt.is_set():
            self._shutdown()
            raise SystemExit(self._auth_help())

    def _on_ready(self) -> None:
        self._ready_reached = True

    def _spawn(self) -> None:
        use_pw = self.sudo_password is not None
        remote_cmd = _build_remote_cmd(
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
        ssh = _ssh_base(self.user, self.host, opts, multiplex=False) + [remote_cmd]
        self._say(
            f"starting executor on {self.host}:{self._bind_port} "
            f"(tunnel {self.addr} -> remote:{self._bind_port})"
        )
        self._say(f"remote: {remote_cmd}", level=2)
        _logcmd(ssh)
        self.proc = subprocess.Popen(
            ssh,
            stdin=(subprocess.PIPE if use_pw else subprocess.DEVNULL),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=_PDEATHSIG,
        )
        if use_pw:  # feed the sudo password straight into `sudo -S` on stdin
            assert self.proc.stdin is not None and self.sudo_password is not None
            with contextlib.suppress(BrokenPipeError, OSError):
                self.proc.stdin.write(self.sudo_password + "\n")
                self.proc.stdin.flush()

    def _auth_help(self) -> str:
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
        # Backstop kill of OUR executor (closing the -tt ssh already SIGHUPs it). Only when we
        # actually bound this port — on a port collision the process there is someone else's, and a
        # port-scoped pkill would wrongly kill it.
        if not self._ready_reached:
            return
        pat = f"catalyst-executor.*--bind=0.0.0.0:{self._bind_port}"
        with contextlib.suppress(Exception):
            if not self.sudo:
                subprocess.call(
                    _ssh_base(self.user, self.host) + [f"pkill -f {shlex.quote(pat)}"],
                    timeout=15,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif self.sudo_password is not None:
                subprocess.run(
                    _ssh_base(self.user, self.host)
                    + [f"sudo -S -p '' pkill -f {shlex.quote(pat)}"],
                    input=self.sudo_password + "\n",
                    text=True,
                    timeout=15,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.call(
                    _ssh_base(self.user, self.host) + [f"sudo -n pkill -f {shlex.quote(pat)}"],
                    timeout=15,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

    def teardown_workspace(self) -> None:
        """Remove the auto-generated remote workspace. Guarded by the ``catalyst-exec-`` prefix so it
        can never wipe a user-pinned dir; a no-op for a pinned workspace."""
        if not self.cleanup_ws:
            return
        if not self.workspace.rsplit("/", 1)[-1].startswith("catalyst-exec-"):
            return
        self._say(f"removing remote workspace {self.workspace}", level=2)
        with contextlib.suppress(Exception):
            subprocess.call(
                _ssh_base(self.user, self.host) + [f"rm -rf {self.workspace}"],
                timeout=15,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def stop(self) -> None:
        self._say("stopping executor + closing tunnel")
        super().stop()


def _start_with_retry(make_process, pinned_port: int | None) -> _ExecutorProcess:
    """Start a process from ``make_process(port)``, picking a random port and retrying on a
    collision unless ``pinned_port`` is given."""
    tries = 1 if pinned_port is not None else _PORT_TRIES
    last: Exception | None = None
    for _ in range(tries):
        port = pinned_port if pinned_port is not None else _random_port()
        proc = make_process(port)
        try:
            return proc.start()
        except PortInUse as e:
            last = e
            proc._say(f"port {port} is busy on the host (another user?) — trying another")
    raise SystemExit(
        f"couldn't get a free executor port after {tries} tries ({last}). Pin one with port=."
        if pinned_port is None
        else f"port {pinned_port} is busy on the host ({last}). Pick another port=."
    )


# --------------------------------------------------------------------------------------------------
# The public Executor
# --------------------------------------------------------------------------------------------------

# Process-wide registry of launched executors, torn down at process exit. Keyed by name so a repeated
# launch for the same role reuses the running one.
_SESSIONS: dict[str, _ExecutorProcess] = {}
_atexit_registered = False


def _shutdown_sessions() -> None:
    for proc in list(_SESSIONS.values()):
        with contextlib.suppress(Exception):
            proc.stop()
        proc.teardown_workspace()
    _SESSIONS.clear()


class Executor:
    """A ``catalyst-executor`` you launch and talk to over TCP — local, remote, or an already-running
    one. Pass its :attr:`address` to ``target(address=...)``.

    Construction is inert; :meth:`launch` deploys it and :meth:`stop` tears it down (both idempotent),
    and it is a context manager so ``with Executor(...) as ex:`` launches on entry and stops on exit.
    Everything is an explicit argument — no environment variables. The three modes:

    * ``local=True``  — run catalyst-executor as a local subprocess on ``127.0.0.1`` (no SSH). Uses
      the shipped/built binary unless ``executor_bin=`` overrides it.
    * ``host=<addr>`` — run it on that host over a forwarded SSH (``user``/``sudo``/``sudo_password``
      as needed; ``copy=True`` + ``bundle=<dir>`` first scp's the bundle there, cross-building it via
      ``build=`` if given).
    * neither         — carry ``address`` for an executor already running/tunnelled there.

    ``name`` labels it: output streams as ``[<name>]`` and the log is
    ``catalyst-executor-<name>-<host>-<ts>.log``. ``plugins`` are the device backends / runtime_call
    libraries to load; ``env`` is extra environment for the executor process (e.g.
    ``LD_LIBRARY_PATH``); ``verbose`` (0-3) sets launcher detail; ``triple`` overrides the
    auto-detected target triple (see :attr:`triple`). ``build`` is an optional ``build(triple,
    bundle_dir)`` recipe invoked on every ``copy=True`` deploy to (re)produce the bundle for the
    target; it must be idempotent — it is called on each deploy, not only when the bundle is missing,
    so return fast when nothing changed. The port is randomized per launch unless pinned via ``port``.
    """

    def __init__(
        self,
        address: str = "127.0.0.1:1373",
        *,
        host: str | None = None,
        local: bool = False,
        user: str = "",
        port: int | None = None,
        local_port: int | None = None,
        workspace: str | None = None,
        bundle=None,
        plugins: list[str] | None = None,
        copy: bool = False,
        build=None,
        ready_timeout: float = 60.0,
        name: str = "executor",
        sudo: bool = True,
        sudo_password: str | None = None,
        executor_bin: str | None = None,
        triple: str | None = None,
        env: dict[str, str] | None = None,
        verbose: int = 1,
    ):
        self.host = host
        self.name = name
        self._local = local
        self._address = address
        self._user = user
        self._port = port
        self._local_port = local_port
        self._workspace = workspace
        self._bundle = bundle
        self._plugins = plugins
        self._copy = copy
        self._build = build
        self._ready_timeout = ready_timeout
        self._sudo = sudo
        self._sudo_password = sudo_password
        self._executor_bin = executor_bin
        self._triple = triple
        self._detected_triple: str | None = None
        self._triple_detected = False
        self._env = env
        self._verbose = verbose
        self._proc: _ExecutorProcess | None = None
        self._launched = False

    @property
    def address(self) -> str:
        if not self._launched:
            raise RuntimeError("Executor not launched — call .launch() or use `with Executor(...)`")
        return self._address

    @property
    def triple(self) -> str | None:
        """The executor's LLVM target triple, for cross-compiling a ``target`` to its architecture.

        The explicit ``triple=`` if one was given, otherwise auto-detected: the local host's triple
        for ``local=True``, or a ``uname`` probe over SSH for a remote ``host``. ``None`` when it
        can't be determined (an attach-only executor); the compiler then falls back to the host
        triple."""
        if self._triple is not None:
            return self._triple
        if not self._triple_detected:
            self._detected_triple = self._detect_triple()
            self._triple_detected = True
        return self._detected_triple

    def _detect_triple(self) -> str | None:
        if self._local:
            return _triple_from_uname(platform.system(), platform.machine())
        if self.host:
            user = self._user or getpass.getuser()
            cmd = _ssh_base(
                user, self.host.strip(), ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
            ) + ["uname -sm"]
            try:
                out = subprocess.check_output(
                    cmd, text=True, timeout=15, stderr=subprocess.DEVNULL
                ).strip()
            except Exception:
                return None
            system, _, machine = out.partition(" ")
            return _triple_from_uname(system, machine)
        return None

    def _remote_target(self) -> tuple[str, str, str]:
        """Resolve ``(user, host, workspace)`` for a remote deploy or run. Remote only."""
        assert self.host is not None
        host = self.host.strip()
        user = self._user or getpass.getuser()
        workspace = self._workspace or _default_workspace()
        return user, host, workspace

    def launch(self) -> "Executor":
        """Deploy the executor (idempotent). Returns ``self`` so ``ex = Executor(...).launch()`` works."""
        if self._launched:
            return self
        if not (self._local or self.host):
            self._launched = True  # manual mode: nothing to deploy, use the given address
            return self
        _set_verbosity(self._verbose)
        plugins = self._plugins if self._plugins is not None else []

        if self._local:

            def make(port: int) -> _ExecutorProcess:
                return _LocalProcess(
                    port=port,
                    executor_bin=self._executor_bin or _default_executor_bin(),
                    plugins=plugins,
                    env=self._env,
                    ready_timeout=self._ready_timeout,
                    name=self.name,
                    log_path=_resolve_log_path("localhost", name=self.name),
                )

        else:
            user, host, workspace = self._remote_target()
            ws_pinned = self._workspace is not None  # a pinned dir is left in place on teardown
            sudo_pw = (
                _resolve_sudo_password(user, host, self._sudo_password) if self._sudo else None
            )
            if self._copy and self._bundle:
                bundle = Path(self._bundle)
                if self._build is not None:
                    self._build(
                        self.triple, bundle
                    )  # idempotent recipe (see build=); may cross-build
                _copy_bundle(bundle, user, host, workspace)

            def make(port: int) -> _ExecutorProcess:
                return _RemoteProcess(
                    host=host,
                    user=user,
                    port=port,
                    local_port=self._local_port,
                    workspace=workspace,
                    plugins=plugins,
                    env=self._env,
                    sudo=self._sudo,
                    sudo_password=sudo_pw,
                    executor_bin=self._executor_bin or "catalyst-executor",
                    cleanup_ws=(not ws_pinned),
                    ready_timeout=self._ready_timeout,
                    name=self.name,
                    log_path=_resolve_log_path(host, name=self.name),
                )

        self._proc = _start_with_retry(make, self._port)
        self._address = self._proc.addr
        self._launched = True
        _SESSIONS[self.name] = self._proc
        global _atexit_registered
        if not _atexit_registered:
            atexit.register(_shutdown_sessions)
            _atexit_registered = True
        return self

    def stop(self) -> None:
        """Tear down the executor + tunnel (idempotent; a no-op in manual mode)."""
        self._launched = False
        if self._proc is None:
            return
        with contextlib.suppress(Exception):
            self._proc.stop()
        self._proc.teardown_workspace()
        _SESSIONS.pop(self.name, None)
        self._proc = None

    def setup_workspace(self) -> "Executor":
        """Deploy the bundle to a persistent remote workspace *without* starting the executor.

        Requires a remote ``host``, a pinned ``workspace=`` (so later runs can reuse it), and a
        ``bundle``. Idempotent — re-run to redeploy after rebuilding the bundle. Afterwards
        ``launch()`` this instance, or a fresh ``Executor(..., workspace=<same>)`` from another run;
        neither re-copies (``copy`` defaults off). Delete it with :meth:`remove_workspace`. Copies
        as the login user (no sudo needed)."""
        if not self.host:
            raise ValueError("setup_workspace() needs a remote host= (nothing to deploy locally)")
        if self._workspace is None:
            raise ValueError(
                "setup_workspace() needs a pinned workspace= so later runs can reuse it"
            )
        if not self._bundle:
            raise ValueError("setup_workspace() needs a bundle= to deploy")
        _set_verbosity(self._verbose)
        user, host, workspace = self._remote_target()
        bundle = Path(self._bundle)
        if self._build is not None:
            self._build(self.triple, bundle)  # idempotent recipe (see build=); may cross-build
        _copy_bundle(bundle, user, host, workspace)
        self._copy = False  # bundle is deployed; launch() on this instance won't re-copy
        return self

    def remove_workspace(self, force: bool = False) -> None:
        """Delete a pinned remote workspace (directory + bundle) — explicit teardown for a persistent
        workspace, which is never auto-removed. Refuses to delete ``/`` or the home directory. SSH
        errors are ignored unless ``force`` re-raises them; the safety refusal always raises."""
        if not self.host:
            raise ValueError("remove_workspace() needs a remote host=")
        if self._workspace is None:
            raise ValueError("remove_workspace() needs a pinned workspace= to remove")
        _set_verbosity(self._verbose)
        user, host, workspace = self._remote_target()
        try:
            _remove_remote_dir(user, host, workspace)
        except RuntimeError:
            if force:
                raise

    def __enter__(self) -> "Executor":
        return self.launch()

    def __exit__(self, *exc) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (
            f"Executor(name={self.name!r}, host={self.host!r}, local={self._local}, "
            f"launched={self._launched}, address={self._address!r})"
        )
