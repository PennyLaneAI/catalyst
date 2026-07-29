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

Implementation is split across this package: :mod:`._util` (logging, regexes, small helpers),
:mod:`.ssh` (SSH/scp/auth orchestration), :mod:`.process` (the subprocess-lifecycle classes). This
module holds the public :class:`Executor` and the process registry.
"""

from __future__ import annotations

import atexit
import contextlib
import getpass
import platform
import subprocess
from pathlib import Path

from ._util import (
    _default_executor_bin,
    _default_workspace,
    _PORT_TRIES,
    _random_port,
    _resolve_log_path,
    _set_verbosity,
    _triple_from_uname,
    PortInUse,
)
from .process import _ExecutorProcess, _LocalProcess, _RemoteProcess
from .ssh import _copy_bundle, _remove_remote_dir, _resolve_sudo_password, _ssh_base

__all__ = ["Executor", "PortInUse"]


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
                    # copied bundle -> run it from the workspace (./); sudo's secure_path would miss a
                    # bare name. Bare only when attaching to a remote that has it on PATH.
                    executor_bin=self._executor_bin
                    or ("./catalyst-executor" if self._copy else "catalyst-executor"),
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
