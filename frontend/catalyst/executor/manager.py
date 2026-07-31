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

An out-of-process server this talks to over TCP: a QNode compiled for ``target(address=...)`` or a
``kernel.declare(remote=...)`` call is dispatched to it. Construction is inert; :meth:`Executor.launch`
deploys, :meth:`Executor.stop` tears down, and it works as a context manager::

    from catalyst import Executor

    # local: run catalyst-executor as a subprocess on 127.0.0.1 (no SSH):
    with Executor(local=True, plugins=[...]) as ex:
        dev = target(qml.device(...), address=ex.address)

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
import getpass
import platform
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Self

from .utils import (
    MAX_PORT_TRIES,
    Paths,
    PortInUse,
    random_port,
    set_verbose,
    triple_from_uname,
)
from .process import _ExecutorProcess, _LocalProcess, _RemoteProcess
from .ssh import SCP, SSH


def _start_on_free_port(
    make_process, pinned_port: int | None, max_tries: int = MAX_PORT_TRIES
) -> _ExecutorProcess:
    """Start a process on a free port and return it. Tries ``pinned_port`` first if given, then
    up to ``max_tries`` random ports; retries on :class:`PortInUse`.

    Callers who need the pinned port to *stick* should treat :attr:`Executor.address` as
    authoritative — a pinned port that was busy silently falls back to a random one.

    Args:
        make_process: Callable ``port -> _ExecutorProcess`` that builds the process bound to
            ``port``. Called once per attempt; must not open the socket until ``.start()``.
        pinned_port: A specific port to try first; ``None`` for random-only.
        max_tries: Number of random ports to try after the (optional) pinned one.

    Returns:
        _ExecutorProcess: The started process, listening on the chosen port.

    Raises:
        SystemExit: If no port bound after all attempts.
    """
    ports_to_try = ([pinned_port] if pinned_port is not None else []) + [
        random_port() for _ in range(max_tries)
    ]
    last: PortInUse | None = None
    for port in ports_to_try:
        proc = make_process(port)
        try:
            return proc.start()
        except PortInUse as e:
            last = e
            proc._say(f"port {port} is busy on the host — trying another")
    raise SystemExit(
        f"couldn't get a free executor port after {len(ports_to_try)} tries ({last}). "
        "Pin one with port=."
    )


class _SessionRegistry:
    """Process-wide registry of launched executors, torn down at process exit. Keyed by name so a
    repeated launch for the same role reuses the running one. Registers its shutdown hook with
    :mod:`atexit` on construction, so nothing leaks when the Python process exits without an
    explicit :meth:`Executor.stop`."""

    def __init__(self) -> None:
        self._procs: dict[str, _ExecutorProcess] = {}
        atexit.register(self._shutdown_all)

    def register(self, name: str, proc: _ExecutorProcess) -> None:
        self._procs[name] = proc

    def unregister(self, name: str) -> None:
        self._procs.pop(name, None)

    def _shutdown_all(self) -> None:
        for proc in list(self._procs.values()):
            with contextlib.suppress(Exception):
                proc.stop()
            proc.teardown_workspace()
        self._procs.clear()


_sessions = _SessionRegistry()


@dataclass
class ExecutorConfig:
    """User-supplied configuration for an :class:`Executor`. Identity (``host``, ``name``,
    ``local``) and connection state live on :class:`Executor` itself."""

    user: str = ""
    port: int | None = None
    local_port: int | None = None
    workspace: str | None = None
    bundle: Any = None
    plugins: list[str] | None = None
    copy: bool = False
    build: Any = None
    ready_timeout: float = 60.0
    sudo: bool = True
    sudo_password: str | None = None
    executor_bin: str | None = None
    triple: str | None = None
    env: dict[str, str] | None = None
    verbose: int = 1


class Executor:
    """A ``catalyst-executor`` you launch and talk to over TCP — local, remote, or an already-running
    one. Pass its :attr:`address` to ``target(address=...)``. Construction is inert; :meth:`launch`
    and :meth:`stop` are idempotent, and ``with Executor(...) as ex:`` launches on entry, stops on
    exit. Three modes:

    * ``local=True`` — local subprocess on ``127.0.0.1`` (no SSH); uses the shipped binary unless
      ``executor_bin=`` overrides.
    * ``host=<addr>`` — remote over forwarded SSH (``user``/``sudo``/``sudo_password`` as needed;
      ``copy=True`` + ``bundle=<dir>`` first scp's the bundle, cross-built via ``build=`` if given).
    * neither — carry ``address`` for an executor already running or tunnelled there.

    Other kwargs: ``name`` tags stream output as ``[<name>]`` and the per-launch log; ``plugins``
    are device backends / runtime_call libs; ``env`` extends the process environment; ``verbose``
    (0-3) sets launcher detail; ``triple`` overrides auto-detection (see :attr:`triple`);
    ``build(triple, bundle_dir)`` runs on every ``copy=True`` deploy — must be idempotent. Port is
    random unless pinned via ``port=``.
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
        self._cfg = ExecutorConfig(
            user=user,
            port=port,
            local_port=local_port,
            workspace=workspace,
            bundle=bundle,
            plugins=plugins,
            copy=copy,
            build=build,
            ready_timeout=ready_timeout,
            sudo=sudo,
            sudo_password=sudo_password,
            executor_bin=executor_bin,
            triple=triple,
            env=env,
            verbose=verbose,
        )
        self._proc: _ExecutorProcess | None = None
        self._launched = False

    @property
    def address(self) -> str:
        """The ``host:port`` for ``target(address=...)``. Raises :class:`RuntimeError` if not launched."""
        if not self._launched:
            raise RuntimeError("Executor not launched — call .launch() or use `with Executor(...)`")
        return self._address

    @cached_property
    def triple(self) -> str | None:
        """LLVM target triple for cross-compilation — explicit ``triple=`` or auto-detected via
        ``uname``. ``None`` if undetectable (compiler falls back to the host triple)."""
        return self._cfg.triple if self._cfg.triple is not None else self._detect_triple()

    def _detect_triple(self) -> str | None:
        """Auto-detect the LLVM target triple via ``uname`` — locally for ``local=True``, remotely
        over SSH for ``host=``. Returns ``None`` in attach-only mode or if the remote probe fails."""
        if self._local:
            return triple_from_uname(platform.system(), platform.machine())
        if self.host:
            user = self._cfg.user or getpass.getuser()
            if (out := SSH.capture(user, self.host.strip(), "uname -sm")) is None:
                return None
            system, _, machine = out.partition(" ")
            return triple_from_uname(system, machine)
        return None

    def _remote_target(self) -> tuple[str, str, str]:
        """Resolve ``(user, host, workspace)`` for a remote deploy or run, filling in defaults
        where the caller didn't — ``user`` from ``getpass.getuser()``, ``workspace`` from
        :meth:`Paths.default_workspace`. Remote only; asserts ``self.host`` is set."""
        assert self.host is not None
        host = self.host.strip()
        user = self._cfg.user or getpass.getuser()
        workspace = self._cfg.workspace or Paths.default_workspace()
        return user, host, workspace

    def _scp_bundle(self, user: str, host: str, workspace: str) -> None:
        """Cross-build (if ``build=``) then delegate to :meth:`SCP.bundle` when ``copy=True`` and
        a ``bundle=`` was supplied. No-op otherwise. The build recipe is called every deploy —
        must be idempotent."""
        if not (self._cfg.copy and self._cfg.bundle):
            return
        bundle = Path(self._cfg.bundle)
        if self._cfg.build is not None:
            self._cfg.build(self.triple, bundle)
        SCP.bundle(user, host, bundle, workspace)

    def _local_maker(self) -> Callable[[int], _ExecutorProcess]:
        """Return a ``make(port)`` closure that builds a :class:`_LocalProcess`. Config-derived
        values (executor binary, log path) are captured once so port-retries share them."""
        default_bin = Paths.default_executor_bin()
        log_path = Paths.resolve_log("localhost", name=self.name)
        ready_timeout = self._cfg.ready_timeout

        def make(port: int) -> _ExecutorProcess:
            return _LocalProcess(
                port=port,
                executor_bin=self._cfg.executor_bin or default_bin,
                plugins=self._cfg.plugins or [],
                env=self._cfg.env,
                ready_timeout=ready_timeout,
                name=self.name,
                log_path=log_path,
            )

        return make

    def _remote_maker(self) -> Callable[[int], _ExecutorProcess]:
        """Prep the remote deploy (resolve target, sudo, scp bundle) and return a ``make(port)``
        closure that builds a :class:`_RemoteProcess`. Deploy-once state is captured so
        port-retries reuse the same authenticated context — no re-prompt, no re-scp."""
        user, host, workspace = self._remote_target()
        ws_pinned = self._cfg.workspace is not None  # pinned dirs are left in place on teardown
        sudo_pw = SSH.resolve_sudo(user, host, self._cfg.sudo_password) if self._cfg.sudo else None
        self._scp_bundle(user, host, workspace)
        # copied bundle -> run it from the workspace (./); sudo's secure_path would miss a
        # bare name. Bare only when attaching to a remote that has it on PATH.
        default_bin = f"./{Paths.EXECUTOR_BIN}" if self._cfg.copy else Paths.EXECUTOR_BIN
        log_path = Paths.resolve_log(host, name=self.name)
        ready_timeout = self._cfg.ready_timeout

        def make(port: int) -> _ExecutorProcess:
            return _RemoteProcess(
                host=host,
                user=user,
                port=port,
                local_port=self._cfg.local_port,
                workspace=workspace,
                plugins=self._cfg.plugins or [],
                env=self._cfg.env,
                sudo=self._cfg.sudo,
                sudo_password=sudo_pw,
                executor_bin=self._cfg.executor_bin or default_bin,
                cleanup_ws=(not ws_pinned),
                ready_timeout=ready_timeout,
                name=self.name,
                log_path=log_path,
            )

        return make

    def launch(self) -> Self:
        """Deploy the executor and return ``self`` (idempotent, chainable). See the class docstring
        for the three modes."""
        # Short-circuit: already launched, or attach-only mode with nothing to deploy.
        if self._launched or not (self._local or self.host):
            self._launched = True
            return self
        set_verbose(self._cfg.verbose)
        make = self._local_maker() if self._local else self._remote_maker()
        self._proc = _start_on_free_port(make, self._cfg.port)
        self._address = self._proc.addr
        self._launched = True
        _sessions.register(self.name, self._proc)
        return self

    def stop(self) -> None:
        """Tear down the executor + tunnel and deregister from the atexit shutdown hook.
        Idempotent; no-op in attach-only mode. Auto-generated workspaces are removed; a pinned
        ``workspace=`` is left for :meth:`remove_workspace`. Subprocess errors during shutdown
        are swallowed (best-effort)."""
        self._launched = False
        if self._proc is None:
            return
        with contextlib.suppress(Exception):
            self._proc.stop()
        self._proc.teardown_workspace()
        _sessions.unregister(self.name)
        self._proc = None

    def setup_workspace(self) -> Self:
        """Remote only. Deploy the bundle to a persistent workspace *without* starting the
        executor. Requires ``host``, pinned ``workspace=``, and ``bundle``. Idempotent — later
        ``launch()`` on this instance or a fresh ``Executor(..., workspace=<same>)`` reuses it
        (``copy`` defaults off). Delete via :meth:`remove_workspace`. Copies as the login user
        (no sudo). Returns ``self`` for chaining; raises :class:`ValueError` if any required
        arg is missing."""
        if not self.host:
            raise ValueError("setup_workspace() needs a remote host= (nothing to deploy locally)")
        if self._cfg.workspace is None:
            raise ValueError(
                "setup_workspace() needs a pinned workspace= so later runs can reuse it"
            )
        if not self._cfg.bundle:
            raise ValueError("setup_workspace() needs a bundle= to deploy")
        set_verbose(self._cfg.verbose)
        user, host, workspace = self._remote_target()
        bundle = Path(self._cfg.bundle)
        if self._cfg.build is not None:
            self._cfg.build(self.triple, bundle)  # idempotent recipe (see build=); may cross-build
        SCP.bundle(user, host, bundle, workspace)
        self._cfg.copy = False  # bundle is deployed; launch() on this instance won't re-copy
        return self

    def remove_workspace(self, force: bool = False) -> None:
        """Remote only. Delete a pinned workspace (dir + bundle) — explicit teardown for a
        persistent workspace, never auto-removed. Refuses to delete ``/`` or ``$HOME``.
        ``force=True`` re-raises SSH errors instead of swallowing (the safety refusal always
        raises). Raises :class:`ValueError` on missing ``host``/``workspace``."""
        if not self.host:
            raise ValueError("remove_workspace() needs a remote host=")
        if self._cfg.workspace is None:
            raise ValueError("remove_workspace() needs a pinned workspace= to remove")
        set_verbose(self._cfg.verbose)
        user, host, workspace = self._remote_target()
        SSH.rmdir(user, host, workspace, force=force)

    def __enter__(self) -> Self:
        return self.launch()

    def __exit__(self, *exc) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (
            f"Executor(name={self.name!r}, host={self.host!r}, local={self._local}, "
            f"launched={self._launched}, address={self._address!r})"
        )
