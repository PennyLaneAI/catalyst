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

"""The :class:`Executor` launches a ``catalyst-executor`` process and exposes the address it serves.

Compiled programs are dispatched to that address to run out-of-process.

Construction is inert; :meth:`Executor.launch` deploys, :meth:`Executor.stop` tears down, and it
works as a context manager::

    from catalyst import Executor

    # local subprocess on 127.0.0.1 (no SSH):
    with Executor(plugins=[...]) as ex:
        print(ex.address)               # dispatch compiled programs here

    # remote over forwarded SSH:
    ex = Executor(host="10.0.0.9", user="me", plugins=[...]).launch()
    print(ex.address)                   # ... ex.stop() (also at process exit)

    # attach to an externally managed executor (address required, no default):
    ex = Executor("127.0.0.1:1234").launch()

    # persistent remote workspace:
    Executor(host="10.0.0.9", workspace="~/cat-ws", deploy=["..."]).setup_workspace()
    Executor(host="10.0.0.9", workspace="~/cat-ws").launch()
    Executor(host="10.0.0.9", workspace="~/cat-ws").remove_workspace()
"""

from __future__ import annotations

import atexit
import contextlib
import getpass
import platform
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Callable, Self

from .process import _ExecutorProcess, _LocalProcess, _RemoteProcess
from .ssh import SCP, RemoteOps
from .utils import (
    MAX_PORT_TRIES,
    ExecutorPaths,
    random_port,
    set_verbose,
    triple_from_uname,
)


def _teardown_process_and_workspace(proc: _ExecutorProcess) -> None:
    """Tear down a process that never became the live executor, its generated workspace included.
    Only a started process is registered for shutdown at exit, so nothing else would reach this
    one."""
    with contextlib.suppress(Exception):
        proc.stop()
    with contextlib.suppress(Exception):
        proc.teardown_workspace()


def _start_on_free_port(
    make_process: Callable[[int], _ExecutorProcess],
    pinned_port: int | None,
    max_tries: int = MAX_PORT_TRIES,
    strict: bool = False,
) -> _ExecutorProcess:
    """Start a process on a free port. Tries ``pinned_port`` first, then up to ``max_tries``
    random ports; retries while the executor reports the port already bound.

    A pinned port that was busy silently falls back to a random one; treat
    :attr:`Executor.address` as authoritative. ``strict`` (which requires ``pinned_port``) tries
    that port alone, for an executor whose address was published before it launched.

    Raises:
        RuntimeError: If no port bound after all attempts. Any other launch failure propagates
            unchanged.
    """
    if strict:
        assert pinned_port is not None, "strict needs a pinned port"
        ports_to_try = [pinned_port]
    else:
        ports_to_try = ([pinned_port] if pinned_port is not None else []) + [
            random_port() for _ in range(max_tries)
        ]
    last = ""
    proc: _ExecutorProcess | None = None
    for port in ports_to_try:
        proc = make_process(port)
        try:
            return proc.start()
        except BaseException as e:  # BaseException: an interrupted deploy needs teardown too
            if not proc.port_conflict:
                _teardown_process_and_workspace(proc)  # another port would fail the same way
                raise
            last = str(e)
            proc._log_message(f"port {port} busy, trying another")
            with contextlib.suppress(Exception):
                proc.stop()  # workspace left in place: the next attempt reuses it
    if proc is not None:
        _teardown_process_and_workspace(proc)
    raise RuntimeError(
        f"pinned executor port {pinned_port} is already in use"
        if strict
        else f"no free executor port after {len(ports_to_try)} tries ({last}). Pin one with port=."
    )


class _SessionRegistry:
    """Process-wide registry of launched executors. Registers an :mod:`atexit` shutdown hook so
    nothing leaks when Python exits without an explicit :meth:`Executor.stop`."""

    def __init__(self) -> None:
        self._procs: list[_ExecutorProcess] = []
        atexit.register(self._shutdown_all)

    def register(self, proc: _ExecutorProcess) -> None:
        self._procs.append(proc)

    def unregister(self, proc: _ExecutorProcess) -> None:
        if proc in self._procs:
            self._procs.remove(proc)

    def _shutdown_all(self) -> None:
        for proc in list(self._procs):
            with contextlib.suppress(Exception):
                proc.stop()
            with contextlib.suppress(Exception):
                proc.teardown_workspace()
        self._procs.clear()


_sessions = _SessionRegistry()


class _Mode(Enum):
    """How an :class:`Executor` reaches the process it addresses. What the caller gives picks one,
    so there is no combination that names none."""

    LOCAL = "local"
    """Neither host nor address: spawn a subprocess here."""

    REMOTE = "remote"
    """``host=``: ssh to it, deploy, and tunnel back."""

    ATTACHED = "attached"
    """``address=`` alone: something else launched it, so there is nothing to deploy."""


@dataclass
class ExecutorConfig:
    """User-supplied configuration for an :class:`Executor`. The deployment fields apply to a remote
    executor only; a local subprocess runs in the current directory."""

    user: str = ""
    """SSH account on the remote host. Empty means the local username."""

    port: int | None = None
    """Port the executor binds on the target, and the local end of the tunnel reaching it.
    ``ssh -L`` uses the number at both ends, so two executors sharing one collide on this machine.
    ``None`` picks a free one at launch, so the address is only known afterwards."""

    workspace: str | None = None
    """Directory on the target the executor runs in. ``None`` generates a ``catalyst-exec-*`` one
    per launch and removes it on teardown; a directory named here is left in place."""

    plugins: list[str] | None = None
    """Shared libraries the executor ``dlopen``s at startup, in this order. A bare filename resolves
    against the workspace; ``~`` and absolute paths are taken as given. They share the global
    namespace, so the first definition of a symbol wins."""

    ready_timeout: float = 60.0
    """Seconds to wait for the executor to report that it bound its port."""

    sudo: bool = False
    """Run the executor as root, for a target whose devices are not world-accessible."""

    sudo_password: str | None = None
    """Password piped to ``sudo -S``; unneeded with passwordless sudo. Supplying one drops the SSH
    TTY, so closing the session no longer signals the executor and teardown falls to ``pkill``."""

    executor_bin: str | None = None
    """Command that starts the executor, for wrapping it in something like ``numactl``. Defaults to
    the workspace binary when the workspace holds one (:attr:`copy`, or a pinned :attr:`workspace`),
    else the bare name from ``PATH``."""

    triple: str | None = None
    """LLVM target triple to cross-compile this node's code for. ``None`` detects it from the
    target's ``uname``, which requires reaching the host."""

    env: dict[str, str] | None = None
    """Environment for the executor process. Applied after any privilege change, since ``sudo`` is
    setuid and the dynamic linker drops ``LD_*`` across it."""

    verbose: int = 1
    """How much launcher narration to log: ``0`` quiet, ``1`` normal, ``2`` per-command detail."""

    deploy: list[str | Path] | None = None
    """What to place in the workspace before the executor starts: directories, files, or both.
    A directory contributes the files inside it, which is how a cross-built set of artifacts
    travels; a file is copied as itself."""


class Executor:
    """A ``catalyst-executor`` process, addressable over TCP at :attr:`address`. Construction is
    inert; :meth:`launch` / :meth:`stop` are idempotent, and ``with Executor(...) as ex:``
    launches on entry and stops on exit.

    Three modes:

    * ``host=<addr>``: remote over forwarded SSH. ``deploy=[...]`` first scp's those
      directories and files into the workspace.
    * ``address``: attach to an executor whose lifetime is managed elsewhere.
    * neither: subprocess on ``127.0.0.1`` (no SSH).
    """

    def __init__(
        self,
        address: str | None = None,
        *,
        host: str | None = None,
        user: str = "",
        port: int | None = None,
        workspace: str | None = None,
        plugins: list[str] | None = None,
        deploy: list[str | Path] | None = None,
        ready_timeout: float = 60.0,
        name: str = "executor",
        sudo: bool = False,
        sudo_password: str | None = None,
        executor_bin: str | None = None,
        triple: str | None = None,
        env: dict[str, str] | None = None,
        verbose: int = 1,
    ):
        self.host = host
        self.name = name
        self._address = address
        self._mode = _Mode.REMOTE if host else _Mode.ATTACHED if address else _Mode.LOCAL
        self._cfg = ExecutorConfig(
            user=user,
            port=port,
            workspace=workspace,
            plugins=plugins,
            deploy=deploy,
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
        self._committed = False

    @property
    def address(self) -> str:
        """The ``host:port`` the executor serves on. Raises :class:`RuntimeError` if it is neither
        launched nor committed to an address by :meth:`resolve`."""
        if not (self._launched or self._committed):
            raise RuntimeError("Executor not launched; call .launch() or use `with Executor(...)`")
        return self._address

    def resolve(self) -> Executor | None:
        """Commit to the address this executor will serve on, without deploying it.

        Compiled programs carry their executor's address, so it has to be known before the executor
        runs; committing to it here lets the deployment wait until execution. Only possible when the
        address is predictable: an attached executor already has one, and a deployed one needs
        ``port=`` pinned, since a free-port search settles the address only at launch.

        Returns:
            Executor | None: ``self`` if an address could be committed to, ``None`` if it is only
            knowable by launching and the caller should do that.
        """
        if self._launched or self._committed:
            return self
        if self._mode is _Mode.ATTACHED:  # the address came from the caller
            self._committed = True
            return self
        if self._cfg.port is None:
            return None
        # Local or remote, a deployed executor is reached on loopback -- directly, or at the ssh
        # tunnel's local end -- which is the address both process classes report.
        self._address = f"{_ExecutorProcess.LOCALHOST}:{self._cfg.port}"
        self._committed = True
        return self

    @cached_property
    def triple(self) -> str | None:
        """LLVM target triple: explicit ``triple=`` if given, else auto-detected via ``uname``.
        ``None`` when undetectable (the compiler falls back to the host triple)."""
        return self._cfg.triple if self._cfg.triple is not None else self._detect_triple()

    def _detect_triple(self) -> str | None:
        """Auto-detect the LLVM triple via ``uname``: local for a subprocess, remote over SSH
        for ``host=``. Returns ``None`` in attach-only mode or if the remote probe fails."""
        if self._mode is _Mode.LOCAL:
            return triple_from_uname(platform.system(), platform.machine())
        if self.host:
            user = self._cfg.user or getpass.getuser()
            if (out := RemoteOps.capture(user, self.host.strip(), "uname -sm")) is None:
                return None
            system, _, machine = out.partition(" ")
            return triple_from_uname(system, machine)
        return None

    def _remote_target(self) -> tuple[str, str, str]:
        """Resolve ``(user, host, workspace)`` for a remote op. Defaults: ``user`` from
        ``getpass.getuser()``, ``workspace`` from :meth:`ExecutorPaths.default_workspace`."""
        assert self.host is not None
        host = self.host.strip()
        user = self._cfg.user or getpass.getuser()
        workspace = self._cfg.workspace or ExecutorPaths.default_workspace()
        return user, host, workspace

    def _deploy_sources(self, user: str, host: str, workspace: str) -> None:
        """:meth:`SCP.deploy` everything :attr:`ExecutorConfig.deploy` names; no-op if it names
        nothing."""
        sources = [Path(src) for src in (self._cfg.deploy or [])]
        if sources:
            SCP.deploy(user, host, sources, workspace)

    def _local_maker(self) -> Callable[[int], _ExecutorProcess]:
        """``make(port) -> _LocalProcess`` closure. Config-derived values are captured once so
        port retries share them."""
        default_bin = ExecutorPaths.default_executor_bin()
        log_path = ExecutorPaths.resolve_log("localhost", name=self.name)
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
        """``make(port) -> _RemoteProcess`` closure. Runs the one-time prep (sudo resolve, scp)
        so port retries reuse the same auth context with no re-prompt or re-scp."""
        user, host, workspace = self._remote_target()
        ws_pinned = self._cfg.workspace is not None  # pinned dirs are left in place on teardown
        sudo_pw = (
            RemoteOps.resolve_sudo(user, host, self._cfg.sudo_password) if self._cfg.sudo else None
        )
        RemoteOps.mkdir(user, host, workspace)  # the launch command cd's into it
        self._deploy_sources(user, host, workspace)
        # A copied or pinned workspace holds the binary, so run it from there; sudo's secure_path
        # would miss a bare name. Bare name only when attaching to a remote that has it on PATH.
        ws_holds_bin = bool(self._cfg.deploy) or ws_pinned
        default_bin = (
            f"./{ExecutorPaths.EXECUTOR_BIN}" if ws_holds_bin else ExecutorPaths.EXECUTOR_BIN
        )
        log_path = ExecutorPaths.resolve_log(host, name=self.name)
        ready_timeout = self._cfg.ready_timeout

        def make(port: int) -> _ExecutorProcess:
            return _RemoteProcess(
                host=host,
                user=user,
                port=port,
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
        """Deploy the executor and return ``self``. Idempotent, chainable. See the class docstring
        for the three modes."""
        if self._launched:
            return self
        # Attach-only mode: nothing to deploy, just carry the address the caller supplied.
        if self._mode is _Mode.ATTACHED:
            self._launched = True
            return self
        set_verbose(self._cfg.verbose)
        make = self._local_maker() if self._mode is _Mode.LOCAL else self._remote_maker()
        self._proc = _start_on_free_port(make, self._cfg.port, strict=self._committed)
        self._address = self._proc.addr
        self._launched = True
        _sessions.register(self._proc)
        return self

    def stop(self) -> None:
        """Tear down the executor + tunnel and deregister from the atexit hook. Idempotent.
        Auto-generated workspaces are removed; a pinned ``workspace=`` stays for
        :meth:`remove_workspace`. Shutdown errors are swallowed (best-effort)."""
        self._launched = False
        if self._proc is None:
            return
        with contextlib.suppress(Exception):
            self._proc.stop()
        with contextlib.suppress(Exception):
            self._proc.teardown_workspace()
        _sessions.unregister(self._proc)
        self._proc = None

    def setup_workspace(self) -> Self:
        """Remote only. Deploy the bundle to a persistent workspace without starting the executor.

        Requires ``host``, pinned ``workspace=``, and ``deploy=``. Idempotent; a later
        :meth:`launch` (or a fresh ``Executor(..., workspace=<same>)``) reuses it. Delete via
        :meth:`remove_workspace`.

        Raises:
            ValueError: If ``host``, ``workspace``, or ``deploy`` is missing.
        """
        if not self.host:
            raise ValueError("setup_workspace() requires host=")
        if self._cfg.workspace is None:
            raise ValueError("setup_workspace() requires a pinned workspace=")
        if not self._cfg.deploy:
            raise ValueError("setup_workspace() requires deploy=")
        set_verbose(self._cfg.verbose)
        user, host, workspace = self._remote_target()
        self._deploy_sources(user, host, workspace)
        self._cfg.deploy = []  # already placed; launch() on this instance won't copy again
        return self

    def remove_workspace(self, force: bool = False) -> None:
        """Remote only. Delete a pinned workspace. Refuses to delete ``/`` or ``$HOME``.

        ``force=True`` re-raises SSH errors; default swallows them. The safety refusal always
        raises.

        Raises:
            ValueError: If ``host`` or ``workspace`` is missing.
        """
        if not self.host:
            raise ValueError("remove_workspace() requires host=")
        if self._cfg.workspace is None:
            raise ValueError("remove_workspace() requires a pinned workspace=")
        set_verbose(self._cfg.verbose)
        user, host, workspace = self._remote_target()
        RemoteOps.rmdir(user, host, workspace, force=force)

    def __enter__(self) -> Self:
        return self.launch()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (
            f"Executor(name={self.name!r}, host={self.host!r}, mode={self._mode.value}, "
            f"launched={self._launched}, address={self._address!r})"
        )
