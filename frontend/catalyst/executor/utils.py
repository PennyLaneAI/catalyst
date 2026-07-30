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

"""Shared low-level helpers for the :class:`~catalyst.executor.Executor`.

Grouped into three namespace classes so related items sit together:

* :class:`Log` — verbosity level + tagged stderr logging (single source of truth for the level).
* :class:`Patterns` — regexes for scanning executor / ssh output (ready, auth prompt, port busy).
* :class:`Paths` — default workspace, executor binary, and log-file path resolution.

Plus a few free helpers (:func:`random_port`, :func:`triple_from_uname`, :data:`pdeathsig`) and
the domain types (:class:`PortInUse`, :class:`SSHError`, :class:`Raw`).
Imported by :mod:`.ssh`, :mod:`.process`, and :mod:`.manager`."""

from __future__ import annotations

import contextlib
import ctypes
import faulthandler
import getpass
import os
import random
import re
import shlex
import signal
import sys
import time
from pathlib import Path

from catalyst.utils.runtime_environment import get_lib_path

# A segfault in a loaded plugin (e.g. a device lib reacting to a remote abort) otherwise core-dumps
# silently — this prints a C-level traceback instead.
with contextlib.suppress(Exception):
    faulthandler.enable()

# Random bind port so concurrent launches on a shared host don't collide; retried on conflict.
PORT_TRIES = 6


class PortInUse(Exception):
    """The chosen executor port was already taken (likely another user on the host)."""


class SSHError(RuntimeError):
    """A remote SSH/scp/sudo operation failed (auth, bind, transport, or command exit).

    A domain exception (not :class:`SystemExit`) so callers can catch and recover — e.g. try a
    different host, or fall back to a local executor."""


class Raw(str):
    """Marker for a string that must NOT be shell-quoted when embedded into a remote command.

    Wrap values in :attr:`~catalyst.Executor.env` (or a plugin path) that need ``$VAR`` to expand
    on the remote — bare :class:`str` values are shell-quoted so metacharacters can't break out.

    Example::

        Executor(host="h", env={"LD_LIBRARY_PATH": Raw("$HOME/lib")}, plugins=[Raw("$LIBDIR/x.so")])
    """


class Patterns:
    """Regexes for scanning executor / ssh output. Grouped so a reader sees at a glance the full
    set of output-matching rules the launcher relies on."""

    # Executor stderr lines that mean "bound and accepting". The first launch prints "Listening on
    # <h>:<p>"; "executor ready, ..." only recurs after a client disconnects.
    READY = re.compile(r"Listening on \S+:\d+|executor ready, waiting for next connection")
    # ssh login still wanting a password/passphrase means key auth isn't set up — we can't feed it.
    SSH_PW = re.compile(r"'s password:|Enter passphrase for key")
    # sudo telling us the password we fed (via sudo -S) was wrong.
    SUDO_FAIL = re.compile(
        r"Sorry, try again|incorrect password|authentication failure|sudo: \d+ incorrect"
    )
    # A port collision — the remote bind or the local -L forward is already taken.
    PORT = re.compile(r"Address already in use|Could not request local forwarding")


class Log:
    """Verbosity-gated stderr logger for the launcher. Single source of truth for the level so
    submodules always read the live value (a plain ``from utils import _VERBOSITY`` would snapshot
    the int at import time).

    Levels: 0 quiet (errors only) · 1 phases + executor stream (default) · 2 full ssh/scp commands
    + scp -v + timings · 3+ adds ``ssh -v``. The executor's own output always streams regardless."""

    _level: int = 1

    @classmethod
    def set_level(cls, level: int) -> None:
        """Set the launcher's output verbosity (also settable per launch via ``Executor(verbose=)``)."""
        cls._level = level

    @classmethod
    def level(cls) -> int:
        """The current verbosity."""
        return cls._level

    @classmethod
    def info(cls, msg: str, level: int = 1) -> None:
        """Print ``msg`` to stderr tagged ``[remote-exec]`` when verbosity is at least ``level``.

        ``level`` defaults to 1 (the main narrative — phases, ready/stop). Pass ``level=2`` for
        verbose-only detail (full commands, per-step timings)."""
        if level <= cls._level:
            print(f"[remote-exec] {msg}", file=sys.stderr, flush=True)

    @classmethod
    def cmd(cls, argv: list[str]) -> None:
        """Echo a command we're about to run (verbosity >= 2)."""
        cls.info("$ " + " ".join(shlex.quote(c) for c in argv), level=2)


class Paths:
    """Default paths — workspace, executor binary, per-launch log file — resolved from the current
    environment (installed wheel vs source build, login user, host, timestamp)."""

    # Workspace-name prefix — used both to build a fresh default workspace and (by
    # `_RemoteProcess.teardown_workspace`) to gate the `rm -rf` cleanup, so a user-pinned dir
    # (without this prefix) can never be wiped.
    WORKSPACE_PREFIX = "catalyst-exec-"

    # The catalyst-executor binary name — single source of truth for the filename we look for in
    # the runtime lib dir, embed in log filenames, and refer to as `./{EXECUTOR_BIN}` when the
    # binary sits in a scp'd workspace.
    EXECUTOR_BIN = "catalyst-executor"

    # Filesystem-safe but clearly separated (no bare run-together digits): 2026-06-30_04-48-15.
    _TS_FMT = "%Y-%m-%d_%H-%M-%S"

    @staticmethod
    def _timestamp() -> str:
        """Filesystem-safe timestamp used to tag workspace names and log files so runs don't
        overwrite each other."""
        return time.strftime(Paths._TS_FMT, time.localtime())

    @staticmethod
    def _random_suffix() -> str:
        """A random 12-bit hex tag (3 chars, 4096 possibilities) so two launches within the same
        second don't collide on the workspace name."""
        return f"{random.randint(0, 0xfff):03x}"

    @staticmethod
    def default_workspace() -> str:
        """Per-run remote dir so concurrent launches on a shared account don't clobber each other.
        The :attr:`WORKSPACE_PREFIX` also gates the ``rm -rf`` cleanup (safety)."""
        return (
            f"~/{Paths.WORKSPACE_PREFIX}{getpass.getuser()}-"
            f"{Paths._timestamp()}-{Paths._random_suffix()}"
        )

    @staticmethod
    def default_executor_bin() -> str:
        """Locate the shipped/built ``catalyst-executor`` binary (for ``local=True``).

        In an installed wheel it sits in the packaged lib dir; in a source build it is under the
        runtime build's ``remote/`` subdir. Falls back to the name on ``PATH``."""
        rt_lib = Path(get_lib_path("runtime", "RUNTIME_LIB_DIR"))
        for candidate in (rt_lib / Paths.EXECUTOR_BIN, rt_lib / "remote" / Paths.EXECUTOR_BIN):
            if candidate.exists():
                return str(candidate)
        return Paths.EXECUTOR_BIN

    @staticmethod
    def resolve_log(
        host: str, explicit: str | None = None, disabled: bool = False, name: str = "executor"
    ) -> str | None:
        """Host-side log file for an executor's output — one per launch in the cwd, named by the
        executor (so several executors each get their own file). ``explicit`` pins a path;
        ``disabled`` turns it off."""
        if disabled:
            return None
        if explicit:
            return explicit
        tag = "" if name == "executor" else f"-{name}"
        return f"{Paths.EXECUTOR_BIN}{tag}-{host}-{Paths._timestamp()}.log"


def random_port() -> int:
    """A random ephemeral-range port to try binding on, so concurrent launches on a shared host
    seldom collide."""
    return random.randint(20000, 59999)


def triple_from_uname(system: str, machine: str) -> str | None:
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


def _set_pdeathsig() -> None:
    """preexec_fn: ask the kernel to SIGTERM this child when the parent (python) dies — so a host
    crash (segfault/SIGKILL, which skip atexit) doesn't leak the ssh tunnel + executor."""
    with contextlib.suppress(Exception):
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG=1


# preexec_fn only exists on POSIX; harmless no-op reference elsewhere. Lowercase because it is a
# callable-or-None reference, not an immutable constant.
pdeathsig = _set_pdeathsig if hasattr(os, "fork") else None
