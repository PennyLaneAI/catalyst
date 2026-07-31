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

Grouped into namespace classes so related items sit together:

* :class:`Patterns` — regexes for scanning executor / ssh output (ready, auth prompt, port busy).
* :class:`Paths` — default workspace, executor binary, and log-file path resolution.
* :class:`ShellCommand` — generic POSIX shell fragments (``pkill``, ``sudo``, ``rm -rf``,
  ``mkdir -p``) and safe path quoting.
* :class:`ExecutorCli` — CLI flag constants for the ``catalyst-executor`` binary.

Plus stdlib logging setup — :data:`logger` (a :class:`logging.Logger` under
``catalyst.executor``), :func:`set_verbose`, :func:`verbose_level`, and :func:`log_cmd`.
And a few free helpers (:func:`random_port`, :func:`triple_from_uname`, :data:`pdeathsig`) and
domain types (:class:`PortInUse`, :class:`Raw`).
Imported by :mod:`.ssh`, :mod:`.process`, and :mod:`.manager`."""

from __future__ import annotations

import contextlib
import ctypes
import faulthandler
import getpass
import logging
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
MAX_PORT_TRIES = 6


class PortInUse(Exception):
    """The chosen executor port was already taken (likely another user on the host)."""


class Raw(str):
    """Marker for a string that must NOT be shell-quoted when embedded into a remote command.

    Wrap values in :attr:`~catalyst.Executor.env` (or a plugin path) that need ``$VAR`` to expand
    on the remote — bare :class:`str` values are shell-quoted so metacharacters can't break out.

    Example::

        Executor(host="h", env={"LD_LIBRARY_PATH": Raw("$HOME/lib")}, plugins=[Raw("$LIBDIR/x.so")])
    """


class Patterns:
    """Regex-based classifiers for lines of executor / ssh output. Callers use the ``is_*``
    predicates — the raw regexes are implementation detail."""

    # Executor stderr lines that mean "bound and accepting". The first launch prints "Listening
    # on <h>:<p>"; "executor ready, ..." only recurs after a client disconnects.
    _READY = re.compile(r"Listening on \S+:\d+|executor ready, waiting for next connection")
    # ssh login still wanting a password/passphrase means key auth isn't set up — we can't feed it.
    _SSH_PW = re.compile(r"'s password:|Enter passphrase for key")
    # sudo telling us the password we fed (via sudo -S) was wrong.
    _SUDO_FAIL = re.compile(
        r"Sorry, try again|incorrect password|authentication failure|sudo: \d+ incorrect"
    )
    # A port collision — the remote bind or the local -L forward is already taken.
    _PORT = re.compile(r"Address already in use|Could not request local forwarding")

    @staticmethod
    def is_ready(line: str) -> bool:
        """True if ``line`` signals the executor has bound its port and is accepting."""
        return bool(Patterns._READY.search(line))

    @staticmethod
    def is_port_conflict(line: str) -> bool:
        """True if ``line`` signals a port bind failure (remote bind or local ``-L`` forward)."""
        return bool(Patterns._PORT.search(line))

    @staticmethod
    def is_ssh_prompt(line: str) -> bool:
        """True if ``line`` is an SSH password/passphrase prompt — key auth isn't set up."""
        return bool(Patterns._SSH_PW.search(line))

    @staticmethod
    def is_sudo_fail(line: str) -> bool:
        """True if ``line`` is a ``sudo`` password-rejection message."""
        return bool(Patterns._SUDO_FAIL.search(line))


class ShellCommand:
    """Generic POSIX shell fragments and path helpers reused by remote ops — sudo/kill/rm
    building blocks and safe path quoting. Nothing here is catalyst-executor-specific;
    everything runs on any Bourne-family shell."""

    @staticmethod
    def sudo_probe() -> str:
        """Non-interactive sudo check: ``sudo -n true`` returns 0 iff sudo needs no password.
        Stderr is redirected to ``/dev/null`` to silence the "a password is required" message
        when it does."""
        return "sudo -n true 2>/dev/null"

    @staticmethod
    def sudo_pw(cmd: str) -> str:
        """Wrap ``cmd`` in ``sudo -S -p ''`` — sudo reads its password from stdin (the caller
        must pipe it via ``input=``) with an empty prompt so it never leaks to the terminal.
        ``cmd`` is a pre-built shell fragment and is inserted verbatim (no re-quoting)."""
        return f"sudo -S -p '' {cmd}"

    @staticmethod
    def sudo_np(cmd: str) -> str:
        """Wrap ``cmd`` in ``sudo -n`` — non-interactive: sudo fails immediately if a password
        is required (NOPASSWD only). ``cmd`` is inserted verbatim (no re-quoting)."""
        return f"sudo -n {cmd}"

    @staticmethod
    def pkill(pat: str) -> str:
        """Shell command to kill any process whose ``ps``-visible argv matches ``pat`` (a regex,
        per ``pkill -f``). ``pat`` is shell-quoted so metacharacters can't break out."""
        return f"pkill -f {shlex.quote(pat)}"

    @staticmethod
    def rm_rf(path: str) -> str:
        """Shell command to recursively remove ``path``, quoted via :meth:`path` so spaces and
        ``~`` survive. Callers are responsible for whatever safety-gating the path deserves —
        this helper only handles quoting."""
        return f"rm -rf {ShellCommand.path(path)}"

    @staticmethod
    def mkdir_p(path: str) -> str:
        """Shell command to create ``path`` (and any missing parents) on the remote, with the
        path itself quoted via :meth:`path` so spaces and ``~`` survive."""
        return f"mkdir -p {ShellCommand.path(path)}"

    @staticmethod
    def path(path: str) -> str:
        """Shell expression for ``path`` that expands a leading ``~`` via ``$HOME`` and quotes the
        rest, so it survives ``cd``/``rm`` without tilde-in-quotes breakage or word-splitting."""
        if path == "~":
            return '"$HOME"'
        if path.startswith("~/"):
            return '"$HOME"/' + shlex.quote(path[2:])
        return shlex.quote(path)


class ExecutorCli:
    """CLI interface for the ``catalyst-executor`` binary — flag constants used to invoke it,
    local or remote. Kept as class data so a rename on the executor side is a one-line change
    here instead of hunting through f-strings."""

    PLUGIN_FLAG = "--plugin="
    BIND_FLAG = "--bind="


# --- logging -----------------------------------------------------------------------------------
# Uses stdlib :mod:`logging` — matches the rest of catalyst (compiler/jit/etc. all do
# ``getLogger(__name__)``). Users can attach their own handlers / filters via
# ``logging.getLogger("catalyst.executor")``.
#
# Verbosity levels (set via :func:`set_verbose` or ``Executor(verbose=)``):
#   0 quiet — WARNING only (errors)
#   1 default — INFO (phases, ready/stop, executor stdout stream)
#   2 verbose — DEBUG (full ssh/scp commands, per-step timings)
#   3+ trace — DEBUG plus extra ``ssh -v`` flags on the wire
logger = logging.getLogger("catalyst.executor")
logger.setLevel(logging.INFO)
_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setFormatter(logging.Formatter("[remote-exec] %(message)s"))
logger.addHandler(_stderr_handler)
logger.propagate = False  # avoid double-logging when a caller configures the root logger

_verbose = 1  # 0-3, for external tool verbosity (``ssh -v`` flag count, ``scp -v`` vs ``-q``)


def set_verbose(level: int) -> None:
    """Set launcher output verbosity (also settable per launch via ``Executor(verbose=)``).

    Maps to :mod:`logging` levels: 0 → WARNING, 1 → INFO, ≥2 → DEBUG. Higher values (3+) still
    map to DEBUG but bump the ``ssh -v`` flag count and enable ``scp -v``."""
    global _verbose
    _verbose = level
    if level <= 0:
        logger.setLevel(logging.WARNING)
    elif level == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)


def verbose_level() -> int:
    """Current numeric verbosity (0-3), for external-tool flag decisions (``ssh -v``, ``scp -v``)."""
    return _verbose


def log_cmd(argv: list[str]) -> None:
    """DEBUG-level echo of a command we're about to run."""
    logger.debug("$ %s", " ".join(shlex.quote(c) for c in argv))


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
