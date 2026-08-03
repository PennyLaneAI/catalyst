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

"""Shared helpers for the :class:`~catalyst.executor.Executor`.

* :class:`Patterns`: regexes for scanning executor/ssh output.
* :class:`Paths`: workspace, binary, and log-file path resolvers.
* :class:`ShellCommand`: POSIX shell fragments and path quoting.
* :class:`ExecutorCli`: CLI flag constants for ``catalyst-executor``.

Plus stdlib logging (:data:`logger`, :func:`set_verbose`, :func:`verbose_level`, :func:`log_cmd`),
free helpers (:func:`random_port`, :func:`triple_from_uname`, :data:`pdeathsig`), and domain
types (:class:`PortInUse`, :class:`Raw`).
"""

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

# Print a C-level traceback on a plugin segfault instead of a silent core dump.
with contextlib.suppress(Exception):
    faulthandler.enable()

# Random bind-port retries on collision.
MAX_PORT_TRIES = 6


class PortInUse(Exception):
    """The chosen executor port was already taken."""


class Raw(str):
    """Marker for a string that must not be shell-quoted when embedded into a remote command.

    Use for :attr:`~catalyst.Executor.env` values or plugin paths that need ``$VAR`` to expand
    on the remote. Bare :class:`str` values are shell-quoted.

    Example::

        Executor(host="h", env={"LD_LIBRARY_PATH": Raw("$HOME/lib")}, plugins=[Raw("$LIBDIR/x.so")])
    """


class Patterns:
    """Regex classifiers for executor/ssh output lines. Use the ``is_*`` predicates."""

    # "bound and accepting". First launch prints "Listening on <h>:<p>"; the other recurs after
    # a client disconnects.
    _READY = re.compile(r"Listening on \S+:\d+|executor ready, waiting for next connection")
    # ssh still asking for a password means key auth isn't set up.
    _SSH_PW = re.compile(r"'s password:|Enter passphrase for key")
    # sudo rejecting the password piped via sudo -S.
    _SUDO_FAIL = re.compile(
        r"Sorry, try again|incorrect password|authentication failure|sudo: \d+ incorrect"
    )
    # Port collision on the remote bind or the local -L forward.
    _PORT = re.compile(r"Address already in use|Could not request local forwarding")

    @staticmethod
    def is_ready(line: str) -> bool:
        """True if ``line`` signals the executor bound its port."""
        return bool(Patterns._READY.search(line))

    @staticmethod
    def is_port_conflict(line: str) -> bool:
        """True if ``line`` signals a port bind failure."""
        return bool(Patterns._PORT.search(line))

    @staticmethod
    def is_ssh_prompt(line: str) -> bool:
        """True if ``line`` is an SSH password/passphrase prompt."""
        return bool(Patterns._SSH_PW.search(line))

    @staticmethod
    def is_sudo_fail(line: str) -> bool:
        """True if ``line`` is a sudo password rejection."""
        return bool(Patterns._SUDO_FAIL.search(line))


class ShellCommand:
    """Generic POSIX shell fragments and path quoting for remote ops."""

    @staticmethod
    def sudo_probe() -> str:
        """``sudo -n true`` — exits 0 iff sudo needs no password. Stderr silenced."""
        return "sudo -n true 2>/dev/null"

    @staticmethod
    def sudo_pw(cmd: str) -> str:
        """Wrap ``cmd`` in ``sudo -S -p ''``. Caller pipes the password via ``input=``."""
        return f"sudo -S -p '' {cmd}"

    @staticmethod
    def sudo_np(cmd: str) -> str:
        """Wrap ``cmd`` in ``sudo -n`` (NOPASSWD only; fails if a password is required)."""
        return f"sudo -n {cmd}"

    @staticmethod
    def pkill(pat: str) -> str:
        """``pkill -f <pat>``. ``pat`` is shell-quoted."""
        return f"pkill -f {shlex.quote(pat)}"

    @staticmethod
    def rm_rf(path: str) -> str:
        """``rm -rf <path>``. Caller is responsible for safety-gating."""
        return f"rm -rf {ShellCommand.path(path)}"

    @staticmethod
    def mkdir_p(path: str) -> str:
        """``mkdir -p <path>``."""
        return f"mkdir -p {ShellCommand.path(path)}"

    @staticmethod
    def path(path: str) -> str:
        """Shell expression for ``path`` with ``~`` expanded via ``$HOME`` and the rest quoted."""
        if path == "~":
            return '"$HOME"'
        if path.startswith("~/"):
            return '"$HOME"/' + shlex.quote(path[2:])
        return shlex.quote(path)


class ExecutorCli:
    """CLI flag constants for the ``catalyst-executor`` binary."""

    PLUGIN_FLAG = "--plugin="
    BIND_FLAG = "--bind="


# --- logging -----------------------------------------------------------------------------------
# Verbosity levels (via :func:`set_verbose` or ``Executor(verbose=)``):
#   0 quiet    — WARNING only
#   1 default  — INFO (phases, ready/stop, executor stdout stream)
#   2 verbose  — DEBUG (ssh/scp commands, timings)
#   3+ trace   — DEBUG + extra ``ssh -v`` flags on the wire
logger = logging.getLogger("catalyst.executor")
logger.setLevel(logging.INFO)
_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setFormatter(logging.Formatter("[remote-exec] %(message)s"))
logger.addHandler(_stderr_handler)
logger.propagate = False  # avoid double-logging when a caller configures the root logger

_verbose = 1  # 0-3, drives external-tool flags (``ssh -v``, ``scp -v``)


def set_verbose(level: int) -> None:
    """Set launcher verbosity. Maps to :mod:`logging`: 0 → WARNING, 1 → INFO, ≥2 → DEBUG.
    Higher values (3+) also bump the ``ssh -v`` flag count."""
    global _verbose
    _verbose = level
    if level <= 0:
        logger.setLevel(logging.WARNING)
    elif level == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)


def verbose_level() -> int:
    """Current numeric verbosity (0-3)."""
    return _verbose


def log_cmd(argv: list[str]) -> None:
    """DEBUG-level echo of a command about to run."""
    logger.debug("$ %s", " ".join(shlex.quote(c) for c in argv))


class Paths:
    """Default paths for workspace, executor binary, and per-launch log file."""

    # Guards the ``rm -rf`` teardown so a user-pinned workspace can never be wiped.
    WORKSPACE_PREFIX = "catalyst-exec-"

    # Executor binary name; also used as ``./<name>`` inside a scp'd workspace.
    EXECUTOR_BIN = "catalyst-executor"

    # Filesystem-safe timestamp: 2026-06-30_04-48-15.
    _TS_FMT = "%Y-%m-%d_%H-%M-%S"

    @staticmethod
    def _timestamp() -> str:
        """Filesystem-safe timestamp for workspace and log names."""
        return time.strftime(Paths._TS_FMT, time.localtime())

    @staticmethod
    def _random_suffix() -> str:
        """Random 12-bit hex tag (3 chars) to break same-second name collisions."""
        return f"{random.randint(0, 0xfff):03x}"

    @staticmethod
    def default_workspace() -> str:
        """Per-run remote workspace path under ``~/``, tagged with user/timestamp/random suffix."""
        return (
            f"~/{Paths.WORKSPACE_PREFIX}{getpass.getuser()}-"
            f"{Paths._timestamp()}-{Paths._random_suffix()}"
        )

    @staticmethod
    def default_executor_bin() -> str:
        """Locate the ``catalyst-executor`` binary for ``local=True``.

        Search order: packaged lib dir, then its ``remote/`` subdir. Defaults to the name on
        ``$PATH``.
        """
        rt_lib = Path(get_lib_path("runtime", "RUNTIME_LIB_DIR"))
        for candidate in (rt_lib / Paths.EXECUTOR_BIN, rt_lib / "remote" / Paths.EXECUTOR_BIN):
            if candidate.exists():
                return str(candidate)
        return Paths.EXECUTOR_BIN

    @staticmethod
    def resolve_log(
        host: str, explicit: str | None = None, disabled: bool = False, name: str = "executor"
    ) -> str | None:
        """Host-side log-file path for one launch. ``explicit`` pins a path; ``disabled`` returns
        ``None``."""
        if disabled:
            return None
        if explicit:
            return explicit
        tag = "" if name == "executor" else f"-{name}"
        return f"{Paths.EXECUTOR_BIN}{tag}-{host}-{Paths._timestamp()}.log"


def random_port() -> int:
    """A random ephemeral-range port (20000-59999)."""
    return random.randint(20000, 59999)


def triple_from_uname(system: str, machine: str) -> str | None:
    """Map ``uname -s`` / ``uname -m`` to an LLVM target triple, or ``None`` if unknown."""
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
    """preexec_fn: kernel SIGTERMs this child when the parent dies, so a host crash doesn't
    leak the ssh tunnel + executor."""
    with contextlib.suppress(Exception):
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG=1


# preexec_fn only exists on POSIX; None elsewhere. Lowercased since it's a callable-or-None.
pdeathsig = _set_pdeathsig if hasattr(os, "fork") else None
