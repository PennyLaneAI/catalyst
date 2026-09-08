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

* :class:`OutputPatterns`: regexes for scanning executor/ssh output.
* :class:`ExecutorPaths`: workspace, binary, and log-file path resolvers.
* :class:`ShellText`: POSIX shell fragments and path quoting.
* :class:`ExecutorFlags`: CLI flag constants for ``catalyst-executor``.

Plus stdlib logging (:data:`logger`, :func:`set_verbose`, :func:`verbose_level`, :func:`log_cmd`)
and free helpers (:func:`random_port`, :func:`triple_from_uname`).
"""

from __future__ import annotations

import getpass
import logging
import random
import re
import shlex
import time
from pathlib import Path
from typing import Final

from catalyst.utils.runtime_environment import get_lib_path

# Random fallback ports to try on a bind collision.
MAX_PORT_TRIES = 6


class OutputPatterns:
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
    # sudo refusing `-E` because the sudoers policy grants no SETENV.
    _SUDO_SETENV = re.compile(
        r"you are not allowed to (set the following environment variables|preserve the environment)"
    )
    # Port collision on the remote bind or the local -L forward.
    _PORT = re.compile(r"Address already in use|Could not request local forwarding")

    @staticmethod
    def is_ready(line: str) -> bool:
        """True if ``line`` signals the executor bound its port."""
        return bool(OutputPatterns._READY.search(line))

    @staticmethod
    def is_port_conflict(line: str) -> bool:
        """True if ``line`` signals a port bind failure."""
        return bool(OutputPatterns._PORT.search(line))

    @staticmethod
    def is_ssh_prompt(line: str) -> bool:
        """True if ``line`` is an SSH password/passphrase prompt."""
        return bool(OutputPatterns._SSH_PW.search(line))

    @staticmethod
    def is_sudo_fail(line: str) -> bool:
        """True if ``line`` is a sudo password rejection."""
        return bool(OutputPatterns._SUDO_FAIL.search(line))

    @staticmethod
    def is_sudo_setenv_refusal(line: str) -> bool:
        """True if ``line`` is sudo refusing to carry the environment across (no ``SETENV``)."""
        return bool(OutputPatterns._SUDO_SETENV.search(line))


class ShellText:
    """Generic POSIX shell fragments and path quoting for remote ops."""

    # Shell expression that expands to ``$HOME`` on the remote; quoted to survive one round of
    # shell parsing.
    HOME: Final = '"$HOME"'

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
        """``pkill -f <pat>``. ``pat`` is shell-quoted.

        The first character is wrapped in a bracket expression, which matches the same text while
        making the pattern unable to match itself.
        """
        if pat[:1].isalnum():
            pat = f"[{pat[0]}]{pat[1:]}"
        return f"pkill -f {shlex.quote(pat)}"

    @staticmethod
    def rm_rf(path: str) -> str:
        """``rm -rf <path>``. Caller is responsible for safety-gating."""
        return f"rm -rf {ShellText.path(path)}"

    @staticmethod
    def mkdir_p(path: str) -> str:
        """``mkdir -p <path>``."""
        return f"mkdir -p {ShellText.path(path)}"

    @staticmethod
    def path(path: str) -> str:
        """Shell expression for ``path`` with ``~`` expanded via ``$HOME`` and the rest quoted."""
        if path == "~":
            return ShellText.HOME
        if path.startswith("~/"):
            return f"{ShellText.HOME}/{shlex.quote(path[2:])}"
        return shlex.quote(path)


class ExecutorFlags:
    """CLI flag constants for the ``catalyst-executor`` binary."""

    PLUGIN_FLAG = "--plugin="
    BIND_FLAG = "--bind="
    BIND_HOST: Final = "127.0.0.1"


# --- logging -----------------------------------------------------------------------------------
# PennyLane convention: library-silent by default. Attach a handler to the ``catalyst.executor``
# logger tree (or call :func:`pennylane.logging.enable_logging`) to see output.
#
# ``Executor(verbose=)`` / :func:`set_verbose` maps an int 0-3 to log level and ssh-verbosity:
#   0 quiet    → WARNING
#   1 default  → INFO   (launch phases, ready/stop, remote stdout)
#   2 verbose  → DEBUG  (ssh/scp commands, timings)
#   3+ trace   → DEBUG + ``ssh -v`` flags on the wire
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# set_verbose targets ``catalyst.executor`` so every child module inherits the level.
_pkg_logger = logging.getLogger("catalyst.executor")
_verbose = 1


def set_verbose(level: int) -> None:
    """Set launcher verbosity (0-3). See the module-level table for the mapping."""
    global _verbose
    _verbose = level
    if level <= 0:
        _pkg_logger.setLevel(logging.WARNING)
    elif level == 1:
        _pkg_logger.setLevel(logging.INFO)
    else:
        _pkg_logger.setLevel(logging.DEBUG)


def verbose_level() -> int:
    """Current verbosity (0-3)."""
    return _verbose


def log_cmd(argv: list[str]) -> None:
    """DEBUG-level echo of a command about to run."""
    logger.debug("$ %s", " ".join(shlex.quote(c) for c in argv))


class ExecutorPaths:
    """Default paths for workspace, executor binary, and per-launch log file."""

    # Guards the ``rm -rf`` teardown so a user-pinned workspace can never be wiped.
    WORKSPACE_PREFIX: Final = "catalyst-exec-"

    # Executor binary name; also used as ``./<name>`` inside a scp'd workspace.
    EXECUTOR_BIN: Final = "catalyst-executor"

    # Filesystem-safe timestamp: 2026-06-30_04-48-15.
    _TS_FMT: Final = "%Y-%m-%d_%H-%M-%S"

    @staticmethod
    def _timestamp() -> str:
        """Filesystem-safe timestamp for workspace and log names."""
        return time.strftime(ExecutorPaths._TS_FMT, time.localtime())

    @staticmethod
    def _random_suffix() -> str:
        """Random 12-bit hex tag (3 chars) to break same-second name collisions."""
        return f"{random.randint(0, 0xfff):03x}"

    @staticmethod
    def default_workspace() -> str:
        """Per-run remote workspace path under ``~/``, tagged with user/timestamp/random suffix."""
        return (
            f"~/{ExecutorPaths.WORKSPACE_PREFIX}{getpass.getuser()}-"
            f"{ExecutorPaths._timestamp()}-{ExecutorPaths._random_suffix()}"
        )

    @staticmethod
    def default_executor_bin() -> str:
        """Locate the ``catalyst-executor`` binary for a local subprocess.

        Search order: packaged lib dir, then its ``remote/`` subdir. Defaults to the name on
        ``$PATH``.
        """
        rt_lib = Path(get_lib_path("runtime", "RUNTIME_LIB_DIR"))
        for candidate in (
            rt_lib / ExecutorPaths.EXECUTOR_BIN,
            rt_lib / "remote" / ExecutorPaths.EXECUTOR_BIN,
        ):
            if candidate.exists():
                return str(candidate)
        return ExecutorPaths.EXECUTOR_BIN

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
        return f"{ExecutorPaths.EXECUTOR_BIN}{tag}-{host}-{ExecutorPaths._timestamp()}.log"


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
