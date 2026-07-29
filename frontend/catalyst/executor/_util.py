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

"""Shared low-level helpers for the :class:`~catalyst.executor.Executor`: readiness/auth regexes,
verbosity-aware logging, target-triple detection, workspace/port/timestamp helpers, and the
``PR_SET_PDEATHSIG`` child-cleanup hook. Imported by :mod:`.ssh`, :mod:`.process`, and the package
:mod:`~catalyst.executor` itself. Verbosity lives here as the single source of truth; submodules that
need the live value read it via :func:`verbosity` (a plain import would snapshot the int)."""

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


def verbosity() -> int:
    """The current verbosity. Submodules call this rather than importing ``_VERBOSITY`` directly, so
    they always see the live value after :func:`_set_verbosity`."""
    return _VERBOSITY


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


def _set_pdeathsig() -> None:
    """preexec_fn: ask the kernel to SIGTERM this child when the parent (python) dies — so a host
    crash (segfault/SIGKILL, which skip atexit) doesn't leak the ssh tunnel + executor."""
    with contextlib.suppress(Exception):
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG=1


# preexec_fn only exists on POSIX; harmless no-op reference elsewhere.
_PDEATHSIG = _set_pdeathsig if hasattr(os, "fork") else None
