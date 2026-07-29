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

"""SSH/scp orchestration for a remote :class:`~catalyst.executor.Executor`: the base ssh command
(with connection multiplexing), bundle copy, guarded remote-dir removal, key/sudo auth probing, and
the remote shell command that launches the executor. Consumed by :mod:`.process` (the remote process)
and the package :mod:`~catalyst.executor` (deploy/teardown)."""

from __future__ import annotations

import getpass
import shlex
import subprocess
import time
from pathlib import Path

from ._util import _log, _logcmd, verbosity

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


def _ssh_base(
    user: str, host: str, opts: list[str] | None = None, multiplex: bool = True
) -> list[str]:
    cmd = ["ssh", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
    if multiplex:
        cmd += _ctl_opts()
    cmd += ["-v"] * max(0, verbosity() - 2)  # ssh protocol debug at -vvv (verbosity 3+)
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

    def _scp(extra: list[str]) -> int:
        cmd = [
            "scp",
            *extra,
            *_ctl_opts(),
            "-v" if verbosity() >= 2 else "-q",
            *[str(f) for f in files],
            f"{user}@{host}:{workspace}/",
        ]
        _logcmd(cmd)
        return subprocess.call(cmd)

    t0 = time.monotonic()
    rc = _scp([])
    if rc != 0:
        # A modern scp uses the SFTP backend; hosts with no sftp subsystem reject it ("subsystem
        # request failed"). Retry with the legacy SCP protocol (-O), which only needs remote scp.
        _log("scp failed — retrying with the legacy protocol (scp -O)")
        rc = _scp(["-O"])
    if rc != 0:
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
