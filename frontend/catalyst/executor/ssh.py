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
import os
import shlex
import subprocess
import time
from pathlib import Path

from .utils import Log, Raw, SSHError


def _ctl_dir() -> Path:
    """Where to keep the SSH control sockets. Prefers ``$XDG_RUNTIME_DIR`` (tmpfs, per-user,
    auto-cleared at logout) and falls back to ``~/.cache/catalyst/ssh-cm``. Kept out of
    ``~/.ssh`` so hardened setups (strict perms/allowlist) don't complain."""
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(xdg) / "catalyst" if xdg else Path.home() / ".cache" / "catalyst" / "ssh-cm"
    base.mkdir(parents=True, exist_ok=True)
    return base

# Connection multiplexing: the first short op opens a master, the rest reuse it (no re-handshake),
# and it self-expires. Applied to the chatty control ops (probe/mkdir/scp/pkill), NOT the long-lived
# executor session — that keeps one clean connection whose close SIGHUPs the executor.
def _ctl_opts() -> list[str]:
    """SSH ``ControlMaster``/``ControlPath``/``ControlPersist`` flags for connection multiplexing.

    The first short op opens a master socket under :func:`_ctl_dir` and later ops (probe / mkdir /
    scp / pkill) reuse it, skipping the auth handshake. ``%C`` (a hash of ``%l%h%p%r``) keeps the
    path short enough for the Unix-socket length limit. The socket self-expires 30s after the
    last user."""
    return [
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPath={_ctl_dir()}/cm-%C",
        "-o",
        "ControlPersist=30",
    ]


def _ssh_base(
    user: str, host: str, opts: list[str] | None = None, multiplex: bool = True
) -> list[str]:
    """Build the shared ``ssh`` command prefix used by every remote op.

    Sets ``ServerAliveInterval``/``ServerAliveCountMax`` so a dead peer surfaces within ~1 min,
    optionally layers on the multiplexing control socket (see :func:`_ctl_opts`), and adds
    ``-v`` flags at verbosity 3+ for SSH protocol debug.

    Args:
        user: The remote user (e.g. from ``getpass.getuser()``).
        host: The remote host.
        opts: Extra ``ssh`` flags to append after the base options and before the target.
        multiplex: Whether to include the ``ControlMaster`` multiplexing options. Set to ``False``
            for the long-lived executor session so its close SIGHUPs the executor cleanly.

    Returns:
        list[str]: The ``ssh ... user@host`` argv prefix, ready to append a remote command."""
    cmd = ["ssh", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
    if multiplex:
        cmd += _ctl_opts()
    cmd += ["-v"] * max(0, Log.level() - 2)  # ssh protocol debug at -vvv (verbosity 3+)
    if opts:
        cmd += opts
    cmd.append(f"{user}@{host}")
    return cmd


def _copy_bundle(bundle: Path, user: str, host: str, workspace: str) -> None:
    """mkdir the remote workspace and scp every artifact in ``bundle`` into it.

    ``workspace`` is shell-quoted (with ``~`` expansion) before it is embedded in the remote
    ``mkdir``; the scp target is passed as its own argv element and needs no quoting."""
    files = sorted(p for p in bundle.iterdir() if p.is_file() and p.name != "README.md")
    if not files:
        raise SSHError(
            f"no artifacts in {bundle} — pass build=<recipe> to cross-compile the executor + "
            "runtime libs for the target, or point bundle= at a prebuilt directory."
        )
    total = sum(f.stat().st_size for f in files)
    Log.say(f"copying {len(files)} artifact(s), {total/1e6:.1f} MB -> {user}@{host}:{workspace}/")
    for f in files:
        Log.say(f"  - {f.name}  ({f.stat().st_size/1e6:.2f} MB)", level=2)
    mkdir = _ssh_base(user, host) + [f"mkdir -p {_remote_path_expr(workspace)}"]
    Log.cmd(mkdir)
    if subprocess.call(mkdir) != 0:
        raise SSHError(f"failed to create remote workspace {workspace!r}")

    def _scp(extra: list[str]) -> int:
        cmd = [
            "scp",
            *extra,
            *_ctl_opts(),
            "-v" if Log.level() >= 2 else "-q",
            *[str(f) for f in files],
            f"{user}@{host}:{workspace}/",
        ]
        Log.cmd(cmd)
        return subprocess.call(cmd)

    t0 = time.monotonic()
    rc = _scp([])
    if rc != 0:
        # A modern scp uses the SFTP backend; hosts with no sftp subsystem reject it ("subsystem
        # request failed"). Retry with the legacy SCP protocol (-O), which only needs remote scp.
        Log.say("scp failed — retrying with the legacy protocol (scp -O)")
        rc = _scp(["-O"])
    if rc != 0:
        raise SSHError(f"scp of bundle to {user}@{host}:{workspace}/ failed")
    Log.say(f"copied in {time.monotonic() - t0:.1f}s", level=2)


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
    Log.cmd(cmd)
    rc = subprocess.call(cmd, timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc == 3:
        raise ValueError(
            f"refusing to remove workspace {workspace!r}: it resolves to '/' or the home directory"
        )
    if rc != 0:
        raise SSHError(f"failed to remove remote workspace {workspace!r} (ssh rc={rc})")


def _probe_auth(user: str, host: str) -> bool:
    """Non-interactively check that key-based SSH works and whether sudo needs no password.

    ``BatchMode=yes`` means a missing key fails fast (rc 255) instead of prompting; ``sudo -n true``
    returns 0 only when sudo needs no password.

    Args:
        user: The remote user.
        host: The remote host.

    Returns:
        bool: ``True`` if remote sudo needs no password, ``False`` if it does.

    Raises:
        SSHError: If SSH itself failed (no usable key, host unreachable, …) — with an
            ``ssh-copy-id`` hint the caller can surface to the user.
    """
    cmd = _ssh_base(user, host, ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]) + [
        "sudo -n true 2>/dev/null"
    ]
    Log.cmd(cmd)
    rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc == 255:
        raise SSHError(
            f"can't SSH to {user}@{host} without a password — install your key once:\n"
            f"    ssh-copy-id {user}@{host}\n"
            "  (this only ADDS your key; it does not affect other users of the account.)"
        )
    return rc == 0


def _resolve_sudo_password(user: str, host: str, sudo_password: str | None = None) -> str | None:
    """The remote sudo password, when NOPASSWD isn't set: the explicit ``sudo_password`` if given,
    else a one-time getpass prompt. Returns ``None`` when sudo needs no password (nothing to do).

    Raises:
        SSHError: If SSH is unreachable (from :func:`_probe_auth`), or the user aborted the
            interactive prompt without supplying a password.
    """
    if _probe_auth(user, host):
        return None
    if sudo_password is not None:
        return sudo_password
    Log.say("remote sudo needs a password (no NOPASSWD) — prompting once", level=1)
    try:
        return getpass.getpass(f"[remote] sudo password for {user}@{host}: ")
    except (EOFError, KeyboardInterrupt) as e:
        raise SSHError(
            "no sudo password provided — pass sudo_password= or run interactively"
        ) from e


def _build_remote_cmd(
    workspace: str,
    remote_port: int,
    plugins: list[str],
    env: dict[str, str],
    use_password: bool = False,
    sudo: bool = True,
    executor_bin: str = "./catalyst-executor",
) -> str:
    """The shell command run on the remote host: ``cd``, export env, exec the executor.

    Quoting: ``env`` values, plugin paths, and ``workspace`` are shell-quoted so metacharacters
    (spaces, ``;``, ``$``, …) can't break out. Values that need ``$VAR`` to expand on the remote
    must be wrapped in :class:`~catalyst.executor.utils.Raw` — bare :class:`str` is treated as a
    literal. ``executor_bin`` is used as-is (it is either a repo-controlled default like
    ``./catalyst-executor`` or an explicit override the caller vouches for).

    ``sudo`` wraps the executor in sudo (some devices need root): with ``use_password`` use
    ``sudo -S`` (reads the password piped to stdin) with an empty prompt, otherwise plain
    ``sudo -E`` under a PTY (NOPASSWD). ``sudo=False`` runs it as the login user.
    """

    def _q(v) -> str:
        # Raw values expand `$VAR` on the remote; everything else is a literal.
        return v if isinstance(v, Raw) else shlex.quote(v)

    env_prefix = " ".join(f"{k}={_q(v)}" for k, v in env.items())

    def _plugin_arg(p) -> str:
        if isinstance(p, Raw):
            return f"--plugin={p}"
        # A bare filename resolves against the workspace (a scp'd bundle); an absolute or ~-rooted
        # path is quoted with tilde expansion; everything else is safely shell-quoted.
        if "/" not in p and not p.startswith("~"):
            return f"--plugin=$PWD/{shlex.quote(p)}"
        return f"--plugin={_remote_path_expr(p)}"

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
        f"cd {_remote_path_expr(workspace)} && {chmod}{env_prefix} "
        f"{launcher} {executor_bin} --bind=0.0.0.0:{remote_port} {plugin_args}"
    )
