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

"""SSH/scp orchestration for a remote :class:`~catalyst.executor.Executor`.

Grouped into five namespace classes so related items sit together:

* :class:`SSH` — the ``ssh`` binary wrapper: command line (control socket, base flags), a runner
  with rc + optional-raise, non-interactive auth probing, sudo-password resolution, and remote
  filesystem verbs (``mkdir``, ``rmdir``, ``scp_bundle``).
* :class:`SCP` — the ``scp`` binary wrapper, riding on SSH's multiplexed control socket.
* :class:`ShellCommand` — generic POSIX shell fragments (``pkill``, ``sudo``, ``rm -rf``,
  ``mkdir -p``) and safe path quoting.
* :class:`ExecutorCli` — flag constants for the ``catalyst-executor`` binary itself; shared
  between local and remote invocations.
* :class:`RemoteLauncher` — the shell command that launches ``catalyst-executor`` on a remote
  host (``cd + env + exec``, sudo-wrapped when needed).

Consumed by :mod:`.process` (the remote process) and :mod:`.manager` (deploy/teardown).
"""

from __future__ import annotations

import getpass
import os
import shlex
import subprocess
import time
from pathlib import Path

from .utils import Log, Paths, Raw


class SSH:
    """Local ``ssh`` command line + non-interactive auth for the remote host: control-socket
    location, multiplexing flags, the shared ``ssh ... user@host`` prefix reused by every remote
    op, a fire-and-forget runner, plus SSH/sudo auth probing and sudo-password resolution."""

    # Seconds a multiplexed control socket lingers idle after the last user before self-closing.
    # Long enough to fold the follow-up teardown ops (pkill / rm) into the same master; short
    # enough that sockets don't outlive the launch by much.
    CONTROL_PERSIST = 30

    # Argv prefix used by every ``base()`` invocation: the ``ssh`` binary plus keep-alive tuning
    # (ping every 15s, give up after 4 misses → dead peer surfaces in ~1 min). Tuple so it can't
    # be mutated by accident; ``base()`` copies it into a list before appending.
    BASE_CMD: tuple[str, ...] = ("ssh", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4")

    # Extra ``-o`` flags for one-shot probes (:meth:`probe`, ``_detect_triple``): ``BatchMode=yes``
    # fails fast with rc 255 instead of prompting for a password, and a short ``ConnectTimeout``
    # keeps launches from hanging on an unreachable host.
    PROBE_OPTS: tuple[str, ...] = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")

    @staticmethod
    def _fallback_ctl_dir() -> Path:
        """Filesystem fallback for the SSH control-socket dir when ``$XDG_RUNTIME_DIR`` isn't set.
        Kept out of ``~/.ssh`` so hardened setups (strict perms/allowlist) don't complain."""
        return Path.home() / ".cache" / "catalyst" / "ssh-cm"

    @staticmethod
    def _ctl_dir() -> Path:
        """Where to keep the SSH control sockets. Prefers ``$XDG_RUNTIME_DIR`` (tmpfs, per-user,
        auto-cleared at logout) and falls back to :meth:`_fallback_ctl_dir`."""
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        base = Path(xdg) / "catalyst" if xdg else SSH._fallback_ctl_dir()
        base.mkdir(parents=True, exist_ok=True)
        return base

    @staticmethod
    def ctl_opts() -> list[str]:
        """SSH ``ControlMaster``/``ControlPath``/``ControlPersist`` flags for connection
        multiplexing.

        The first short op opens a master socket under :meth:`_ctl_dir` and later ops (probe /
        mkdir / scp / pkill) reuse it, skipping the auth handshake. ``%C`` (a hash of
        ``%l%h%p%r``) keeps the path short enough for the Unix-socket length limit. The socket
        self-expires ``CONTROL_PERSIST`` seconds after the last user.

        Applied to the chatty control ops (probe/mkdir/scp/pkill), NOT the long-lived executor
        session — that keeps one clean connection whose close SIGHUPs the executor."""
        return [
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={SSH._ctl_dir()}/cm-%C",
            "-o",
            f"ControlPersist={SSH.CONTROL_PERSIST}",
        ]

    @staticmethod
    def base(
        user: str, host: str, opts: list[str] | None = None, multiplex: bool = True
    ) -> list[str]:
        """Build the shared ``ssh`` command prefix used by every remote op.

        Sets ``ServerAliveInterval``/``ServerAliveCountMax`` so a dead peer surfaces within
        ~1 min, optionally layers on the multiplexing control socket (see :meth:`ctl_opts`), and
        adds ``-v`` flags at verbosity 3+ for SSH protocol debug.

        Args:
            user: The remote user (e.g. from ``getpass.getuser()``).
            host: The remote host.
            opts: Extra ``ssh`` flags to append after the base options and before the target.
            multiplex: Whether to include the ``ControlMaster`` multiplexing options. Set to
                ``False`` for the long-lived executor session so its close SIGHUPs the executor
                cleanly.

        Returns:
            list[str]: The ``ssh ... user@host`` argv prefix, ready to append a remote command."""
        cmd = list(SSH.BASE_CMD)
        if multiplex:
            cmd += SSH.ctl_opts()
        cmd += ["-v"] * max(0, Log.level() - 2)  # ssh protocol debug at -vvv (verbosity 3+)
        if opts:
            cmd += opts
        cmd.append(f"{user}@{host}")
        return cmd

    @staticmethod
    def capture(user: str, host: str, remote_cmd: str, *, timeout: float = 15) -> str | None:
        """Run ``ssh user@host <remote_cmd>`` non-interactively (with :data:`PROBE_OPTS`:
        ``BatchMode=yes`` + short ``ConnectTimeout``) and return stdout stripped, or ``None`` on
        any failure (SSH error, non-zero exit, timeout, spawn failure). Used for state probes
        (e.g. ``uname -sm``) where "can't tell" is a valid answer."""
        cmd = SSH.base(user, host, list(SSH.PROBE_OPTS)) + [remote_cmd]
        try:
            return subprocess.check_output(
                cmd, text=True, timeout=timeout, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    @staticmethod
    def run(
        user: str,
        host: str,
        remote_cmd: str,
        *,
        input: str | None = None,
        timeout: float = 15,
        opts: list[str] | None = None,
        quiet: bool = True,
        log: bool = False,
        error: str | None = None,
    ) -> int:
        """Run ``ssh user@host <remote_cmd>`` and return its exit code.

        Two flavors, selected by ``quiet``:

        * ``quiet=True`` (default) — teardown-style: stdout/stderr → ``/dev/null``, and any
          subprocess exception (spawn failure, ``TimeoutExpired``) is swallowed and reported as
          rc ``-1``. A failure here isn't actionable — the caller is already cleaning up, and
          the SSH connection close is the primary signal that stops the remote.
        * ``quiet=False`` — deploy-style: stdout/stderr inherit the terminal so the user sees
          remote output (``mkdir: Permission denied``, scp progress, …), and any subprocess
          exception propagates.

        ``log=True`` records the argv via :meth:`Log.cmd`; ``opts`` are extra ``-o`` flags
        passed through to :meth:`base`.

        ``error``: if given and the exit code is non-zero, raise :class:`RuntimeError` with this
        message (the actual rc is appended). Sugar for the ``if SSH.run(...) != 0: raise
        RuntimeError(...)`` pattern when there's no rc-specific branching."""
        cmd = SSH.base(user, host, opts) + [remote_cmd]
        if log:
            Log.cmd(cmd)
        kwargs = {"input": input, "text": input is not None, "timeout": timeout}
        if quiet:
            kwargs["stdout"] = subprocess.DEVNULL
            kwargs["stderr"] = subprocess.DEVNULL
            try:
                rc = subprocess.run(cmd, **kwargs).returncode
            except Exception:
                rc = -1
        else:
            rc = subprocess.run(cmd, **kwargs).returncode
        if error is not None and rc != 0:
            raise RuntimeError(f"{error} (rc={rc})")
        return rc

    @staticmethod
    def pkill(
        user: str,
        host: str,
        pat: str,
        *,
        sudo: bool = False,
        sudo_password: str | None = None,
    ) -> int:
        """Fire-and-forget ``pkill -f <pat>`` on the remote. Silent; see :meth:`run`.

        With ``sudo=True``, wraps in ``sudo`` — using ``sudo -S`` if ``sudo_password`` is given
        (piped on stdin, empty prompt), else ``sudo -n`` (NOPASSWD only — fails immediately if
        a password is required)."""
        cmd = ShellCommand.pkill(pat)
        if not sudo:
            return SSH.run(user, host, cmd)
        if sudo_password is not None:
            return SSH.run(user, host, ShellCommand.sudo_pw(cmd), input=sudo_password + "\n")
        return SSH.run(user, host, ShellCommand.sudo_np(cmd))

    @staticmethod
    def mkdir(user: str, host: str, path: str) -> None:
        """Create ``path`` (and any missing parents) on the remote host. Inherits stdout/stderr
        so the user sees any ``mkdir: Permission denied`` from the remote.

        Raises:
            RuntimeError: If the remote ``mkdir`` returned non-zero.
        """
        SSH.run(
            user,
            host,
            ShellCommand.mkdir_p(path),
            quiet=False,
            log=True,
            error=f"failed to create remote directory {path!r}",
        )

    @staticmethod
    def rm_rf(user: str, host: str, path: str) -> None:
        """Recursively remove ``path`` on the remote host — no safety guards; the caller is
        responsible for verifying ``path`` is safe to delete. Best-effort: errors are swallowed.
        For guarded removal (refuses ``/`` and ``$HOME``) use :meth:`rmdir` instead."""
        SSH.run(user, host, ShellCommand.rm_rf(path))

    @staticmethod
    def rmdir(user: str, host: str, path: str, *, force: bool = False) -> None:
        """Remove a remote directory recursively, guarded so it can never delete ``/`` or the
        home directory. Resolves ``path`` (including a leading ``~``) to a canonical path on
        the remote and refuses if it is empty, ``/``, or ``$HOME``. A missing directory is a
        no-op.

        ``force=True`` re-raises SSH/remote errors on failure; default swallows them (best-effort
        teardown). The safety refusal (``/`` or ``$HOME``) always raises regardless of ``force``.

        Raises:
            ValueError: If ``path`` resolves to ``/`` or ``$HOME``.
            RuntimeError: If ``force`` is set and the remote ``rm`` returned non-zero.
        """
        remote = (
            f"ws={ShellCommand.path(path)}; "
            'd=$(cd "$ws" 2>/dev/null && pwd) || exit 0; '
            'if [ -z "$d" ] || [ "$d" = "/" ] || [ "$d" = "$HOME" ]; then exit 3; fi; '
            'rm -rf "$d"'
        )
        rc = SSH.run(user, host, remote, timeout=30, log=True)
        if rc == 3:
            raise ValueError(
                f"refusing to remove {path!r}: it resolves to '/' or the home directory"
            )
        if rc != 0 and force:
            raise RuntimeError(f"failed to remove remote directory {path!r} (ssh rc={rc})")

    @staticmethod
    def probe(user: str, host: str) -> bool:
        """Non-interactively check that key-based SSH works and whether sudo needs no password.

        ``BatchMode=yes`` means a missing key fails fast (rc 255) instead of prompting;
        ``sudo -n true`` returns 0 only when sudo needs no password.

        Args:
            user: The remote user.
            host: The remote host.

        Returns:
            bool: ``True`` if remote sudo needs no password, ``False`` if it does.

        Raises:
            RuntimeError: If SSH itself failed (no usable key, host unreachable, …) — with an
                ``ssh-copy-id`` hint the caller can surface to the user.
        """
        rc = SSH.run(user, host, ShellCommand.sudo_probe(), opts=list(SSH.PROBE_OPTS), log=True)
        if rc == 255:
            raise RuntimeError(
                f"can't SSH to {user}@{host} without a password — install your key once:\n"
                f"    ssh-copy-id {user}@{host}\n"
                "  (this only ADDS your key; it does not affect other users of the account.)"
            )
        return rc == 0

    @staticmethod
    def resolve_sudo(user: str, host: str, sudo_password: str | None = None) -> str | None:
        """The remote sudo password, when NOPASSWD isn't set: the explicit ``sudo_password`` if
        given, else a one-time getpass prompt. Returns ``None`` when sudo needs no password
        (nothing to do).

        Raises:
            RuntimeError: If SSH is unreachable (from :meth:`probe`), or the user aborted the
                interactive prompt without supplying a password.
        """
        if SSH.probe(user, host):
            return None
        if sudo_password is not None:
            return sudo_password
        Log.info("remote sudo needs a password (no NOPASSWD) — prompting once")
        try:
            return getpass.getpass(f"[remote] sudo password for {user}@{host}: ")
        except (EOFError, KeyboardInterrupt) as e:
            raise RuntimeError(
                "no sudo password provided — pass sudo_password= or run interactively"
            ) from e


class SCP:
    """Wrapper for the ``scp`` binary — sibling to :class:`SSH`, riding on the same
    control-socket multiplexing (see :meth:`SSH.ctl_opts`) so a copy piggybacks on the already
    authenticated connection instead of opening a fresh one."""

    @staticmethod
    def run(
        user: str, host: str, files: list[Path], dest: str, *, log: bool = True
    ) -> None:
        """Copy ``files`` into ``user@host:dest/`` via scp. Tries the modern SFTP backend first;
        on failure, retries with the legacy SCP protocol (``-O``) for hosts without an SFTP
        subsystem ("subsystem request failed"). Verbosity mirrors :class:`Log`: ``-v`` at level
        ≥ 2, ``-q`` otherwise.

        Raises:
            RuntimeError: If both the modern and legacy attempts fail.
        """

        def _once(legacy: bool) -> int:
            cmd = [
                "scp",
                *(["-O"] if legacy else []),
                *SSH.ctl_opts(),
                "-v" if Log.level() >= 2 else "-q",
                *[str(f) for f in files],
                f"{user}@{host}:{dest}/",
            ]
            if log:
                Log.cmd(cmd)
            return subprocess.run(cmd).returncode

        if _once(legacy=False) == 0:
            return
        Log.info("scp failed — retrying with the legacy protocol (scp -O)")
        if _once(legacy=True) != 0:
            raise RuntimeError(f"scp to {user}@{host}:{dest}/ failed")

    @staticmethod
    def bundle(user: str, host: str, bundle: Path, workspace: str) -> None:
        """Copy every artifact in ``bundle`` to ``user@host:workspace/``, first creating the
        workspace via :meth:`SSH.mkdir`. Composite of :meth:`SSH.mkdir` + :meth:`run`.

        Raises:
            RuntimeError: If ``bundle`` has no artifacts to copy.
        """
        files = sorted(p for p in bundle.iterdir() if p.is_file() and p.name != "README.md")
        if not files:
            raise RuntimeError(
                f"no artifacts in {bundle} — pass build=<recipe> to cross-compile the executor + "
                "runtime libs for the target, or point bundle= at a prebuilt directory."
            )
        total = sum(f.stat().st_size for f in files)
        Log.info(
            f"copying {len(files)} artifact(s), {total/1e6:.1f} MB -> {user}@{host}:{workspace}/"
        )
        for f in files:
            Log.info(f"  - {f.name}  ({f.stat().st_size/1e6:.2f} MB)", level=2)
        SSH.mkdir(user, host, workspace)
        t0 = time.monotonic()
        SCP.run(user, host, files, workspace)
        Log.info(f"copied in {time.monotonic() - t0:.1f}s", level=2)


class ShellCommand:
    """Generic POSIX shell fragments and path helpers reused by remote ops — the sudo/kill/rm
    building blocks assembled by :class:`RemoteLauncher` (executor launcher), :class:`SSH` (auth
    probe / mkdir), and :mod:`.process` (teardown). Nothing here is catalyst-executor-specific;
    everything runs on any Bourne-family shell."""

    # Shell-command fragments reused by teardown probes (see :mod:`.process`), auth probing,
    # and directory ops.
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


class RemoteLauncher:
    """Builders for the shell command that launches ``catalyst-executor`` on the *remote* host —
    ``cd + env + exec`` line, optionally wrapped in sudo. Uses :class:`ExecutorCli` for flags
    and :class:`ShellCommand` for generic shell fragments and path quoting."""

    @staticmethod
    def build(
        workspace: str,
        remote_port: int,
        plugins: list[str],
        env: dict[str, str],
        use_password: bool = False,
        sudo: bool = True,
        executor_bin: str = f"./{Paths.EXECUTOR_BIN}",
    ) -> str:
        """The shell command run on the remote host: ``cd``, export env, exec the executor.

        Quoting: ``env`` values, plugin paths, and ``workspace`` are shell-quoted so
        metacharacters (spaces, ``;``, ``$``, …) can't break out. Values that need ``$VAR`` to
        expand on the remote must be wrapped in :class:`~catalyst.executor.utils.Raw` — bare
        :class:`str` is treated as a literal. ``executor_bin`` is used as-is (it is either a
        repo-controlled default like ``./catalyst-executor`` or an explicit override the caller
        vouches for).

        ``sudo`` wraps the executor in sudo (some devices need root): with ``use_password`` use
        ``sudo -S`` (reads the password piped to stdin) with an empty prompt, otherwise plain
        ``sudo -E`` under a PTY (NOPASSWD). ``sudo=False`` runs it as the login user.
        """

        def _q(v) -> str:
            # Raw values expand `$VAR` on the remote; everything else is a literal.
            return v if isinstance(v, Raw) else shlex.quote(v)

        env_prefix = " ".join(f"{k}={_q(v)}" for k, v in env.items())

        def _plugin_arg(p) -> str:
            flag = ExecutorCli.PLUGIN_FLAG
            if isinstance(p, Raw):
                return f"{flag}{p}"
            # A bare filename resolves against the workspace (a scp'd bundle); an absolute or
            # ~-rooted path is quoted with tilde expansion; everything else is safely
            # shell-quoted.
            if "/" not in p and not p.startswith("~"):
                return f"{flag}$PWD/{shlex.quote(p)}"
            return f"{flag}{ShellCommand.path(p)}"

        plugin_args = " ".join(_plugin_arg(p) for p in plugins)
        # scp drops the +x bit; only relevant for a workspace-local executor binary.
        chmod = (
            f"chmod +x ./{Paths.EXECUTOR_BIN} 2>/dev/null; "
            if executor_bin == f"./{Paths.EXECUTOR_BIN}"
            else ""
        )
        if sudo:
            launcher = "exec sudo -S -E -p ''" if use_password else "exec sudo -E"
        else:
            launcher = "exec"
        return (
            f"cd {ShellCommand.path(workspace)} && {chmod}{env_prefix} "
            f"{launcher} {executor_bin} {ExecutorCli.BIND_FLAG}0.0.0.0:{remote_port} {plugin_args}"
        )




