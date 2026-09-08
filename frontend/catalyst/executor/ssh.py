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

* :class:`SSHArgv`: builds the ``ssh`` argv — constants, control-socket options, base command.
* :class:`RemoteOps`: runs commands on the remote via ssh — generic ``run``/``capture`` plus
  named verbs (``pkill``, ``mkdir``, ``rmdir``) and sudo probing.
* :class:`SCP`: ``scp`` wrapper that shares :class:`SSHArgv`'s multiplexed control socket.
* :class:`RemoteLauncher`: builds the remote ``catalyst-executor`` launch command.

.. warning::

    Experimental, and not intended for production use. This module shells out to ``ssh`` and
    ``scp``, may run the remote executor under ``sudo``, and deploys files into a workspace on the
    target host. It is built for driving development hardware on a trusted network, and makes no
    attempt at the isolation or auditing that running untrusted code would call for.
"""

from __future__ import annotations

import getpass
import logging
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .utils import ExecutorFlags, ExecutorPaths, ShellText, log_cmd, verbose_level

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SSHArgv:
    """Builders for the local ``ssh`` argv: base command, control-socket options, argv prefix."""

    # Idle lifetime of a multiplexed control socket, so follow-up ops (pkill/rm) reuse it.
    CONTROL_PERSIST = 30

    # Shared ``ssh`` argv prefix; keep-alive tuning surfaces a dead peer within ~1 min.
    BASE_CMD: tuple[str, ...] = (
        "ssh",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=4",
    )

    # One-shot probe flags: fail fast on a missing key, short connect timeout. ssh reports its own
    # failures as exit code 255, which is how they are told apart from the remote command's status.
    PROBE_OPTS: tuple[str, ...] = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")
    CONTROL_PATH_MAX = 104
    CONTROL_PATH_RESERVE = 20

    # Per-user directory for control sockets, relative to the user's home. Anyone able to reach a
    # control socket can open sessions on its remote host without authenticating, the master having
    # done so already, so the directory it sits in is what protects it.
    CONTROL_DIR = Path(".catalyst") / "cm"

    @staticmethod
    def _private_dir(home: Path | None = None) -> Path | None:
        """:data:`CONTROL_DIR` under ``home``, created ``0700``, or ``None`` if it cannot be.

        Args:
            home: Directory to place it under, defaulting to the user's own. Taken as an argument so
                that creating it and locking it down is checkable against a real directory.
        """
        path = (home or Path.home()) / SSHArgv.CONTROL_DIR
        try:
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o700)  # mkdir's mode is masked by umask, so set it outright
        except OSError as exc:
            logger.debug("control-socket dir %s unusable (%s); falling back", path, exc)
            return None
        return path

    @staticmethod
    def ctl_opts() -> list[str]:
        """``ControlMaster``/``ControlPath``/``ControlPersist`` flags for connection multiplexing.

        The first op opens a master socket under :meth:`_private_dir`; later ops
        (probe/mkdir/scp/pkill) reuse it, skipping the auth handshake. ``%C`` (a hash of
        ``%l%h%p%r``) keeps the name short. Not applied to the long-lived executor session, whose
        close SIGHUPs the executor cleanly.
        """
        private = SSHArgv._private_dir()
        return SSHArgv._ctl_flags(private) if private is not None else []

    @staticmethod
    def _ctl_flags(base: Path) -> list[str]:
        """:meth:`ctl_opts` for a given socket dir, split out so the length rule is decidable
        without touching the environment or the filesystem.
        """
        path = f"{base}/%C"
        # %C expands to a 40-character hash.
        expanded = len(path) - len("%C") + 40 + SSHArgv.CONTROL_PATH_RESERVE
        if expanded > SSHArgv.CONTROL_PATH_MAX:
            logger.debug(
                "ControlPath %r would exceed the %d-byte socket limit; disabling multiplexing",
                path,
                SSHArgv.CONTROL_PATH_MAX,
            )
            return []
        return [
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={path}",
            "-o",
            f"ControlPersist={SSHArgv.CONTROL_PERSIST}",
        ]

    @staticmethod
    def base(
        user: str, host: str, opts: list[str] | None = None, multiplex: bool = True
    ) -> list[str]:
        """Return the ``ssh ... user@host`` argv prefix.

        Args:
            user: Remote user.
            host: Remote host.
            opts: Extra ``ssh`` flags, appended after the base options.
            multiplex: Include the :meth:`ctl_opts` multiplexing flags. Set ``False`` for the
                long-lived executor session so its close SIGHUPs the executor.
        """
        cmd = list(SSHArgv.BASE_CMD)
        if multiplex:
            cmd += SSHArgv.ctl_opts()
        cmd += ["-v"] * max(0, verbose_level() - 2)  # ssh protocol debug at -vvv (verbosity 3+)
        if opts:
            cmd += opts
        cmd.append(f"{user}@{host}")
        return cmd


class RemoteOps:
    """Operations performed on the remote via ssh: generic runners, named verbs, sudo probing."""

    @staticmethod
    def capture(user: str, host: str, remote_cmd: str, *, timeout: float = 15) -> str | None:
        """Run ``remote_cmd`` non-interactively and return stripped stdout, or ``None`` on any
        failure. Used for state probes (e.g. ``uname -sm``) where "can't tell" is a valid answer."""
        cmd = SSHArgv.base(user, host, list(SSHArgv.PROBE_OPTS)) + [remote_cmd]
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
        """Run ``remote_cmd`` on the remote and return its exit code.

        ``quiet=True`` (default): stdout/stderr → ``/dev/null``, spawn/timeout exceptions become
        rc ``-1``. ``quiet=False``: inherit the terminal, exceptions propagate.

        ``error``: if set and rc is non-zero, raise :class:`RuntimeError` with this message.
        """
        cmd = SSHArgv.base(user, host, opts) + [remote_cmd]
        if log:
            log_cmd(cmd)
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
        """Fire-and-forget ``pkill -f <pat>`` on the remote.

        With ``sudo=True``: uses ``sudo -S`` when ``sudo_password`` is given, else ``sudo -n``
        (NOPASSWD only).
        """
        cmd = ShellText.pkill(pat)
        if not sudo:
            return RemoteOps.run(user, host, cmd)
        if sudo_password is not None:
            return RemoteOps.run(user, host, ShellText.sudo_pw(cmd), input=sudo_password + "\n")
        return RemoteOps.run(user, host, ShellText.sudo_np(cmd))

    @staticmethod
    def mkdir(user: str, host: str, path: str) -> None:
        """``mkdir -p`` on the remote.

        Raises:
            RuntimeError: If the remote ``mkdir`` returned non-zero.
        """
        RemoteOps.run(
            user,
            host,
            ShellText.mkdir_p(path),
            quiet=False,
            log=True,
            error=f"failed to create remote directory {path!r}",
        )

    @staticmethod
    def rmdir(user: str, host: str, path: str, *, force: bool = False) -> None:
        """Recursively remove a remote directory. Refuses to delete ``/`` or ``$HOME``.

        ``force=True`` re-raises SSH/remote errors; default swallows them (best-effort teardown).
        The safety refusal always raises.

        Raises:
            ValueError: If ``path`` resolves to ``/`` or ``$HOME``.
            RuntimeError: If ``force`` is set and the remote ``rm`` returned non-zero.
        """
        remote = (
            f"ws={ShellText.path(path)}; "
            'd=$(cd "$ws" 2>/dev/null && pwd) || exit 0; '
            'if [ -z "$d" ] || [ "$d" = "/" ] || [ "$d" = "$HOME" ]; then exit 3; fi; '
            'rm -rf "$d"'
        )
        rc = RemoteOps.run(user, host, remote, timeout=30, log=True)
        if rc == 3:
            raise ValueError(
                f"refusing to remove {path!r}: it resolves to '/' or the home directory"
            )
        if rc != 0 and force:
            raise RuntimeError(f"failed to remove remote directory {path!r} (ssh rc={rc})")

    @staticmethod
    def needs_sudo_password(user: str, host: str) -> bool:
        """Check key-based SSH works, and whether remote sudo needs a password.

        Returns:
            bool: ``True`` if remote sudo needs a password, ``False`` if NOPASSWD.

        Raises:
            RuntimeError: If SSH itself failed (rc 255), with an ``ssh-copy-id`` hint.
        """
        rc = RemoteOps.run(
            user, host, ShellText.sudo_probe(), opts=list(SSHArgv.PROBE_OPTS), log=True
        )
        if rc == 255:
            raise RuntimeError(
                f"SSH to {user}@{host} needs a password. Install your key:\n"
                f"    ssh-copy-id {user}@{host}"
            )
        return rc != 0

    @staticmethod
    def resolve_sudo(user: str, host: str, sudo_password: str | None = None) -> str | None:
        """Return the remote sudo password, or ``None`` if NOPASSWD is set.

        Precedence: explicit ``sudo_password`` > one-time getpass prompt.

        Raises:
            RuntimeError: If SSH is unreachable or the user aborted the prompt.
        """
        if not RemoteOps.needs_sudo_password(user, host):
            return None
        if sudo_password is not None:
            return sudo_password
        logger.info("remote sudo needs a password (no NOPASSWD) — prompting once")
        try:
            return getpass.getpass(f"[remote] sudo password for {user}@{host}: ")
        except (EOFError, KeyboardInterrupt) as e:
            raise RuntimeError(
                "no sudo password provided — pass sudo_password= or run interactively"
            ) from e


class SCP:
    """``scp`` wrapper. Shares :class:`SSHArgv`'s control-socket multiplexing so copies piggyback on
    an already-authenticated connection."""

    @staticmethod
    def copy(user: str, host: str, files: list[Path], dest: str) -> None:
        """Copy ``files`` into ``user@host:dest/``. Tries the modern SFTP backend, then retries
        with the legacy protocol (``-O``) on hosts without an SFTP subsystem.

        Raises:
            RuntimeError: If both attempts fail.
        """

        def _once(legacy: bool) -> int:
            cmd = [
                "scp",
                *(["-O"] if legacy else []),
                *SSHArgv.ctl_opts(),
                "-v" if verbose_level() >= 2 else "-q",
                *[str(f) for f in files],
                f"{user}@{host}:{dest}/",
            ]
            log_cmd(cmd)
            return subprocess.run(cmd).returncode

        if _once(legacy=False) == 0:
            return
        # scp prints its own failure straight to stderr, so the explanation has to go there too:
        # this module's logger carries a NullHandler, which suppresses logging's last-resort output.
        print(
            "[scp] the SFTP backend is declining. Retrying it with the legacy protocol (scp -O), "
            "which hosts without an SFTP subsystem need",
            file=sys.stderr,
        )
        if _once(legacy=True) != 0:
            raise RuntimeError(f"scp to {user}@{host}:{dest}/ failed")

    @staticmethod
    def deploy(user: str, host: str, sources: list[Path], workspace: str) -> None:
        """Create ``workspace`` on the remote, then copy every source into it.

        Raises:
            RuntimeError: If a source is missing, or if nothing would be copied.
        """
        missing = [str(p) for p in sources if not p.exists()]
        if missing:
            raise RuntimeError(f"nothing to deploy at: {missing}")
        files: list[Path] = []
        for src in sources:
            if src.is_dir():
                files += sorted(p for p in src.iterdir() if p.is_file() and p.name != "README.md")
            else:
                files.append(src)
        if not files:
            raise RuntimeError(
                f"no artifacts in {[str(p) for p in sources]} — cross-compile the executor + "
                "runtime libs for the target first, and point deploy= at what holds them."
            )
        total = sum(f.stat().st_size for f in files)
        logger.info(
            f"copying {len(files)} artifact(s), {total/1e6:.1f} MB -> {user}@{host}:{workspace}/"
        )
        for f in files:
            logger.debug(f"  - {f.name}  ({f.stat().st_size/1e6:.2f} MB)")
        RemoteOps.mkdir(user, host, workspace)
        t0 = time.monotonic()
        SCP.copy(user, host, files, workspace)
        logger.debug(f"copied in {time.monotonic() - t0:.1f}s")


class RemoteLauncher:
    """Builds the argv opening an SSH tunnel and launching ``catalyst-executor`` on a remote host."""

    @staticmethod
    def ssh_argv(
        user: str,
        host: str,
        workspace: str,
        port: int,
        plugins: list[str],
        env: dict[str, str],
        *,
        sudo: bool = False,
        sudo_password: str | None = None,
        executor_bin: str = f"./{ExecutorPaths.EXECUTOR_BIN}",
    ) -> list[str]:
        """Full ``ssh -L ...`` argv that opens a port-forward and starts ``catalyst-executor`` on
        the remote host.

        ``port`` is used at both ends of the forward: the executor binds it on the remote, and the
        tunnel listens on it here. ``env`` and ``plugins`` values are shell-quoted.
        """
        use_pw = sudo_password is not None
        remote_cmd = RemoteLauncher._remote_cmd(
            workspace,
            port,
            plugins,
            env,
            sudo=sudo,
            use_password=use_pw,
            executor_bin=executor_bin,
        )
        logger.debug(f"remote: {remote_cmd}")
        opts = RemoteLauncher._ssh_opts(port, use_pw)
        return SSHArgv.base(user, host, opts, multiplex=False) + [remote_cmd]

    @staticmethod
    def _remote_cmd(
        workspace: str,
        remote_port: int,
        plugins: list[str],
        env: dict[str, str],
        *,
        sudo: bool,
        use_password: bool,
        executor_bin: str,
    ) -> str:
        """Remote shell command: ``cd`` into workspace, exec the executor with ``env`` applied."""
        env_prefix = RemoteLauncher._env_prefix(env)
        return (
            f"cd {ShellText.path(workspace)} "
            f"&& {RemoteLauncher._chmod_prefix(executor_bin)}"
            f"{RemoteLauncher._exec_prefix(sudo, use_password)} "
            f"{f'env {env_prefix} ' if env_prefix else ''}"
            f"{executor_bin} {ExecutorFlags.BIND_FLAG}{ExecutorFlags.BIND_HOST}:{remote_port} "
            f"{RemoteLauncher._plugin_args(plugins)}"
        )

    @staticmethod
    def _ssh_opts(port: int, use_password: bool) -> list[str]:
        """Local-side ssh options for the port-forward. ``-tt`` on NOPASSWD so SSH close
        SIGHUPs the executor; omitted with a password so ``sudo -S`` sees an unechoed pipe."""
        opts = [
            "-o",
            "ExitOnForwardFailure=yes",
            "-L",
            f"{port}:localhost:{port}",
        ]
        if not use_password:
            opts = ["-tt"] + opts
        return opts

    @staticmethod
    def _env_prefix(env: dict[str, str]) -> str:
        """``K=V K=V ...`` prefix, with the values shell-quoted."""
        return " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())

    @staticmethod
    def _plugin_args(plugins: list[str]) -> str:
        """``--plugin=<path>`` args. Bare filenames resolve against ``$PWD``; ``~``/absolute
        paths are quoted with tilde expansion."""

        def arg(p: str) -> str:
            flag = ExecutorFlags.PLUGIN_FLAG
            if "/" not in p and not p.startswith("~"):
                return f"{flag}$PWD/{shlex.quote(p)}"
            return f"{flag}{ShellText.path(p)}"

        return " ".join(arg(p) for p in plugins)

    @staticmethod
    def _chmod_prefix(executor_bin: str) -> str:
        """``chmod +x`` for a workspace-local binary (scp drops the +x bit); empty otherwise."""
        local_bin = f"./{ExecutorPaths.EXECUTOR_BIN}"
        if local_bin not in executor_bin:
            return ""
        return f"chmod +x {local_bin} 2>/dev/null; "

    @staticmethod
    def _exec_prefix(sudo: bool, use_password: bool) -> str:
        """``exec [sudo ...]`` prefix. ``sudo=False``: plain ``exec``; NOPASSWD: ``sudo -E``;
        with password: ``sudo -S -E`` (piped, no PTY)."""
        if not sudo:
            return "exec"
        return "exec sudo -S -E -p ''" if use_password else "exec sudo -E"
