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

* :class:`SSH`: ``ssh`` wrapper for argv, auth probing, and remote filesystem verbs.
* :class:`SCP`: ``scp`` wrapper that shares SSH's multiplexed control socket.
* :class:`RemoteLauncher`: builds the remote ``catalyst-executor`` launch command.
"""

from __future__ import annotations

import getpass
import os
import shlex
import subprocess
import time
from pathlib import Path

from .utils import ExecutorCli, Paths, Raw, ShellCommand, log_cmd, logger, verbose_level


class SSH:
    """Local ``ssh`` command builder and runner: argv, auth probing, sudo-password resolution,
    and remote filesystem verbs."""

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

    # One-shot probe flags: fail fast on missing key (rc 255), short connect timeout.
    PROBE_OPTS: tuple[str, ...] = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")

    # Bytes available to a ControlPath: ``sun_path`` is 104 on macOS (108 on Linux), and while the
    # master socket is being created ssh appends a random suffix to the resolved path. Exceeding it
    # fails every multiplexed op with "unix_listener: path ... too long", so an over-long path
    # disables multiplexing instead (see :meth:`ctl_opts`).
    CONTROL_PATH_MAX = 104
    CONTROL_PATH_RESERVE = 20

    @staticmethod
    def _fallback_ctl_dir() -> Path:
        """Control-socket dir when ``$XDG_RUNTIME_DIR`` isn't set."""
        return Path.home() / ".cache" / "catalyst" / "ssh-cm"

    @staticmethod
    def _ctl_dir() -> Path:
        """Control-socket dir under ``$XDG_RUNTIME_DIR``. Defaults to :meth:`_fallback_ctl_dir`."""
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        base = Path(xdg) / "catalyst" if xdg else SSH._fallback_ctl_dir()
        base.mkdir(parents=True, exist_ok=True)
        return base

    @staticmethod
    def ctl_opts() -> list[str]:
        """``ControlMaster``/``ControlPath``/``ControlPersist`` flags for connection multiplexing.

        The first op opens a master socket under :meth:`_ctl_dir`; later ops
        (probe/mkdir/scp/pkill) reuse it, skipping the auth handshake. ``%C`` (a hash of
        ``%l%h%p%r``) keeps the name short. The socket self-expires ``CONTROL_PERSIST`` seconds
        after its last user. Not applied to the long-lived executor session, whose close SIGHUPs
        the executor cleanly.
        """
        return SSH._ctl_flags(SSH._ctl_dir())

    @staticmethod
    def _ctl_flags(base: Path) -> list[str]:
        """:meth:`ctl_opts` for a socket dir, split out so the length rule is decidable
        without touching the environment or the filesystem.

        Returns no flags when the path would not fit a Unix socket: multiplexing is an
        optimisation, and one auth handshake per op still works.
        """
        path = f"{base}/cm-%C"
        # %C expands to a 40-character hash, plus the suffix ssh appends while creating the master.
        expanded = len(path) - len("%C") + 40 + SSH.CONTROL_PATH_RESERVE
        if expanded > SSH.CONTROL_PATH_MAX:
            logger.debug(
                "ControlPath %r would exceed the %d-byte socket limit; disabling multiplexing",
                path,
                SSH.CONTROL_PATH_MAX,
            )
            return []
        return [
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={path}",
            "-o",
            f"ControlPersist={SSH.CONTROL_PERSIST}",
        ]

    @staticmethod
    def base_cmd(
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
        cmd = list(SSH.BASE_CMD)
        if multiplex:
            cmd += SSH.ctl_opts()
        cmd += ["-v"] * max(0, verbose_level() - 2)  # ssh protocol debug at -vvv (verbosity 3+)
        if opts:
            cmd += opts
        cmd.append(f"{user}@{host}")
        return cmd

    @staticmethod
    def capture(user: str, host: str, remote_cmd: str, *, timeout: float = 15) -> str | None:
        """Run ``remote_cmd`` non-interactively and return stripped stdout, or ``None`` on any
        failure. Used for state probes (e.g. ``uname -sm``) where "can't tell" is a valid answer."""
        cmd = SSH.base_cmd(user, host, list(SSH.PROBE_OPTS)) + [remote_cmd]
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
        cmd = SSH.base_cmd(user, host, opts) + [remote_cmd]
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
        cmd = ShellCommand.pkill(pat)
        if not sudo:
            return SSH.run(user, host, cmd)
        if sudo_password is not None:
            return SSH.run(user, host, ShellCommand.sudo_pw(cmd), input=sudo_password + "\n")
        return SSH.run(user, host, ShellCommand.sudo_np(cmd))

    @staticmethod
    def mkdir(user: str, host: str, path: str) -> None:
        """``mkdir -p`` on the remote.

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
    def rmdir(user: str, host: str, path: str, *, force: bool = False) -> None:
        """Recursively remove a remote directory. Refuses to delete ``/`` or ``$HOME``.

        ``force=True`` re-raises SSH/remote errors; default swallows them (best-effort teardown).
        The safety refusal always raises.

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
    def needs_sudo_password(user: str, host: str) -> bool:
        """Check key-based SSH works, and whether remote sudo needs a password.

        Returns:
            bool: ``True`` if remote sudo needs a password, ``False`` if NOPASSWD.

        Raises:
            RuntimeError: If SSH itself failed (rc 255), listing the causes that produce it.
        """
        rc = SSH.run(user, host, ShellCommand.sudo_probe(), opts=list(SSH.PROBE_OPTS), log=True)
        if rc == 255:
            raise RuntimeError(
                f"ssh to {user}@{host} failed (exit 255). Common causes:\n"
                f"  * no usable key — install yours once: ssh-copy-id {user}@{host}\n"
                f"  * the host key changed (e.g. reimaged): ssh-keygen -R {host}\n"
                "  * the host is unreachable or refusing connections.\n"
                f"Run 'ssh -v {user}@{host} true' to see which."
            )
        return rc != 0

    @staticmethod
    def resolve_sudo(user: str, host: str, sudo_password: str | None = None) -> str | None:
        """Return the remote sudo password, or ``None`` if NOPASSWD is set.

        Precedence: explicit ``sudo_password`` > one-time getpass prompt.

        Raises:
            RuntimeError: If SSH is unreachable or the user aborted the prompt.
        """
        if not SSH.needs_sudo_password(user, host):
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
    """``scp`` wrapper. Shares :class:`SSH`'s control-socket multiplexing so copies piggyback on
    an already-authenticated connection."""

    @staticmethod
    def run(user: str, host: str, files: list[Path], dest: str, *, log: bool = True) -> None:
        """Copy ``files`` into ``user@host:dest/``. Tries the modern SFTP backend, then retries
        with the legacy protocol (``-O``) on hosts without an SFTP subsystem.

        Raises:
            RuntimeError: If both attempts fail.
        """

        def _once(legacy: bool) -> int:
            cmd = [
                "scp",
                *(["-O"] if legacy else []),
                *SSH.ctl_opts(),
                "-v" if verbose_level() >= 2 else "-q",
                *[str(f) for f in files],
                f"{user}@{host}:{dest}/",
            ]
            if log:
                log_cmd(cmd)
            return subprocess.run(cmd).returncode

        if _once(legacy=False) == 0:
            return
        logger.info("scp failed — retrying with the legacy protocol (scp -O)")
        if _once(legacy=True) != 0:
            raise RuntimeError(f"scp to {user}@{host}:{dest}/ failed")

    @staticmethod
    def deploy(user: str, host: str, bundle: Path, workspace: str) -> None:
        """Create ``workspace`` on the remote, then copy every artifact in ``bundle`` to it.

        Raises:
            RuntimeError: If ``bundle`` has no artifacts.
        """
        files = sorted(p for p in bundle.iterdir() if p.is_file() and p.name != "README.md")
        if not files:
            raise RuntimeError(
                f"no artifacts in {bundle} — pass build=<recipe> to cross-compile the executor + "
                "runtime libs for the target, or point bundle= at a prebuilt directory."
            )
        total = sum(f.stat().st_size for f in files)
        logger.info(
            f"copying {len(files)} artifact(s), {total/1e6:.1f} MB -> {user}@{host}:{workspace}/"
        )
        for f in files:
            logger.debug(f"  - {f.name}  ({f.stat().st_size/1e6:.2f} MB)")
        SSH.mkdir(user, host, workspace)
        t0 = time.monotonic()
        SCP.run(user, host, files, workspace)
        logger.debug(f"copied in {time.monotonic() - t0:.1f}s")


class RemoteLauncher:
    """Builds the argv that opens an SSH tunnel and launches ``catalyst-executor`` on a remote host."""

    @staticmethod
    def ssh_argv(
        user: str,
        host: str,
        workspace: str,
        remote_port: int,
        local_port: int,
        plugins: list[str],
        env: dict[str, str],
        *,
        sudo: bool = True,
        sudo_password: str | None = None,
        executor_bin: str = f"./{Paths.EXECUTOR_BIN}",
    ) -> list[str]:
        """Full ``ssh -L ...`` argv that opens a port-forward and starts ``catalyst-executor`` on
        the remote host.

        Bare-string values in ``env``/``plugins`` are shell-quoted; wrap in
        :class:`~catalyst.executor.utils.Raw` to expand ``$VAR`` on the remote instead.
        """
        use_pw = sudo_password is not None
        remote_cmd = RemoteLauncher._remote_cmd(
            workspace,
            remote_port,
            plugins,
            env,
            sudo=sudo,
            use_password=use_pw,
            executor_bin=executor_bin,
        )
        logger.debug(f"remote: {remote_cmd}")
        opts = RemoteLauncher._ssh_opts(local_port, remote_port, use_pw)
        return SSH.base_cmd(user, host, opts, multiplex=False) + [remote_cmd]

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
        """Remote shell command: ``cd`` into workspace, export env, exec the executor."""
        return (
            f"cd {ShellCommand.path(workspace)} "
            f"&& {RemoteLauncher._chmod_prefix(executor_bin)}"
            f"{RemoteLauncher._env_prefix(env)} "
            f"{RemoteLauncher._exec_prefix(sudo, use_password)} "
            f"{executor_bin} {ExecutorCli.BIND_FLAG}0.0.0.0:{remote_port} "
            f"{RemoteLauncher._plugin_args(plugins)}"
        )

    @staticmethod
    def _ssh_opts(local_port: int, remote_port: int, use_password: bool) -> list[str]:
        """Local-side ssh options for the port-forward. ``-tt`` on NOPASSWD so SSH close
        SIGHUPs the executor; omitted with a password so ``sudo -S`` sees an unechoed pipe."""
        opts = [
            "-o",
            "ExitOnForwardFailure=yes",
            "-L",
            f"{local_port}:localhost:{remote_port}",
        ]
        if not use_password:
            opts = ["-tt"] + opts
        return opts

    @staticmethod
    def _env_prefix(env: dict[str, str]) -> str:
        """``K=V K=V ...`` prefix. :class:`Raw` values expand on the remote; bare strings are quoted."""

        def q(v):
            return v if isinstance(v, Raw) else shlex.quote(v)

        return " ".join(f"{k}={q(v)}" for k, v in env.items())

    @staticmethod
    def _plugin_args(plugins: list[str]) -> str:
        """``--plugin=<path>`` args. Bare filenames resolve against ``$PWD``; ``~``/absolute
        paths are quoted with tilde expansion."""

        def arg(p):
            flag = ExecutorCli.PLUGIN_FLAG
            if isinstance(p, Raw):
                return f"{flag}{p}"
            if "/" not in p and not p.startswith("~"):
                return f"{flag}$PWD/{shlex.quote(p)}"
            return f"{flag}{ShellCommand.path(p)}"

        return " ".join(arg(p) for p in plugins)

    @staticmethod
    def _chmod_prefix(executor_bin: str) -> str:
        """``chmod +x`` for a workspace-local binary (scp drops the +x bit); empty otherwise."""
        if executor_bin != f"./{Paths.EXECUTOR_BIN}":
            return ""
        return f"chmod +x ./{Paths.EXECUTOR_BIN} 2>/dev/null; "

    @staticmethod
    def _exec_prefix(sudo: bool, use_password: bool) -> str:
        """``exec [sudo ...]`` prefix. ``sudo=False``: plain ``exec``; NOPASSWD: ``sudo -E``;
        with password: ``sudo -S -E`` (piped, no PTY)."""
        if not sudo:
            return "exec"
        return "exec sudo -S -E -p ''" if use_password else "exec sudo -E"
