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

"""Portable process execution and filesystem helpers.

These replace the POSIX shell verbs the Makefiles relied on (`cp --dereference`,
`rm -rf`, `find ... -exec`, glob expansion, `mkdir -p`) with cross-platform
Python equivalents, and provide a single `run` entry point that honors dry-run
and verbose modes. Subprocesses are launched WITHOUT a shell (list argv), so
there is no dependency on bash/sh/cmd quoting rules.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

StrPath = Union[str, os.PathLike]


class BuildError(RuntimeError):
    """Raised when a build step fails (non-zero exit, missing artifact, etc.)."""


def _fmt_cmd(cmd: Sequence[str]) -> str:
    return " ".join(str(c) for c in cmd)


def run(
    cmd: Sequence[StrPath],
    *,
    cwd: Optional[StrPath] = None,
    env: Optional[Mapping[str, str]] = None,
    extra_env: Optional[Mapping[str, str]] = None,
    dry_run: bool = False,
    check: bool = True,
) -> int:
    """Run a subprocess with no shell, streaming output to the console.

    Args:
        cmd: argv list (never a shell string).
        cwd: working directory.
        env: full environment to use; defaults to the current environment.
        extra_env: overlay applied on top of `env`/os.environ.
        dry_run: when True, print the command and skip execution.
        check: raise BuildError on non-zero exit.
    """
    argv = [str(c) for c in cmd]
    loc = f" (cwd={cwd})" if cwd else ""
    print(f"[run]{loc} {_fmt_cmd(argv)}", flush=True)
    if dry_run:
        return 0

    run_env = dict(env if env is not None else os.environ)
    if extra_env:
        run_env.update(extra_env)

    try:
        proc = subprocess.run(argv, cwd=str(cwd) if cwd else None, env=run_env, check=False)
    except FileNotFoundError as err:
        raise BuildError(f"Executable not found: {argv[0]} ({err})") from err

    if check and proc.returncode != 0:
        raise BuildError(f"Command failed (exit {proc.returncode}): {_fmt_cmd(argv)}")
    return proc.returncode


def run_capture(cmd: Sequence[StrPath], *, cwd: Optional[StrPath] = None) -> str:
    """Run and return stdout as text (used for tool-version probing)."""
    argv = [str(c) for c in cmd]
    return subprocess.check_output(argv, cwd=str(cwd) if cwd else None).decode()


# --------------------------------------------------------------------------
# Filesystem helpers (portable replacements for cp/rm/mkdir/find)
# --------------------------------------------------------------------------
def ensure_dir(path: StrPath, *, dry_run: bool = False) -> None:
    """`mkdir -p`."""
    print(f"[mkdir] {path}", flush=True)
    if not dry_run:
        Path(path).mkdir(parents=True, exist_ok=True)


def remove(path: StrPath, *, dry_run: bool = False) -> None:
    """`rm -rf` for a file or directory; silent if absent."""
    p = Path(path)
    print(f"[rm] {p}", flush=True)
    if dry_run:
        return
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p, ignore_errors=True)
    elif p.exists() or p.is_symlink():
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def copy(src: StrPath, dst: StrPath, *, follow_symlinks: bool = True, dry_run: bool = False) -> None:
    """Copy a single file, creating the destination directory if needed.

    `follow_symlinks=True` reproduces the Makefile `cp --dereference` behaviour
    used on Linux for the LLVM libs.
    """
    src_p, dst_p = Path(src), Path(dst)
    print(f"[cp] {src_p} -> {dst_p}", flush=True)
    if dry_run:
        return
    if dst_p.is_dir():
        dst_p = dst_p / src_p.name
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_p, dst_p, follow_symlinks=follow_symlinks)


def copy_glob(
    pattern: str,
    dst_dir: StrPath,
    *,
    root: Optional[StrPath] = None,
    optional: bool = False,
    follow_symlinks: bool = True,
    dry_run: bool = False,
) -> int:
    """Copy every file matching a glob into a destination directory.

    Replaces shell `cp $(BUILD)/lib/foo* dst/`. Returns the number of files
    copied; raises if zero matches and not `optional`.
    """
    base = Path(root) if root else Path(".")
    matches = sorted(base.glob(pattern))
    if not matches:
        msg = f"[cp-glob] no match for {pattern} (under {base})"
        if optional:
            print(msg + " (optional, skipped)", flush=True)
            return 0
        raise BuildError(msg)
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    for m in matches:
        copy(m, Path(dst_dir) / m.name, follow_symlinks=follow_symlinks, dry_run=dry_run)
    return len(matches)


def copy_tree(src: StrPath, dst: StrPath, *, dry_run: bool = False) -> None:
    """Recursively copy a directory tree (`cp -R`)."""
    print(f"[cp -R] {src} -> {dst}", flush=True)
    if dry_run:
        return
    shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=False)


def find_files(roots: Sequence[StrPath], suffixes: Sequence[str]) -> list[Path]:
    """Recursive file search by suffix (replaces `find ... -name '*.ext'`)."""
    out: list[Path] = []
    for r in roots:
        rp = Path(r)
        if not rp.exists():
            continue
        for f in rp.rglob("*"):
            if f.is_file() and any(f.name.endswith(s) for s in suffixes):
                out.append(f)
    return out


def remove_dirs_named(root: StrPath, name: str, *, dry_run: bool = False) -> None:
    """Remove every directory named `name` under root (e.g. __pycache__)."""
    for d in Path(root).rglob(name):
        if d.is_dir():
            remove(d, dry_run=dry_run)


def remove_glob(pattern: str, *, root: Optional[StrPath] = None, dry_run: bool = False) -> None:
    """`rm -rf` every path matching a glob; silent if none."""
    base = Path(root) if root else Path(".")
    for m in base.glob(pattern):
        remove(m, dry_run=dry_run)


def write_text(path: StrPath, content: str, *, dry_run: bool = False) -> None:
    """Write a small text file (replaces `echo ... > file`)."""
    print(f"[write] {path}", flush=True)
    if not dry_run:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
