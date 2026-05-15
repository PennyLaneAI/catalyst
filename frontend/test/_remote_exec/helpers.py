import os
import pathlib
import platform
import re
import subprocess
import sys

SHARED_LIB_EXT = ".dylib" if platform.system() == "Darwin" else ".so"


def build_shared_lib(src: pathlib.Path, dst: pathlib.Path) -> pathlib.Path:
    """Compile `src` (.c) into a platform shared library at `dst`. Skipped if up to date."""
    if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst

    cc = os.environ.get("CC", "clang")
    cmd = [cc, "-shared", "-fPIC", "-O2"]
    if platform.system() == "Darwin":
        cmd += ["-dynamiclib", "-install_name", str(dst)]
    cmd += ["-o", str(dst), str(src)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"Failed to build {dst.name}:\n{result.stderr}")
    return dst


def patch_artifact_path(ir_text: str, lib_path: pathlib.Path) -> str:
    """Replace the single path inside `catalyst.runtime_artifacts = [...]` with `lib_path`."""
    return re.sub(
        r'(catalyst\.runtime_artifacts\s*=\s*\[)"[^"]*"(\])',
        lambda m: f'{m.group(1)}"{lib_path}"{m.group(2)}',
        ir_text,
        count=1,
    )
