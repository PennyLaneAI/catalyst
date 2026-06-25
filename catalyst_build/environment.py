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

"""Build environment discovery and configuration.

This module centralizes everything the Makefiles previously computed with shell
substitutions (`$(shell which ...)`, `$(shell uname -s)`, `$(abspath ...)`),
replacing them with portable Python (`shutil.which`, `platform.system`,
`pathlib`). All paths are derived relative to the repository root, mirroring the
`MK_DIR`-relative layout of the original Makefiles.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _bool_env(name: str, default: bool) -> bool:
    """Parse an ON/OFF/1/0/true/false environment override; fall back to default."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"on", "1", "true", "yes"}


def _str_env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val is not None and val != "" else default


def find_tool(*candidates: str, env_var: Optional[str] = None) -> Optional[str]:
    """Locate a build tool, honoring an environment override first.

    Mirrors the Makefile idiom `TOOL ?= $(shell which tool)` but cross-platform:
    on Windows `shutil.which` also resolves `.exe`/`.bat`/`.cmd` via PATHEXT.
    """
    if env_var:
        override = os.environ.get(env_var)
        if override:
            return override
    for cand in candidates:
        found = shutil.which(cand)
        if found:
            return found
    return None


def default_nproc() -> int:
    """Number of parallel jobs, matching the Makefile `os.cpu_count()` default."""
    try:
        n = os.cpu_count()
        return n if n and n > 0 else 1
    except NotImplementedError:  # pragma: no cover
        return 1


@dataclass
class BuildEnv:
    """Resolved build configuration shared by all components.

    Field defaults reproduce the Makefile variable defaults exactly; each can be
    overridden by an environment variable (Makefile `?=` semantics) or by a
    command-line flag handled in `cli.py`.
    """

    # --- repository layout (MK_DIR-relative in the Makefiles) ---
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # --- toolchain ---
    python: str = field(default_factory=lambda: sys.executable)
    c_compiler: Optional[str] = None
    cxx_compiler: Optional[str] = None
    compiler_launcher: Optional[str] = None  # ccache, optional
    cmake: Optional[str] = None
    ninja: Optional[str] = None

    # --- common knobs (shared across components) ---
    build_type: str = field(default_factory=lambda: _str_env("BUILD_TYPE", ""))
    enable_asan: bool = field(default_factory=lambda: _bool_env("ENABLE_ASAN", False))
    strict_warnings: bool = field(default_factory=lambda: _bool_env("STRICT_WARNINGS", True))
    nproc: int = field(default_factory=lambda: int(_str_env("NPROC", str(default_nproc()))))
    verbose: bool = field(default_factory=lambda: _bool_env("VERBOSE", False))

    # --- runtime component knobs ---
    enable_openqasm: bool = field(default_factory=lambda: _bool_env("ENABLE_OPENQASM", True))
    enable_oqd: bool = field(default_factory=lambda: _bool_env("ENABLE_OQD", False))
    code_coverage: bool = field(default_factory=lambda: _bool_env("CODE_COVERAGE", False))

    # --- mlir component knobs ---
    enable_lld: Optional[bool] = None  # platform-dependent default, resolved below
    enable_zlib: bool = field(default_factory=lambda: _bool_env("ENABLE_ZLIB", True))
    enable_zstd: bool = field(default_factory=lambda: _bool_env("ENABLE_ZSTD", False))
    llvm_targets_to_build: str = field(
        default_factory=lambda: _str_env("LLVM_TARGETS_TO_BUILD", "host")
    )

    # --- frontend knobs ---
    pip_verbose: bool = field(default_factory=lambda: _bool_env("VERBOSE", False))

    # --- dry-run plumbing (set by CLI) ---
    dry_run: bool = False

    def __post_init__(self) -> None:
        # Resolve toolchain with env overrides, mirroring the Makefile `?=` chain.
        # On Windows the conventional default compiler is clang-cl/cl; we still
        # prefer clang/clang++ when present for parity with the POSIX build.
        if self.c_compiler is None:
            self.c_compiler = find_tool("clang", "cl", "gcc", "cc", env_var="C_COMPILER")
        if self.cxx_compiler is None:
            self.cxx_compiler = find_tool("clang++", "cl", "g++", "c++", env_var="CXX_COMPILER")
        if self.compiler_launcher is None:
            # ccache is optional; Makefiles pass an empty value when absent.
            self.compiler_launcher = find_tool("ccache", env_var="COMPILER_LAUNCHER")
        if self.cmake is None:
            self.cmake = find_tool("cmake", env_var="CMAKE")
        if self.ninja is None:
            self.ninja = find_tool("ninja", env_var="NINJA")

        # LLD default: ON everywhere except macOS (matches mlir/Makefile).
        if self.enable_lld is None:
            env_lld = os.environ.get("ENABLE_LLD")
            if env_lld is not None:
                self.enable_lld = env_lld.strip().lower() in {"on", "1", "true", "yes"}
            else:
                self.enable_lld = self.platform != "Darwin"

    # ----- platform helpers (replacing `$(shell uname -s)`) -----
    @property
    def platform(self) -> str:
        """'Linux', 'Darwin', or 'Windows' (platform.system() values)."""
        return platform.system()

    @property
    def is_windows(self) -> bool:
        return self.platform == "Windows"

    @property
    def is_macos(self) -> bool:
        return self.platform == "Darwin"

    @property
    def is_linux(self) -> bool:
        return self.platform == "Linux"

    # ----- canonical build directories (MK_DIR-relative) -----
    @property
    def mlir_dir(self) -> Path:
        return self.root / "mlir"

    @property
    def llvm_build_dir(self) -> Path:
        return Path(_str_env("LLVM_BUILD_DIR", str(self.mlir_dir / "llvm-project" / "build")))

    @property
    def stablehlo_build_dir(self) -> Path:
        return Path(_str_env("STABLEHLO_BUILD_DIR", str(self.mlir_dir / "stablehlo" / "build")))

    @property
    def enzyme_build_dir(self) -> Path:
        return Path(_str_env("ENZYME_BUILD_DIR", str(self.mlir_dir / "Enzyme" / "build")))

    @property
    def dialects_build_dir(self) -> Path:
        return Path(_str_env("DIALECTS_BUILD_DIR", str(self.mlir_dir / "build")))

    @property
    def dialects_docs_build_dir(self) -> Path:
        return Path(_str_env("DIALECTS_DOCS_BUILD_DIR", str(self.mlir_dir / "build-docs")))

    @property
    def runtime_dir(self) -> Path:
        return self.root / "runtime"

    @property
    def rt_build_dir(self) -> Path:
        return Path(_str_env("RT_BUILD_DIR", str(self.runtime_dir / "build")))

    @property
    def oqc_src_dir(self) -> Path:
        return self.root / "frontend" / "catalyst" / "third_party" / "oqc" / "src"

    @property
    def oqc_build_dir(self) -> Path:
        return Path(_str_env("OQC_BUILD_DIR", str(self.oqc_src_dir / "build")))

    @property
    def frontend_dir(self) -> Path:
        return self.root / "frontend"

    # ----- derived toolchain knobs -----
    def numpy_include(self) -> str:
        """Resolve numpy's include dir via the configured interpreter."""
        out = subprocess.check_output(
            [self.python, "-c", "import numpy as np; print(np.get_include())"]
        )
        return out.decode().strip()

    def libpython_path(self) -> str:
        """Path to the python shared lib, used by the lit test target."""
        out = subprocess.check_output(
            [
                self.python,
                "-c",
                "from catalyst.utils.runtime_environment import get_libpython_path;"
                " print(get_libpython_path())",
            ]
        )
        return out.decode().strip()

    def using_clang(self) -> bool:
        return bool(self.c_compiler) and "clang" in os.path.basename(self.c_compiler).lower()

    def require(self, *tools: str) -> None:
        """Raise a clear error if a required tool was not resolved.

        Under dry-run we only warn: previewing the planned commands must not
        depend on the toolchain actually being installed.
        """
        missing = []
        for t in tools:
            if getattr(self, t, None) in (None, ""):
                missing.append(t)
        if not missing:
            return
        pretty = ", ".join(missing)
        if self.dry_run:
            print(f"[catalyst-build] (dry-run) warning: tool(s) not found: {pretty}",
                  flush=True)
            return
        raise EnvironmentError(
            f"Required build tool(s) not found: {pretty}. "
            f"Install them or set the matching environment override "
            f"(e.g. CMAKE=/path/to/cmake)."
        )
