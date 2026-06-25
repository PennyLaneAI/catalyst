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

"""Wheel packaging target.

Reproduces the (large) `wheel` recipe from the top-level Makefile: stage the
runtime/MLIR/OQC libraries and headers into the frontend tree, build a
bootstrap wheel to generate the decomposition-rules resource, then build the
final wheel. All `cp`/`find`/`mkdir`/`rm` shell verbs are replaced with the
portable helpers in `runner.py`.
"""

from __future__ import annotations

from .environment import BuildEnv
from .runner import (
    copy,
    copy_glob,
    copy_tree,
    ensure_dir,
    find_files,
    remove,
    remove_dirs_named,
    run,
    write_text,
)

# Linux dereferences symlinks when copying LLVM libs (Makefile COPY_FLAGS).
def _follow(env: BuildEnv) -> bool:
    return env.is_linux


def build(env: BuildEnv) -> None:
    """`wheel` target."""
    env.require("cmake")
    root = env.root
    lib_dst = root / "frontend" / "catalyst" / "lib"
    rt_lib = env.rt_build_dir / "lib"
    oqc = env.oqc_build_dir
    llvm_lib = env.llvm_build_dir / "lib"
    dialects = env.dialects_build_dir
    follow = _follow(env)

    write_text(root / "frontend" / "catalyst" / "_configuration.py", "INSTALLED = True\n",
               dry_run=env.dry_run)

    ensure_dir(lib_dst / "backend", dry_run=env.dry_run)

    # Runtime libraries (required unless marked optional).
    copy_glob("librtd*", lib_dst, root=rt_lib, dry_run=env.dry_run)
    for name in ("catalyst_callback_registry.so", "openqasm_python_module.so",
                 "librt_capi.*", "librt_rsdecomp.*", "librt_decoder.*"):
        copy_glob(name, lib_dst, root=rt_lib, dry_run=env.dry_run)
    copy_glob("liblapacke.*", lib_dst, root=rt_lib, optional=True, dry_run=env.dry_run)
    copy_glob("*.toml", lib_dst / "backend", root=rt_lib / "backend", dry_run=env.dry_run)

    # OQC libraries.
    copy_glob("librtd_oqc*", lib_dst, root=oqc, dry_run=env.dry_run)
    copy_glob("oqc_python_module.so", lib_dst, root=oqc, dry_run=env.dry_run)
    copy_glob("*.toml", lib_dst / "backend", root=oqc / "backend", dry_run=env.dry_run)

    # LLVM/MLIR runtime support libraries.
    for name in ("libmlir_float16_utils.*", "libmlir_c_runner_utils.*", "libmlir_async_runtime.*"):
        copy_glob(name, lib_dst, root=llvm_lib, follow_symlinks=follow, dry_run=env.dry_run)
    copy_glob("libmlir_apfloat_wrappers.*", lib_dst, root=llvm_lib,
              optional=True, follow_symlinks=follow, dry_run=env.dry_run)

    # Dialect libraries.
    for name in ("default_pipelines.*", "libQuantumPythonDecompositions.*"):
        copy_glob(name, lib_dst, root=dialects / "lib", follow_symlinks=follow, dry_run=env.dry_run)

    # MLIR python bindings + compiler driver.
    mlir_quantum = root / "frontend" / "mlir_quantum"
    ensure_dir(mlir_quantum / "dialects", dry_run=env.dry_run)
    copy_tree(dialects / "python_packages" / "quantum" / "mlir_quantum" / "runtime",
              mlir_quantum / "runtime", dry_run=env.dry_run)
    dialects_py = dialects / "python_packages" / "quantum" / "mlir_quantum" / "dialects"
    for token in ("gradient", "qref", "quantum", "_ods_common", "catalyst",
                  "mbqc", "mitigation", "pbc", "_transform"):
        copy_glob(f"*{token}*", mlir_quantum / "dialects", root=dialects_py,
                  follow_symlinks=follow, optional=True, dry_run=env.dry_run)
    ensure_dir(root / "frontend" / "bin", dry_run=env.dry_run)
    copy(dialects / "bin" / "catalyst", root / "frontend" / "bin" / "catalyst",
         follow_symlinks=follow, dry_run=env.dry_run)
    remove_dirs_named(root / "frontend", "__pycache__", dry_run=env.dry_run)

    # Stage selected dialect headers into the wheel.
    _stage_headers(env)

    # Bootstrap wheel -> install -> generate resources -> final wheel.
    run([env.python, "-m", "pip", "wheel", ".", "-w", "bootstrap_dist",
         "--extra-index-url", "https://test.pypi.org/simple"], cwd=root, dry_run=env.dry_run)
    copy_glob("*.whl", root / "_bootstrap_install_tmp", root=root / "bootstrap_dist",
              optional=True, dry_run=env.dry_run)
    run([env.python, "-m", "pip", "install"] + _glob_list(root / "bootstrap_dist", "*.whl", env),
        cwd=root, dry_run=env.dry_run)
    run([env.python, "-m", "catalyst.utils.precompile_decomposition_rules"],
        cwd=root, dry_run=env.dry_run)

    ensure_dir(root / "frontend" / "catalyst" / "resources", dry_run=env.dry_run)
    bytecode = _bytecode_path(env)
    if bytecode:
        copy(bytecode, root / "frontend" / "catalyst" / "resources",
             follow_symlinks=follow, dry_run=env.dry_run)

    run([env.python, "-m", "pip", "wheel", "--no-deps", ".", "-w", "dist"],
        cwd=root, dry_run=env.dry_run)

    remove(root / "build", dry_run=env.dry_run)
    remove(root / "bootstrap_dist", dry_run=env.dry_run)
    remove(root / "_bootstrap_install_tmp", dry_run=env.dry_run)
    remove(root / "frontend" / "pennylane_catalyst.egg-info", dry_run=env.dry_run)


def plugin_wheel(env: BuildEnv) -> None:
    """`plugin-wheel` target — package the standalone plugin wheel."""
    root = env.root
    spw = root / "standalone_plugin_wheel"
    ensure_dir(spw / "standalone_plugin" / "lib", dry_run=env.dry_run)
    copy_glob("StandalonePlugin.*", spw / "standalone_plugin" / "lib",
              root=env.dialects_build_dir / "lib", follow_symlinks=_follow(env), dry_run=env.dry_run)
    run([env.python, "-m", "pip", "wheel", "--no-deps", str(spw), "-w", str(spw / "dist")],
        dry_run=env.dry_run)
    remove(spw / "standalone_plugin" / "lib", dry_run=env.dry_run)
    remove(spw / "standalone_plugin.egg-info", dry_run=env.dry_run)
    remove(spw / "build", dry_run=env.dry_run)


def _stage_headers(env: BuildEnv) -> None:
    """Copy *.h and *.h.inc for selected dialects into frontend/catalyst/include.

    Replaces the Makefile `find ... -exec sh -c '...'` header-staging block.
    Headers are taken from the build tree when present, otherwise the source
    tree, preserving their path under .../include/.
    """
    src_dir = env.mlir_dir
    build_dir = env.dialects_build_dir
    include_dst = env.root / "frontend" / "catalyst" / "include"
    follow = _follow(env)
    roots = []
    for d in ("Quantum", "Gradient", "Mitigation"):
        roots.append(src_dir / "include" / d)
        roots.append(build_dir / "include" / d)
    for hfile in find_files(roots, [".h", ".h.inc"]):
        # Determine base (build tree wins, else source tree) to preserve relative path.
        s = str(hfile)
        if str(build_dir) in s:
            base = build_dir
        else:
            base = src_dir
        try:
            rel = hfile.relative_to(base / "include")
        except ValueError:
            continue
        dest = include_dst / rel
        copy(hfile, dest, follow_symlinks=follow, dry_run=env.dry_run)


def _glob_list(directory, pattern, env: BuildEnv) -> list:
    from pathlib import Path
    if env.dry_run:
        return [str(directory / pattern)]
    return [str(p) for p in sorted(Path(directory).glob(pattern))]


def _bytecode_path(env: BuildEnv):
    if env.dry_run:
        return None
    try:
        from .runner import run_capture
        out = run_capture(
            [env.python, "-c",
             "from catalyst.utils.runtime_environment import BYTECODE_FILE_PATH;"
             " print(BYTECODE_FILE_PATH)"]
        ).strip()
        return out or None
    except Exception:  # pylint: disable=broad-except
        return None
