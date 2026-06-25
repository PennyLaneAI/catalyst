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

"""MLIR component: build LLVM/MLIR, StableHLO, Enzyme, and Catalyst dialects.

Mirrors `mlir/Makefile`. Git patch application (previously done with shell
`git apply --check && git apply`) is reproduced with an explicit, idempotent
helper so re-runs do not fail on already-applied patches.
"""

from __future__ import annotations

import os

from ..environment import BuildEnv
from ..runner import BuildError, copy, copy_tree, ensure_dir, remove, run, run_capture

LLVM_PROJECTS = "mlir"
LLVM_TARGETS = ["check-mlir", "llvm-symbolizer"]


def _build_type(env: BuildEnv) -> str:
    return env.build_type or "Release"


def _on(flag: bool) -> str:
    return "ON" if flag else "OFF"


def _visibility(env: BuildEnv) -> str:
    return "default"


def _sanitizer_names(env: BuildEnv) -> str:
    return "Address" if env.enable_asan else ""


def _sanitizer_flags(env: BuildEnv) -> str:
    return "-fsanitize=address" if env.enable_asan else ""


def _launcher_args(env: BuildEnv) -> list[str]:
    if not env.compiler_launcher:
        return []
    return [
        f"-DCMAKE_C_COMPILER_LAUNCHER={env.compiler_launcher}",
        f"-DCMAKE_CXX_COMPILER_LAUNCHER={env.compiler_launcher}",
    ]


def _git_apply_if_needed(env: BuildEnv, repo_dir, patch_file) -> None:
    """Apply a git patch only if it is not already applied (idempotent).

    Replaces the Makefile shell idiom:
        if git apply --check PATCH; then git apply PATCH; fi
    """
    check = run(
        ["git", "apply", "--check", str(patch_file)],
        cwd=repo_dir, dry_run=env.dry_run, check=False,
    )
    if check == 0:
        run(["git", "apply", str(patch_file)], cwd=repo_dir, dry_run=env.dry_run)
    else:
        print(f"[patch] skipping already-applied/incompatible patch {patch_file}", flush=True)


def llvm(env: BuildEnv) -> None:
    """`llvm` target — build LLVM/MLIR with Python bindings."""
    env.require("cmake", "ninja")
    print("build LLVM and MLIR enabling Python bindings", flush=True)
    patches = env.mlir_dir / "patches"
    llvm_src = env.mlir_dir / "llvm-project"
    _git_apply_if_needed(env, llvm_src, patches / "llvm-bufferization-segfault.patch")
    _git_apply_if_needed(env, llvm_src, patches / "llvm-python-bindinggen-annotations.patch")

    args = [
        env.cmake, "-G", "Ninja", "-S", str(llvm_src / "llvm"),
        "-B", str(env.llvm_build_dir),
        f"-DCMAKE_BUILD_TYPE={_build_type(env)}",
        "-DLLVM_BUILD_EXAMPLES=OFF",
        f"-DLLVM_TARGETS_TO_BUILD={env.llvm_targets_to_build}",
        f"-DLLVM_ENABLE_PROJECTS={LLVM_PROJECTS}",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
        f"-DPython_EXECUTABLE={env.python}",
        f"-DPython3_EXECUTABLE={env.python}",
        f"-DPython3_NumPy_INCLUDE_DIRS={env.numpy_include()}",
        f"-DCMAKE_C_COMPILER={env.c_compiler}",
        f"-DCMAKE_CXX_COMPILER={env.cxx_compiler}",
        *_launcher_args(env),
        f"-DLLVM_USE_SANITIZER={_sanitizer_names(env)}",
        f"-DLLVM_ENABLE_LLD={_on(env.enable_lld)}",
        f"-DLLVM_ENABLE_ZLIB={_on(env.enable_zlib)}",
        f"-DLLVM_ENABLE_ZSTD={_on(env.enable_zstd)}",
        f"-DCMAKE_CXX_VISIBILITY_PRESET={_visibility(env)}",
    ]
    run(args, dry_run=env.dry_run)
    # Skip flaky/irrelevant lit tests (matches LIT_FILTER_OUT in the Makefile).
    run(
        [env.cmake, "--build", str(env.llvm_build_dir), "--target", *LLVM_TARGETS],
        extra_env={"LIT_FILTER_OUT": "Bytecode|tosa-to-tensor|execution_engine|python_test"},
        dry_run=env.dry_run,
    )


def stablehlo(env: BuildEnv) -> None:
    """`stablehlo` target."""
    env.require("cmake", "ninja")
    print("build stablehlo", flush=True)
    args = [
        env.cmake, "-G", "Ninja", "-S", str(env.mlir_dir / "stablehlo"),
        "-B", str(env.stablehlo_build_dir),
        f"-DSTABLEHLO_ENABLE_LLD={_on(env.enable_lld)}",
        f"-DCMAKE_BUILD_TYPE={_build_type(env)}",
        f"-DMLIR_DIR={env.llvm_build_dir / 'lib' / 'cmake' / 'mlir'}",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        f"-DLLVM_ENABLE_LLD={_on(env.enable_lld)}",
        f"-DLLVM_ENABLE_ZLIB={_on(env.enable_zlib)}",
        "-DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF",
        "-DSTABLEHLO_ENABLE_SPLIT_DWARF=ON",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DCMAKE_C_COMPILER={env.c_compiler}",
        f"-DCMAKE_CXX_COMPILER={env.cxx_compiler}",
        *_launcher_args(env),
        f"-DCMAKE_EXE_LINKER_FLAGS={_sanitizer_flags(env)}",
        f"-DCMAKE_CXX_VISIBILITY_PRESET={_visibility(env)}",
    ]
    run(args, dry_run=env.dry_run)
    run([env.cmake, "--build", str(env.stablehlo_build_dir)], dry_run=env.dry_run)


def enzyme(env: BuildEnv) -> None:
    """`enzyme` target."""
    env.require("cmake", "ninja")
    print("build enzyme", flush=True)
    enzyme_src = env.mlir_dir / "Enzyme"
    _git_apply_if_needed(env, enzyme_src, env.mlir_dir / "patches" / "enzyme-nvvm-fabs-intrinsics.patch")
    args = [
        env.cmake, "-G", "Ninja", "-S", str(enzyme_src / "enzyme"),
        "-B", str(env.enzyme_build_dir),
        "-DENZYME_STATIC_LIB=ON",
        f"-DCMAKE_BUILD_TYPE={_build_type(env)}",
        f"-DLLVM_DIR={env.llvm_build_dir / 'lib' / 'cmake' / 'llvm'}",
        f"-DCMAKE_C_COMPILER={env.c_compiler}",
        f"-DCMAKE_CXX_COMPILER={env.cxx_compiler}",
        *_launcher_args(env),
        f"-DCMAKE_EXE_LINKER_FLAGS={_sanitizer_flags(env)}",
        f"-DCMAKE_CXX_VISIBILITY_PRESET={_visibility(env)}",
        "-DCMAKE_POLICY_DEFAULT_CMP0116=NEW",
    ]
    run(args, dry_run=env.dry_run)
    run(
        [env.cmake, "--build", str(env.enzyme_build_dir), "--target", "EnzymeStatic-22"],
        dry_run=env.dry_run,
    )


def dialects(env: BuildEnv) -> None:
    """`dialects` target — build the custom Catalyst MLIR dialects."""
    env.require("cmake", "ninja")
    print("build quantum-lsp and dialects", flush=True)
    args = [
        env.cmake, "-G", "Ninja", "-S", str(env.mlir_dir), "-B", str(env.dialects_build_dir),
        f"-DCMAKE_BUILD_TYPE={_build_type(env)}",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DQUANTUM_ENABLE_BINDINGS_PYTHON=ON",
        f"-DPython_EXECUTABLE={env.python}",
        f"-DPython3_EXECUTABLE={env.python}",
        f"-DPython3_NumPy_INCLUDE_DIRS={env.numpy_include()}",
        f"-DEnzyme_DIR={env.enzyme_build_dir}",
        f"-DENZYME_SRC_DIR={env.mlir_dir / 'Enzyme'}",
        f"-DMLIR_DIR={env.llvm_build_dir / 'lib' / 'cmake' / 'mlir'}",
        f"-DSTABLEHLO_DIR={env.mlir_dir / 'stablehlo'}",
        f"-DSTABLEHLO_BUILD_DIR={env.stablehlo_build_dir}",
        f"-DRUNTIME_LIB_DIR={env.rt_build_dir / 'lib'}",
        f"-DMLIR_LIB_DIR={env.llvm_build_dir / 'lib'}",
        f"-DCMAKE_C_COMPILER={env.c_compiler}",
        f"-DCMAKE_CXX_COMPILER={env.cxx_compiler}",
        *_launcher_args(env),
        f"-DLLVM_USE_SANITIZER={_sanitizer_names(env)}",
        f"-DLLVM_ENABLE_LLD={_on(env.enable_lld)}",
        f"-DLLVM_ENABLE_ZLIB={_on(env.enable_zlib)}",
        f"-DLLVM_ENABLE_ZSTD={_on(env.enable_zstd)}",
        f"-DCATALYST_ENABLE_WARNINGS={_on(env.strict_warnings)}",
    ]
    run(args, dry_run=env.dry_run)
    run(
        [env.cmake, "--build", str(env.dialects_build_dir), "--target",
         "check-dialects", "quantum-lsp-server", "catalyst-cli",
         "check-unit-tests", "runner_tests_dgsolver"],
        dry_run=env.dry_run,
    )
    # Catch2 unit tests
    run(
        [env.dialects_build_dir / "unittests" / "DecompGraphSolver" / "runner_tests_dgsolver",
         "--reporter", "compact"],
        dry_run=env.dry_run, check=False,
    )


def dialect_docs(env: BuildEnv) -> None:
    """`dialect-docs` target."""
    env.require("cmake")
    print("build dialect documentation", flush=True)
    args = [
        env.cmake, "-G", "Ninja", "-S", str(env.mlir_dir), "-B", str(env.dialects_docs_build_dir),
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DQUANTUM_ENABLE_BINDINGS_PYTHON=ON",
        f"-DPython3_EXECUTABLE={env.python}",
        f"-DPython3_NumPy_INCLUDE_DIRS={env.numpy_include()}",
        f"-DEnzyme_DIR={env.enzyme_build_dir}",
        f"-DENZYME_SRC_DIR={env.mlir_dir / 'Enzyme'}",
        f"-DMLIR_DIR={env.llvm_build_dir / 'lib' / 'cmake' / 'mlir'}",
        f"-DRUNTIME_LIB_DIR={env.rt_build_dir / 'lib'}",
        f"-DMLIR_LIB_DIR={env.llvm_build_dir / 'lib'}",
        "-DCATALYST_DOCS_ONLY=ON",
    ]
    run(args, dry_run=env.dry_run)
    run([env.cmake, "--build", str(env.dialects_docs_build_dir), "--target", "mlir-doc"],
        dry_run=env.dry_run)


def plugin(env: BuildEnv) -> None:
    """`plugin` target — build the standalone MLIR plugin example."""
    env.require("cmake", "ninja")
    standalone = env.mlir_dir / "standalone"
    if not standalone.exists():
        copy_tree(
            env.mlir_dir / "llvm-project" / "mlir" / "examples" / "standalone",
            standalone, dry_run=env.dry_run,
        )
    # Apply the catalyst plugin patch (patch -p0, idempotent via --dry-run).
    patch = env.mlir_dir / "patches" / "test-plugin-with-catalyst.patch"
    dryrun = run(["patch", "-p0", "--dry-run", "-N", "-i", str(patch)],
                 cwd=env.mlir_dir, dry_run=env.dry_run, check=False)
    if dryrun == 0:
        run(["patch", "-p0", "-i", str(patch)], cwd=env.mlir_dir, dry_run=env.dry_run)

    args = [
        env.cmake, "-B", str(standalone / "build"), "-G", "Ninja",
        f"-DCMAKE_C_COMPILER={env.c_compiler}",
        f"-DCMAKE_CXX_COMPILER={env.cxx_compiler}",
        *_launcher_args(env),
        f"-DMLIR_DIR={env.llvm_build_dir / 'lib' / 'cmake' / 'mlir'}",
        f"-DLLVM_EXTERNAL_LIT={env.llvm_build_dir / 'bin' / 'llvm-lit'}",
        f"-DCATALYST_TOOLS_DIR={env.dialects_build_dir / 'bin'}",
        f"-DPython_EXECUTABLE={env.python}",
        f"-DPython3_EXECUTABLE={env.python}",
        f"-DPython3_NumPy_INCLUDE_DIRS={env.numpy_include()}",
        str(standalone),
    ]
    run(args, dry_run=env.dry_run)
    run([env.cmake, "--build", str(standalone / "build"), "--target", "check-standalone"],
        dry_run=env.dry_run)
    ensure_dir(env.dialects_build_dir / "lib", dry_run=env.dry_run)
    from ..runner import copy_glob
    copy_glob("StandalonePlugin.*", env.dialects_build_dir / "lib",
              root=standalone / "build" / "lib", dry_run=env.dry_run)


def all_(env: BuildEnv) -> None:
    """`all` target — llvm, stablehlo, enzyme, dialects, dialect-docs, plugin."""
    llvm(env)
    stablehlo(env)
    enzyme(env)
    dialects(env)
    dialect_docs(env)
    plugin(env)


def test(env: BuildEnv) -> None:
    """`test` target — run the dialects lit suite."""
    env.require("cmake")
    run([env.cmake, "--build", str(env.dialects_build_dir), "--target", "check-dialects"],
        dry_run=env.dry_run)


# --------------------------- clean targets ---------------------------
def clean_dialects(env: BuildEnv) -> None:
    print("clean catalyst dialect build files", flush=True)
    remove(env.dialects_build_dir, dry_run=env.dry_run)


def clean_llvm(env: BuildEnv) -> None:
    print("clean llvm/mlir build files", flush=True)
    remove(env.llvm_build_dir, dry_run=env.dry_run)
    _git_reset(env, env.mlir_dir / "llvm-project")


def reset_llvm(env: BuildEnv) -> None:
    print("reset llvm git state without deleting builds", flush=True)
    _git_reset(env, env.mlir_dir / "llvm-project")


def clean_stablehlo(env: BuildEnv) -> None:
    print("clean Stablehlo dialect build files", flush=True)
    remove(env.stablehlo_build_dir, dry_run=env.dry_run)
    _git_reset(env, env.mlir_dir / "stablehlo")


def clean_enzyme(env: BuildEnv) -> None:
    print("clean enzyme build files", flush=True)
    remove(env.enzyme_build_dir, dry_run=env.dry_run)
    _git_reset(env, env.mlir_dir / "Enzyme")


def clean_plugin(env: BuildEnv) -> None:
    print("clean plugin", flush=True)
    remove(env.mlir_dir / "standalone", dry_run=env.dry_run)
    from ..runner import remove_glob
    remove_glob("StandalonePlugin.*", root=env.dialects_build_dir / "lib", dry_run=env.dry_run)


def clean_dialect_docs(env: BuildEnv) -> None:
    print("clean dialect docs", flush=True)
    remove(env.dialects_docs_build_dir, dry_run=env.dry_run)


def clean(env: BuildEnv) -> None:
    """`clean` target — clean dialects, llvm, stablehlo, enzyme, plugin, docs."""
    clean_dialects(env)
    clean_llvm(env)
    clean_stablehlo(env)
    clean_enzyme(env)
    clean_plugin(env)
    clean_dialect_docs(env)


def _git_reset(env: BuildEnv, repo_dir) -> None:
    if env.dry_run:
        print(f"[git] clean+checkout {repo_dir}", flush=True)
        return
    run(["git", "clean", "-fd"], cwd=repo_dir, check=False)
    run(["git", "checkout", "."], cwd=repo_dir, check=False)
