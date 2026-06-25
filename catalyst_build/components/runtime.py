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

"""Runtime component: build the Catalyst Runtime (QIR C-API + devices).

Mirrors `runtime/Makefile`. The CMake configure/build invocations and target
lists are reproduced verbatim; the ASAN LD_LIBRARY_PATH plumbing is handled in
Python so the test recipe works without a POSIX shell.
"""

from __future__ import annotations

import os

from ..environment import BuildEnv
from ..runner import BuildError, remove, run

# Build/test target lists from runtime/Makefile.
BASE_BUILD_TARGETS = ["rt_capi", "rtd_null_qubit", "rt_rsdecomp", "rt_decoder"]
BASE_TEST_TARGETS = [
    "runner_tests_qir_runtime",
    "runner_tests_mbqc_runtime",
    "runner_tests_rsdecomp_runtime",
    "runner_tests_decoder_runtime",
]


def _build_targets(env: BuildEnv) -> list[str]:
    targets = list(BASE_BUILD_TARGETS)
    if env.enable_openqasm:
        targets.append("rtd_openqasm")
    if env.enable_oqd:
        targets += ["rt_OQD_capi", "rtd_oqd_device"]
    return targets


def _test_targets(env: BuildEnv) -> list[str]:
    targets = list(BASE_TEST_TARGETS)
    if env.enable_openqasm:
        targets.append("runner_tests_openqasm")
    if env.enable_oqd:
        targets.append("runner_tests_oqd")
    return targets


def _build_type(env: BuildEnv) -> str:
    return env.build_type or "RelWithDebInfo"


def _llvm_dir(env: BuildEnv) -> str:
    return os.environ.get("LLVM_DIR", str(env.mlir_dir / "llvm-project"))


def configure(env: BuildEnv, build_dir=None) -> None:
    """`configure` target — CMake configure of the runtime."""
    env.require("cmake", "ninja")
    print("Configure Catalyst Runtime", flush=True)
    build_dir = build_dir or env.rt_build_dir
    args = [
        env.cmake, "-G", "Ninja", "-B", str(build_dir), ".",
        f"-DCMAKE_BUILD_TYPE={_build_type(env)}",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_dir / 'lib'}",
        f"-DCMAKE_C_COMPILER={env.c_compiler}",
        f"-DMLIR_INCLUDE_DIRS={env.mlir_dir / 'llvm-project' / 'mlir' / 'include'}",
        f"-DCMAKE_CXX_COMPILER={env.cxx_compiler}",
        f"-DENABLE_OPENQASM={_on(env.enable_openqasm)}",
        f"-DENABLE_OQD={_on(env.enable_oqd)}",
        f"-DENABLE_CODE_COVERAGE={_on(env.code_coverage)}",
        f"-DPython_EXECUTABLE={env.python}",
        f"-DENABLE_ADDRESS_SANITIZER={_on(env.enable_asan)}",
        f"-DRUNTIME_ENABLE_WARNINGS={_on(env.strict_warnings)}",
    ]
    if env.compiler_launcher:
        args.insert(-1, f"-DCMAKE_C_COMPILER_LAUNCHER={env.compiler_launcher}")
        args.insert(-1, f"-DCMAKE_CXX_COMPILER_LAUNCHER={env.compiler_launcher}")
    run(args, cwd=env.runtime_dir, dry_run=env.dry_run)


def build(env: BuildEnv) -> None:
    """`runtime` target — configure then build the runtime libraries."""
    configure(env)
    run(
        [env.cmake, "--build", str(env.rt_build_dir),
         "--target", *_build_targets(env), f"-j{env.nproc}"],
        dry_run=env.dry_run,
    )


def build_test_runners(env: BuildEnv) -> None:
    """`test_runner` target — build the C++ test executables."""
    configure(env)
    run(
        [env.cmake, "--build", str(env.rt_build_dir),
         "--target", *_test_targets(env), f"-j{env.nproc}"],
        dry_run=env.dry_run,
    )


def test(env: BuildEnv) -> None:
    """`test` target — build and run the runtime C++ test suite."""
    build_test_runners(env)
    extra_env = {}
    if env.enable_asan and env.is_linux:
        extra_env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(env.rt_build_dir / "lib"), os.environ.get("LD_LIBRARY_PATH", "")]
        )
    tests_dir = env.rt_build_dir / "tests"
    suites = [
        ("NullQubit", "runner_tests_qir_runtime"),
        ("MBQC", "runner_tests_mbqc_runtime"),
        ("RSDecomp", "runner_tests_rsdecomp_runtime"),
        ("Decoder", "runner_tests_decoder_runtime"),
    ]
    for label, exe in suites:
        print(f"Catalyst runtime test suite - {label}", flush=True)
        run([tests_dir / exe], extra_env=extra_env, dry_run=env.dry_run)
    if env.enable_openqasm:
        py_asan = dict(extra_env)
        if env.is_linux:
            py_asan["ASAN_OPTIONS"] = "detect_leaks=0"
        run([tests_dir / "runner_tests_openqasm"], extra_env=py_asan, dry_run=env.dry_run)
    if env.enable_oqd:
        run([tests_dir / "runner_tests_oqd"], extra_env=extra_env, dry_run=env.dry_run)


def clean(env: BuildEnv) -> None:
    """`clean` target — remove runtime build artifacts."""
    print("clean build files", flush=True)
    remove(env.rt_build_dir, dry_run=env.dry_run)
    remove(str(env.rt_build_dir) + "_cov", dry_run=env.dry_run)
    remove(env.runtime_dir / "cov", dry_run=env.dry_run)
    remove(env.runtime_dir / "coverage.info", dry_run=env.dry_run)
    remove(env.runtime_dir / "BuildTidy", dry_run=env.dry_run)


def _on(flag: bool) -> str:
    return "ON" if flag else "OFF"
