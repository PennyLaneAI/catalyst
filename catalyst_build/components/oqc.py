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

"""OQC component: build the third-party OQC runtime device.

Mirrors `frontend/catalyst/third_party/oqc/src/Makefile`.
"""

from __future__ import annotations

from ..environment import BuildEnv
from ..runner import remove, run


def configure(env: BuildEnv) -> None:
    """`configure` target."""
    env.require("cmake", "ninja")
    print("Configure OQC Runtime", flush=True)
    run(
        [env.cmake, "-G", "Ninja", "-B", str(env.oqc_build_dir),
         f"-DCMAKE_C_COMPILER={env.c_compiler}",
         f"-DCMAKE_CXX_COMPILER={env.cxx_compiler}",
         f"-DRUNTIME_BUILD_DIR={env.rt_build_dir}",
         f"-DPython_EXECUTABLE={env.python}"],
        cwd=env.oqc_src_dir, dry_run=env.dry_run,
    )


def build(env: BuildEnv) -> None:
    """`oqc` target — build librtd_oqc."""
    configure(env)
    run([env.cmake, "--build", str(env.oqc_build_dir), "--target", "rtd_oqc", f"-j{env.nproc}"],
        dry_run=env.dry_run)


def test(env: BuildEnv) -> None:
    """`test` target — build and run the OQC C++ tests."""
    configure(env)
    run([env.cmake, "--build", str(env.oqc_build_dir),
         "--target", "runner_tests_oqc", f"-j{env.nproc}"], dry_run=env.dry_run)
    print("test the Catalyst runtime test suite", flush=True)
    run([env.oqc_build_dir / "tests" / "runner_tests_oqc"], dry_run=env.dry_run)


def clean(env: BuildEnv) -> None:
    """`clean` target."""
    print("clean build files", flush=True)
    remove(env.oqc_build_dir, dry_run=env.dry_run)
