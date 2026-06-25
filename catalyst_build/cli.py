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

"""Command-line control plane for the Catalyst build.

This is the cross-platform replacement for the recursive GNU Makefiles. Every
public Makefile target is exposed here as a subcommand of the same name (with
`-`/`_` accepted interchangeably). Global flags populate a single `BuildEnv`,
reproducing the Makefile `?=` override variables (which also still work as
environment variables).

Examples (parity with `make`):

    python build.py frontend            # == make frontend  (the test point)
    python build.py all                 # == make all
    python build.py runtime --enable-oqd
    python build.py llvm --build-type Release -j 8
    python build.py --dry-run wheel     # print every step, execute nothing
    python build.py clean-all
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

from .environment import BuildEnv
from .runner import BuildError, run
from .components import frontend as frontend_mod
from .components import mlir as mlir_mod
from .components import oqc as oqc_mod
from .components import runtime as runtime_mod
from . import wheel as wheel_mod


# ---------------------------------------------------------------------------
# Aggregate targets that compose component functions (mirror Makefile deps).
# ---------------------------------------------------------------------------
def _all(env: BuildEnv) -> None:
    """`make all` -> runtime oqc mlir frontend."""
    runtime_mod.build(env)
    oqc_mod.build(env)
    mlir_mod.all_(env)
    frontend_mod.install(env)


def _catalyst(env: BuildEnv) -> None:
    """`make catalyst` -> runtime dialects plugin frontend oqc."""
    runtime_mod.build(env)
    mlir_mod.dialects(env)
    mlir_mod.plugin(env)
    frontend_mod.install(env)
    oqc_mod.build(env)


def _builtin_decomp_rules(env: BuildEnv) -> None:
    """`make builtin-decomp-rules` -> dialects runtime frontend + precompile."""
    mlir_mod.dialects(env)
    runtime_mod.build(env)
    frontend_mod.install(env)
    # Matches the Makefile `builtin-decomp-rules` recipe verbatim, which uses the
    # source-tree module path (run from the repo root).
    run([env.python, "-m", "frontend.catalyst.utils.precompile_decomposition_rules"],
        cwd=env.root, dry_run=env.dry_run)


def _clean_all(env: BuildEnv) -> None:
    """`make clean-all` -> clean clean-mlir clean-runtime clean-oqc."""
    frontend_mod.clean(env)
    mlir_mod.clean(env)
    runtime_mod.clean(env)
    oqc_mod.clean(env)


def _clean_catalyst(env: BuildEnv) -> None:
    """`make clean-catalyst` -> clean clean-dialects clean-runtime clean-oqc."""
    frontend_mod.clean(env)
    mlir_mod.clean_dialects(env)
    runtime_mod.clean(env)
    oqc_mod.clean(env)


# ---------------------------------------------------------------------------
# Target registry: name -> callable(env). `-` and `_` are interchangeable.
# ---------------------------------------------------------------------------
TARGETS: dict[str, Callable[[BuildEnv], None]] = {
    # aggregates
    "all": _all,
    "catalyst": _catalyst,
    "builtin-decomp-rules": _builtin_decomp_rules,
    # frontend (the designated test point)
    "frontend": frontend_mod.install,
    # mlir layer
    "mlir": mlir_mod.all_,
    "llvm": mlir_mod.llvm,
    "stablehlo": mlir_mod.stablehlo,
    "enzyme": mlir_mod.enzyme,
    "dialects": mlir_mod.dialects,
    "dialect-docs": mlir_mod.dialect_docs,
    "plugin": mlir_mod.plugin,
    # runtime + oqc
    "runtime": runtime_mod.build,
    "test-runner": runtime_mod.build_test_runners,
    "oqc": oqc_mod.build,
    # wheels
    "wheel": wheel_mod.build,
    "plugin-wheel": wheel_mod.plugin_wheel,
    # tests
    "test": runtime_mod.test,            # closest single-component test point
    "test-runtime": runtime_mod.test,
    "test-mlir": mlir_mod.test,
    "test-oqc": oqc_mod.test,
    # clean variants
    "clean": frontend_mod.clean,
    "clean-all": _clean_all,
    "clean-catalyst": _clean_catalyst,
    "clean-mlir": mlir_mod.clean,
    "clean-dialects": mlir_mod.clean_dialects,
    "clean-llvm": mlir_mod.clean_llvm,
    "reset-llvm": mlir_mod.reset_llvm,
    "clean-stablehlo": mlir_mod.clean_stablehlo,
    "clean-enzyme": mlir_mod.clean_enzyme,
    "clean-plugin": mlir_mod.clean_plugin,
    "clean-dialect-docs": mlir_mod.clean_dialect_docs,
    "clean-runtime": runtime_mod.clean,
    "clean-oqc": oqc_mod.clean,
}


def _normalize(name: str) -> str:
    return name.replace("_", "-")


def _apply_overrides(args: argparse.Namespace) -> BuildEnv:
    """Build a BuildEnv, letting explicit CLI flags win over env/auto-detect."""
    overrides: dict = {}
    if args.python is not None:
        overrides["python"] = args.python
    if args.c_compiler is not None:
        overrides["c_compiler"] = args.c_compiler
    if args.cxx_compiler is not None:
        overrides["cxx_compiler"] = args.cxx_compiler
    if args.compiler_launcher is not None:
        overrides["compiler_launcher"] = args.compiler_launcher
    if args.cmake is not None:
        overrides["cmake"] = args.cmake
    if args.ninja is not None:
        overrides["ninja"] = args.ninja
    if args.build_type is not None:
        overrides["build_type"] = args.build_type
    if args.nproc is not None:
        overrides["nproc"] = args.nproc
    if args.llvm_targets_to_build is not None:
        overrides["llvm_targets_to_build"] = args.llvm_targets_to_build

    # Tri-state booleans: only override when the flag was actually given.
    for attr, val in (
        ("enable_asan", args.enable_asan),
        ("enable_oqd", args.enable_oqd),
        ("enable_openqasm", args.enable_openqasm),
        ("code_coverage", args.code_coverage),
        ("strict_warnings", args.strict_warnings),
        ("enable_lld", args.enable_lld),
        ("enable_zlib", args.enable_zlib),
        ("enable_zstd", args.enable_zstd),
    ):
        if val is not None:
            overrides[attr] = val

    if args.verbose:
        overrides["verbose"] = True
        overrides["pip_verbose"] = True

    overrides["dry_run"] = args.dry_run
    return BuildEnv(**overrides)


def _add_bool_flag(parser: argparse.ArgumentParser, name: str, help_text: str) -> None:
    """Add a tri-state --flag/--no-flag pair defaulting to None (unset)."""
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true", default=None,
                        help=help_text)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false", default=None,
                        help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build.py",
        description="Cross-platform Python control plane for the Catalyst build "
                    "(drop-in replacement for the GNU Makefiles).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "targets", nargs="+", metavar="TARGET",
        help="One or more targets to run in order. Available: "
             + ", ".join(sorted(TARGETS)),
    )

    g = parser.add_argument_group("global options")
    g.add_argument("-n", "--dry-run", action="store_true",
                   help="Print every command/filesystem action without executing it.")
    g.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose pip/build output.")
    g.add_argument("-j", "--nproc", type=int, default=None,
                   help="Parallel build jobs (default: CPU count).")
    g.add_argument("--build-type", default=None,
                   help="CMAKE_BUILD_TYPE (component defaults: Release for mlir, "
                        "RelWithDebInfo for runtime).")
    g.add_argument("--llvm-targets-to-build", default=None,
                   help="LLVM_TARGETS_TO_BUILD (default: host).")

    t = parser.add_argument_group("toolchain overrides (else auto-detected / env)")
    t.add_argument("--python", default=None, help="Python interpreter to use.")
    t.add_argument("--c-compiler", default=None, help="C compiler (clang/cl/gcc).")
    t.add_argument("--cxx-compiler", default=None, help="C++ compiler.")
    t.add_argument("--compiler-launcher", default=None, help="e.g. ccache.")
    t.add_argument("--cmake", default=None, help="Path to cmake.")
    t.add_argument("--ninja", default=None, help="Path to ninja.")

    f = parser.add_argument_group("feature toggles (--flag / --no-flag)")
    _add_bool_flag(f, "enable-asan", "Build with AddressSanitizer.")
    _add_bool_flag(f, "enable-oqd", "Build the OQD device.")
    _add_bool_flag(f, "enable-openqasm", "Build the OpenQASM device (default on).")
    _add_bool_flag(f, "code-coverage", "Enable code coverage instrumentation.")
    _add_bool_flag(f, "strict-warnings", "Treat warnings as errors (default on).")
    _add_bool_flag(f, "enable-lld", "Use the LLD linker (default on except macOS).")
    _add_bool_flag(f, "enable-zlib", "Enable zlib (default on).")
    _add_bool_flag(f, "enable-zstd", "Enable zstd (default off).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate targets up front so a typo fails before any work happens.
    requested = [_normalize(t) for t in args.targets]
    unknown = [t for t in requested if t not in TARGETS]
    if unknown:
        parser.error(
            "unknown target(s): " + ", ".join(unknown)
            + "\nAvailable targets: " + ", ".join(sorted(TARGETS))
        )

    env = _apply_overrides(args)
    print(f"[catalyst-build] platform={env.platform} python={env.python}", flush=True)
    print(f"[catalyst-build] targets: {', '.join(requested)}"
          + (" (dry-run)" if env.dry_run else ""), flush=True)

    for name in requested:
        print(f"\n=== target: {name} ===", flush=True)
        try:
            TARGETS[name](env)
        except (BuildError, EnvironmentError) as err:
            print(f"\n[catalyst-build] FAILED at target '{name}': {err}", file=sys.stderr,
                  flush=True)
            return 1
    print("\n[catalyst-build] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
