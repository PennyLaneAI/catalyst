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

"""Frontend component: install the Catalyst Python package (editable).

Reproduces the `frontend` target of the top-level Makefile:

    frontend:
        $(PYTHON) -m pip uninstall -y pennylane
        $(PYTHON) -m pip install -e . --extra-index-url https://test.pypi.org/simple
        $(PYTHON) -m catalyst.utils.precompile_decomposition_rules
        rm -r frontend/pennylane_catalyst.egg-info

This is the agreed validation test point. The behaviour is preserved 1:1; the
only differences are that the steps run as portable Python calls (no `rm`
shell-out, no `make`), and pip uninstall of a not-installed package is treated
as non-fatal (the Makefile relied on `pip uninstall -y` being a no-op error,
which we make explicit and portable).
"""

from __future__ import annotations

from ..environment import BuildEnv
from ..runner import remove, run

TESTPYPI = "https://test.pypi.org/simple"


def _pip(env: BuildEnv) -> list[str]:
    return [env.python, "-m", "pip"]


def install(env: BuildEnv) -> None:
    """`make frontend` — editable install of the Catalyst frontend."""
    print("install Catalyst Frontend", flush=True)

    # Uninstall pennylane before updating Catalyst, since pip will not replace two
    # development versions of a package with the same version tag (e.g. 0.38-dev0).
    # `pip uninstall -y <pkg>` errors if the package is absent; that is harmless
    # here, so we do not fail the build on it (check=False).
    run(_pip(env) + ["uninstall", "-y", "pennylane"], dry_run=env.dry_run, check=False)

    pip_install = _pip(env) + ["install", "-e", ".", "--extra-index-url", TESTPYPI]
    if env.pip_verbose:
        pip_install.append("--verbose")
    run(pip_install, cwd=env.root, dry_run=env.dry_run)

    # AOT-compile PennyLane's decomposition rules to MLIR bytecode.
    run(
        [env.python, "-m", "catalyst.utils.precompile_decomposition_rules"],
        cwd=env.root,
        dry_run=env.dry_run,
    )

    # Clean up the egg-info directory produced by the editable install.
    remove(env.frontend_dir / "pennylane_catalyst.egg-info", dry_run=env.dry_run)


def clean(env: BuildEnv) -> None:
    """`make clean` — uninstall and delete frontend build/cache artifacts.

    Portable reimplementation of the top-level Makefile `clean` recipe.
    """
    print("uninstall catalyst and delete all temporary and cache files", flush=True)
    run(_pip(env) + ["uninstall", "-y", "pennylane-catalyst"], dry_run=env.dry_run, check=False)

    # find frontend/catalyst -name "*.so" -not -path "*/third_party/*" -exec rm {} +
    catalyst_pkg = env.frontend_dir / "catalyst"
    for so in catalyst_pkg.rglob("*.so"):
        if "third_party" in so.parts:
            continue
        remove(so, dry_run=env.dry_run)

    # git restore frontend/catalyst/_configuration.py
    run(
        ["git", "restore", "frontend/catalyst/_configuration.py"],
        cwd=env.root,
        dry_run=env.dry_run,
        check=False,
    )

    for rel in (
        "frontend/catalyst/_revision.py",
        "frontend/catalyst/include",
        "frontend/catalyst/lib",
        "frontend/bin",
        "frontend/catalyst/resources",
        "frontend/test/lit/GraphDecomposition/test_rules.mlirbc",
        "frontend/mlir_quantum",
        "dist",
        "__pycache__",
        ".coverage",
        "coverage_html_report",
        ".benchmarks",
    ):
        remove(env.root / rel, dry_run=env.dry_run)
