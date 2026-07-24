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

"""Thin launcher exposed as the ``catalyst`` console script.

Console scripts are automatically installed by Python package managers such that they are
available on PATH, however the exact location is system-dependent. Using a shim allows us
to place the real Catalyst executable into a known location within the package/wheel,
making it easier for us to find and handle dynamically linked dependencies.

This module deliberately lives outside the ``catalyst`` package and loads Catalyst
functionality only via importlib, in order to avoid the expensive overhead of importing
all of Catalyst and all its dependencies.
"""

import importlib.util
import os
import sys


def _get_cli_path() -> str:
    """Resolve the Catalyst CLI path without importing Catalyst via the usual machinery.

    Relies on the runtime_environment module not importing Catalyst directly either.
    """
    spec = importlib.util.find_spec("catalyst")
    if spec is None or spec.origin is None:
        raise ImportError(
            "Could not locate the 'catalyst' package, please make sure it is installed. "
            "Refer to https://docs.pennylane.ai/projects/catalyst/en/stable/dev/installation.html "
            "for additional instructions."
        )
    catalyst_path = os.path.dirname(spec.origin)

    runtime_env_path = os.path.join(catalyst_path, "utils", "runtime_environment.py")
    spec = importlib.util.spec_from_file_location("", runtime_env_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_cli_path()


def main():
    """Run the Catalyst CLI."""
    binary = _get_cli_path()
    if not os.path.isfile(binary):  # pragma: nocover
        raise FileNotFoundError(
            f"Could not locate the `catalyst` executable at {binary}. "
            "Please verify your installation or report the issue on GitHub."
        )
    os.execv(binary, [binary, *sys.argv[1:]])
