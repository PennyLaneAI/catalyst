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
making it easier for us to find, and distribute shared library dependencies for
(predictable RPATH).
"""

import os
import sys

from catalyst.utils.runtime_environment import get_cli_path


def main():
    """Run the Catalyst CLI."""
    binary = get_cli_path()
    if not os.path.isfile(binary):  # pragma: nocover
        raise FileNotFoundError(
            f"Could not locate the `catalyst` executable at {binary}. "
            "Please verify your installation or report the issue on GitHub."
        )
    os.execv(binary, [binary, *sys.argv[1:]])
