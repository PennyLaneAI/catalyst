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

"""Catalyst Python build control plane.

A cross-platform (POSIX + Windows) replacement for the recursive GNU Makefiles
that drive the Catalyst frontend, MLIR layer, runtime, and OQC components. The
control plane exposes the same targets and the same configuration knobs as the
Makefiles, but it drives CMake/Ninja/pip directly from Python instead of from
`make` recipes, so it does not depend on a POSIX shell, `uname`, `find`, `cp`,
or `rm`.

Entry point: `python -m catalyst_build <target> [options]` or the thin
`build.py` launcher at the repository root.
"""

__version__ = "0.1.0"
