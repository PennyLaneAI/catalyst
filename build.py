#!/usr/bin/env python3
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

"""Root launcher for the Catalyst Python build control plane.

Cross-platform replacement for `make`. Usage mirrors the Makefile targets:

    python build.py frontend          # the designated test point (== make frontend)
    python build.py all
    python build.py llvm --build-type Release -j 8
    python build.py --dry-run wheel

Run `python build.py --help` for the full list of targets and flags.
"""

import sys
from pathlib import Path

# Ensure the repo root is importable regardless of the invoking cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalyst_build.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
