# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example module with entry points"""

import platform
from pathlib import Path

from catalyst.utils.runtime_environment import get_bin_path

ext = "so" if platform.system() == "Linux" else "dylib"
plugin_path = get_bin_path("cli", "CATALYST_BIN_DIR") + f"/../lib/StandalonePlugin.{ext}"
plugin = Path(plugin_path)


def name2pass(_name):
    """Example entry point for standalone plugin"""
    return plugin, "standalone-switch-bar-foo"
