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

"""
Utility code for keeping paths
"""
import os
import os.path

from catalyst._configuration import INSTALLED

package_root = os.path.dirname(__file__)

# Default paths to dep libraries
DEFAULT_LIB_PATHS = {
    "llvm": os.path.join(package_root, "../../../mlir/llvm-project/build/lib"),
    "runtime": os.path.join(package_root, "../../../runtime/build/lib"),
    "enzyme": os.path.join(package_root, "../../../mlir/Enzyme/build/Enzyme"),
}


def get_lib_path(project, env_var):
    """Get the library path."""
    if INSTALLED:
        return os.path.join(package_root, "..", "lib")  # pragma: no cover
    return os.getenv(env_var, DEFAULT_LIB_PATHS.get(project, ""))
