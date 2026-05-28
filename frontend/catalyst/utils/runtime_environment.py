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
import sys
import sysconfig
from pathlib import Path

from catalyst._configuration import INSTALLED
from catalyst._revision import __revision__
from catalyst._version import __version__

package_root = os.path.dirname(__file__)

# Default paths to dep libraries
DEFAULT_LIB_PATHS = {
    "llvm": os.path.join(package_root, "../../../mlir/llvm-project/build/lib"),
    "catalyst": os.path.join(package_root, "../../../mlir/build/lib"),
    "runtime": os.path.join(package_root, "../../../runtime/build/lib"),
    "enzyme": os.path.join(package_root, "../../../mlir/Enzyme/build/Enzyme"),
    "oqc_runtime": os.path.join(package_root, "../../catalyst/third_party/oqc/src/build"),
    "oqd_runtime": os.path.join(package_root, "../../../runtime/build/lib"),
    "callbacks_lib": os.path.join(package_root, "../../mlir/build/lib"),
}

DEFAULT_INCLUDE_PATHS = {
    "mlir": os.path.join(package_root, "../../../mlir/include"),
}

DEFAULT_BIN_PATHS = {
    "cli": os.path.join(package_root, "../../../mlir/build/bin"),
}

BYTECODE_FILE_PATH = (
    Path(__file__).parent.parent
    / Path("resources")
    / Path("decomposition_rules_" + (__revision__ if __revision__ else __version__) + ".mlirbc")
)


def get_libpython_path() -> Path | str:
    """Return the path to the python shared library, or emptystring if failed to find."""
    libdir = sysconfig.get_config_var("LIBDIR")
    ldlibrary = sysconfig.get_config_var("LDLIBRARY")
    framework_prefix = sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX")

    # TODO: prefer INSTSONAME for linux
    print("[CI-DEBUG] libdir:", libdir)
    print("[CI-DEBUG] ldlibrary:", ldlibrary)
    print("[CI-DEBUG] framework_prefix:", framework_prefix)

    if framework_prefix:
        print("[CI-DEBUG] found framework install:", Path(framework_prefix) / Path(ldlibrary))
        return Path(framework_prefix) / Path(ldlibrary)

    if not (libdir and ldlibrary):
        print("[CI-DEBUG] frontend could not find libpython")
        return ""

    # standard installation
    ldlibrary_path = Path(libdir) / ldlibrary
    if ldlibrary_path.exists():
        print("[CI-DEBUG] found python at", ldlibrary_path.resolve(), "(standard installation)")
        return ldlibrary_path.resolve()

    return ""


def get_lib_path(project, env_var):
    """Get the library path."""
    if INSTALLED:
        return os.path.join(package_root, "..", "lib")  # pragma: no cover
    return os.getenv(env_var, DEFAULT_LIB_PATHS.get(project, ""))


def get_include_path():
    """Return the path to Catalyst's include directory."""
    if INSTALLED:
        return os.path.join(package_root, "..", "include")  # pragma: no cover
    return os.getenv("CATALYST_INCLUDE_DIRS", DEFAULT_INCLUDE_PATHS.get("mlir", ""))


def get_bin_path(project, env_var):
    """Get the library path."""
    if INSTALLED:
        return os.path.join(package_root, "..", "bin")  # pragma: no cover
    return os.getenv(env_var, DEFAULT_BIN_PATHS.get(project, ""))


def get_cli_path() -> str:  # pragma: nocover
    """Method to obtain the Catalyst CLI path packaged via the data_files mechanism."""
    catalyst_cli = "catalyst"

    if not INSTALLED:
        return os.path.join(
            os.getenv("CATALYST_BIN_DIR", DEFAULT_BIN_PATHS.get("cli", "")), catalyst_cli
        )

    # Default path
    path = os.path.join(sysconfig.get_path("scripts"), catalyst_cli)
    if os.path.isfile(path):
        return path

    # User path
    user_scheme = sysconfig.get_preferred_scheme("user")
    path = os.path.join(sysconfig.get_path("scripts", scheme=user_scheme), catalyst_cli)
    if os.path.isfile(path):
        return path

    # Fallback to python location
    path = os.path.join(os.path.dirname(sys.executable), catalyst_cli)
    if os.path.isfile(path):
        return path

    raise RuntimeError(
        "Could not locate the Catalyst executable, please report this issue on GitHub."
    )
