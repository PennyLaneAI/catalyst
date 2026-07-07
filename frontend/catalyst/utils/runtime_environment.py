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

"""Utility code for keeping paths."""

import importlib.util
import os
import os.path
import sysconfig

package_root = os.path.join(os.path.dirname(__file__), "..")


def _load_config(filename, attr):
    """Read Catalyst config flags without importing Catalyst via the usual machinery.

    This enables us to use this utility module from outside the Catalyst package as well.
    """
    path = os.path.join(package_root, filename)
    spec = importlib.util.spec_from_file_location("", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


INSTALLED = _load_config("_configuration.py", "INSTALLED")
__revision__ = _load_config("_revision.py", "__revision__")
__version__ = _load_config("_version.py", "__version__")

# Default paths to dep libraries
DEFAULT_LIB_PATHS = {
    "llvm": os.path.join(package_root, "..", "..", "mlir", "llvm-project", "build", "lib"),
    "catalyst": os.path.join(package_root, "..", "..", "mlir", "build", "lib"),
    "runtime": os.path.join(package_root, "..", "..", "runtime", "build", "lib"),
    "enzyme": os.path.join(package_root, "..", "..", "mlir", "Enzyme", "build", "Enzyme"),
    "oqc_runtime": os.path.join(package_root, "third_party", "oqc", "src", "build"),
    "oqd_runtime": os.path.join(package_root, "..", "..", "runtime", "build", "lib"),
}

DEFAULT_INCLUDE_PATHS = {
    "mlir": os.path.join(package_root, "..", "..", "mlir", "include"),
}

DEFAULT_BIN_PATHS = {
    "cli": os.path.join(package_root, "..", "..", "mlir", "build", "bin"),
}

BYTECODE_FILE_PATH = os.path.join(
    package_root,
    "resources",
    "decomposition_rules_" + (__revision__ if __revision__ else __version__) + ".mlirbc",
)


def get_libpython_path() -> str:  # pragma: no cover
    """Return the path to the python shared library, or empty string if failed to find."""
    libdir = sysconfig.get_config_var("LIBDIR")
    ldlibrary = sysconfig.get_config_var("LDLIBRARY")
    framework_prefix = sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX")

    # macOS framework-style installations
    if framework_prefix:
        return os.path.join(framework_prefix, ldlibrary)

    if not (libdir and ldlibrary):
        return ""

    # standard installation
    ldlibrary_path = os.path.join(libdir, ldlibrary)
    if os.path.exists(ldlibrary_path):
        return ldlibrary_path

    return ""


def get_lib_path(project, env_var):
    """Get the path to Catalyst's shared libraries."""
    if INSTALLED:  # pragma: no cover
        return os.path.join(package_root, "lib")

    return os.getenv(env_var, DEFAULT_LIB_PATHS.get(project, ""))


def get_include_path():
    """Return the path to Catalyst's include directory."""
    if INSTALLED:  # pragma: no cover
        return os.path.join(package_root, "include")

    return os.getenv("CATALYST_INCLUDE_DIRS", DEFAULT_INCLUDE_PATHS.get("mlir", ""))


def get_cli_path() -> str:
    """Method to obtain the Catalyst CLI path whether installed or locally built."""
    catalyst_cli = "catalyst"

    if INSTALLED:  # pragma: nocover
        return os.path.join(package_root, "bin", catalyst_cli)

    return os.path.join(
        os.getenv("CATALYST_BIN_DIR", DEFAULT_BIN_PATHS.get("cli", "")), catalyst_cli
    )
