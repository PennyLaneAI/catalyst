# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import platform
import subprocess
from distutils import sysconfig
from os import environ, path

import numpy as np
from pybind11.setup_helpers import intree_extensions
from setuptools import (  # pylint: disable=wrong-import-order
    Extension,
    find_namespace_packages,
    setup,
)
from setuptools.command.build_ext import build_ext

system_platform = platform.system()

with open(path.join("frontend", "catalyst", "_version.py")) as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

with open(".dep-versions") as f:
    jax_version = [line[4:].strip() for line in f.readlines() if "jax=" in line][0]

pl_version = environ.get("PL_VERSION", ">=0.32,<=0.34")
requirements = [
    f"pennylane{pl_version}",
    f"jax=={jax_version}",
    f"jaxlib=={jax_version}",
    "tomlkit;python_version<'3.11'",
    "scipy",
    "diastatic-malt",
]

# TODO: Once PL version 0.35 is released:
# * remove this special handling
# * make pennylane>=0.35 a requirement
# * Close this ticket https://github.com/PennyLaneAI/catalyst/issues/494
one_compiler_per_distribution = pl_version == ">=0.32,<=0.34"
if one_compiler_per_distribution:
    entry_points = {
        "pennylane.plugins": "cudaq = catalystcuda:CudaQDevice",
        "pennylane.compilers": [
            "context = catalyst.tracing.contexts:EvaluationContext",
            "ops = catalyst:pennylane_extensions",
            "qjit = catalyst:qjit",
        ],
    }
else:
    entry_points = {
        "pennylane.plugins": "cudaq = catalystcuda:CudaQDevice",
        "pennylane.compilers": [
            "catalyst.context = catalyst.tracing.contexts:EvaluationContext",
            "catalyst.ops = catalyst:pennylane_extensions",
            "catalyst.qjit = catalyst:qjit",
        ],
    }

classifiers = [
    "Environment :: Console",
    "Natural Language :: English",
    "Intended Audience :: Science/Research",
    "Development Status :: 3 - Alpha",
    "Operating System :: POSIX",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
]


description = {
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "https://github.com/PennyLaneAI/catalyst",
    "description": "A JIT compiler for hybrid quantum programs in PennyLane",
    "long_description": open("README.md").read(),
    "long_description_content_type": "text/markdown",
    "license": "Apache License 2.0",
}


class CustomBuildExtLinux(build_ext):
    """Override build ext from setuptools in order to remove the architecture/python
    version suffix of the library name."""

    def get_ext_filename(self, fullname):
        filename = super().get_ext_filename(fullname)
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        extension = path.splitext(filename)[1]
        return filename.replace(suffix, "") + extension


class CustomBuildExtMacos(build_ext):
    """Override build ext from setuptools in order to change to remove the architecture/python
    version suffix of the library name and to change the LC_ID_DYLIB that otherwise is constant
    and equal to where the shared library was created."""

    def get_ext_filename(self, fullname):
        filename = super().get_ext_filename(fullname)
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        extension = path.splitext(filename)[1]
        return filename.replace(suffix, "") + extension

    def run(self):
        # Run the original build_ext command
        build_ext.run(self)

        # Construct library name based on ext suffix (contains python version, architecture and .so)
        library_name = "libcustom_calls.so"

        package_root = path.dirname(__file__)
        frontend_path = glob.glob(
            path.join(package_root, "frontend", "**", library_name), recursive=True
        )
        build_path = glob.glob(path.join("build", "**", library_name), recursive=True)
        lib_with_r_path = "@rpath/libcustom_calls.so"

        original_path = frontend_path[0] if frontend_path else build_path[0]

        # Run install_name_tool to modify LC_ID_DYLIB(other the rpath stays in vars/folder)
        subprocess.run(
            ["/usr/bin/install_name_tool", "-id", lib_with_r_path, original_path],
            check=False,
        )


# Compile the library of custom calls in the frontend
if system_platform == "Linux":
    custom_calls_extension = Extension(
        "catalyst.utils.libcustom_calls",
        sources=["frontend/catalyst/utils/libcustom_calls.cpp"],
    )
    cmdclass = {"build_ext": CustomBuildExtLinux}

elif system_platform == "Darwin":
    variables = sysconfig.get_config_vars()
    # Here we need to switch the deault to MacOs dynamic lib
    variables["LDSHARED"] = variables["LDSHARED"].replace("-bundle", "-dynamiclib")
    custom_calls_extension = Extension(
        "catalyst.utils.libcustom_calls",
        sources=["frontend/catalyst/utils/libcustom_calls.cpp"],
    )
    cmdclass = {"build_ext": CustomBuildExtMacos}


ext_modules = [custom_calls_extension]

lib_path_npymath = path.join(np.get_include(), "..", "lib")
intree_extension_list = intree_extensions(["frontend/catalyst/utils/wrapper.cpp"])
for ext in intree_extension_list:
    ext._add_ldflags(["-L", lib_path_npymath])  # pylint: disable=protected-access
    ext._add_ldflags(["-lnpymath"])  # pylint: disable=protected-access
    ext._add_cflags(["-I", np.get_include()])  # pylint: disable=protected-access
    ext._add_cflags(["-std=c++17"])  # pylint: disable=protected-access
ext_modules.extend(intree_extension_list)
# For any compiler packages seeking to be registered in PennyLane, it is imperative that they
# expose the entry_points metadata under the designated group name `pennylane.compilers`, with
# the following entry points:
# - `context`: Path to the compilation evaluation context manager.
# - `ops`: Path to the compiler operations module.
# - `qjit`: Path to the JIT compiler decorator provided by the compiler.
setup(
    classifiers=classifiers,
    name="PennyLane-Catalyst",
    provides=["catalyst"],
    version=version,
    python_requires=">=3.9",
    entry_points=entry_points,
    install_requires=requirements,
    packages=find_namespace_packages(
        where="frontend",
        include=["catalyst", "catalyst.*", "mlir_quantum"],
    ),
    package_dir={"": "frontend"},
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    **description,
)
