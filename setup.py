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

"""Build the Catalyst frontend, dependencies, and binary wheels."""

# pylint: disable=wrong-import-order

import glob
import platform
import subprocess
from os import path
from typing import Optional

import numpy as np
from pybind11.setup_helpers import intree_extensions
from setuptools import Extension, find_namespace_packages, setup
from setuptools._distutils import sysconfig
from setuptools.command.build_ext import build_ext

system_platform = platform.system()


REVISION: Optional[str]
try:
    from subprocess import check_output

    REVISION = (
        check_output(["/usr/bin/env", "git", "rev-parse", "HEAD"], cwd=path.dirname(__file__))
        .decode()
        .strip()
    )
except Exception:  # pylint: disable=broad-exception-caught  # pragma: no cover
    REVISION = None

with open(path.join("frontend", "catalyst", "_version.py"), encoding="utf-8") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

with open(path.join("frontend", "catalyst", "_revision.py"), "w", encoding="utf-8") as f:
    f.write("# AUTOGENERATED by setup.py!\n")
    f.write(f"__revision__ = '{REVISION}'\n")

with open(".dep-versions", encoding="utf-8") as f:
    lines = f.readlines()
    jax_version = next(line[4:].strip() for line in lines if "jax=" in line)
    pl_version = next((line[10:].strip() for line in lines if "pennylane=" in line), None)
    lq_version = next((line[10:].strip() for line in lines if "lightning=" in line), None)

pl_min_release = 0.37
lq_min_release = pl_min_release

if pl_version is not None:
    pennylane_dep = f"pennylane @ git+https://github.com/pennylaneai/pennylane@{pl_version}"
else:
    pennylane_dep = f"pennylane>={pl_min_release}"
if lq_version is not None:
    lightning_dep = f"pennylane-lightning=={lq_version}"  # use TestPyPI wheels to avoid rebuild
    kokkos_dep = f"pennylane-lightning-kokkos=={lq_version}"
else:
    lightning_dep = f"pennylane-lightning>={lq_min_release}"
    kokkos_dep = ""

requirements = [
    pennylane_dep,
    lightning_dep,
    kokkos_dep,
    f"jax=={jax_version}",
    f"jaxlib=={jax_version}",
    "tomlkit; python_version < '3.11'",
    "scipy<1.13",
    "numpy<2",
    "diastatic-malt>=2.15.2",
]

entry_points = {
    "pennylane.plugins": [
        "oqc.cloud = catalyst.third_party.oqc:OQCDevice",
        "softwareq.qpp = catalyst.third_party.cuda:SoftwareQQPP",
        "nvidia.custatevec = catalyst.third_party.cuda:NvidiaCuStateVec",
        "nvidia.cutensornet = catalyst.third_party.cuda:NvidiaCuTensorNet",
    ],
    "pennylane.compilers": [
        "catalyst.context = catalyst.tracing.contexts:EvaluationContext",
        "catalyst.ops = catalyst.api_extensions",
        "catalyst.qjit = catalyst:qjit",
        "cuda_quantum.context = catalyst.tracing.contexts:EvaluationContext",
        "cuda_quantum.ops = catalyst.api_extensions",
        "cuda_quantum.qjit = catalyst.third_party.cuda:cudaqjit",
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
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3 :: Only",
]

with open("README.md", encoding="utf-8") as f:
    readme_content = f.read()

description = {
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "https://github.com/PennyLaneAI/catalyst",
    "description": "A JIT compiler for hybrid quantum programs in PennyLane",
    "long_description": readme_content,
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
        exclude=["catalyst.third_party.oqc.*"],
    ),
    package_dir={"": "frontend"},
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    **description,
)
