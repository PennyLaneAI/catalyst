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
import os
import platform
import re
import subprocess
import sys
from typing import Optional

# build deps
from setuptools import Extension, find_namespace_packages, setup
from setuptools._distutils import sysconfig
from setuptools.command.build_ext import build_ext

system_platform = platform.system()


REVISION: Optional[str]
try:
    from subprocess import check_output

    REVISION = (
        check_output(["/usr/bin/env", "git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__))
        .decode()
        .strip()
    )
except Exception:  # pylint: disable=broad-exception-caught  # pragma: no cover
    REVISION = None

with open(os.path.join("frontend", "catalyst", "_version.py"), encoding="utf-8") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

with open(os.path.join("frontend", "catalyst", "_revision.py"), "w", encoding="utf-8") as f:
    f.write("# AUTOGENERATED by setup.py!\n")
    f.write(f"__revision__ = '{REVISION}'\n")


def parse_dep_versions():
    """Parse the version strings of the Catalyst package dependencies based on the values found in
    the .dep-versions file.

    Search patterns must be added manually; this function will not automatically parse patterns like

        <package>=x.y.z

    Example
    -------

    If the .dep-versions file is

        jax=0.4.28
        pennylane=0.40.0
        # lightning=0.40.0

    The version strings for jax and pennylane will be returned, but the version string for lightning
    is omitted because it has been commented out:

        >>> parse_dep_version()
        {"jax": "0.4.28", "pennylane": "0.40.0"}

    Returns:
        dict: Dictionary of dependency versions
    """
    results = {}
    pattern_jax = re.compile(r"^jax=(\S+)", re.MULTILINE)
    pattern_pl = re.compile(r"^pennylane=(\S+)", re.MULTILINE)
    pattern_lq = re.compile(r"^lightning=(\S+)", re.MULTILINE)

    with open(".dep-versions", encoding="utf-8") as fin:
        lines = fin.read()

        match_jax = pattern_jax.search(lines)
        match_pl = pattern_pl.search(lines)
        match_lq = pattern_lq.search(lines)

        if match_jax is not None:
            results["jax"] = match_jax.group(1)

        if match_pl is not None:
            results["pennylane"] = match_pl.group(1)

        if match_lq is not None:
            results["lightning"] = match_lq.group(1)

    return results


dep_versions = parse_dep_versions()
jax_version = dep_versions.get("jax")
pl_version = dep_versions.get("pennylane")
lq_version = dep_versions.get("lightning")

pl_min_release = 0.40
lq_min_release = pl_min_release

if pl_version is not None:
    pennylane_dep = f"pennylane=={pl_version}"  # use TestPyPI wheels, git is not allowed on PyPI
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
    "numpy!=2.0.0",
    "scipy-openblas32>=0.3.26",  # symbol and library name
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


class CMakeExtension(Extension):
    """A setuptools Extension class for modules with a CMake configuration."""

    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class UnifiedBuildExt(build_ext):
    """Custom build extension class for the Catalyst Frontend.

    This class overrides a number of methods from its parent class
    setuptools.command.build_ext.build_ext, the most important of which are:

        1. `get_ext_filename`, in order to remove the architecture/python
           version suffix of the library name.
        2. `build_extension`, in order to handle the compilation of extensions
           with CMake configurations, namely the catalyst.utils.wrapper module,
           and of generic C/C++ extensions without a CMake configuration, namely
           the catalyst.utils.libcustom_calls module, which is currently built
           as a plain setuptools Extension.

    TODO: Eventually it would be better to build the utils.libcustom_calls
    module using a CMake configuration as well, rather than as a setuptools
    Extension.
    """

    def initialize_options(self):
        super().initialize_options()
        self.define = None
        self.verbosity = ""

    def finalize_options(self):
        # Parse the custom CMake options and store them in a new attribute
        defines = [] if self.define is None else self.define.split(";")
        self.cmake_defines = [  # pylint: disable=attribute-defined-outside-init
            f"-D{define}" for define in defines
        ]
        if self.verbosity != "":
            self.verbosity = "--verbose"  # pylint: disable=attribute-defined-outside-init

        super().finalize_options()

    def get_ext_filename(self, fullname):
        filename = super().get_ext_filename(fullname)
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        extension = os.path.splitext(filename)[1]
        return filename.replace(suffix, "") + extension

    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            self.build_cmake_extension(ext)
        else:
            super().build_extension(ext)

    def build_cmake_extension(self, ext: CMakeExtension):
        """Configure and build CMake extension."""
        cmake_path = "cmake"
        ninja_path = "ninja"

        try:
            subprocess.check_output([cmake_path, "--version"])
        except subprocess.CalledProcessError as err:
            raise RuntimeError(
                f"'{cmake_path} --version' failed: check CMake installation"
            ) from err

        try:
            subprocess.check_output([ninja_path, "--version"])
        except subprocess.CalledProcessError as err:
            raise RuntimeError(
                f"'{ninja_path} --version' failed: check Ninja installation"
            ) from err

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        build_type = "Debug" if debug else "RelWithDebInfo"
        configure_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_MAKE_PROGRAM={ninja_path}",
        ]
        configure_args += [
            f"-DPython_EXECUTABLE={sys.executable}",
        ]

        configure_args += self.cmake_defines

        if "CMAKE_ARGS" in os.environ:
            configure_args += os.environ["CMAKE_ARGS"].split(" ")

        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)

        build_args = ["--config", "Debug"] if debug else ["--config", "RelWithDebInfo"]
        build_args += ["--", f"-j{os.cpu_count()}"]

        subprocess.check_call(
            [cmake_path, "-G", "Ninja", ext.sourcedir] + configure_args, cwd=build_temp
        )
        subprocess.check_call([cmake_path, "--build", "."] + build_args, cwd=build_temp)


class CustomBuildExtLinux(UnifiedBuildExt):
    """Custom build extension class for Linux platforms

    Currently no extra work needs to be performed with respect to the base class
    UnifiedBuildExt.
    """


class CustomBuildExtMacos(UnifiedBuildExt):
    """Custom build extension class for macOS platforms

    In addition to the work performed by the base class UnifiedBuildExt, this
    class also changes the LC_ID_DYLIB that is otherwise constant and equal to
    where the shared library was created.
    """

    def run(self):
        # Run the original build_ext command
        super().run()

        # Construct library name based on ext suffix (contains python version, architecture and .so)
        library_name = "libcustom_calls.so"

        package_root = os.path.dirname(__file__)
        frontend_path = glob.glob(
            os.path.join(package_root, "frontend", "**", library_name), recursive=True
        )
        build_path = glob.glob(os.path.join("build", "**", library_name), recursive=True)
        lib_with_r_path = "@rpath/libcustom_calls.so"

        original_path = frontend_path[0] if frontend_path else build_path[0]

        # Run install_name_tool to modify LC_ID_DYLIB(other the rpath stays in vars/folder)
        subprocess.run(
            ["/usr/bin/install_name_tool", "-id", lib_with_r_path, original_path],
            check=False,
        )


Py_LIMITED_API_macros = [("Py_LIMITED_API", "0x030C0000")]

# Compile the library of custom calls in the frontend
if system_platform == "Linux":
    custom_calls_extension = Extension(
        "catalyst.utils.libcustom_calls",
        sources=[
            "frontend/catalyst/utils/libcustom_calls.cpp",
            "frontend/catalyst/utils/jax_cpu_lapack_kernels/lapack_kernels.cpp",
            "frontend/catalyst/utils/jax_cpu_lapack_kernels/lapack_kernels_using_lapack.cpp",
        ],
        extra_compile_args=["-std=c++17"],
        define_macros=Py_LIMITED_API_macros,
    )
    cmdclass = {"build_ext": CustomBuildExtLinux}

elif system_platform == "Darwin":
    variables = sysconfig.get_config_vars()
    # Here we need to switch the deault to MacOs dynamic lib
    variables["LDSHARED"] = variables["LDSHARED"].replace("-bundle", "-dynamiclib")
    if sysconfig.get_config_var("LDCXXSHARED"):
        variables["LDCXXSHARED"] = variables["LDCXXSHARED"].replace("-bundle", "-dynamiclib")
    custom_calls_extension = Extension(
        "catalyst.utils.libcustom_calls",
        sources=[
            "frontend/catalyst/utils/libcustom_calls.cpp",
            "frontend/catalyst/utils/jax_cpu_lapack_kernels/lapack_kernels.cpp",
            "frontend/catalyst/utils/jax_cpu_lapack_kernels/lapack_kernels_using_lapack.cpp",
        ],
        extra_compile_args=["-std=c++17"],
        define_macros=Py_LIMITED_API_macros,
    )
    cmdclass = {"build_ext": CustomBuildExtMacos}


project_root_dir = os.path.abspath(os.path.dirname(__file__))
frontend_dir = os.path.join(project_root_dir, "frontend")

ext_modules = [
    custom_calls_extension,
    CMakeExtension("catalyst.utils.wrapper", sourcedir=frontend_dir),
]

options = {"bdist_wheel": {"py_limited_api": "cp312"}} if sys.hexversion >= 0x030C0000 else {}
# For any compiler packages seeking to be registered in PennyLane, it is imperative that they
# expose the entry_points metadata under the designated group name `pennylane.compilers`, with
# the following entry points:
# - `context`: Path to the compilation evaluation context manager.
# - `ops`: Path to the compiler operations module.
# - `qjit`: Path to the JIT compiler decorator provided by the compiler.

# Install the `catalyst` binary into the user's Python environment so it is accessible on the PATH.
# Does not work with editable installs. Requires the Catalyst mlir module to be built.
if os.path.exists("frontend/bin/catalyst"):
    catalyst_cli = ["frontend/bin/catalyst"]
elif os.path.exists("mlir/build/bin/catalyst"):
    catalyst_cli = ["mlir/build/bin/catalyst"]
else:
    catalyst_cli = []

setup(
    classifiers=classifiers,
    name="PennyLane-Catalyst",
    version=version,
    python_requires=">=3.10",
    entry_points=entry_points,
    install_requires=requirements,
    packages=find_namespace_packages(
        where="frontend",
        include=["catalyst", "catalyst.*", "mlir_quantum"],
        exclude=[
            "catalyst.third_party.oqc.*",
            "catalyst.third_party.oqd.*",
            "catalyst.third_party.oqd",  # Exclude OQD from wheels as it is still under development
        ],
    ),
    package_dir={"": "frontend"},
    include_package_data=True,
    data_files=[
        ("bin", catalyst_cli),
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    **description,
    options=options,
)
