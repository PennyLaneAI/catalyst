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

import os

import numpy as np
from pybind11.setup_helpers import intree_extensions
from setuptools import (  # pylint: disable=wrong-import-order
    find_namespace_packages,
    setup,
)

# To build the frontend without any other Catalyst components or dependencies:
build_all_modules = not os.getenv("READTHEDOCS") and os.getenv("BUILDALLMODULES")

if build_all_modules:
    import platform
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    from setuptools import Extension
    from setuptools.command.build_ext import build_ext

    class CMakeExtension(Extension):
        """
        Extension that uses cpp files in place of pyx files derived from ``setuptools.Extension``
        """

        def __init__(self, name, sourcedir=""):
            """The initial method"""
            Extension.__init__(self, name, sources=[])
            self.sourcedir = Path(sourcedir).absolute()

    class BuildExtension(build_ext):
        """Build infrastructure for Catalyst C++ modules."""

        # The set of supported backends at runtime
        BACKENDS = {
            "lightning": "ENABLE_LIGHTNING",
            "lightning.kokkos": "ENABLE_LIGHTNING_KOKKOS",
            "openqasm3": "ENABLE_OPENQASM",
            "braket": "ENABLE_OPENQASM",
        }

        def build_extension(self, ext: CMakeExtension) -> None:
            """Build extension steps."""

            if not hasattr(ext, "sourcedir"):  # e.g., Pybind11Extension
                super().build_extension(ext)
                return

            # Build the runtime
            self.build_ext_catalyst_runtime(ext)

        def build_ext_catalyst_runtime(self, ext: CMakeExtension) -> None:
            """Build Catalyst Runtime"""

            extdir = str(
                Path(self.get_ext_fullpath(ext.name)).parent.joinpath("catalyst", "lib").absolute()
            )
            cfg = "Debug" if int(os.environ.get("DEBUG", 0)) else "Release"
            ninja_bin = self.get_executable("ninja")
            configure_args = [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
                f"-DCMAKE_C_COMPILER={os.environ.get('C_COMPILER', 'clang')}",
                f"-DCMAKE_CXX_COMPILER={os.environ.get('CXX_COMPILER', 'clang++')}",
                f"-DPYTHON_EXECUTABLE={sys.executable}",
                f"-DCMAKE_BUILD_TYPE={cfg}",
                "-GNinja",
                f"-DCMAKE_MAKE_PROGRAM={str(ninja_bin)}",
            ]

            # additional conf args
            configure_args += [
                "-DCMAKE_CXX_FLAGS=-fno-lto",
                f"-DCMAKE_C_COMPILER_LAUNCHER={os.environ.get('COMPILER_LAUNCHER', 'ccache')}",
                f"-DCMAKE_CXX_COMPILER_LAUNCHER={os.environ.get('COMPILER_LAUNCHER', 'ccache')}",
                f"-DENABLE_OPENMP={os.environ.get('ENABLE_OPENMP', 'ON')}",
                "-DENABLE_WARNINGS=ON",
            ]

            selected_backends = (
                os.getenv("BACKEND").split(";") if os.getenv("BACKEND") else ["lightning"]
            )
            for key in selected_backends:
                if key in BuildExtension.BACKENDS:
                    configure_args.append(f"-D{BuildExtension.BACKENDS[key]}=ON")
                else:
                    raise RuntimeError(f"Unsupported backend device: {key}'")

            if platform.system() != "Linux":
                raise RuntimeError(f"Unsupported '{platform.system()}' platform")

            for var, opt in zip(["C_COMPILER", "CXX_COMPILER"], ["C", "CXX"]):
                if os.getenv(var):
                    configure_args += [f"-DCMAKE_{opt}_COMPILER={os.getenv(var)}"]
            if not Path(self.build_temp).exists():
                os.makedirs(self.build_temp)

            qir_stdlib_dir = None
            qir_stdlib_includes_dir = None
            if os.getenv("QIR_STDLIB_DIR") and os.getenv("QIR_STDLIB_INCLUDES_DIR"):
                qir_stdlib_dir = os.getenv("QIR_STDLIB_DIR")
                qir_stdlib_includes_dir = os.getenv("QIR_STDLIB_INCLUDES_DIR")
            else:
                manifest_path = str(Path("runtime", "qir-stdlib", "Cargo.toml").absolute())
                cargo_bin = self.get_executable("cargo")
                subprocess.check_call(
                    [str(cargo_bin), "build", "--release", "--manifest-path", f"{manifest_path}"],
                    cwd=self.build_temp,
                )
                qir_stdlib_dir = str(Path("runtime", "qir-stdlib", "target", "release").absolute())
                qir_stdlib_includes_dir = str(
                    Path(
                        "runtime", "qir-stdlib", "target", "release", "build", "include"
                    ).absolute()
                )

            configure_args += [
                f"-DQIR_STDLIB_LIB={qir_stdlib_dir}",
                f"-DQIR_STDLIB_INCLUDES={qir_stdlib_includes_dir}",
            ]

            if not Path(self.build_temp).exists():
                os.makedirs(self.build_temp)

            cmake_bin = self.get_executable("cmake")
            subprocess.check_call(
                [cmake_bin, str(ext.sourcedir)] + configure_args, cwd=self.build_temp
            )

            subprocess.check_call(
                [cmake_bin, "--build", ".", "--target rt_capi"], cwd=self.build_temp
            )

        def get_executable(self, name: str) -> str:
            """Get the absolute path of an executable using shutil.which"""

            name_bin = shutil.which(name)
            if not name_bin:
                raise RuntimeError(f"Not found executable: {name}")
            return str(name_bin)


with open(os.path.join("frontend", "catalyst", "_version.py"), encoding="utf-8") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

with open(".dep-versions", encoding="utf-8") as f:
    jax_version = [line[4:].strip() for line in f.readlines() if "jax=" in line][0]

requirements = [
    "pennylane>=0.31",
    f"jax=={jax_version}",
    f"jaxlib=={jax_version}",
]

classifiers = [
    "Environment :: Console",
    "Natural Language :: English",
    "Intended Audience :: Science/Research",
    "Development Status :: 3 - Alpha",
    "Operating System :: POSIX",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
]

lib_path_npymath = os.path.join(np.get_include(), "..", "lib")
intree_extension_list = intree_extensions(["frontend/catalyst/utils/wrapper.cpp"])
for iext in intree_extension_list:
    iext._add_ldflags(["-L", lib_path_npymath])  # pylint: disable=protected-access
    iext._add_ldflags(["-lnpymath"])  # pylint: disable=protected-access
    iext._add_cflags(["-I", np.get_include()])  # pylint: disable=protected-access

attrs = {
    "name": "pennylane-catalyst",
    "provides": ["catalyst"],
    "version": version,
    "python_requires": ">=3.8",
    "install_requires": requirements,
    "packages": find_namespace_packages(
        where="frontend", include=["catalyst", "catalyst.*", "mlir_quantum"]
    ),
    "package_dir": {"": "frontend"},
    "include_package_data": True,
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "https://github.com/PennyLaneAI/catalyst",
    "description": "A JIT compiler for hybrid quantum programs in PennyLane",
    "long_description": open("README.md", encoding="utf-8").read(),
    "long_description_content_type": "text/markdown",
    "license": "Apache License 2.0",
    "ext_modules": [*intree_extension_list],
}

if build_all_modules:
    attrs["ext_modules"].append(CMakeExtension("catalyst", "runtime"))
    attrs["cmdclass"] = {"build_ext": BuildExtension}

<<<<<<< HEAD
setup(classifiers=classifiers, **(attrs))
=======
lib_path_npymath = path.join(np.get_include(), "..", "lib")
intree_extension_list = intree_extensions(["frontend/catalyst/utils/wrapper.cpp"])
for ext in intree_extension_list:
    ext._add_ldflags(["-L", lib_path_npymath])  # pylint: disable=protected-access
    ext._add_ldflags(["-lnpymath"])  # pylint: disable=protected-access
    ext._add_cflags(["-I", np.get_include()])  # pylint: disable=protected-access
    ext._add_cflags(["-std=c++17"])  # pylint: disable=protected-access
ext_modules = intree_extension_list

setup(
    classifiers=classifiers,
    name="pennylane-catalyst",
    provides=["catalyst"],
    version=version,
    python_requires=">=3.8",
    install_requires=requirements,
    packages=find_namespace_packages(
        where="frontend",
        include=["catalyst", "catalyst.*", "mlir_quantum"],
    ),
    package_dir={"": "frontend"},
    include_package_data=True,
    ext_modules=ext_modules,
    **description,
)
>>>>>>> 087554cf94d864f2931db0fcc053569bf89f7baa
