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

if not os.getenv("READTHEDOCS"):
    import platform
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    from setuptools import Extension
    from setuptools.command.build_ext import build_ext

    class CMakeExtension(Extension):
        """Extension that uses cpp files in place of pyx files derived from ``setuptools.Extension``"""

        def __init__(self, name, sourcedir=""):
            """The initial method"""
            Extension.__init__(self, name, sources=[])
            self.sourcedir = Path(sourcedir).absolute()

    class BuildExtension(build_ext):
        """Build infrastructure for Catalyst C++ modules."""

        user_options = build_ext.user_options + [
            ("define=", "D", "Define variables for CMake"),
            ("verbosity", "V", "Increase CMake build verbosity"),
            ("backend=", "B", "Define compiled backend device"),
        ]

        backends = {"LIGHTNING", "LIGHTNING_KOKKOS", "OPENQASM"}

        def initialize_options(self) -> None:
            """Set default values for all the options that this command supports."""
            super().initialize_options()
            self.define = None
            self.verbosity = ""
            self.backend = None

        def finalize_options(self) -> None:
            """Set final values for all the options that this command supports."""

            defines = [] if self.define is None else self.define.split(";")
            self.cmake_defines = [f"-D{define}" for define in defines]
            if self.verbosity != "":
                self.verbosity = "--verbose"
            super().finalize_options()

        def build_extension(self, ext: CMakeExtension) -> None:
            """Build extension steps."""

            if not hasattr(ext, "sourcedir"):  # e.g., Pybind11Extension
                super().build_extension(ext)
                return

            extdir = str(Path(self.get_ext_fullpath(ext.name)).parent.absolute())
            debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
            cfg = "Debug" if debug else "Release"
            ninja_path = str(shutil.which("ninja"))
            configure_args = [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
                f"-DPYTHON_EXECUTABLE={sys.executable}",
                f"-DCMAKE_BUILD_TYPE={cfg}",
                "-GNinja",
                f"-DCMAKE_MAKE_PROGRAM={ninja_path}",
                *(self.cmake_defines),
            ]

            # additional conf args
            configure_args += [
                "-DCMAKE_CXX_FLAGS=-fno-lto",
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DENABLE_WARNINGS=ON",
                "-DENABLE_OPENMP=ON",
            ]

            if not self.backend:
                self.backend = (
                    os.getenv("BACKEND").split(";") if os.getenv("BACKEND") else ["LIGHTNING"]
                )

            for device in self.backend:
                if device in self.backends:
                    configure_args.append(f"-DENABLE_{device}=ON")
                else:
                    raise RuntimeError(f"Unsupported backend device: '{device}'")

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
                subprocess.check_call(
                    ["cargo", "build", "--release", "--manifest-path", f"{manifest_path}"],
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

            subprocess.check_call(
                ["cmake", str(ext.sourcedir)] + configure_args, cwd=self.build_temp
            )

            subprocess.check_call(
                ["cmake", "--build", ".", "--target rt_capi"], cwd=self.build_temp
            )


with open(os.path.join("frontend", "catalyst", "_version.py")) as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

with open(".dep-versions") as f:
    jax_version = [line[4:].strip() for line in f.readlines() if "jax=" in line][0]

requirements = [
    # TODO: Change to "pennylane>=0.31" when next version releases.
    "pennylane @ git+https://github.com/PennyLaneAI/pennylane/@master",
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


description = {
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "https://github.com/PennyLaneAI/catalyst",
    "description": "A JIT compiler for hybrid quantum programs in PennyLane",
    "long_description": open("README.md").read(),
    "long_description_content_type": "text/markdown",
    "license": "Apache License 2.0",
}


lib_path_npymath = os.path.join(np.get_include(), "..", "lib")
intree_extension_list = intree_extensions(["frontend/catalyst/utils/wrapper.cpp"])
for ext in intree_extension_list:
    ext._add_ldflags(["-L", lib_path_npymath])  # pylint: disable=protected-access
    ext._add_ldflags(["-lnpymath"])  # pylint: disable=protected-access
    ext._add_cflags(["-I", np.get_include()])  # pylint: disable=protected-access
ext_modules = [*intree_extension_list, CMakeExtension("catalyst_runtime", "runtime")]

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
    cmdclass={"build_ext": BuildExtension},
    **description,
)
