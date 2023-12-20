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
import importlib.util
from os import path

import numpy as np
from pybind11.setup_helpers import intree_extensions
from setuptools import (  # pylint: disable=wrong-import-order
    Extension,
    find_namespace_packages,
    setup,
)

with open(path.join("frontend", "catalyst", "_version.py")) as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

with open(".dep-versions") as f:
    jax_version = [line[4:].strip() for line in f.readlines() if "jax=" in line][0]

requirements = [
    "pennylane>=0.32",
    f"jax=={jax_version}",
    f"jaxlib=={jax_version}",
    "tomlkit;python_version<'3.11'",
]

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

package_name = "scipy"
file_path_within_package = "../scipy.libs/"

scipy_package = importlib.util.find_spec(package_name)

if scipy_package is not None:
    package_directory = path.dirname(scipy_package.origin)
    scipy_lib_path = path.join(package_directory, file_path_within_package)
    file_prefix = "libopenblasp"
    file_extension = ".so"
    search_pattern = f"{file_prefix}*{file_extension}"
    openblas_so_file = glob.glob(f"{search_pattern}", root_dir=scipy_lib_path)[0]
    openblas_lib_name = openblas_so_file[3:-3]

openblas_so = scipy_lib_path + openblas_so_file

custom_calls_extension = Extension(
    "catalyst.utils.custom_calls",
    sources=["frontend/catalyst/utils/custom_calls.cpp"],
    libraries=[openblas_lib_name],
    library_dirs=[scipy_lib_path],
)

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
    name="pennylane-catalyst",
    provides=["catalyst"],
    version=version,
    python_requires=">=3.9",
    entry_points={
        "pennylane.compilers": [
            "context = catalyst.utils.contexts:EvaluationContext",
            "ops = catalyst:pennylane_extensions",
            "qjit = catalyst:qjit",
        ]
    },
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
