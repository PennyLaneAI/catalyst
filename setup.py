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

from os import path

from setuptools import setup, find_namespace_packages

with open(path.join("frontend", "catalyst", "_version.py")) as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

with open(".dep-versions") as f:
    jax_version = [line[4:].strip() for line in f.readlines() if "jax=" in line][0]

requirements = [
    "pennylane>=0.28",
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
    **description,
)
