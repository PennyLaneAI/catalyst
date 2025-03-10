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

"""Build the Standalone Plugin wheel."""

from setuptools import setup

entry_points = {
    "catalyst.passes_resolution": [
        "standalone.passes = standalone_plugin",
    ],
}

setup(
    name="standalone_plugin",
    version="0.1.0",
    description="The standalone plugin example as a python package",
    packages=["standalone_plugin"],
    entry_points=entry_points,
    include_package_data=True,
    install_requires=["PennyLane", "PennyLane-Catalyst"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
    ],
)
