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
"""
This package contains the Catalyst Python interface.
"""

# pylint: disable=missing-module-docstring

from catalyst._version import __version__
from catalyst.compiler import compile
from catalyst.compilation_pipelines import qjit, QJIT
from catalyst.pennylane_extensions import for_loop, while_loop, cond, measure, grad


__all__ = (
    "qjit",
    "QJIT",
    "for_loop",
    "while_loop",
    "cond",
    "measure",
    "grad",
)
