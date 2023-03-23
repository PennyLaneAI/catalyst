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


from catalyst._version import __version__
from catalyst._configuration import INSTALLED

if not INSTALLED:
    import os

    default_bindings_path = os.path.join(
        os.path.dirname(__file__), "../../mlir/build/python_packages/quantum"
    )
    if os.path.exists(default_bindings_path):  # pragma: no cover
        import sys

        sys.path.insert(0, default_bindings_path)

from catalyst.compilation_pipelines import (  # pylint: disable=wrong-import-position
    qjit,
    QJIT,
    CompileOptions,
)
from catalyst.pennylane_extensions import (  # pylint: disable=wrong-import-position
    for_loop,
    while_loop,
    cond,
    measure,
    grad,
)
from catalyst.utils.exceptions import CompileError  # pylint: disable=wrong-import-position


__all__ = (
    "qjit",
    "QJIT",
    "for_loop",
    "while_loop",
    "cond",
    "measure",
    "grad",
    "CompileError",
    "CompileOptions"
)
