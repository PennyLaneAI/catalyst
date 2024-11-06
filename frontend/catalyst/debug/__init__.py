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

"""Catalyst's debug module contains functions useful for user program debugging."""

from catalyst.debug.callback import callback
from catalyst.debug.compiler_functions import (
    compile_executable,
    get_cmain,
    get_compilation_stage,
    replace_ir,
)
from catalyst.debug.instruments import instrumentation
from catalyst.debug.printing import (  # pylint: disable=redefined-builtin
    print,
    print_memref,
)

__all__ = (
    "callback",
    "print",
    "print_memref",
    "get_compilation_stage",
    "get_cmain",
    "instrumentation",
    "replace_ir",
    "compile_executable",
)
