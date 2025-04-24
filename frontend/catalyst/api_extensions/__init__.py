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

"""
This module is a collection of public API extensions for programming with Catalyst frontends.
"""

from catalyst.api_extensions.callbacks import accelerate, pure_callback
from catalyst.api_extensions.control_flow import (
    Cond,
    ForLoop,
    WhileLoop,
    cond,
    for_loop,
    while_loop,
)
from catalyst.api_extensions.differentiation import (
    grad,
    jacobian,
    jvp,
    value_and_grad,
    vjp,
)
from catalyst.api_extensions.error_mitigation import mitigate_with_zne
from catalyst.api_extensions.function_maps import vmap
from catalyst.api_extensions.quantum_operators import (
    HybridAdjoint,
    HybridCtrl,
    MidCircuitMeasure,
    adjoint,
    ctrl,
    measure,
)

__all__ = (
    "accelerate",
    "pure_callback",
    "cond",
    "for_loop",
    "while_loop",
    "ctrl",
    "grad",
    "value_and_grad",
    "jacobian",
    "vjp",
    "jvp",
    "mitigate_with_zne",
    "vmap",
    "measure",
    "adjoint",
    "ctrl",
)
