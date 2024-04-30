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
"""This module contains for program verification.
"""

from dataclasses import dataclass
from typing import List

from pennylane.tape import QuantumTape

from catalyst.jax_extras import DynamicJaxprTrace
from catalyst.utils.exceptions import CompileError
from catalyst.utils.toml import DeviceCapabilities


@dataclass
class ProgramRepresentation:
    """Hybrid quantum program representation used in Catalyst"""

    jax_trace: DynamicJaxprTrace
    quantum_tape: QuantumTape


def verify_program(config: DeviceCapabilities, program: ProgramRepresentation):
    """Verify quantum program against the device capabilities.

    Raises: CompileError
    """
    verify_inverses(config, program)
    verify_control(config, program)


def verify_inverses(config: DeviceCapabilities, program: ProgramRepresentation) -> None:
    """Verify quantum program against the device capabilities.

    Raises: CompileError
    """
    pass


def verify_control(config: DeviceCapabilities, program: ProgramRepresentation) -> None:
    """Verify quantum program against the device capabilities.

    Raises: CompileError
    """
    pass
