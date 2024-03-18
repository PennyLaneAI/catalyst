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

from typing import List
from dataclasses import dataclass
from pennylane import QuantumTape
from catalyst.utils.toml import TOMLDocument, ProgramFeatures
from catalyst.utils.exceptions import CompileError


@dataclass
class ProgramRepresentation:
    """ Hybrid quantum program representation used in Catalyst """
    jax_eqns: List[Any]
    quantum_tape: QuantumTape


def verify_program(
    config: TOMLDocument,
    program_features: ProgramFeatures,
    program: ProgramRepresentation
):
    """ Verify quantum program against the device capabilities.

    Raises: CompileError
    """
    verify_inverses(config, program_features, program)
    verify_control(config, program_features, program)


def verify_inverses(
    config: TOMLDocument,
    program_features: ProgramFeatures,
    program: ProgramRepresentation
) -> None:
    """ Verify quantum program against the device capabilities.

    Raises: CompileError
    """
    pass


def verify_control(
    config: TOMLDocument,
    program_features: ProgramFeatures,
    program: ProgramRepresentation
) -> None:
    """ Verify quantum program against the device capabilities.

    Raises: CompileError
    """
    pass



