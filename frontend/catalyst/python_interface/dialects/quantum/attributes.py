# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Attributes for the xDSL Quantum dialect, which mirrors Catalyst's MLIR
Quantum dialect."""

from typing import TypeAlias

from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.ir import (
    EnumAttribute,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    StrEnum,
    TypeAttribute,
)
from xdsl.irdl import irdl_attr_definition

################################################################
######################## ATTRIBUTES ############################
################################################################


@irdl_attr_definition
class ObservableType(ParametrizedAttribute, TypeAttribute):
    """A quantum observable for use in measurements."""

    name = "quantum.obs"


@irdl_attr_definition
class QubitType(ParametrizedAttribute, TypeAttribute):
    """A value-semantic qubit (state)."""

    name = "quantum.bit"


@irdl_attr_definition
class QuregType(ParametrizedAttribute, TypeAttribute):
    """An array of value-semantic qubits (i.e. quantum register)."""

    name = "quantum.reg"


@irdl_attr_definition
class ResultType(ParametrizedAttribute, TypeAttribute):
    """A quantum measurement result."""

    name = "quantum.res"


class NamedObservable(StrEnum):
    """Known named observables"""

    Identity = "Identity"
    PauliX = "PauliX"
    PauliY = "PauliY"
    PauliZ = "PauliZ"
    Hadamard = "Hadamard"


@irdl_attr_definition
class NamedObservableAttr(EnumAttribute[NamedObservable], SpacedOpaqueSyntaxAttribute):
    """Known named observables"""

    name = "quantum.named_observable"


# Type aliases
QubitSSAValue: TypeAlias = SSAValue[QubitType]
QuregSSAValue: TypeAlias = SSAValue[QuregType]
ObservableSSAValue: TypeAlias = SSAValue[ObservableType]
PauliWord: TypeAlias = ArrayAttr[StringAttr]
