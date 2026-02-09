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

from functools import partial
from typing import TypeAlias

from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.ir import (
    Attribute,
    EnumAttribute,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    StrEnum,
    TypeAttribute,
)
from xdsl.irdl import irdl_attr_definition, param_def, ParamAttrConstraint
from xdsl.parser import AttrParser, GenericParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError

################################################################
######################## ATTRIBUTES ############################
################################################################


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


class QubitLevel(StrEnum):
    """Qubit levels enum in the hierarchical qubit representation."""

    Abstract = "abstract"
    Logical = "logical"
    QEC = "qec"
    Physical = "physical"


@irdl_attr_definition
class QubitLevelAttr(EnumAttribute[QubitLevel], SpacedOpaqueSyntaxAttribute):
    """Qubit levels in the hierarchical qubit representation."""

    name = "quantum.qubit_level"


class QubitRole(StrEnum):
    """Qubit roles enum for further specialization in the hierarchical qubit representation."""

    Null = "null"
    Data = "data"
    XCheck = "xcheck"
    ZCheck = "zcheck"


@irdl_attr_definition
class QubitRoleAttr(EnumAttribute[QubitRole], SpacedOpaqueSyntaxAttribute):
    """Qubit roles for further specialization in the hierarchical qubit representation."""

    name = "quantum.qubit_role"


#############################################################
########################## Types ############################
#############################################################


@irdl_attr_definition
class ObservableType(ParametrizedAttribute, TypeAttribute):
    """A quantum observable for use in measurements."""

    name = "quantum.obs"


@irdl_attr_definition
class QubitType(ParametrizedAttribute, TypeAttribute):
    """A value-semantic qubit (state)."""

    name = "quantum.bit"

    level: QubitLevelAttr

    role: QubitRoleAttr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:

        def parse_fn():
            try:
                res = parser.parse_optional_str_enum(QubitLevel)
            except ParseError:
                try:
                    res = parser.parse_optional_str_enum(QubitRole)
                except ParseError:
                    res = None

            return res

        optional_params = parser.parse_optional_comma_separated_list(
            delimiter=parser.Delimiter.ANGLE, parse=parse_fn
        )

        final_params = []
        match len(optional_params):
            case 0:
                final_params = [QubitLevelAttr(QubitLevel.Abstract), QubitRoleAttr(QubitRole.Null)]
            case 1:
                level = QubitLevelAttr(optional_params[0] if isinstance(optional_params[0], QubitLevel) else QubitLevel.Abstract)
                role = QubitRoleAttr(optional_params[0] if isinstance(optional_params[0], QubitRole) else QubitRole.Null)
                final_params = [level, role]
            case 2:
                final_params = optional_params
            case _:
                raise ParseError(f"Expected 2 or less parameters for QubitType, got {optional_params}.")

        return final_params

@irdl_attr_definition
class QuregType(ParametrizedAttribute, TypeAttribute):
    """An array of value-semantic qubits (i.e. quantum register)."""

    name = "quantum.reg"

    level: QubitLevelAttr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parse_fn = partial(parser.parse_optional_str_enum, QubitLevel)
        optional_params = parser.parse_optional_comma_separated_list(
            delimiter=parser.Delimiter.ANGLE, parse=parse_fn
        )

        if len(optional_params) > 1:
            raise ParseError(f"Expected 1 or less parameters for QuregType, got {optional_params}.")

        if len(optional_params) == 1:
            return [QubitLevelAttr(optional_params[0])]

        return [QubitLevelAttr(QubitLevel.Abstract)]


@irdl_attr_definition
class ResultType(ParametrizedAttribute, TypeAttribute):
    """A quantum measurement result."""

    name = "quantum.res"


################################################################
######################## Type aliases ##########################
################################################################


# Type aliases
QubitSSAValue: TypeAlias = SSAValue[QubitType]
QuregSSAValue: TypeAlias = SSAValue[QuregType]
ObservableSSAValue: TypeAlias = SSAValue[ObservableType]
PauliWord: TypeAlias = ArrayAttr[StringAttr]

# Constraints
AnyQubitTypeConstr = ParamAttrConstraint(base_attr=QubitType, param_constrs=(None, None))
AnyQuregTypeConstr = ParamAttrConstraint(base_attr=QuregType, param_constrs=(None,))
