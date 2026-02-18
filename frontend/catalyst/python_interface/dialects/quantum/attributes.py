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

from collections.abc import Sequence
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
from xdsl.irdl import AnyOf, AttrConstraint, irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError, VerifyException

################################################################
######################## ATTRIBUTES ############################
################################################################


class QubitLevel(StrEnum):
    """Qubit levels enum in the hierarchical qubit representation."""

    Abstract = "abstract"
    Logical = "logical"
    PBC = "pbc"
    Physical = "physical"


class QubitRole(StrEnum):
    """Qubit roles enum for further specialization in the hierarchical qubit representation."""

    Null = "null"
    Data = "data"
    XCheck = "xcheck"
    ZCheck = "zcheck"


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

    level: StringAttr

    role: StringAttr

    def __init__(
        self,
        level: str | StringAttr = QubitLevel.Abstract.value,
        role: str | StringAttr = QubitRole.Null.value,
    ):
        level = level if isinstance(level, StringAttr) else StringAttr(level)
        role = role if isinstance(role, StringAttr) else StringAttr(role)
        super().__init__(level, role)

    def verify(self) -> None:
        """Verify that the attribute is defined correctly."""
        level = self.level.data
        role = self.role.data
        allowed_levels = list(QubitLevel.__members__.values())
        allowed_roles = list(QubitRole.__members__.values())

        if level not in allowed_levels:
            raise VerifyException(
                f"Invalid value {level} for 'QubitType.level'. Allowed values are {allowed_levels}."
            )

        if role not in allowed_roles:
            raise VerifyException(
                f"Invalid value {role} for 'QubitType.role'. Allowed values are {allowed_roles}."
            )

    def print_parameters(self, printer: Printer):
        """Print type parameters."""
        params_to_print = []
        if self.level.data != QubitLevel.Abstract.value:
            params_to_print.append(self.level.data)
        if self.role.data != QubitRole.Null.value:
            params_to_print.append(self.role.data)

        if params_to_print:
            with printer.in_angle_brackets():
                printer.print_list(params_to_print, printer.print_string)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        """Parse type parameters."""
        optional_params = parser.parse_optional_comma_separated_list(
            delimiter=parser.Delimiter.ANGLE, parse=parser.parse_str_literal
        )
        optional_params = optional_params or []

        final_params = []
        match len(optional_params):
            case 0:
                final_params = [
                    StringAttr(QubitLevel.Abstract.value),
                    StringAttr(QubitRole.Null.value),
                ]

            case 1:
                param = optional_params[0]
                if param in QubitLevel.__members__.values():
                    level = StringAttr(param)
                    role = StringAttr(QubitRole.Null.value)
                elif param in QubitRole.__members__.values():
                    level = StringAttr(QubitLevel.Abstract.value)
                    role = StringAttr(param)
                else:
                    raise ParseError(f"Invalid parameter for 'QubitType': {param}.")
                final_params = [level, role]

            case 2:
                final_params = optional_params

            case _:
                raise ParseError(
                    f"Expected 2 or fewer parameters for 'QubitType', got {optional_params}."
                )

        return final_params


@irdl_attr_definition
class QuregType(ParametrizedAttribute, TypeAttribute):
    """An array of value-semantic qubits (i.e. quantum register)."""

    name = "quantum.reg"

    level: StringAttr

    def __init__(self, level: str | StringAttr = QubitLevel.Abstract.value):
        level = level if isinstance(level, StringAttr) else StringAttr(level)
        super().__init__(level)

    def verify(self) -> None:
        """Verify that the attribute is defined correctly."""
        level = self.level.data
        allowed_levels = list(QubitLevel.__members__.values())

        if level not in allowed_levels:
            raise VerifyException(
                f"Invalid value {level} for 'QuregType.level'. Allowed values are {allowed_levels}."
            )

    def print_parameters(self, printer: Printer):
        """Print type parameters."""
        if self.level.data == QubitLevel.Abstract.value:
            return

        with printer.in_angle_brackets():
            printer.print_string(self.level.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        """Parse type parameters."""
        optional_params = parser.parse_optional_comma_separated_list(
            delimiter=parser.Delimiter.ANGLE, parse=parser.parse_str_literal
        )
        optional_params = optional_params or []

        final_params = []
        match len(optional_params):
            case 0:
                final_params = [StringAttr(QubitLevel.Abstract.value)]
            case 1:
                final_params = [StringAttr(optional_params[0])]
            case _:
                raise ParseError(
                    f"Expected 1 or fewer parameters for 'QuregType', got {optional_params}."
                )

        return final_params


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


###############################################################
######################## Constraints ##########################
###############################################################
# pylint: disable=unused-argument


class QubitTypeConstraint(AttrConstraint):
    """Constraint to make QubitType be inferrable during IRDL declaration."""

    level_constr: AttrConstraint
    """The qubit level constraint."""

    role_constr: AttrConstraint
    """The qubit role constraint."""

    def __init__(
        self,
        *,
        level_constr: Sequence[str | StringAttr] | None = None,
        role_constr: Sequence[str | StringAttr] | None = None,
    ):
        if not level_constr:
            self.level_constr = AnyOf([StringAttr(v) for v in QubitLevel.__members__.values()])
        else:
            attr_list = []
            for s in level_constr:
                if isinstance(s, str):
                    assert s in QubitLevel.__members__.values()
                    attr_list.append(StringAttr(s))
                elif isinstance(s, StringAttr):
                    assert s.value in QubitLevel.__members__.values()
                    attr_list.append(s)
                else:
                    raise ValueError(f"Invalid value for 'QubitType.level' constraint: {s}.")
            self.level_constr = AnyOf(attr_list)

        if not role_constr:
            self.role_constr = AnyOf([StringAttr(v) for v in QubitRole.__members__.values()])
        else:
            attr_list = []
            for s in role_constr:
                if isinstance(s, str):
                    assert s in QubitRole.__members__.values()
                    attr_list.append(StringAttr(s))
                elif isinstance(s, StringAttr):
                    assert s.value in QubitRole.__members__.values()
                    attr_list.append(s)
                else:
                    raise ValueError(f"Invalid value for 'QubitType.role' constraint: {s}.")
            self.role_constr = AnyOf(attr_list)

    def can_infer(self, var_constraint_names) -> bool:
        """Check if there is enough information to infer the attribute given the
        constraint variables that are already set.
        """
        if len(self.level_constr.attr_constrs) not in (1, len(QubitLevel.__members__)):
            return False
        if len(self.role_constr.attr_constrs) not in (1, len(QubitRole.__members__)):
            return False
        return True

    def infer(self, context):
        """Infer the attribute given the the values for all variables."""
        if len(self.level_constr.attr_constrs) == 1:
            level = self.level_constr.attr_constrs[0].attr
        else:
            level = StringAttr(QubitLevel.Abstract.value)
        if len(self.role_constr.attr_constrs) == 1:
            role = self.role_constr.attr_constrs[0].attr
        else:
            role = StringAttr(QubitRole.Null.value)

        return QubitType(level, role)

    def verify(self, attr, constraint_context):
        """Verify the constraint."""
        self.level_constr.verify(attr.level, constraint_context)
        self.role_constr.verify(attr.role, constraint_context)

    def mapping_type_vars(self, type_var_mapping):
        """A helper function to make type vars used in attribute definitions concrete when
        creating constraints for new attributes or operations.
        """
        return self


class QuregTypeConstraint(AttrConstraint):
    """Constraint to make QuregType be inferrable during IRDL declaration."""

    level_constr: AttrConstraint
    """The qubit level constraint."""

    def __init__(self, *, level_constr: Sequence[str | StringAttr] | None = None):
        if not level_constr:
            self.level_constr = AnyOf([StringAttr(v) for v in QubitLevel.__members__.values()])
        else:
            attr_list = []
            for s in level_constr:
                if isinstance(s, str):
                    assert s in QubitLevel.__members__.values()
                    attr_list.append(StringAttr(s))
                elif isinstance(s, StringAttr):
                    assert s.value in QubitLevel.__members__.values()
                    attr_list.append(s)
                else:
                    raise ValueError(f"Invalid value for 'QubitType.level' constraint: {s}.")
            self.level_constr = AnyOf(attr_list)

    def can_infer(self, var_constraint_names) -> bool:
        """Check if there is enough information to infer the attribute given the
        constraint variables that are already set.
        """
        if len(self.level_constr.attr_constrs) not in (1, len(QubitLevel.__members__)):
            return False
        return True

    def infer(self, context):
        """Infer the attribute given the the values for all variables."""
        if len(self.level_constr.attr_constrs) == 1:
            level = self.level_constr.attr_constrs[0].attr
        else:
            level = StringAttr(QubitLevel.Abstract.value)

        return QuregType(level)

    def verify(self, attr, constraint_context):
        """Verify the constraint."""
        self.level_constr.verify(attr.level, constraint_context)

    def mapping_type_vars(self, type_var_mapping):
        """A helper function to make type vars used in attribute definitions concrete when
        creating constraints for new attributes or operations.
        """
        return self
