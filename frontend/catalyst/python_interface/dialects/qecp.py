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

"""
This module contains the experimental QEC physical dialect for the xDSL Python interface to Catalyst.

This dialect is a mirror of the ``qecp`` MLIR dialect, which should be taken as the source of truth.
For a complete description of this dialect, please see

    mlir/include/QecPhysical/IR/QecPhysicalDialect.td
"""

from xdsl.dialects.builtin import I64, IntegerAttr
from xdsl.ir import (
    Dialect,
    EnumAttribute,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    StrEnum,
    TypeAttribute,
)
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import Parser
from xdsl.printer import Printer


class QecPhysicalQubitRole(StrEnum):
    """Enum for the role specialization of QEC physical qubits"""

    Data = "data"
    Aux = "aux"


@irdl_attr_definition
class QecPhysicalQubitRoleAttr(EnumAttribute[QecPhysicalQubitRole], SpacedOpaqueSyntaxAttribute):
    """Role specialization of QEC physical qubits"""

    name = "qecp.qubit_role"


@irdl_attr_definition
class QecPhysicalQubitType(ParametrizedAttribute, TypeAttribute):
    """A value-semantic QEC physical qubit."""

    name = "qecp.qubit"

    role: QecPhysicalQubitRoleAttr

    def __init__(self, role: str | QecPhysicalQubitRoleAttr):
        role_attr = (
            role if isinstance(role, QecPhysicalQubitRoleAttr) else QecPhysicalQubitRoleAttr(role)
        )
        super().__init__(role_attr)

    def print_parameters(self, printer: Printer) -> None:
        """Print the attribute parameters."""
        with printer.in_angle_brackets():
            printer.print_string(self.role.data)

    @classmethod
    def parse_parameters(cls, parser: Parser) -> list[QecPhysicalQubitRoleAttr]:
        """Parse the attribute parameters."""
        with parser.in_angle_brackets():
            role = parser.parse_identifier_or_str_literal()

        return [QecPhysicalQubitRoleAttr(role)]


@irdl_attr_definition
class PhysicalCodeblockType(ParametrizedAttribute, TypeAttribute):
    """A value-semantic physical codeblock."""

    name = "qecp.codeblock"

    k: IntegerAttr[I64]
    n: IntegerAttr[I64]

    def __init__(self, k: int | IntegerAttr[I64], n: int | IntegerAttr[I64]):
        k_attr = IntegerAttr(k, 64) if isinstance(k, int) else k
        n_attr = IntegerAttr(n, 64) if isinstance(n, int) else n
        super().__init__(k_attr, n_attr)

    def print_parameters(self, printer: Printer) -> None:
        """Print the attribute parameters."""
        with printer.in_angle_brackets():
            printer.print_int(self.k.value.data)
            printer.print_string(" x ")
            printer.print_int(self.n.value.data)

    @classmethod
    def parse_parameters(cls, parser: Parser) -> list[IntegerAttr]:
        """Parse the attribute parameters."""
        with parser.in_angle_brackets():
            k = parser.parse_integer()
            parser.parse_characters("x")
            n = parser.parse_integer()

        return [IntegerAttr(k, 64), IntegerAttr(n, 64)]


@irdl_attr_definition
class PhysicalHyperRegisterType(ParametrizedAttribute, TypeAttribute):
    """A value-semantic physical hyper-register."""

    name = "qecp.hyperreg"

    a: IntegerAttr[I64]
    k: IntegerAttr[I64]
    n: IntegerAttr[I64]

    def __init__(
        self, a: int | IntegerAttr[I64], k: int | IntegerAttr[I64], n: int | IntegerAttr[I64]
    ):
        a_attr = IntegerAttr(a, 64) if isinstance(a, int) else a
        k_attr = IntegerAttr(k, 64) if isinstance(k, int) else k
        n_attr = IntegerAttr(n, 64) if isinstance(n, int) else n
        super().__init__(a_attr, k_attr, n_attr)

    def print_parameters(self, printer: Printer) -> None:
        """Print the attribute parameters."""
        with printer.in_angle_brackets():
            printer.print_int(self.a.value.data)
            # TODO: We need to print with whitespace around 'x' for compatibility with MLIR parser
            printer.print_string(" x ")
            printer.print_int(self.k.value.data)
            printer.print_string(" x ")
            printer.print_int(self.n.value.data)

    @classmethod
    def parse_parameters(cls, parser: Parser) -> list[IntegerAttr]:
        """Parse the attribute parameters."""
        with parser.in_angle_brackets():
            a = parser.parse_integer()
            parser.parse_characters("x")
            k = parser.parse_integer()
            parser.parse_characters("x")
            n = parser.parse_integer()

        return [IntegerAttr(a, 64), IntegerAttr(k, 64), IntegerAttr(n, 64)]


QecPhysical = Dialect(
    "qecp",
    [
        # Ops
    ],
    [
        QecPhysicalQubitRoleAttr,
        QecPhysicalQubitType,
        PhysicalCodeblockType,
        PhysicalHyperRegisterType,
    ],
)
