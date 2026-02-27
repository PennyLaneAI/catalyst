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
This module contains the experimental QEC logical dialect for the xDSL Python interface to Catalyst.

This dialect is a mirror of the ``qecl`` MLIR dialect, which should be taken as the source of truth.
For a complete description of this dialect, please see

    mlir/include/QecLogical/IR/QecLogicalDialect.td
"""

from xdsl.dialects.builtin import I64, IntegerAttr
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import Parser
from xdsl.printer import Printer


@irdl_attr_definition
class LogicalCodeblockType(ParametrizedAttribute, TypeAttribute):
    """A value-semantic logical codeblock."""

    name = "qecl.codeblock"

    k: IntegerAttr[I64]

    def __init__(self, k: int | IntegerAttr[I64]):
        k_attr = IntegerAttr(k, 64) if isinstance(k, int) else k
        super().__init__(k_attr)

    def print_parameters(self, printer: Printer) -> None:
        """Print the attribute parameters."""
        with printer.in_angle_brackets():
            printer.print_int(self.k.value.data)

    @classmethod
    def parse_parameters(cls, parser: Parser) -> list[IntegerAttr]:
        """Parse the attribute parameters."""
        with parser.in_angle_brackets():
            k = parser.parse_integer()

        return [IntegerAttr(k, 64)]


@irdl_attr_definition
class LogicalHyperRegisterType(ParametrizedAttribute, TypeAttribute):
    """A value-semantic logical hyper-register."""

    name = "qecl.hyperreg"

    a: IntegerAttr[I64]
    k: IntegerAttr[I64]

    def __init__(self, a: int | IntegerAttr[I64], k: int | IntegerAttr[I64]):
        a_attr = IntegerAttr(a, 64) if isinstance(a, int) else a
        k_attr = IntegerAttr(k, 64) if isinstance(k, int) else k
        super().__init__(a_attr, k_attr)

    def print_parameters(self, printer: Printer) -> None:
        """Print the attribute parameters."""
        with printer.in_angle_brackets():
            printer.print_int(self.a.value.data)
            # TODO: We need to print with whitespace around 'x' for compatibility with MLIR parser
            printer.print_string(" x ")
            printer.print_int(self.k.value.data)

    @classmethod
    def parse_parameters(cls, parser: Parser) -> list[IntegerAttr]:
        """Parse the attribute parameters."""
        with parser.in_angle_brackets():
            a = parser.parse_integer()
            parser.parse_characters("x")
            k = parser.parse_integer()

        return [IntegerAttr(a, 64), IntegerAttr(k, 64)]


QecLogical = Dialect(
    "qecl",
    [
        # Ops
    ],
    [
        LogicalCodeblockType,
        LogicalHyperRegisterType,
    ],
)
