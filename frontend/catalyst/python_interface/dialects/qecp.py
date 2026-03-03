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
This module contains the experimental QEC physical dialect for the xDSL Python interface to
Catalyst.

This dialect is a mirror of the ``qecp`` MLIR dialect, which should be taken as the source of truth.
For a complete description of this dialect, please see

    mlir/include/QecPhysical/IR/QecPhysicalDialect.td
"""

from collections.abc import Set as AbstractSet
from typing import TypeAlias

from xdsl.dialects.builtin import I64, IndexType, IntegerAttr, IntegerType
from xdsl.ir import (
    Attribute,
    Dialect,
    EnumAttribute,
    Operation,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    StrEnum,
    TypeAttribute,
)
from xdsl.irdl import (
    AtLeast,
    AttrConstraint,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    result_def,
)
from xdsl.irdl.constraints import ConstraintContext
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
    """A value-semantic QEC physical qubit"""

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
    """A value-semantic physical codeblock"""

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

    width: IntegerAttr[I64]
    k: IntegerAttr[I64]
    n: IntegerAttr[I64]

    def __init__(
        self, width: int | IntegerAttr[I64], k: int | IntegerAttr[I64], n: int | IntegerAttr[I64]
    ):
        width_attr = IntegerAttr(width, 64) if isinstance(width, int) else width
        k_attr = IntegerAttr(k, 64) if isinstance(k, int) else k
        n_attr = IntegerAttr(n, 64) if isinstance(n, int) else n
        super().__init__(width_attr, k_attr, n_attr)

    def print_parameters(self, printer: Printer) -> None:
        """Print the attribute parameters."""
        with printer.in_angle_brackets():
            printer.print_int(self.width.value.data)
            # TODO: We need to print with whitespace around 'x' for compatibility with MLIR parser
            printer.print_string(" x ")
            printer.print_int(self.k.value.data)
            printer.print_string(" x ")
            printer.print_int(self.n.value.data)

    @classmethod
    def parse_parameters(cls, parser: Parser) -> list[IntegerAttr]:
        """Parse the attribute parameters."""
        with parser.in_angle_brackets():
            width = parser.parse_integer()
            parser.parse_characters("x")
            k = parser.parse_integer()
            parser.parse_characters("x")
            n = parser.parse_integer()

        return [IntegerAttr(width, 64), IntegerAttr(k, 64), IntegerAttr(n, 64)]


PhysicalCodeBlockSSAValue: TypeAlias = SSAValue[PhysicalCodeblockType]
PhysicalHyperRegisterSSAValue: TypeAlias = SSAValue[PhysicalHyperRegisterType]


class PhysicalHyperRegisterTypeConstraint(AttrConstraint):
    """Constraint to make PhysicalHyperRegisterType inferrable during IRDL declaration."""

    # This is a bit of a hack for ops that both consume and return a LogicalHyperRegisterType.
    # See comment for LogicalHyperRegisterTypeConstraint for an explanation of what's happening.

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        """Verify the constraint and add resolved values to the ConstraintContext."""
        constraint_context.set_attr_variable("hyper_reg_type", attr)

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        """Check if there is enough information to infer the attribute given the constraint
        variables that are already set.
        """
        # Assume we can always infer
        return True

    def infer(self, context: ConstraintContext) -> PhysicalHyperRegisterType:
        """Infer the attribute given the the values for all variables."""
        hyper_reg_type = context.get_variable("hyper_reg_type")
        assert isinstance(
            hyper_reg_type, PhysicalHyperRegisterType
        ), f"Expected a PhysicalHyperRegisterType from constraint context, but got {hyper_reg_type}"
        return hyper_reg_type

    def mapping_type_vars(self, type_var_mapping):
        """A helper function to make type vars used in attribute definitions concrete when creating
        constraints for new attributes or operations.
        """
        return self


@irdl_op_definition
class AllocOp(IRDLOperation):
    """Allocate a physical hyper-register containing a sequence of physical codeblocks."""

    name = "qecp.alloc"

    assembly_format = """
            `(` `)` attr-dict `:` type($hyper_reg)
        """

    hyper_reg = result_def(PhysicalHyperRegisterType)


@irdl_op_definition
class DeallocOp(IRDLOperation):
    """Deallocate a physical hyper-register."""

    name = "qecp.dealloc"

    assembly_format = """
            $hyper_reg attr-dict `:` type($hyper_reg)
        """

    hyper_reg = operand_def(PhysicalHyperRegisterType)


@irdl_op_definition
class ExtractCodeblockOp(IRDLOperation):
    """Extract a physical codeblock value from a hyper-register."""

    name = "qecp.extract_block"

    assembly_format = """
            $hyper_reg `[` ($idx^):($idx_attr)? `]` attr-dict `:` type($hyper_reg) `->` type($codeblock)
        """

    hyper_reg = operand_def(PhysicalHyperRegisterType)

    idx = opt_operand_def(IndexType)

    idx_attr = opt_prop_def(IntegerAttr.constr(type=IndexType, value=AtLeast(0)))

    codeblock = result_def(PhysicalCodeblockType)

    def __init__(
        self,
        hyper_reg: PhysicalHyperRegisterType | Operation,
        idx: int | SSAValue[IntegerType] | Operation | IntegerAttr,
    ):
        if isinstance(idx, int):
            idx = IntegerAttr.from_int_and_width(idx, 64)

        if isinstance(idx, IntegerAttr):
            operands = (hyper_reg, None)
            properties = {"idx_attr": idx}
        else:
            operands = (hyper_reg, idx)
            properties = {}

        if isinstance(hyper_reg, PhysicalHyperRegisterType):
            result_type = PhysicalCodeblockType(k=hyper_reg.k, n=hyper_reg.n)
        else:
            result_type = PhysicalCodeblockType(k=hyper_reg.type.k, n=hyper_reg.type.n)

        super().__init__(
            operands=operands,
            result_types=(result_type,),
            properties=properties,
        )


@irdl_op_definition
class InsertCodeblockOp(IRDLOperation):
    """Update the physical codeblock value of a hyper-register."""

    name = "qecp.insert_block"

    assembly_format = """
            $in_hyper_reg `[` ($idx^):($idx_attr)? `]` `,` $codeblock attr-dict `:` type($in_hyper_reg) `,` type($codeblock)
        """

    in_hyper_reg = operand_def(PhysicalHyperRegisterTypeConstraint())

    idx = opt_operand_def(IndexType)

    idx_attr = opt_prop_def(IntegerAttr.constr(type=IndexType, value=AtLeast(0)))

    codeblock = operand_def(PhysicalCodeblockType)

    out_hyper_reg = result_def(PhysicalHyperRegisterTypeConstraint())

    def __init__(
        self,
        in_hyper_reg: PhysicalCodeBlockSSAValue | Operation,
        idx: SSAValue[IntegerType] | Operation | int | IntegerAttr,
        codeblock: PhysicalCodeBlockSSAValue | Operation,
    ):
        if isinstance(idx, int):
            idx = IntegerAttr.from_int_and_width(idx, 64)

        if isinstance(idx, IntegerAttr):
            operands = (in_hyper_reg, None, codeblock)
            properties = {"idx_attr": idx}
        else:
            operands = (in_hyper_reg, idx, codeblock)
            properties = {}

        super().__init__(
            operands=operands, properties=properties, result_types=(in_hyper_reg.type,)
        )


@irdl_op_definition
class AllocAuxQubitOp(IRDLOperation):
    """Allocate a single auxiliary QEC physical qubit."""

    name = "qecp.alloc_aux"

    assembly_format = """
            attr-dict `:` type($qubit)
        """

    qubit = result_def(QecPhysicalQubitType(role=QecPhysicalQubitRole.Aux))


@irdl_op_definition
class DeallocAuxQubitOp(IRDLOperation):
    """Deallocate a single auxiliary QEC physical qubit."""

    name = "qecp.dealloc_aux"

    assembly_format = """
            $qubit attr-dict `:` type($qubit)
        """

    qubit = operand_def(QecPhysicalQubitType(role=QecPhysicalQubitRole.Aux))


QecPhysical = Dialect(
    "qecp",
    [
        AllocOp,
        DeallocOp,
        ExtractCodeblockOp,
        InsertCodeblockOp,
        AllocAuxQubitOp,
        DeallocAuxQubitOp,
    ],
    [
        QecPhysicalQubitRoleAttr,
        QecPhysicalQubitType,
        PhysicalCodeblockType,
        PhysicalHyperRegisterType,
    ],
)
