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

from collections.abc import Sequence
from typing import ClassVar, TypeAlias

from xdsl.dialects.builtin import I64, ContainerOf, IndexType, IntegerAttr
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
    IRDLOperation,
    TypeAttributeInvT,
    VarConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer


class QecPhysicalQubitRole(StrEnum):
    """Enum for the role specialization of QEC physical qubits"""

    Data = "data"
    Aux = "aux"


@irdl_attr_definition
class QecPhysicalQubitRoleAttr(EnumAttribute[QecPhysicalQubitRole], SpacedOpaqueSyntaxAttribute):
    """Role specialization of QEC physical qubits"""

    name = "qecp.qubit_role"

    def __init__(self, role: str | QecPhysicalQubitRole):
        role_enum = role if isinstance(role, QecPhysicalQubitRole) else QecPhysicalQubitRole(role)
        super().__init__(role_enum)


@irdl_attr_definition
class QecPhysicalQubitType(ParametrizedAttribute, TypeAttribute):
    """A value-semantic QEC physical qubit"""

    name = "qecp.qubit"

    role: QecPhysicalQubitRoleAttr

    def __init__(self, role: str | QecPhysicalQubitRole | QecPhysicalQubitRoleAttr):
        role_attr = (
            role if isinstance(role, QecPhysicalQubitRoleAttr) else QecPhysicalQubitRoleAttr(role)
        )
        super().__init__(role_attr)

    def print_parameters(self, printer: Printer) -> None:
        """Print the attribute parameters."""
        with printer.in_angle_brackets():
            printer.print_string(self.role.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[QecPhysicalQubitRoleAttr]:
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
    def parse_parameters(cls, parser: AttrParser) -> Sequence[IntegerAttr]:
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
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """Parse the attribute parameters."""
        with parser.in_angle_brackets():
            width = parser.parse_integer()
            parser.parse_characters("x")
            k = parser.parse_integer()
            parser.parse_characters("x")
            n = parser.parse_integer()

        return [IntegerAttr(width, 64), IntegerAttr(k, 64), IntegerAttr(n, 64)]


QecPhysicalQubitSSAValue: TypeAlias = SSAValue[QecPhysicalQubitType]
PhysicalCodeBlockSSAValue: TypeAlias = SSAValue[PhysicalCodeblockType]
PhysicalHyperRegisterSSAValue: TypeAlias = SSAValue[PhysicalHyperRegisterType]

anyPhysicalQubit = ContainerOf(QecPhysicalQubitType)
anyPhysicalCodeblock = ContainerOf(PhysicalCodeblockType)
anyPhysicalHyperRegister = ContainerOf(PhysicalHyperRegisterType)


def _get_type_from_ssa_value_or_operation(
    arg: SSAValue | Operation, expected_type: TypeAttributeInvT
):
    """Helper function that returns the type of an SSA value or of an operation's returned value.

    Args:
        arg (SSAValue | Operation): An SSA value or an operation that returns exactly one SSA value.

    Returns:
        TypeAttribute: The type.
    """
    if isinstance(arg, Operation):
        arg_types = arg.result_types
        assert len(arg_types) == 1, f"Expected operation '{arg}' to have exactly one result type"
        arg_type = arg_types[0]
        assert isinstance(
            arg_type, expected_type
        ), f"Expected operation '{arg}' to have result type '{expected_type.name}'"

    else:
        arg_type = arg.type
        assert isinstance(
            arg_type, expected_type
        ), f"Expected value '{arg}' to have type '{expected_type.name}'"

    return arg_type


def get_physical_hyper_reg_type(
    hyper_reg: PhysicalHyperRegisterSSAValue | Operation,
) -> PhysicalHyperRegisterType:
    """Helper function to return the physical hyper-register type given an SSA value or operation."""
    return _get_type_from_ssa_value_or_operation(hyper_reg, PhysicalHyperRegisterType)


def get_physical_codeblock_type(
    codeblock: PhysicalCodeBlockSSAValue | Operation,
) -> PhysicalCodeblockType:
    """Helper function to return the physical codeblock type given an SSA value or operation."""
    return _get_type_from_ssa_value_or_operation(codeblock, PhysicalCodeblockType)


def get_physical_qubit_type(
    qubit: QecPhysicalQubitSSAValue | Operation,
) -> QecPhysicalQubitType:
    """Helper function to return the physical qubit type given an SSA value or operation."""
    return _get_type_from_ssa_value_or_operation(qubit, QecPhysicalQubitType)


@irdl_op_definition
class AllocOp(IRDLOperation):
    """Allocate a physical hyper-register containing a sequence of physical codeblocks."""

    name = "qecp.alloc"

    assembly_format = """
            `(` `)` attr-dict `:` type($hyper_reg)
        """

    hyper_reg = result_def(PhysicalHyperRegisterType)

    def __init__(self, hyper_reg: PhysicalHyperRegisterType):
        super().__init__(result_types=(hyper_reg,))


@irdl_op_definition
class DeallocOp(IRDLOperation):
    """Deallocate a physical hyper-register."""

    name = "qecp.dealloc"

    assembly_format = """
            $hyper_reg attr-dict `:` type($hyper_reg)
        """

    hyper_reg = operand_def(PhysicalHyperRegisterType)

    def __init__(self, hyper_reg: PhysicalHyperRegisterSSAValue | Operation):
        super().__init__(operands=(hyper_reg,))


@irdl_op_definition
class AllocAuxQubitOp(IRDLOperation):
    """Allocate a single auxiliary QEC physical qubit."""

    name = "qecp.alloc_aux"

    assembly_format = """
            attr-dict `:` type($qubit)
        """

    qubit = result_def(QecPhysicalQubitType(role=QecPhysicalQubitRole.Aux))

    def __init__(self):
        super().__init__(result_types=(QecPhysicalQubitType(role=QecPhysicalQubitRole.Aux),))


@irdl_op_definition
class DeallocAuxQubitOp(IRDLOperation):
    """Deallocate a single auxiliary QEC physical qubit."""

    name = "qecp.dealloc_aux"

    assembly_format = """
            $qubit attr-dict `:` type($qubit)
        """

    qubit = operand_def(QecPhysicalQubitType(role=QecPhysicalQubitRole.Aux))

    def __init__(self, qubit: QecPhysicalQubitSSAValue | Operation):
        qubit_type = get_physical_qubit_type(qubit)
        if qubit_type.role.data != str(QecPhysicalQubitRole.Aux):
            raise ValueError(
                f"{self.name} op expected a qubit with role '{str(QecPhysicalQubitRole.Aux)}', "
                f"but got '{qubit_type.role.data}'"
            )

        super().__init__(operands=(qubit,))


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
        hyper_reg: PhysicalHyperRegisterSSAValue | Operation,
        idx: int | IntegerAttr | SSAValue[IndexType] | Operation,
    ):
        if isinstance(idx, int):
            idx = IntegerAttr.from_int_and_width(idx, 64)

        if isinstance(idx, IntegerAttr):
            operands = (hyper_reg, None)
            properties = {"idx_attr": idx}
        else:
            operands = (hyper_reg, idx)
            properties = {}

        hyper_reg_type = get_physical_hyper_reg_type(hyper_reg)
        result_type = PhysicalCodeblockType(k=hyper_reg_type.k, n=hyper_reg_type.n)

        super().__init__(
            operands=operands,
            result_types=(result_type,),
            properties=properties,
        )


@irdl_op_definition
class InsertCodeblockOp(IRDLOperation):
    """Update the physical codeblock value of a hyper-register."""

    T: ClassVar = VarConstraint("T", anyPhysicalHyperRegister)

    name = "qecp.insert_block"

    assembly_format = """
            $in_hyper_reg `[` ($idx^):($idx_attr)? `]` `,` $codeblock attr-dict `:` type($in_hyper_reg) `,` type($codeblock)
        """

    in_hyper_reg = operand_def(T)

    idx = opt_operand_def(IndexType)

    idx_attr = opt_prop_def(IntegerAttr.constr(type=IndexType, value=AtLeast(0)))

    codeblock = operand_def(PhysicalCodeblockType)

    out_hyper_reg = result_def(T)

    def __init__(
        self,
        in_hyper_reg: PhysicalHyperRegisterSSAValue | Operation,
        idx: int | IntegerAttr | SSAValue[IndexType] | Operation,
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

        in_hyper_reg_type = get_physical_hyper_reg_type(in_hyper_reg)

        super().__init__(
            operands=operands, properties=properties, result_types=(in_hyper_reg_type,)
        )


@irdl_op_definition
class ExtractQubitOp(IRDLOperation):
    """Extract a physical qubit value from a codeblock.

    This operation extracts a QEC physical qubit value from a physical codeblock. The qubit value is
    restricted to have the 'data' role; in other words, an auxiliary qubit cannot be extracted from
    a physical codeblock.
    """

    name = "qecp.extract"

    assembly_format = """
            $codeblock `[` ($idx^):($idx_attr)? `]` attr-dict `:` type($codeblock) `->` type($qubit)
        """

    codeblock = operand_def(PhysicalCodeblockType)

    idx = opt_operand_def(IndexType)

    idx_attr = opt_prop_def(IntegerAttr.constr(type=IndexType, value=AtLeast(0)))

    qubit = result_def(QecPhysicalQubitType)

    def __init__(
        self,
        codeblock: PhysicalCodeBlockSSAValue | Operation,
        idx: int | IntegerAttr | SSAValue[IndexType] | Operation,
    ):
        if isinstance(idx, int):
            idx = IntegerAttr.from_int_and_width(idx, 64)

        if isinstance(idx, IntegerAttr):
            operands = (codeblock, None)
            properties = {"idx_attr": idx}
        else:
            operands = (codeblock, idx)
            properties = {}

        result_type = QecPhysicalQubitType(role=QecPhysicalQubitRole.Data)

        super().__init__(
            operands=operands,
            result_types=(result_type,),
            properties=properties,
        )


@irdl_op_definition
class InsertQubitOp(IRDLOperation):
    """Update the physical qubit value of a codeblock.

    This operation updates the value of a QEC physical qubit in a physical codeblock. The qubit
    value is restricted to have the 'data' role; in other words, an auxiliary qubit cannot be
    inserted into a physical codeblock.
    """

    T: ClassVar = VarConstraint("T", anyPhysicalCodeblock)

    name = "qecp.insert"

    assembly_format = """
            $in_codeblock `[` ($idx^):($idx_attr)? `]` `,` $qubit attr-dict `:` type($in_codeblock) `,` type($qubit)
        """

    in_codeblock = operand_def(T)

    idx = opt_operand_def(IndexType)

    idx_attr = opt_prop_def(IntegerAttr.constr(type=IndexType, value=AtLeast(0)))

    qubit = operand_def(QecPhysicalQubitType)

    out_codeblock = result_def(T)

    def __init__(
        self,
        in_codeblock: PhysicalCodeBlockSSAValue | Operation,
        idx: int | IntegerAttr | SSAValue[IndexType] | Operation,
        qubit: QecPhysicalQubitSSAValue | Operation,
    ):
        if isinstance(idx, int):
            idx = IntegerAttr.from_int_and_width(idx, 64)

        if isinstance(idx, IntegerAttr):
            operands = (in_codeblock, None, qubit)
            properties = {"idx_attr": idx}
        else:
            operands = (in_codeblock, idx, qubit)
            properties = {}

        in_codeblock_type = get_physical_codeblock_type(in_codeblock)

        super().__init__(
            operands=operands, properties=properties, result_types=(in_codeblock_type,)
        )


QecPhysical = Dialect(
    "qecp",
    [
        AllocOp,
        DeallocOp,
        AllocAuxQubitOp,
        DeallocAuxQubitOp,
        ExtractCodeblockOp,
        InsertCodeblockOp,
        ExtractQubitOp,
        InsertQubitOp,
    ],
    [
        QecPhysicalQubitRoleAttr,
        QecPhysicalQubitType,
        PhysicalCodeblockType,
        PhysicalHyperRegisterType,
    ],
)
