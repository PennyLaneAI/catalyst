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
from collections.abc import Sequence
from typing import ClassVar, TypeAlias

from xdsl.dialects.builtin import I64, ContainerOf, IndexType, IntegerAttr
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
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


@irdl_attr_definition
class LogicalCodeblockType(ParametrizedAttribute, TypeAttribute):
    """A value-semantic logical codeblock"""

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
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """Parse the attribute parameters."""
        with parser.in_angle_brackets():
            k = parser.parse_integer()

        return [IntegerAttr(k, 64)]


@irdl_attr_definition
class LogicalHyperRegisterType(ParametrizedAttribute, TypeAttribute):
    """A value-semantic logical hyper-register"""

    name = "qecl.hyperreg"

    width: IntegerAttr[I64]
    k: IntegerAttr[I64]

    def __init__(self, width: int | IntegerAttr[I64], k: int | IntegerAttr[I64]):
        width_attr = IntegerAttr(width, 64) if isinstance(width, int) else width
        k_attr = IntegerAttr(k, 64) if isinstance(k, int) else k
        super().__init__(width_attr, k_attr)

    def print_parameters(self, printer: Printer) -> None:
        """Print the attribute parameters."""
        with printer.in_angle_brackets():
            printer.print_int(self.width.value.data)
            # TODO: We need to print with whitespace around 'x' for compatibility with MLIR parser
            printer.print_string(" x ")
            printer.print_int(self.k.value.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """Parse the attribute parameters."""
        with parser.in_angle_brackets():
            width = parser.parse_integer()
            parser.parse_characters("x")
            k = parser.parse_integer()

        return [IntegerAttr(width, 64), IntegerAttr(k, 64)]


LogicalCodeBlockSSAValue: TypeAlias = SSAValue[LogicalCodeblockType]
LogicalHyperRegisterSSAValue: TypeAlias = SSAValue[LogicalHyperRegisterType]

anyLogicalCodeblock = ContainerOf(LogicalCodeblockType)
anyLogicalHyperRegister = ContainerOf(LogicalHyperRegisterType)


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


def get_logical_hyper_reg_type(
    hyper_reg: LogicalHyperRegisterSSAValue | Operation,
) -> LogicalHyperRegisterType:
    """Helper function to return the logical hyper-register type given an SSA value or operation."""
    return _get_type_from_ssa_value_or_operation(hyper_reg, LogicalHyperRegisterType)


@irdl_op_definition
class AllocOp(IRDLOperation):
    """Allocate a logical hyper-register containing a sequence of logical codeblocks."""

    name = "qecl.alloc"

    assembly_format = """
            `(` `)` attr-dict `:` type($hyper_reg)
        """

    hyper_reg = result_def(LogicalHyperRegisterType)

    def __init__(self, hyper_reg: LogicalHyperRegisterType):
        super().__init__(result_types=(hyper_reg,))


@irdl_op_definition
class DeallocOp(IRDLOperation):
    """Deallocate a logical hyper-register."""

    name = "qecl.dealloc"

    assembly_format = """
            $hyper_reg attr-dict `:` type($hyper_reg)
        """

    hyper_reg = operand_def(LogicalHyperRegisterType)

    def __init__(self, hyper_reg: LogicalHyperRegisterSSAValue | Operation):
        super().__init__(operands=(hyper_reg,))


@irdl_op_definition
class ExtractCodeblockOp(IRDLOperation):
    """Extract a logical codeblock value from a hyper-register."""

    name = "qecl.extract_block"

    assembly_format = """
            $hyper_reg `[` ($idx^):($idx_attr)? `]` attr-dict `:` type($hyper_reg) `->` type($codeblock)
        """

    hyper_reg = operand_def(LogicalHyperRegisterType)

    idx = opt_operand_def(IndexType)

    idx_attr = opt_prop_def(IntegerAttr.constr(type=IndexType, value=AtLeast(0)))

    codeblock = result_def(LogicalCodeblockType)

    def __init__(
        self,
        hyper_reg: LogicalHyperRegisterSSAValue | Operation,
        idx: int | IntegerAttr | SSAValue[IndexType] | Operation,
    ):
        if isinstance(idx, int):
            idx = IntegerAttr(idx, IndexType())

        if isinstance(idx, IntegerAttr):
            operands = (hyper_reg, None)
            properties = {"idx_attr": idx}
        else:
            operands = (hyper_reg, idx)
            properties = {}

        hyper_reg_type = get_logical_hyper_reg_type(hyper_reg)
        result_type = LogicalCodeblockType(k=hyper_reg_type.k)

        super().__init__(
            operands=operands,
            result_types=(result_type,),
            properties=properties,
        )


@irdl_op_definition
class InsertCodeblockOp(IRDLOperation):
    """Update the logical codeblock value of a hyper-register."""

    T: ClassVar = VarConstraint("T", anyLogicalHyperRegister)

    name = "qecl.insert_block"

    assembly_format = """
            $in_hyper_reg `[` ($idx^):($idx_attr)? `]` `,` $codeblock attr-dict `:` type($in_hyper_reg) `,` type($codeblock)
        """

    in_hyper_reg = operand_def(T)

    idx = opt_operand_def(IndexType)

    idx_attr = opt_prop_def(IntegerAttr.constr(type=IndexType, value=AtLeast(0)))

    codeblock = operand_def(LogicalCodeblockType)

    out_hyper_reg = result_def(T)

    def __init__(
        self,
        in_hyper_reg: LogicalHyperRegisterSSAValue | Operation,
        idx: int | IntegerAttr | SSAValue[IndexType] | Operation,
        codeblock: LogicalCodeBlockSSAValue | Operation,
    ):
        if isinstance(idx, int):
            idx = IntegerAttr(idx, IndexType())

        if isinstance(idx, IntegerAttr):
            operands = (in_hyper_reg, None, codeblock)
            properties = {"idx_attr": idx}
        else:
            operands = (in_hyper_reg, idx, codeblock)
            properties = {}

        in_hyper_reg_type = get_logical_hyper_reg_type(in_hyper_reg)

        super().__init__(
            operands=operands, properties=properties, result_types=(in_hyper_reg_type,)
        )


QecLogical = Dialect(
    "qecl",
    [
        AllocOp,
        DeallocOp,
        ExtractCodeblockOp,
        InsertCodeblockOp,
    ],
    [
        LogicalCodeblockType,
        LogicalHyperRegisterType,
    ],
)
