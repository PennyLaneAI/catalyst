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
    prop_def,
    result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer


class LogicalCodeblockInitState(StrEnum):
    """Enum for logical codeblock initial state"""

    Zero = "zero"
    # Add other supported codeblock initial states here


@irdl_attr_definition
class LogicalCodeblockInitStateAttr(
    EnumAttribute[LogicalCodeblockInitState], SpacedOpaqueSyntaxAttribute
):
    """Role specialization of QEC physical qubits"""

    name = "qecl.codeblock_init_state"

    def __init__(self, init_state: str | LogicalCodeblockInitState):
        init_state_enum = (
            init_state
            if isinstance(init_state, LogicalCodeblockInitState)
            else LogicalCodeblockInitState(init_state)
        )
        super().__init__(init_state_enum)


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


def get_logical_codeblock_type(
    hyper_reg: LogicalCodeBlockSSAValue | Operation,
) -> LogicalCodeblockType:
    """Helper function to return the logical codeblock type given an SSA value or operation."""
    return _get_type_from_ssa_value_or_operation(hyper_reg, LogicalCodeblockType)


class LogicalHyperRegisterTypeConstraint(AttrConstraint):
    """Constraint to make LogicalHyperRegisterType inferrable during IRDL declaration."""

    # This is a bit of a hack for ops that both consume and return a LogicalHyperRegisterType.
    # Here's what's happening. In the op's assembly format, we typically specify the input
    # hyper-register operand with `type($in_hyper_reg)`, but we don't do the same for the
    # `$out_hyper_reg` result, since we implicitly constrain it to be the same type as the input.
    # When xDSL parses the op's assembly format, in FormatProgram.parse(), it calls a function
    # resolve_constraint_variables(). This function runs the verify() method of every constrained
    # operand, result, attribute, etc. in the op. From these verify() methods, it is possible to
    # update a ConstraintContext variable with data relating to the constraints needed for other
    # operands/results/attributes/etc. When it calls verify() on the input hyper-register operand,
    # we store its type in the context variable. Then, when the assembly parser attempts to resolve
    # the result types, it calls the infer() method, which looks up the hyper-register type in the
    # context variable and uses the same type for the returned hyper-register.

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        """Verify the constraint and add resolved values to the ConstraintContext."""
        constraint_context.set_attr_variable("hyper_reg_type", attr)

    # pylint: disable=unused-argument
    def can_infer(self, var_constraint_names: Set[str]) -> bool:
        """Check if there is enough information to infer the attribute given the constraint
        variables that are already set.
        """
        # Assume we can always infer
        return True

    def infer(self, context: ConstraintContext) -> LogicalHyperRegisterType:
        """Infer the attribute given the the values for all variables."""
        hyper_reg_type = context.get_variable("hyper_reg_type")
        assert isinstance(
            hyper_reg_type, LogicalHyperRegisterType
        ), f"Expected a LogicalHyperRegisterType from constraint context, but got {hyper_reg_type}"
        return hyper_reg_type

    # pylint: disable=unused-argument
    def mapping_type_vars(self, type_var_mapping):
        """A helper function to make type vars used in attribute definitions concrete when creating
        constraints for new attributes or operations.
        """
        return self  # pragma: nocover


class LogicalCodeblockConstraint(AttrConstraint):
    """Constraint to make LogicalCodeblock inferrable during IRDL declaration."""

    # See comment in LogicalHyperRegisterTypeConstraint above for description of what's happening
    # here.

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        """Verify the constraint and add resolved values to the ConstraintContext."""
        constraint_context.set_attr_variable("codeblock_type", attr)

    # pylint: disable=unused-argument
    def can_infer(self, var_constraint_names: Set[str]) -> bool:
        """Check if there is enough information to infer the attribute given the constraint
        variables that are already set.
        """
        # Assume we can always infer
        return True

    def infer(self, context: ConstraintContext) -> LogicalCodeblockType:
        """Infer the attribute given the the values for all variables."""
        codeblock_type = context.get_variable("codeblock_type")
        assert isinstance(
            codeblock_type, LogicalCodeblockType
        ), f"Expected a LogicalCodeblockType from constraint context, but got {codeblock_type}"
        return codeblock_type

    # pylint: disable=unused-argument
    def mapping_type_vars(self, type_var_mapping):
        """A helper function to make type vars used in attribute definitions concrete when creating
        constraints for new attributes or operations.
        """
        return self


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
            idx = IntegerAttr.from_int_and_width(idx, 64)

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
            idx = IntegerAttr.from_int_and_width(idx, 64)

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


@irdl_op_definition
class EncodeOp(IRDLOperation):
    """Encode a logical codeblock to the specified logical state."""

    name = "qecl.encode"

    in_codeblock = operand_def(LogicalCodeblockConstraint())

    init_state = prop_def(LogicalCodeblockInitStateAttr)

    out_codeblock = result_def(LogicalCodeblockConstraint())

    assembly_format = """
            `[` $init_state `]` $in_codeblock attr-dict `:` type($in_codeblock)
        """

    def __init__(
        self,
        in_codeblock: LogicalCodeBlockSSAValue | Operation,
        init_state: str | LogicalCodeblockInitStateAttr,
    ):
        operands = (in_codeblock,)

        init_state_attr = (
            init_state
            if isinstance(init_state, LogicalCodeblockInitStateAttr)
            else LogicalCodeblockInitStateAttr(init_state)
        )
        properties = {"init_state": init_state_attr}

        in_codeblock_type = get_logical_codeblock_type(in_codeblock)

        super().__init__(
            operands=operands, result_types=(in_codeblock_type,), properties=properties
        )


QecLogical = Dialect(
    "qecl",
    [
        AllocOp,
        DeallocOp,
        ExtractCodeblockOp,
        InsertCodeblockOp,
        EncodeOp,
    ],
    [
        LogicalCodeblockInitStateAttr,
        LogicalCodeblockType,
        LogicalHyperRegisterType,
    ],
)
