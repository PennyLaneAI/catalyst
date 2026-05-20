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

from xdsl.dialects.builtin import (
    I64,
    IndexType,
    IntegerAttr,
    IntegerType,
    UnitAttr,
    i1,
)
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
    AttrSizedOperandSegments,
    IRDLOperation,
    TypeAttributeInvT,
    VarConstraint,
    base,
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
            properties = {"idx_attr": IntegerAttr(idx.value.data, IndexType())}
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

    T: ClassVar = VarConstraint("T", base(LogicalHyperRegisterType))

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
            properties = {"idx_attr": IntegerAttr(idx.value.data, IndexType())}
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

    T: ClassVar = VarConstraint("T", base(LogicalCodeblockType))

    name = "qecl.encode"

    in_codeblock = operand_def(T)

    init_state = prop_def(LogicalCodeblockInitStateAttr)

    out_codeblock = result_def(T)

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


@irdl_op_definition
class NoiseOp(IRDLOperation):
    """Inject physical noise on elements of a logical codeblock."""

    T: ClassVar = VarConstraint("T", base(LogicalCodeblockType))

    name = "qecl.noise"

    in_codeblock = operand_def(T)

    out_codeblock = result_def(T)

    assembly_format = """
            $in_codeblock attr-dict `:` type($in_codeblock)
        """

    def __init__(
        self,
        in_codeblock: LogicalCodeBlockSSAValue | Operation,
    ):
        operands = (in_codeblock,)

        in_codeblock_type = get_logical_codeblock_type(in_codeblock)

        super().__init__(operands=operands, result_types=(in_codeblock_type,))


@irdl_op_definition
class QecCycleOp(IRDLOperation):
    """Perform a single cycle of a quantum error-correction protocol."""

    T: ClassVar = VarConstraint("T", base(LogicalCodeblockType))

    name = "qecl.qec"

    in_codeblock = operand_def(T)

    out_codeblock = result_def(T)

    assembly_format = """
            $in_codeblock attr-dict `:` type($in_codeblock)
        """

    def __init__(
        self,
        in_codeblock: LogicalCodeBlockSSAValue | Operation,
    ):
        operands = (in_codeblock,)

        in_codeblock_type = get_logical_codeblock_type(in_codeblock)

        super().__init__(operands=operands, result_types=(in_codeblock_type,))


class SingleQubitLogicalGateOp(IRDLOperation):
    """Base class for single-qubit logical gate operations.

    An operation that inherits from this class represents a logical gate operation applied to the
    logical qubit at the provided index in the logical codeblock. For example,

    ```mlir
    %1 = qecl.hadamard %0[ 1] : !qecl.codeblock<3>
    ```

    represents a logical Hadamard operation applied to the logical qubit at index `1` in the
    codeblock `%0`, which encodes k = 3 logical qubits.

    Adjoint operations are supported by adding the `adj` unit attribute. For example, to represent
    a logical S† gate operation:

    ```mlir
    %1 = qecl.s %0[ 1] adj : !qecl.codeblock<3>
    ```
    """

    T: ClassVar = VarConstraint("T", base(LogicalCodeblockType))

    in_codeblock = operand_def(T)

    idx = opt_operand_def(IndexType)

    idx_attr = opt_prop_def(IntegerAttr.constr(type=IndexType, value=AtLeast(0)))

    adjoint = opt_prop_def(UnitAttr)

    out_codeblock = result_def(T)

    assembly_format = """
            $in_codeblock `[` ($idx^):($idx_attr)? `]` (`adj` $adjoint^)? attr-dict `:` type($in_codeblock)
        """

    def __init__(
        self,
        in_codeblock: LogicalCodeBlockSSAValue | Operation,
        idx: int | IntegerAttr | SSAValue[IndexType] | Operation,
        adjoint: UnitAttr | bool = False,
    ):
        properties: dict[str, Attribute | None] = {}

        if isinstance(idx, int):
            idx = IntegerAttr(idx, IndexType())

        if isinstance(idx, IntegerAttr):
            operands = (in_codeblock, None)
            properties["idx_attr"] = IntegerAttr(idx.value.data, IndexType())

        else:
            operands = (in_codeblock, idx)

        if adjoint:
            properties["adjoint"] = UnitAttr()

        in_codeblock_type = get_logical_codeblock_type(in_codeblock)

        super().__init__(
            operands=operands,
            result_types=(in_codeblock_type,),
            properties=properties,
        )


@irdl_op_definition
class IdentityOp(SingleQubitLogicalGateOp):
    """A logical Identity gate operation.

    Example:

    ```mlir
    %1 = qecl.identity %0[ 1] : !qecl.codeblock<3>
    ```
    """

    name = "qecl.identity"

    def __init__(
        self,
        in_codeblock: LogicalCodeBlockSSAValue | Operation,
        idx: int | IntegerAttr | SSAValue[IndexType] | Operation,
    ):
        super().__init__(in_codeblock, idx)


@irdl_op_definition
class PauliXOp(SingleQubitLogicalGateOp):
    """A logical Pauli X gate operation.

    Example:

    ```mlir
    %1 = qecl.x %0[ 1] : !qecl.codeblock<3>
    ```
    """

    name = "qecl.x"

    def __init__(
        self,
        in_codeblock: LogicalCodeBlockSSAValue | Operation,
        idx: int | IntegerAttr | SSAValue[IndexType] | Operation,
    ):
        super().__init__(in_codeblock, idx)


@irdl_op_definition
class PauliYOp(SingleQubitLogicalGateOp):
    """A logical Pauli Y gate operation.

    Example:

    ```mlir
    %1 = qecl.y %0[ 1] : !qecl.codeblock<3>
    ```
    """

    name = "qecl.y"

    def __init__(
        self,
        in_codeblock: LogicalCodeBlockSSAValue | Operation,
        idx: int | IntegerAttr | SSAValue[IndexType] | Operation,
    ):
        super().__init__(in_codeblock, idx)


@irdl_op_definition
class PauliZOp(SingleQubitLogicalGateOp):
    """A logical Pauli Z gate operation.

    Example:

    ```mlir
    %1 = qecl.z %0[ 1] : !qecl.codeblock<3>
    ```
    """

    name = "qecl.z"

    def __init__(
        self,
        in_codeblock: LogicalCodeBlockSSAValue | Operation,
        idx: int | IntegerAttr | SSAValue[IndexType] | Operation,
    ):
        super().__init__(in_codeblock, idx)


@irdl_op_definition
class HadamardOp(SingleQubitLogicalGateOp):
    """A logical Hadamard gate operation.

    Example:

    ```mlir
    %1 = qecl.hadamard %0[ 1] : !qecl.codeblock<3>
    ```
    """

    name = "qecl.hadamard"

    def __init__(
        self,
        in_codeblock: LogicalCodeBlockSSAValue | Operation,
        idx: int | IntegerAttr | SSAValue[IndexType] | Operation,
    ):
        super().__init__(in_codeblock, idx)


@irdl_op_definition
class SOp(SingleQubitLogicalGateOp):
    """A logical S (π/2 phase) gate operation.

    Example:

    ```mlir
    %1 = qecl.s %0[ 1] : !qecl.codeblock<3>
    %2 = qecl.s %1[ 1] adj : !qecl.codeblock<3>
    ```
    """

    name = "qecl.s"


@irdl_op_definition
class CnotOp(IRDLOperation):
    """A logical inter-codeblock CNOT gate operation.

    This operation represents a logical inter-codeblock CNOT gate operation applied to the
    logical qubits at the provided indices in the respective control and target logical
    codeblocks. For example,

    ```mlir
    %2, %3 = qecl.cnot %0[ 1], %1[ 2] : !qecl.codeblock<3>, !qecl.codeblock<3>
    ```

    represents a logical CNOT operation applied to the logical qubit at index `1` in the
    codeblock `%0` (the control qubit) and the logical qubit at index `2` in the codeblock `%1`
    (the target qubit), where both codeblocks encode k = 3 logical qubits.

    Note that this operation cannot represent an intra-codeblock CNOT operation—that is, a CNOT
    operation where the control and target qubits are encoded in the same logical codeblock (for
    k >= 2).
    """

    T_CTRL: ClassVar = VarConstraint("T_CTRL", base(LogicalCodeblockType))
    T_TRGT: ClassVar = VarConstraint("T_TRGT", base(LogicalCodeblockType))

    name = "qecl.cnot"

    assembly_format = """
            $in_ctrl_codeblock `[` ($idx_ctrl^):($idx_ctrl_attr)? `]` `,`
            $in_trgt_codeblock `[` ($idx_trgt^):($idx_trgt_attr)? `]`
            attr-dict `:` type($in_ctrl_codeblock) `,` type($in_trgt_codeblock)
        """

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    in_ctrl_codeblock = operand_def(T_CTRL)

    idx_ctrl = opt_operand_def(IndexType)

    idx_ctrl_attr = opt_prop_def(IntegerAttr.constr(type=IndexType, value=AtLeast(0)))

    in_trgt_codeblock = operand_def(T_TRGT)

    idx_trgt = opt_operand_def(IndexType)

    idx_trgt_attr = opt_prop_def(IntegerAttr.constr(type=IndexType, value=AtLeast(0)))

    out_ctrl_codeblock = result_def(T_CTRL)

    out_trgt_codeblock = result_def(T_TRGT)

    def __init__(
        self,
        in_ctrl_codeblock: LogicalCodeBlockSSAValue | Operation,
        idx_ctrl: int | IntegerAttr | SSAValue[IndexType] | Operation,
        in_trgt_codeblock: LogicalCodeBlockSSAValue | Operation,
        idx_trgt: int | IntegerAttr | SSAValue[IndexType] | Operation,
    ):
        if isinstance(idx_ctrl, int):
            idx_ctrl = IntegerAttr(idx_ctrl, IndexType())
        if isinstance(idx_trgt, int):
            idx_trgt = IntegerAttr(idx_trgt, IndexType())

        if isinstance(idx_ctrl, IntegerAttr) and isinstance(idx_trgt, IntegerAttr):
            operands = (in_ctrl_codeblock, None, in_trgt_codeblock, None)
            properties = {
                "idx_ctrl_attr": IntegerAttr(idx_ctrl.value.data, IndexType()),
                "idx_trgt_attr": IntegerAttr(idx_trgt.value.data, IndexType()),
            }
        elif isinstance(idx_ctrl, IntegerAttr):
            operands = (in_ctrl_codeblock, None, in_trgt_codeblock, idx_trgt)
            properties = {"idx_ctrl_attr": IntegerAttr(idx_ctrl.value.data, IndexType())}
        elif isinstance(idx_trgt, IntegerAttr):
            operands = (in_ctrl_codeblock, idx_ctrl, in_trgt_codeblock, None)
            properties = {"idx_trgt_attr": IntegerAttr(idx_trgt.value.data, IndexType())}
        else:
            operands = (in_ctrl_codeblock, idx_ctrl, in_trgt_codeblock, idx_trgt)
            properties = {}

        in_ctrl_codeblock_type = get_logical_codeblock_type(in_ctrl_codeblock)
        in_trgt_codeblock_type = get_logical_codeblock_type(in_trgt_codeblock)

        super().__init__(
            operands=operands,
            result_types=(in_ctrl_codeblock_type, in_trgt_codeblock_type),
            properties=properties,
        )


@irdl_op_definition
class MeasureOp(IRDLOperation):
    """A logical single-qubit projective measurement in the computational basis.

    This operation represents a logical, projective computational-basis measurement of the
    logical qubit at the provided index in the logical codeblock. For example,

    ```mlir
    %mres, %1 = qecl.measure %0[ 1] : i1, !qecl.codeblock<3>
    ```

    represents a measurement of the logical qubit at index `1` in the codeblock `%0`, which
    encodes k = 3 logical qubits. The result of the measurement is returned as the value
    `%mres`.

    Note that unlike the `quantum.measure` operation, this operation does not currently support
    the `postselect` attribute to select the basis state of the qubit post-measurement.
    """

    T: ClassVar = VarConstraint("T", base(LogicalCodeblockType))

    name = "qecl.measure"

    assembly_format = """
            $in_codeblock `[` ($idx^):($idx_attr)? `]` attr-dict `:` type($mres) `,` type($in_codeblock)
        """

    in_codeblock = operand_def(T)

    idx = opt_operand_def(IndexType)

    idx_attr = opt_prop_def(IntegerAttr.constr(type=IndexType, value=AtLeast(0)))

    mres = result_def(IntegerType(1))

    out_codeblock = result_def(T)

    def __init__(
        self,
        in_codeblock: LogicalCodeBlockSSAValue | Operation,
        idx: int | IntegerAttr | SSAValue[IndexType] | Operation,
    ):
        properties: dict[str, Attribute | None] = {}

        if isinstance(idx, int):
            idx = IntegerAttr(idx, IndexType())

        if isinstance(idx, IntegerAttr):
            operands = (in_codeblock, None)
            properties = {"idx_attr": IntegerAttr(idx.value.data, IndexType())}
        else:
            operands = (in_codeblock, idx)

        in_codeblock_type = get_logical_codeblock_type(in_codeblock)

        super().__init__(
            operands=operands,
            result_types=(i1, in_codeblock_type),
            properties=properties,
        )


QecLogical = Dialect(
    "qecl",
    [
        AllocOp,
        DeallocOp,
        ExtractCodeblockOp,
        InsertCodeblockOp,
        EncodeOp,
        NoiseOp,
        QecCycleOp,
        IdentityOp,
        PauliXOp,
        PauliYOp,
        PauliZOp,
        HadamardOp,
        SOp,
        CnotOp,
        MeasureOp,
    ],
    [
        LogicalCodeblockInitStateAttr,
        LogicalCodeblockType,
        LogicalHyperRegisterType,
    ],
)
