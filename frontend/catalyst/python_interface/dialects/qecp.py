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

from xdsl.dialects.builtin import (
    I64,
    ContainerType,
    Float64Type,
    IndexType,
    IntegerAttr,
    IntegerType,
    TensorType,
    UnitAttr,
    i1,
)
from xdsl.ir import (
    Attribute,
    AttributeCovT,
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
    BaseAttr,
    IRDLOperation,
    TypeAttributeInvT,
    VarConstraint,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    opt_result_def,
    result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer

from catalyst.python_interface.xdsl_extras import MemRefConstraint, TensorConstraint


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


@irdl_attr_definition
class TannerGraphType(ParametrizedAttribute, TypeAttribute, ContainerType[AttributeCovT]):
    """A Tanner graph represented by its adjacency matrix in CSC form"""

    name = "qecp.tanner_graph"

    row_idx_size: IntegerAttr[I64]
    col_ptr_size: IntegerAttr[I64]
    element_type: AttributeCovT

    def __init__(
        self,
        row_idx_size: int | IntegerAttr[I64],
        col_ptr_size: int | IntegerAttr[I64],
        element_type: AttributeCovT,
    ):
        row_idx_size_attr = (
            IntegerAttr(row_idx_size, 64) if isinstance(row_idx_size, int) else row_idx_size
        )
        col_ptr_size_attr = (
            IntegerAttr(col_ptr_size, 64) if isinstance(col_ptr_size, int) else col_ptr_size
        )
        super().__init__(row_idx_size_attr, col_ptr_size_attr, element_type)

    def get_element_type(self) -> AttributeCovT:
        """Return the element type of the Tanner graph's adjacency matrix."""
        return self.element_type

    def print_parameters(self, printer: Printer) -> None:
        """Print the attribute parameters."""
        with printer.in_angle_brackets():
            printer.print_int(self.row_idx_size.value.data)
            printer.print_string(", ")
            printer.print_int(self.col_ptr_size.value.data)
            printer.print_string(", ")
            printer.print_attribute(self.element_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """Parse the attribute parameters."""
        with parser.in_angle_brackets():
            row_idx_size = parser.parse_integer()
            parser.parse_characters(",")
            col_idx_size = parser.parse_integer()
            parser.parse_characters(",")
            element_type = parser.parse_attribute()

        return [IntegerAttr(row_idx_size, 64), IntegerAttr(col_idx_size, 64), element_type]


QecPhysicalQubitSSAValue: TypeAlias = SSAValue[QecPhysicalQubitType]
PhysicalCodeBlockSSAValue: TypeAlias = SSAValue[PhysicalCodeblockType]
PhysicalHyperRegisterSSAValue: TypeAlias = SSAValue[PhysicalHyperRegisterType]
TannerGraphSSAValue: TypeAlias = SSAValue[TannerGraphType]


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
            idx = IntegerAttr(idx, IndexType())

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

    T: ClassVar = VarConstraint("T", base(PhysicalHyperRegisterType))

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
            idx = IntegerAttr(idx, IndexType())

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
            idx = IntegerAttr(idx, IndexType())

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

    T: ClassVar = VarConstraint("T", base(PhysicalCodeblockType))

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
            idx = IntegerAttr(idx, IndexType())

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


class SingleQubitPhysicalGateOp(IRDLOperation):
    """Base class for single-qubit physical gate operations.

    An operation that inherits from this class represents a physical gate operation applied to a QEC
    physical qubit. For example,

    ```mlir
    %1 = qecp.hadamard %0 : !qecp.qubit<data>
    ```

    represents a physical Hadamard operation applied to the physical data qubit `%0`. Single-qubit
    physical gate operations can be applied to both data and auxiliary qubits.

    Adjoint operations are supported by adding the `adj` unit attribute. For example, a physical S†
    gate operation is represented as follows:

    ```mlir
    %1 = qecp.s %0 adj : !qecp.qubit<data>
    """

    T: ClassVar = VarConstraint("T", base(QecPhysicalQubitType))

    assembly_format = """
            $in_qubit (`adj` $adjoint^)? attr-dict `:` type($out_qubit)
        """

    in_qubit = operand_def(T)

    adjoint = opt_prop_def(UnitAttr)

    out_qubit = result_def(T)

    def __init__(
        self, in_qubit: QecPhysicalQubitSSAValue | Operation, adjoint: UnitAttr | bool = False
    ):
        in_qubit_type = get_physical_qubit_type(in_qubit)

        properties = {}
        if adjoint:
            properties["adjoint"] = UnitAttr()

        super().__init__(operands=(in_qubit,), result_types=(in_qubit_type,), properties=properties)


@irdl_op_definition
class IdentityOp(SingleQubitPhysicalGateOp):
    """A physical Identity gate operation."""

    name = "qecp.identity"

    def __init__(self, in_qubit: QecPhysicalQubitSSAValue | Operation):
        super().__init__(in_qubit)


@irdl_op_definition
class PauliXOp(SingleQubitPhysicalGateOp):
    """A physical Pauli X gate operation."""

    name = "qecp.x"

    def __init__(self, in_qubit: QecPhysicalQubitSSAValue | Operation):
        super().__init__(in_qubit)


@irdl_op_definition
class PauliYOp(SingleQubitPhysicalGateOp):
    """A physical Pauli Y gate operation."""

    name = "qecp.y"

    def __init__(self, in_qubit: QecPhysicalQubitSSAValue | Operation):
        super().__init__(in_qubit)


@irdl_op_definition
class PauliZOp(SingleQubitPhysicalGateOp):
    """A physical Pauli Z gate operation."""

    name = "qecp.z"

    def __init__(self, in_qubit: QecPhysicalQubitSSAValue | Operation):
        super().__init__(in_qubit)


@irdl_op_definition
class HadamardOp(SingleQubitPhysicalGateOp):
    """A physical Hadamard gate operation."""

    name = "qecp.hadamard"

    def __init__(self, in_qubit: QecPhysicalQubitSSAValue | Operation):
        super().__init__(in_qubit)


@irdl_op_definition
class SOp(SingleQubitPhysicalGateOp):
    """A physical S (π/2 phase) gate operation."""

    name = "qecp.s"


@irdl_op_definition
class CnotOp(IRDLOperation):
    """A physical CNOT gate operation."""

    T_CTRL: ClassVar = VarConstraint("T_CTRL", base(QecPhysicalQubitType))
    T_TRGT: ClassVar = VarConstraint("T_TRGT", base(QecPhysicalQubitType))

    name = "qecp.cnot"

    assembly_format = """
            $in_ctrl_qubit `,` $in_trgt_qubit attr-dict `:` type($out_ctrl_qubit) `,` type($out_trgt_qubit)
        """

    in_ctrl_qubit = operand_def(T_CTRL)

    in_trgt_qubit = operand_def(T_TRGT)

    out_ctrl_qubit = result_def(T_CTRL)

    out_trgt_qubit = result_def(T_TRGT)

    def __init__(
        self,
        in_ctrl_qubit: QecPhysicalQubitSSAValue | Operation,
        in_trgt_qubit: QecPhysicalQubitSSAValue | Operation,
    ):
        in_ctrl_qubit_type = get_physical_qubit_type(in_ctrl_qubit)
        in_trgt_qubit_type = get_physical_qubit_type(in_trgt_qubit)

        super().__init__(
            operands=(in_ctrl_qubit, in_trgt_qubit),
            result_types=(in_ctrl_qubit_type, in_trgt_qubit_type),
        )


@irdl_op_definition
class RotOp(IRDLOperation):
    """A physical Rot gate operation.

    ```mlir
    %1 = qecp.rot (%phi, %theta, %omega) %0 : !qecp.qubit<data>
    ```
    NOTE: This operation is for physical noise injection only.
    """

    T: ClassVar = VarConstraint("T", base(QecPhysicalQubitType))

    name = "qecp.rot"

    assembly_format = """
           `(` $phi `,` $theta `,` $omega `)` $in_qubit attr-dict `:` type($in_qubit)
        """

    phi = operand_def(Float64Type())

    theta = operand_def(Float64Type())

    omega = operand_def(Float64Type())

    in_qubit = operand_def(T)

    out_qubit = result_def(T)

    def __init__(
        self,
        phi: SSAValue[Float64Type],
        theta: SSAValue[Float64Type],
        omega: SSAValue[Float64Type],
        in_qubit: QecPhysicalQubitSSAValue | Operation,
    ):
        in_qubit_type = get_physical_qubit_type(in_qubit)

        super().__init__(
            operands=(
                phi,
                theta,
                omega,
                in_qubit,
            ),
            result_types=(in_qubit_type,),
        )


@irdl_op_definition
class MeasureOp(IRDLOperation):
    """A physical single-qubit projective measurement in the computational basis."""

    T: ClassVar = VarConstraint("T", base(QecPhysicalQubitType))

    name = "qecp.measure"

    assembly_format = """
            $in_qubit attr-dict `:` type($mres) `,` type($in_qubit)
        """

    in_qubit = operand_def(T)

    mres = result_def(i1)

    out_qubit = result_def(T)

    def __init__(self, in_qubit: QecPhysicalQubitSSAValue | Operation):
        in_qubit_type = get_physical_qubit_type(in_qubit)
        super().__init__(operands=(in_qubit,), result_types=(i1, in_qubit_type))


@irdl_op_definition
class AssembleTannerGraphOp(IRDLOperation):
    """Assemble a Tanner graph in CSC form from the given input arrays."""

    name = "qecp.assemble_tanner"

    assembly_format = """
            $row_idx `,` $col_ptr attr-dict `:` type($row_idx) `,` type($col_ptr) `->` type($tanner_graph)
        """

    row_idx = operand_def(
        TensorConstraint(element_type=BaseAttr(IntegerType), rank=1)
        | (MemRefConstraint(element_type=BaseAttr(IntegerType), rank=1))
    )

    col_ptr = operand_def(
        TensorConstraint(element_type=BaseAttr(IntegerType), rank=1)
        | (MemRefConstraint(element_type=BaseAttr(IntegerType), rank=1))
    )

    tanner_graph = result_def(TannerGraphType)

    def __init__(
        self,
        row_idx: SSAValue | Operation,
        col_ptr: SSAValue | Operation,
        tanner_graph_type: TannerGraphType,
    ):
        operands = (row_idx, col_ptr)
        super().__init__(operands=operands, result_types=(tanner_graph_type,))


@irdl_op_definition
class DecodeEsmCssOp(IRDLOperation):
    """
    Decode an ESM for a CSS code and return the index (indices) in the codeblock where the error(s)
    occurred.

    .. note::

        The ``err_idx_in`` field is not supported in the Python interface to Catalyst as it is
        needed only after bufferization. It is included in the op definition here for completeness
        and for compatibility with the MLIR op definition.
    """

    name = "qecp.decode_esm_css"

    assembly_format = """
            `(` $tanner_graph `:` type($tanner_graph) `)` $esm
            ( `in` `(` $err_idx_in^ `:` type($err_idx_in) `)` )?
            attr-dict `:` type($esm) ( `->` type($err_idx)^ )?
        """

    esm = operand_def(
        TensorConstraint(element_type=IntegerType(1), rank=1)
        | (MemRefConstraint(element_type=IntegerType(1), rank=1))
    )

    tanner_graph = operand_def(TannerGraphType)

    err_idx_in = opt_operand_def(MemRefConstraint(element_type=IndexType(), rank=1))

    err_idx = opt_result_def(TensorConstraint(element_type=IndexType(), rank=1))

    def __init__(
        self,
        tanner_graph: TannerGraphSSAValue | Operation,
        esm: SSAValue[TensorType] | Operation,
        err_idx_type: TensorType,
    ):
        operands = (tanner_graph, esm, None)
        super().__init__(operands=operands, result_types=(err_idx_type,))


@irdl_op_definition
class DecodePhysicalMeasurementOp(IRDLOperation):
    """
    Decode physical measurement results and return the corresponding logical measurement.

    This operation decodes the results of a transversal measurement operation acting on a
    physical codeblock and returns the corresponding logical measurement result(s) in the
    computational basis. The logical measurement results are returned as a one-dimensional
    tensor with shape=(k,), where k is the number of QEC logical qubits encoded by the physical
    codeblock.

    .. note::

        The ``logical_measurements_in`` field is not supported in the Python interface to Catalyst
        as it is needed only after bufferization. It is included in the op definition here for
        completeness and for compatibility with the MLIR op definition.
    """

    name = "qecp.decode_physical_meas"

    assembly_format = """
            $physical_measurements
            ( `in` `(` $logical_measurements_in^ `:` type($logical_measurements_in) `)` )?
            attr-dict `:` type($physical_measurements) ( `->` type($logical_measurements)^ )?
        """

    physical_measurements = operand_def(
        TensorConstraint(element_type=IntegerType(1), rank=1)
        | (MemRefConstraint(element_type=IntegerType(1), rank=1))
    )

    logical_measurements_in = opt_operand_def(MemRefConstraint(element_type=IntegerType(1), rank=1))

    logical_measurements = opt_result_def(TensorConstraint(element_type=IntegerType(1), rank=1))

    def __init__(
        self,
        physical_measurements: SSAValue[TensorType] | Operation,
        logical_measurements_type: TensorType,
    ):
        super().__init__(
            operands=(physical_measurements, None), result_types=(logical_measurements_type,)
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
        IdentityOp,
        PauliXOp,
        PauliYOp,
        PauliZOp,
        HadamardOp,
        RotOp,
        SOp,
        CnotOp,
        MeasureOp,
        AssembleTannerGraphOp,
        DecodeEsmCssOp,
        DecodePhysicalMeasurementOp,
    ],
    [
        QecPhysicalQubitRoleAttr,
        QecPhysicalQubitType,
        PhysicalCodeblockType,
        PhysicalHyperRegisterType,
        TannerGraphType,
    ],
)
