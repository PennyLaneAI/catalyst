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
This file contains the definition of operations that create/destroy qubits
and quantum registers in the Quantum dialect.
"""
from xdsl.dialects.builtin import I64, IntegerAttr, IntegerType, StringAttr, i64
from xdsl.ir import Operation, SSAValue
from xdsl.irdl import (
    AtLeast,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import NoMemoryEffect

from ..attributes import (
    QubitLevel,
    QubitRole,
    QubitSSAValue,
    QubitType,
    QubitTypeConstraint,
    QuregSSAValue,
    QuregType,
    QuregTypeConstraint,
)


@irdl_op_definition
class AllocOp(IRDLOperation):
    """Allocate n qubits into a quantum register."""

    name = "quantum.alloc"

    assembly_format = """
        `(` ($nqubits^):($nqubits_attr)? `)` attr-dict `:` type(results)
    """

    nqubits = opt_operand_def(i64)

    nqubits_attr = opt_prop_def(IntegerAttr.constr(type=I64, value=AtLeast(0)))

    qreg = result_def(QuregTypeConstraint())

    def __init__(self, nqubits):
        if isinstance(nqubits, int):
            nqubits = IntegerAttr.from_int_and_width(nqubits, 64)

        if isinstance(nqubits, IntegerAttr):
            operands = (None,)
            properties = {"nqubits_attr": nqubits}
        else:
            operands = (nqubits,)
            properties = {}

        super().__init__(
            operands=operands,
            properties=properties,
            result_types=(QuregType(level=StringAttr(QubitLevel.Abstract.value)),),
        )


@irdl_op_definition
class AllocQubitOp(IRDLOperation):
    """Allocate a single qubit."""

    name = "quantum.alloc_qb"

    assembly_format = "attr-dict `:` type(results)"

    qubit = result_def(QubitTypeConstraint())

    def __init__(self):
        super().__init__(
            result_types=(
                QubitType(
                    level=StringAttr(QubitLevel.Abstract.value),
                    role=StringAttr(QubitRole.Null.value),
                ),
            )
        )


@irdl_op_definition
class DeallocOp(IRDLOperation):
    """Deallocate a quantum register."""

    name = "quantum.dealloc"

    assembly_format = "$qreg attr-dict `:` type(operands)"

    qreg = operand_def(QuregTypeConstraint())

    def __init__(self, qreg: QuregSSAValue | Operation):
        super().__init__(operands=(qreg,))


@irdl_op_definition
class DeallocQubitOp(IRDLOperation):
    """Deallocate a single qubit."""

    name = "quantum.dealloc_qb"

    assembly_format = "$qubit attr-dict `:` type(operands)"

    qubit = operand_def(QubitTypeConstraint())

    def __init__(self, qubit: QubitSSAValue | Operation):
        super().__init__(operands=(qubit,))


@irdl_op_definition
class ExtractOp(IRDLOperation):
    """Extract a qubit value from a register."""

    name = "quantum.extract"

    assembly_format = """
        $qreg `[` ($idx^):($idx_attr)? `]` attr-dict `:` type($qreg) `->` type(results)
    """

    qreg = operand_def(QuregTypeConstraint())

    idx = opt_operand_def(i64)

    idx_attr = opt_prop_def(IntegerAttr.constr(type=i64, value=AtLeast(0)))

    qubit = result_def(QubitTypeConstraint())

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        qreg: QuregSSAValue | Operation,
        idx: int | SSAValue[IntegerType] | Operation | IntegerAttr,
    ):
        if isinstance(idx, int):
            idx = IntegerAttr.from_int_and_width(idx, 64)

        if isinstance(idx, IntegerAttr):
            operands = (qreg, None)
            properties = {"idx_attr": idx}
        else:
            operands = (qreg, idx)
            properties = {}

        super().__init__(
            operands=operands,
            result_types=(QubitType(level=qreg.type.level, role=StringAttr(QubitRole.Null.value)),),
            properties=properties,
        )


@irdl_op_definition
class InsertOp(IRDLOperation):
    """Update the qubit value of a register."""

    name = "quantum.insert"

    assembly_format = """
        $in_qreg `[` ($idx^):($idx_attr)? `]` `,` $qubit attr-dict `:` type($in_qreg) `,` type($qubit)
    """

    in_qreg = operand_def(QuregTypeConstraint())

    idx = opt_operand_def(i64)

    idx_attr = opt_prop_def(IntegerAttr.constr(type=i64, value=AtLeast(0)))

    qubit = operand_def(QubitTypeConstraint())

    out_qreg = result_def(QuregTypeConstraint())

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        in_qreg: QuregSSAValue | Operation,
        idx: SSAValue[IntegerType] | Operation | int | IntegerAttr,
        qubit: QubitSSAValue | Operation,
    ):
        if isinstance(idx, int):
            idx = IntegerAttr.from_int_and_width(idx, 64)

        if isinstance(idx, IntegerAttr):
            operands = (in_qreg, None, qubit)
            properties = {"idx_attr": idx}
        else:
            operands = (in_qreg, idx, qubit)
            properties = {}

        super().__init__(operands=operands, properties=properties, result_types=(in_qreg.type,))
