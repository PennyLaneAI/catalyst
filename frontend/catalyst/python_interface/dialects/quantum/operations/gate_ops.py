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
This file contains the definition of operations representing gates in the
Quantum dialect.
"""
from collections.abc import Sequence
from typing import ClassVar

from xdsl.dialects.builtin import (
    ArrayAttr,
    ComplexType,
    Float64Type,
    IntegerType,
    MemRefType,
    StringAttr,
    TensorType,
    UnitAttr,
    i1,
)
from xdsl.ir import Operation, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    AttrSizedResultSegments,
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import NoMemoryEffect
from xdsl.utils.hints import isa

from catalyst.python_interface.xdsl_extras import AllTypesMatch, MemRefConstraint, TensorConstraint

from ..attributes import PauliWord, QubitSSAValue, QubitTypeConstraint

##############################################
################ Base classes ################
##############################################


class GateOp(IRDLOperation):
    """Base class for operations with quantum gate-like semantics."""


class UnitaryGateOp(GateOp):
    """Base class for operations representing unitary gates."""


###########################################
############## Unitary Gates ##############
###########################################


@irdl_op_definition
class CustomOp(UnitaryGateOp):
    """A generic quantum gate on n qubits with m floating point parameters."""

    name = "quantum.custom"

    assembly_format = """
        $gate_name `(` $params `)` $in_qubits
        (`adj` $adjoint^)?
        attr-dict
        ( `ctrls` `(` $in_ctrl_qubits^ `)` )?
        ( `ctrlvals` `(` $in_ctrl_values^ `)` )?
        `:` type($out_qubits) (`ctrls` type($out_ctrl_qubits)^ )?
    """

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        AttrSizedResultSegments(as_property=True),
    )

    params = var_operand_def(Float64Type())

    in_qubits = var_operand_def(QubitTypeConstraint())

    gate_name = prop_def(StringAttr)

    adjoint = opt_prop_def(UnitAttr)

    in_ctrl_qubits = var_operand_def(QubitTypeConstraint())

    in_ctrl_values = var_operand_def(i1)

    out_qubits = var_result_def(QubitTypeConstraint())

    out_ctrl_qubits = var_result_def(QubitTypeConstraint())

    traits = traits_def(
        NoMemoryEffect(),
        AllTypesMatch(
            ("in_qubits", "out_qubits"),
            "Qubit ins and outs must have the same size and types",
        ),
        AllTypesMatch(
            ("in_ctrl_qubits", "out_ctrl_qubits"),
            "Control qubit ins and outs must have the same size and types",
        ),
    )

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        gate_name: str | StringAttr,
        params: SSAValue[Float64Type] | Sequence[SSAValue[Float64Type]] | None = None,
        in_qubits: QubitSSAValue | Operation | Sequence[QubitSSAValue | Operation],
        in_ctrl_qubits: (
            QubitSSAValue | Operation | Sequence[QubitSSAValue | Operation] | None
        ) = None,
        in_ctrl_values: (
            SSAValue[IntegerType]
            | Operation
            | Sequence[SSAValue[IntegerType]]
            | Sequence[Operation]
            | None
        ) = None,
        adjoint: UnitAttr | bool = False,
    ):
        params = () if params is None else params
        in_ctrl_qubits = () if in_ctrl_qubits is None else in_ctrl_qubits
        in_ctrl_values = () if in_ctrl_values is None else in_ctrl_values

        if not isinstance(params, Sequence):
            params = (params,)
        if not isinstance(in_qubits, Sequence):
            in_qubits = (in_qubits,)
        if not isinstance(in_ctrl_qubits, Sequence):
            in_ctrl_qubits = (in_ctrl_qubits,)
        if not isinstance(in_ctrl_values, Sequence):
            in_ctrl_values = (in_ctrl_values,)

        if isinstance(gate_name, str):
            gate_name = StringAttr(data=gate_name)

        out_qubits = tuple(q.type for q in in_qubits)
        out_ctrl_qubits = tuple(q.type for q in in_ctrl_qubits)
        properties = {"gate_name": gate_name}
        if adjoint:
            properties["adjoint"] = UnitAttr()

        super().__init__(
            operands=(params, in_qubits, in_ctrl_qubits, in_ctrl_values),
            result_types=(out_qubits, out_ctrl_qubits),
            properties=properties,
        )


@irdl_op_definition
class GlobalPhaseOp(UnitaryGateOp):
    """Global Phase."""

    name = "quantum.gphase"

    assembly_format = """
        `(` $params `)`
        attr-dict
        ( `ctrls` `(` $in_ctrl_qubits^ `)` )?
        ( `ctrlvals` `(` $in_ctrl_values^ `)` )?
        `:` (`ctrls` type($out_ctrl_qubits)^ )?
    """

    irdl_options = (AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict())

    params = operand_def(Float64Type())

    adjoint = opt_prop_def(UnitAttr)

    in_ctrl_qubits = var_operand_def(QubitTypeConstraint())

    in_ctrl_values = var_operand_def(i1)

    out_ctrl_qubits = var_result_def(QubitTypeConstraint())

    traits = traits_def(
        AllTypesMatch(
            ("in_ctrl_qubits", "out_ctrl_qubits"),
            "Control qubit ins and outs must have the same size and types",
        )
    )

    def __init__(
        self,
        *,
        params: SSAValue[Float64Type],
        in_ctrl_qubits: (
            QubitSSAValue | Operation | Sequence[QubitSSAValue | Operation] | None
        ) = None,
        in_ctrl_values: (
            SSAValue[IntegerType]
            | Operation
            | Sequence[SSAValue[IntegerType]]
            | Sequence[Operation]
            | None
        ) = None,
    ):
        in_ctrl_qubits = () if in_ctrl_qubits is None else in_ctrl_qubits
        in_ctrl_values = () if in_ctrl_values is None else in_ctrl_values

        if not isinstance(in_ctrl_qubits, Sequence):
            in_ctrl_qubits = (in_ctrl_qubits,)
        if not isinstance(in_ctrl_values, Sequence):
            in_ctrl_values = (in_ctrl_values,)

        out_ctrl_qubits = tuple(q.type for q in in_ctrl_qubits)

        super().__init__(
            operands=(params, in_ctrl_qubits, in_ctrl_values),
            result_types=(out_ctrl_qubits,),
        )


@irdl_op_definition
class MultiRZOp(UnitaryGateOp):
    """Apply an arbitrary multi Z rotation"""

    name = "quantum.multirz"

    assembly_format = """
        `(` $theta `)` $in_qubits
        (`adj` $adjoint^)?
        attr-dict
        ( `ctrls` `(` $in_ctrl_qubits^ `)` )?
        ( `ctrlvals` `(` $in_ctrl_values^ `)` )?
        `:` type($out_qubits) (`ctrls` type($out_ctrl_qubits)^ )?
    """

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        AttrSizedResultSegments(as_property=True),
    )

    theta = operand_def(Float64Type())

    in_qubits = var_operand_def(QubitTypeConstraint())

    adjoint = opt_prop_def(UnitAttr)

    in_ctrl_qubits = var_operand_def(QubitTypeConstraint())

    in_ctrl_values = var_operand_def(i1)

    out_qubits = var_result_def(QubitTypeConstraint())

    out_ctrl_qubits = var_result_def(QubitTypeConstraint())

    traits = traits_def(
        NoMemoryEffect(),
        AllTypesMatch(
            ("in_qubits", "out_qubits"),
            "Qubit ins and outs must have the same size and types",
        ),
        AllTypesMatch(
            ("in_ctrl_qubits", "out_ctrl_qubits"),
            "Control qubit ins and outs must have the same size and types",
        ),
    )

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        theta: SSAValue[Float64Type],
        in_qubits: QubitSSAValue | Operation | Sequence[QubitSSAValue | Operation],
        in_ctrl_qubits: (
            QubitSSAValue | Operation | Sequence[QubitSSAValue | Operation] | None
        ) = None,
        in_ctrl_values: (
            SSAValue[IntegerType]
            | Operation
            | Sequence[SSAValue[IntegerType]]
            | Sequence[Operation]
            | None
        ) = None,
        adjoint: UnitAttr | bool = False,
    ):
        in_ctrl_qubits = () if in_ctrl_qubits is None else in_ctrl_qubits
        in_ctrl_values = () if in_ctrl_values is None else in_ctrl_values

        if not isinstance(in_qubits, Sequence):
            in_qubits = (in_qubits,)
        if not isinstance(in_ctrl_qubits, Sequence):
            in_ctrl_qubits = (in_ctrl_qubits,)
        if not isinstance(in_ctrl_values, Sequence):
            in_ctrl_values = (in_ctrl_values,)

        out_qubits = tuple(q.type for q in in_qubits)
        out_ctrl_qubits = tuple(q.type for q in in_ctrl_qubits)
        properties = {"adjoint": UnitAttr()} if adjoint else {}

        super().__init__(
            operands=(theta, in_qubits, in_ctrl_qubits, in_ctrl_values),
            result_types=(out_qubits, out_ctrl_qubits),
            properties=properties,
        )


@irdl_op_definition
class PauliRotOp(UnitaryGateOp):
    """Apply a Pauli Product Rotation."""

    name = "quantum.paulirot"

    assembly_format = """
        $pauli_product `(` $angle `)` $in_qubits
        (`adj` $adjoint^)?
        attr-dict
        ( `ctrls` `(` $in_ctrl_qubits^ `)` )?
        ( `ctrlvals` `(` $in_ctrl_values^ `)` )?
        `:` type($out_qubits) (`ctrls` type($out_ctrl_qubits)^ )?
    """

    VALID_PAULIS: ClassVar[tuple[str]] = ("X", "Y", "Z", "I")

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        AttrSizedResultSegments(as_property=True),
    )

    angle = operand_def(Float64Type())

    pauli_product = prop_def(PauliWord)

    in_qubits = var_operand_def(QubitTypeConstraint())

    adjoint = opt_prop_def(UnitAttr)

    in_ctrl_qubits = var_operand_def(QubitTypeConstraint())

    in_ctrl_values = var_operand_def(i1)

    out_qubits = var_result_def(QubitTypeConstraint())

    out_ctrl_qubits = var_result_def(QubitTypeConstraint())

    traits = traits_def(
        NoMemoryEffect(),
        AllTypesMatch(
            ("in_qubits", "out_qubits"),
            "Qubit ins and outs must have the same size and types",
        ),
        AllTypesMatch(
            ("in_ctrl_qubits", "out_ctrl_qubits"),
            "Control qubit ins and outs must have the same size and types",
        ),
    )

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        angle: SSAValue[Float64Type],
        pauli_product: PauliWord | str | Sequence[str],
        in_qubits: QubitSSAValue | Operation | Sequence[QubitSSAValue | Operation],
        in_ctrl_qubits: (
            QubitSSAValue | Operation | Sequence[QubitSSAValue | Operation] | None
        ) = None,
        in_ctrl_values: (
            SSAValue[IntegerType]
            | Operation
            | Sequence[SSAValue[IntegerType]]
            | Sequence[Operation]
            | None
        ) = None,
        adjoint: UnitAttr | bool = False,
    ):
        in_ctrl_qubits = () if in_ctrl_qubits is None else in_ctrl_qubits
        in_ctrl_values = () if in_ctrl_values is None else in_ctrl_values

        if not isinstance(in_qubits, Sequence):
            in_qubits = (in_qubits,)
        if not isinstance(in_ctrl_qubits, Sequence):
            in_ctrl_qubits = (in_ctrl_qubits,)
        if not isinstance(in_ctrl_values, Sequence):
            in_ctrl_values = (in_ctrl_values,)

        out_qubits = tuple(q.type for q in in_qubits)
        out_ctrl_qubits = tuple(q.type for q in in_ctrl_qubits)

        if not isa(pauli_product, PauliWord):
            pauli_product = ArrayAttr([StringAttr(c) for c in pauli_product])

        properties = {"adjoint": UnitAttr()} if adjoint else {}
        properties["pauli_product"] = pauli_product

        super().__init__(
            operands=(angle, in_qubits, in_ctrl_qubits, in_ctrl_values),
            result_types=(out_qubits, out_ctrl_qubits),
            properties=properties,
        )

    def verify_(self):
        """Verify that the definition of the operation is correct."""
        if len(self.pauli_product) != len(self.in_qubits):
            raise ValueError("The length of the Pauli word must match the number of qubits")

        for p in self.pauli_product.data:
            if p.data not in self.VALID_PAULIS:
                raise ValueError(f"{p} is not a valid Pauli operator.")


@irdl_op_definition
class PCPhaseOp(UnitaryGateOp):
    """Apply a projector-controlled phase gate"""

    name = "quantum.pcphase"

    assembly_format = """
        `(` $theta `,` $dim `)` $in_qubits
        (`adj` $adjoint^)?
        attr-dict
        ( `ctrls` `(` $in_ctrl_qubits^ `)` )?
        ( `ctrlvals` `(` $in_ctrl_values^ `)` )?
        `:` type($out_qubits) (`ctrls` type($out_ctrl_qubits)^ )?
    """

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        AttrSizedResultSegments(as_property=True),
    )

    theta = operand_def(Float64Type())

    dim = operand_def(Float64Type())

    in_qubits = var_operand_def(QubitTypeConstraint())

    adjoint = opt_prop_def(UnitAttr)

    in_ctrl_qubits = var_operand_def(QubitTypeConstraint())

    in_ctrl_values = var_operand_def(i1)

    out_qubits = var_result_def(QubitTypeConstraint())

    out_ctrl_qubits = var_result_def(QubitTypeConstraint())

    traits = traits_def(
        NoMemoryEffect(),
        AllTypesMatch(
            ("in_qubits", "out_qubits"),
            "Qubit ins and outs must have the same size and types",
        ),
        AllTypesMatch(
            ("in_ctrl_qubits", "out_ctrl_qubits"),
            "Control qubit ins and outs must have the same size and types",
        ),
    )

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        theta: SSAValue[Float64Type],
        dim: SSAValue[Float64Type],
        in_qubits: QubitSSAValue | Operation | Sequence[QubitSSAValue | Operation],
        in_ctrl_qubits: (
            QubitSSAValue | Operation | Sequence[QubitSSAValue | Operation] | None
        ) = None,
        in_ctrl_values: (
            SSAValue[IntegerType]
            | Operation
            | Sequence[SSAValue[IntegerType]]
            | Sequence[Operation]
            | None
        ) = None,
        adjoint: UnitAttr | bool = False,
    ):
        in_ctrl_qubits = () if in_ctrl_qubits is None else in_ctrl_qubits
        in_ctrl_values = () if in_ctrl_values is None else in_ctrl_values

        if not isinstance(in_qubits, Sequence):
            in_qubits = (in_qubits,)
        if not isinstance(in_ctrl_qubits, Sequence):
            in_ctrl_qubits = (in_ctrl_qubits,)
        if not isinstance(in_ctrl_values, Sequence):
            in_ctrl_values = (in_ctrl_values,)

        out_qubits = tuple(q.type for q in in_qubits)
        out_ctrl_qubits = tuple(q.type for q in in_ctrl_qubits)
        properties = {"adjoint": UnitAttr()} if adjoint else {}

        super().__init__(
            operands=(theta, dim, in_qubits, in_ctrl_qubits, in_ctrl_values),
            result_types=(out_qubits, out_ctrl_qubits),
            properties=properties,
        )


@irdl_op_definition
class QubitUnitaryOp(UnitaryGateOp):
    """Apply an arbitrary fixed unitary matrix"""

    name = "quantum.unitary"

    assembly_format = """
        `(` $matrix `:` type($matrix) `)` $in_qubits
        (`adj` $adjoint^)?
        attr-dict
        ( `ctrls` `(` $in_ctrl_qubits^ `)` )?
        ( `ctrlvals` `(` $in_ctrl_values^ `)` )?
        `:` type($out_qubits) (`ctrls` type($out_ctrl_qubits)^ )?
    """

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        AttrSizedResultSegments(as_property=True),
    )

    matrix = operand_def(
        (TensorConstraint(element_type=ComplexType(Float64Type()), rank=2))
        | (MemRefConstraint(element_type=ComplexType(Float64Type()), rank=2))
    )

    in_qubits = var_operand_def(QubitTypeConstraint())

    adjoint = opt_prop_def(UnitAttr)

    in_ctrl_qubits = var_operand_def(QubitTypeConstraint())

    in_ctrl_values = var_operand_def(i1)

    out_qubits = var_result_def(QubitTypeConstraint())

    out_ctrl_qubits = var_result_def(QubitTypeConstraint())

    traits = traits_def(
        NoMemoryEffect(),
        AllTypesMatch(
            ("in_qubits", "out_qubits"),
            "Qubit ins and outs must have the same size and types",
        ),
        AllTypesMatch(
            ("in_ctrl_qubits", "out_ctrl_qubits"),
            "Control qubit ins and outs must have the same size and types",
        ),
    )

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        matrix: SSAValue[TensorType | MemRefType],
        in_qubits: QubitSSAValue | Operation | Sequence[QubitSSAValue | Operation],
        in_ctrl_qubits: (
            QubitSSAValue | Operation | Sequence[QubitSSAValue | Operation] | None
        ) = None,
        in_ctrl_values: (
            SSAValue[IntegerType]
            | Operation
            | Sequence[SSAValue[IntegerType]]
            | Sequence[Operation]
            | None
        ) = None,
        adjoint: UnitAttr | bool = False,
    ):
        in_ctrl_qubits = () if in_ctrl_qubits is None else in_ctrl_qubits
        in_ctrl_values = () if in_ctrl_values is None else in_ctrl_values

        if not isinstance(in_qubits, Sequence):
            in_qubits = (in_qubits,)
        if not isinstance(in_ctrl_qubits, Sequence):
            in_ctrl_qubits = (in_ctrl_qubits,)
        if not isinstance(in_ctrl_values, Sequence):
            in_ctrl_values = (in_ctrl_values,)

        out_qubits = tuple(q.type for q in in_qubits)
        out_ctrl_qubits = tuple(q.type for q in in_ctrl_qubits)
        properties = {}
        if adjoint:
            properties["adjoint"] = UnitAttr()

        super().__init__(
            operands=(matrix, in_qubits, in_ctrl_qubits, in_ctrl_values),
            result_types=(out_qubits, out_ctrl_qubits),
            properties=properties,
        )


###########################################
############ State preparation ############
###########################################


@irdl_op_definition
class SetBasisStateOp(GateOp):
    """Set basis state."""

    name = "quantum.set_basis_state"

    assembly_format = """
        `(` $basis_state`)` $in_qubits attr-dict `:` functional-type(operands, results)
    """

    basis_state = operand_def(
        (TensorConstraint(element_type=i1, rank=1)) | (MemRefConstraint(element_type=i1, rank=1))
    )

    in_qubits = var_operand_def(QubitTypeConstraint())

    out_qubits = var_result_def(QubitTypeConstraint())

    traits = traits_def(
        AllTypesMatch(
            ("in_qubits", "out_qubits"),
            "Qubit ins and outs must have the same size and types",
        )
    )


@irdl_op_definition
class SetStateOp(GateOp):
    """Set state to a complex vector."""

    name = "quantum.set_state"

    assembly_format = """
        `(` $in_state `)` $in_qubits attr-dict `:` functional-type(operands, results)
    """

    in_state = operand_def(
        (TensorConstraint(element_type=ComplexType(Float64Type()), rank=1))
        | (MemRefConstraint(element_type=ComplexType(Float64Type()), rank=1))
    )

    in_qubits = var_operand_def(QubitTypeConstraint())

    out_qubits = var_result_def(QubitTypeConstraint())

    traits = traits_def(
        AllTypesMatch(
            ("in_qubits", "out_qubits"),
            "Qubit ins and outs must have the same size and types",
        )
    )
