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
This file contains the definition of operations that represent observables
in the Quantum dialect.
"""
from xdsl.dialects.builtin import ComplexType, Float64Type
from xdsl.ir import Operation
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    prop_def,
    result_def,
    var_operand_def,
)

from catalyst.python_interface.xdsl_extras import MemRefConstraint, TensorConstraint

from ..attributes import (
    NamedObservableAttr,
    ObservableType,
    QubitSSAValue,
    QubitTypeConstraint,
    QuregTypeConstraint,
)

##############################################
################ Base classes ################
##############################################


class ObservableOp(IRDLOperation):
    """Base class for operations representing observables."""

    obs = result_def(ObservableType)


#############################################
################ Observables ################
#############################################


@irdl_op_definition
class ComputationalBasisOp(ObservableOp):
    """Define a pseudo-obeservable of the computational basis for use in measurements"""

    name = "quantum.compbasis"

    assembly_format = """
        (`qubits` $qubits^)? (`qreg` $qreg^)? attr-dict `:` type(results)
    """

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    qubits = var_operand_def(QubitTypeConstraint())

    qreg = opt_operand_def(QuregTypeConstraint())


@irdl_op_definition
class HamiltonianOp(ObservableOp):
    """Define a Hamiltonian observable for use in measurements"""

    name = "quantum.hamiltonian"

    assembly_format = """
        `(` $coeffs `:` type($coeffs) `)` $terms attr-dict `:` type(results)
    """

    coeffs = operand_def(
        TensorConstraint(element_type=Float64Type(), rank=1)
        | (MemRefConstraint(element_type=Float64Type(), rank=1))
    )

    terms = var_operand_def(ObservableType)


@irdl_op_definition
class HermitianOp(ObservableOp):
    """Define a Hermitian observable for use in measurements"""

    name = "quantum.hermitian"

    assembly_format = """
        `(` $matrix `:` type($matrix) `)` $qubits attr-dict `:` type(results)
    """

    matrix = operand_def(
        TensorConstraint(element_type=ComplexType(Float64Type()), rank=2)
        | MemRefConstraint(element_type=ComplexType(Float64Type()), rank=2)
    )

    qubits = var_operand_def(QubitTypeConstraint())


@irdl_op_definition
class NamedObsOp(ObservableOp):
    """Define a Named observable for use in measurements"""

    name = "quantum.namedobs"

    assembly_format = """
        $qubit `[` $type `]` attr-dict  `:` type(results)
    """

    qubit = operand_def(QubitTypeConstraint())

    type = prop_def(NamedObservableAttr)

    def __init__(self, qubit: QubitSSAValue | Operation, obs_type: NamedObservableAttr):
        super().__init__(
            operands=(qubit,), properties={"type": obs_type}, result_types=(ObservableType(),)
        )


@irdl_op_definition
class TensorOp(ObservableOp):
    """Define a tensor product of observables for use in measurements"""

    name = "quantum.tensor"

    assembly_format = "$terms attr-dict `:` type(results)"

    terms = var_operand_def(ObservableType)
