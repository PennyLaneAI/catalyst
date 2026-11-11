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
This file contains the definition of operations representing measurements
in the Quantum dialect.
"""

from xdsl.dialects.builtin import (
    I32,
    ComplexType,
    DictionaryAttr,
    Float64Type,
    IntegerAttr,
    StringAttr,
    i1,
    i64,
)
from xdsl.ir import Operation
from xdsl.irdl import (
    AttrSizedOperandSegments,
    AttrSizedResultSegments,
    IntSetConstraint,
    IRDLOperation,
    SameVariadicResultSize,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    opt_result_def,
    prop_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser

from catalyst.python_interface.xdsl_extras import MemRefConstraint, TensorConstraint

from ..attributes import ObservableSSAValue, ObservableType, QubitSSAValue, QubitType, QuregType

##############################################
################ Base classes ################
##############################################


class TerminalMeasurementOp(IRDLOperation):
    """Base class for operations representing terminal measurement processes."""


############################################
######### Mid-circuit measurements #########
############################################


@irdl_op_definition
class MeasurementOp(IRDLOperation):
    """A named mid-circuit measurement with arbitrary classical parameters and results."""

    name = "quantum.measurement"

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        AttrSizedResultSegments(as_property=True),
    )

    op_name = prop_def(StringAttr)
    params = var_operand_def()
    in_qubits = var_operand_def(QubitType)
    in_qreg = opt_operand_def(QuregType)
    arr_qubit_indices = var_operand_def(TensorConstraint(element_type=i64, rank=1))
    static_data = opt_prop_def(DictionaryAttr)

    mres = var_result_def()
    out_qubits = var_result_def(QubitType)
    out_qreg = opt_result_def(QuregType)

    @classmethod
    def parse(cls, parser: Parser):
        op_name = StringAttr(parser.parse_str_literal())

        def typed_operand():
            operand = parser.parse_operand()
            parser.parse_punctuation(":")
            operand_type = parser.parse_type()
            if operand.type != operand_type:
                parser.raise_error("Measurement operand type does not match its annotation.")
            return operand

        params = parser.parse_comma_separated_list(Parser.Delimiter.PAREN, typed_operand)
        qubits = []
        qreg = None
        indices = []
        if parser.parse_optional_keyword("qubits"):
            qubits = parser.parse_comma_separated_list(Parser.Delimiter.PAREN, parser.parse_operand)
        if parser.parse_optional_keyword("quregs"):
            parser.parse_punctuation("(")
            qreg = parser.parse_operand()
            parser.parse_punctuation(")")
        if parser.parse_optional_keyword("indices"):
            indices = parser.parse_comma_separated_list(Parser.Delimiter.PAREN, typed_operand)

        properties = {"op_name": op_name}
        if parser.parse_optional_keyword("static_data"):
            parser.parse_punctuation("=")
            static_data = parser.parse_attribute()
            if not isinstance(static_data, DictionaryAttr):
                parser.raise_error("Expected a dictionary for measurement static_data.")
            properties["static_data"] = static_data
        attributes = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        if parser.parse_optional_punctuation("("):
            parser.parse_punctuation(")")
            result_types = []
        else:
            result_types = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_type
            )
        num_quantum_results = len(qubits) + (qreg is not None)
        if len(result_types) < num_quantum_results:
            parser.raise_error("Expected a result for each measurement qubit or register.")
        classical_types = result_types[: len(result_types) - num_quantum_results]
        qubit_types = result_types[len(classical_types) : len(classical_types) + len(qubits)]
        qreg_type = result_types[-1] if qreg is not None else None
        return cls.build(
            operands=[params, qubits, qreg, indices],
            result_types=[classical_types, qubit_types, qreg_type],
            properties=properties,
            attributes=attributes,
        )


@irdl_op_definition
class MeasureOp(IRDLOperation):
    """A single-qubit projective measurement in the computational basis."""

    name = "quantum.measure"

    assembly_format = """
        $in_qubit (`postselect` $postselect^)? attr-dict `:` type(results)
    """

    in_qubit = operand_def(QubitType)

    postselect = opt_prop_def(
        IntegerAttr.constr(type=I32, value=IntSetConstraint(frozenset((0, 1))))
    )

    mres = result_def(i1)

    out_qubit = result_def(QubitType)

    def __init__(
        self, in_qubit: QubitSSAValue | Operation, postselect: int | IntegerAttr | None = None
    ):
        if isinstance(postselect, int):
            postselect = IntegerAttr(postselect, 32)

        if postselect is None:
            properties = {}
        else:
            properties = {"postselect": postselect}

        super().__init__(
            operands=(in_qubit,), properties=properties, result_types=(i1, QubitType())
        )


#########################################
######### Terminal measurements #########
#########################################


@irdl_op_definition
class CountsOp(TerminalMeasurementOp):
    """Compute sample counts for the given observable for the current state"""

    name = "quantum.counts"

    assembly_format = """
        $obs ( `shape` $dynamic_shape^ )?
        ( `in` `(` $in_eigvals^ `:` type($in_eigvals) `,` $in_counts `:` type($in_counts) `)` )?
        attr-dict ( `:` type($eigvals)^ `,` type($counts) )?
    """

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        SameVariadicResultSize(),
    )

    obs = operand_def(ObservableType)

    dynamic_shape = opt_operand_def(i64)

    in_eigvals = opt_operand_def(MemRefConstraint(element_type=Float64Type(), rank=1))

    in_counts = opt_operand_def(MemRefConstraint(element_type=i64, rank=1))

    eigvals = opt_result_def(TensorConstraint(element_type=Float64Type(), rank=1))

    counts = opt_result_def(TensorConstraint(element_type=i64, rank=1))


@irdl_op_definition
class ExpvalOp(TerminalMeasurementOp):
    """Compute the expectation value of the given observable for the current state"""

    name = "quantum.expval"

    assembly_format = "$obs attr-dict `:` type(results)"

    obs = operand_def(ObservableType)

    expval = result_def(Float64Type())

    def __init__(self, obs: ObservableSSAValue | Operation):
        super().__init__(operands=(obs,), result_types=(Float64Type(),))


@irdl_op_definition
class ProbsOp(TerminalMeasurementOp):
    """Compute computational basis probabilities for the current state"""

    name = "quantum.probs"

    assembly_format = """
        $obs ( `shape` $dynamic_shape^ )?
        ( `in` `(` $state_in^ `:` type($state_in) `)` )?
        attr-dict ( `:` type($probabilities)^ )?
    """

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    obs = operand_def(ObservableType)

    dynamic_shape = opt_operand_def(i64)

    state_in = opt_operand_def(MemRefConstraint(element_type=Float64Type(), rank=1))

    probabilities = opt_result_def(TensorConstraint(element_type=Float64Type(), rank=1))


@irdl_op_definition
class SampleOp(TerminalMeasurementOp):
    """Sample eigenvalues from the given observable for the current state"""

    name = "quantum.sample"

    assembly_format = """
        $obs ( `shape` $dynamic_shape^ )?
        ( `in` `(` $in_data^ `:` type($in_data) `)` )?
        attr-dict ( `:` type($samples)^ )?
    """

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    obs = operand_def(ObservableType)

    dynamic_shape = var_operand_def(i64)

    in_data = opt_operand_def(MemRefConstraint(element_type=Float64Type(), rank=(1, 2)))

    samples = opt_result_def(TensorConstraint(element_type=Float64Type(), rank=(1, 2)))


@irdl_op_definition
class StateOp(TerminalMeasurementOp):
    """Return the current statevector"""

    name = "quantum.state"

    assembly_format = """
        $obs ( `shape` $dynamic_shape^ )?
        ( `in` `(` $state_in^ `:` type($state_in) `)` )?
        attr-dict ( `:` type($state)^ )?
    """

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    obs = operand_def(ObservableType)

    dynamic_shape = opt_operand_def(i64)

    state_in = opt_operand_def(MemRefConstraint(element_type=ComplexType(Float64Type()), rank=1))

    state = opt_result_def(TensorConstraint(element_type=ComplexType(Float64Type()), rank=1))


@irdl_op_definition
class VarianceOp(TerminalMeasurementOp):
    """Compute the variance of the given observable for the current state"""

    name = "quantum.var"

    assembly_format = "$obs attr-dict `:` type(results)"

    obs = operand_def(ObservableType)

    variance = result_def(Float64Type())

    def __init__(self, obs: ObservableSSAValue | Operation):
        super().__init__(operands=(obs,), result_types=(Float64Type(),))
