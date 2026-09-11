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
This file contains the definition of miscellaneous operations in the
Quantum dialect.
"""

from collections.abc import Sequence
from typing import ClassVar

from xdsl.dialects.builtin import (
    StringAttr,
    UnitAttr,
    i1,
    i64,
)
from xdsl.ir import Block, Operation, Region
from xdsl.irdl import (
    AnyOf,
    AttrSizedOperandSegments,
    AttrSizedResultSegments,
    IRDLOperation,
    ParsePropInAttrDict,
    RangeOf,
    RangeVarConstraint,
    irdl_op_definition,
    lazy_traits_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import (
    HasParent,
    IsTerminator,
    NoMemoryEffect,
    Pure,
    ReturnLike,
    SingleBlockImplicitTerminator,
)

from ..attributes import QubitSSAValue, QubitType, QuregSSAValue, QuregType


@irdl_op_definition
class AdjointOp(IRDLOperation):
    """Calculate the adjoint of the enclosed operations"""

    T: ClassVar = RangeVarConstraint("T", RangeOf(AnyOf([QubitType, QuregType])))

    name = "quantum.adjoint"

    assembly_format = """
        `(` $args `)` attr-dict `:` type($outs) $region
    """

    args = var_operand_def(T)

    outs = var_result_def(T)

    region = region_def("single_block")

    traits = lazy_traits_def(lambda: (NoMemoryEffect(), SingleBlockImplicitTerminator(YieldOp)))

    def __init__(
        self,
        args: Sequence[QuregSSAValue | QubitSSAValue] | Operation,
        region: Region | Sequence[Operation] | Sequence[Block],
    ):
        result_types = tuple(arg.type for arg in args)
        super().__init__(operands=(args,), result_types=(result_types,), regions=(region,))


@irdl_op_definition
class CtrlOp(IRDLOperation):
    """Apply the enclosed operations controlled on the given control qubits."""

    C: ClassVar = RangeVarConstraint("C", RangeOf(AnyOf([QubitType])))
    T: ClassVar = RangeVarConstraint("T", RangeOf(AnyOf([QubitType, QuregType])))

    name = "quantum.ctrl"

    assembly_format = """
        `(` $in_ctrl_qubits `)` `ctrlvals` `(` $in_ctrl_values `)` `(` $args `)`
        attr-dict `:` type($out_ctrl_qubits) `->` type($outs) $region
    """

    in_ctrl_qubits = var_operand_def(C)
    in_ctrl_values = var_operand_def(i1)
    args = var_operand_def(T)

    out_ctrl_qubits = var_result_def(C)
    outs = var_result_def(T)

    region = region_def("single_block")

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        AttrSizedResultSegments(as_property=True),
    )

    traits = lazy_traits_def(lambda: (NoMemoryEffect(), SingleBlockImplicitTerminator(YieldOp)))

    def __init__(
        self,
        in_ctrl_qubits: Sequence[QubitSSAValue] | Operation,
        in_ctrl_values: Sequence[Operation],
        args: Sequence[QuregSSAValue | QubitSSAValue] | Operation,
        region: Region | Sequence[Operation] | Sequence[Block],
    ):
        ctrl_result_types = tuple(q.type for q in in_ctrl_qubits)
        arg_result_types = tuple(a.type for a in args)
        super().__init__(
            operands=(in_ctrl_qubits, in_ctrl_values, args),
            result_types=(ctrl_result_types, arg_result_types),
            regions=(region,),
        )


@irdl_op_definition
class DeviceInitOp(IRDLOperation):
    """Initialize a quantum device."""

    name = "quantum.device"

    assembly_format = """
        (`shots` `(` $shots^ `)`)? `[` $lib `,` $device_name `,` $kwargs `]` attr-dict
    """

    irdl_options = (ParsePropInAttrDict(),)

    shots = opt_operand_def(i64)

    auto_qubit_management = opt_prop_def(UnitAttr)

    lib = prop_def(StringAttr)

    device_name = prop_def(StringAttr)

    kwargs = prop_def(StringAttr)


@irdl_op_definition
class DeviceReleaseOp(IRDLOperation):
    """Release the active quantum device."""

    name = "quantum.device_release"

    assembly_format = "attr-dict"


@irdl_op_definition
class FinalizeOp(IRDLOperation):
    """Teardown the quantum runtime."""

    name = "quantum.finalize"

    assembly_format = "attr-dict"


@irdl_op_definition
class InitializeOp(IRDLOperation):
    """Initialize the quantum runtime."""

    name = "quantum.init"

    assembly_format = "attr-dict"


@irdl_op_definition
class NumQubitsOp(IRDLOperation):
    """Get the number of currently allocated qubits."""

    name = "quantum.num_qubits"

    assembly_format = """
        attr-dict `:` type(results)
    """

    num_qubits = result_def(i64)


@irdl_op_definition
class YieldOp(IRDLOperation):
    """Return results from quantum program regions"""

    name = "quantum.yield"

    assembly_format = "attr-dict ($retvals^ `:` type($retvals))?"

    retvals = var_operand_def(QuregType | QubitType)

    traits = traits_def(HasParent(AdjointOp), IsTerminator(), Pure(), ReturnLike())
