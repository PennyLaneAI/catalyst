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

from xdsl.dialects.builtin import (
    StringAttr,
    UnitAttr,
    i64,
)
from xdsl.ir import Block, Operation, Region
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_op_definition,
    lazy_traits_def,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import (
    HasParent,
    IsTerminator,
    NoMemoryEffect,
    Pure,
    ReturnLike,
    SingleBlockImplicitTerminator,
)

from catalyst.python_interface.xdsl_extras import AllTypesMatch

from ..attributes import QuregSSAValue, QuregType, QuregTypeConstraint


@irdl_op_definition
class AdjointOp(IRDLOperation):
    """Calculate the adjoint of the enclosed operations"""

    name = "quantum.adjoint"

    assembly_format = """
        `(` $qreg `)` attr-dict `:` type(operands) $region
    """

    qreg = operand_def(QuregTypeConstraint())

    out_qreg = result_def(QuregTypeConstraint())

    traits = traits_def(
        AllTypesMatch(
            ("qreg", "out_qreg"),
            "Qreg ins and outs must have the same size and types",
        )
    )

    region = region_def("single_block")

    traits = lazy_traits_def(lambda: (NoMemoryEffect(), SingleBlockImplicitTerminator(YieldOp)))

    def __init__(
        self,
        qreg: QuregSSAValue | Operation,
        region: Region | Sequence[Operation] | Sequence[Block],
    ):
        super().__init__(operands=(qreg,), result_types=(qreg.type,), regions=(region,))


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

    retvals = var_operand_def(QuregTypeConstraint())

    traits = traits_def(HasParent(AdjointOp), IsTerminator(), Pure(), ReturnLike())
