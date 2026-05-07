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

"""QEC Physical to Quantum dialect conversion.

This module contains the implementation of the xDSL convert-qecp-to-quantum dialect-conversion pass.

Known Limitations
-----------------

- The convert-qecp-to-quantum pass does not support the following cases:

  * QEC codes where the number of logical qubits per codeblock, k, is greater than 1.
- This pass is for the debugging purpose only.
"""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
)

from catalyst.python_interface.dialects import qecp
from catalyst.python_interface.dialects.quantum.attributes import QuregType, QubitType
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform



# MARK: Type Conversion Pattern

@dataclass
class PhysicalCodeblockTypeConversion(TypeConversionPattern):
    """Codeblock type conversion pattern from qecp.codeblock -> quantum.qubit."""

    @attr_type_rewrite_pattern
    def convert_type(self, typ: qecp.PhysicalCodeblockType) -> list[QubitType]:
        """Type conversion rewrite pattern for physical codeblock types."""

        return [QubitType()] * typ.n.value.data


@dataclass
class PhysicalHyperRegisterTypeConversion(TypeConversionPattern):
    """Hyper-register type conversion pattern from qecp.hyperreg -> quantum.qreg."""

    @attr_type_rewrite_pattern
    def convert_type(self, typ: qecp.PhysicalHyperRegisterType) -> QuregType:
        """Type conversion rewrite pattern for physical hyperregister types."""

        return QuregType()


@dataclass(frozen=True)
class ConvertQecPhysicalToQuantumPass(ModulePass):
    """
    Convert QEC physical instructions to Quantum instructions.
    """

    name = "convert-qecp-to-quantum"

    # pylint: disable=unused-argument
    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """Apply the convert-qecp-to-quantum pass."""

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    PhysicalCodeblockTypeConversion(),
                    PhysicalHyperRegisterTypeConversion(),
                ]
            )
        ).rewrite_module(op)


convert_qecp_to_quantum_pass = compiler_transform(ConvertQecPhysicalToQuantumPass)
