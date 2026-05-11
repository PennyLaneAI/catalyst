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
from catalyst.python_interface.dialects.quantum.attributes import QubitType
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform

# MARK: Type Conversion Pattern


@dataclass
class QecPhysicalQubitTypeConversion(TypeConversionPattern):
    """Qubit type conversion pattern from qecp.qubit -> quantum.bit."""

    @attr_type_rewrite_pattern
    def convert_type(self, typ: qecp.QecPhysicalQubitType) -> QubitType:
        """Type conversion rewrite pattern for QEC physical qubit types."""

        return QubitType()


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
                    QecPhysicalQubitTypeConversion(),
                ]
            )
        ).rewrite_module(op)


convert_qecp_to_quantum_pass = compiler_transform(ConvertQecPhysicalToQuantumPass)
