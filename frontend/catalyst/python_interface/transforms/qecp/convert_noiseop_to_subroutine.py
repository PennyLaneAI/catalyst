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
This module contains the implementation of the xDSL convert-noiseop-to-subroutine pass.
"""
import math
import random
from dataclasses import dataclass
from typing import NoReturn, cast

from xdsl import builder, context, passes, pattern_rewriter

from xdsl.context import Context
from xdsl.dialects import arith, builtin, scf
from xdsl.dialects.builtin import func
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType
from xdsl.ir import Block, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass

from catalyst.python_interface.dialects import qecl, qecp, quantum
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform
from catalyst.utils.exceptions import CompileError

@dataclass(frozen=True)
class ConvertNoiseOpToSubroutinePass(passes.ModulePass):
    """Pass that converts qecl.noise operations to subroutines in the qecp layer."""

    name = "convert-noiseop-to-subroutine"

    def __init__(self, **options):
        self._number_errors = options.get("number_errors", 1) 


    def _create_noise_subroutine(self, k, n, number_errors):
        """Create a subroutine for injecting physical noise mimic with Rot gates.
        Args:
            gate_name (str): Name of the gate.

        Returns:
            The corresponding subroutine (func.FuncOp).
        """
        # ensure the order of parameters are aligned with customOp
        input_types = (builtin.IntegerType(), qecp.PhysicalCodeblockType(k, n))

        output_types = (qecp.PhysicalCodeblockType(k, n))

        block = Block(arg_types=input_types)

        with builder.ImplicitBuilder(block):
            in_codeblock = block.args[-1]
            number_errors = block.args[0]

            # 2. Define loop bounds
            start = arith.ConstantOp(IndexType.get(), 0)
            stop = number_errors
            step = arith.ConstantOp(IndexType.get(), 1)

            phi = random.uniform(0, math.pi)
            theta = random.uniform(0, math.pi)
            omega = random.uniform(0, math.pi)

            

            loop = scf.ForOp(start, stop, step, iter_args=, body=)



            cz_op = CustomOp(in_qubits=[in_qubits[0], graph_qubit_dict[2]], gate_name="CZ")

            graph_qubit_dict[1], graph_qubit_dict[2] = cz_op.results

            mres, graph_qubit_dict = self._queue_measurements(gate_name, graph_qubit_dict, params)

            # The following could be removed to support Pauli tracker
            by_product_correction = self._insert_byprod_corrections(
                gate_name, mres, graph_qubit_dict[5]
            )

            graph_qubit_dict[5] = by_product_correction

            for node in graph_qubit_dict:
                if node not in [5]:
                    _ = DeallocQubitOp(graph_qubit_dict[node])

            func.ReturnOp(graph_qubit_dict[5])

        region = Region([block])
        # pylint: disable=line-too-long
        # Note that visibility is set as private to ensure the subroutines that are
        # not called (dead code) can be eliminated as the
        # ["symbol-dce"](https://github.com/PennyLaneAI/catalyst/blob/372c376eb821e830da778fdc8af423eeb487eab6/frontend/catalyst/pipelines.py#L248)_
        # pass was added to the pipeline.
        funcOp = func.FuncOp(
            gate_name.lower() + "_in_mbqc",
            (input_types, output_types),
            visibility="private",
            region=region,
        )
        # Add an attribute to the mbqc transform subroutine
        funcOp.attributes["mbqc_transform"] = builtin.NoneAttr()
        return funcOp


    def apply(self, _ctx: context.Context, op: builtin.ModuleOp) -> None:
        """Apply the convert-noiseop-to-subroutine pass."""
        # pylint: disable=line-too-long
        # Insert subroutines for all gates in the MBQC gate set to the module.
        # Note that the visibility of those subroutines are set as private, which ensure the
        # ["symbol-dce"](https://github.com/PennyLaneAI/catalyst/blob/372c376eb821e830da778fdc8af423eeb487eab6/frontend/catalyst/pipelines.py#L248)_
        # pass could eliminate the unreferenced subroutines.
        subroutine_dict = {}

        for gate_name in _MBQC_ONE_QUBIT_GATES:
            funcOp = self._create_single_qubit_gate_subroutine(gate_name)
            op.regions[0].blocks.first.add_op(funcOp)
            subroutine_dict[gate_name] = funcOp

        cnot_funcOp = self._create_cnot_gate_subroutine()
        op.regions[0].blocks.first.add_op(cnot_funcOp)
        subroutine_dict["CNOT"] = cnot_funcOp

        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier(
                [ConvertNoiseOpToSubroutinePattern(subroutine_dict)]
            ),
            apply_recursively=False,
        ).rewrite_module(op)


convert_noiseop_to_subroutine = compiler_transform(ConvertNoiseOpToSubroutinePass)


class ConvertNoiseOpToSubroutinePattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for converting to the MBQC formalism."""

    def __init__(self, subroutines_dict):
        self.subroutine_dict = subroutines_dict

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        noiseop: qecl.NoiseOp,
        rewriter: pattern_rewriter.PatternRewriter,
        /,
    ):
        pass