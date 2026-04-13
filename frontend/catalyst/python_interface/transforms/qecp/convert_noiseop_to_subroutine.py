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
from xdsl.dialects import arith, builtin, scf, tensor
from xdsl.dialects.builtin import func
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType
from xdsl.ir import Block, Operation, SSAValue, Region
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
from catalyst.python_interface.inspection.xdsl_conversion import _tensor_shape_from_ssa

@dataclass(frozen=True)
class ConvertNoiseOpToSubroutinePass(passes.ModulePass):
    """Pass that converts qecl.noise operations to subroutines in the qecp layer."""

    name = "convert-noiseop-to-subroutine"

    def __init__(self, **options):
        self._number_errors = options.get("number_errors", 1) 


    def _create_noise_subroutine(self, k, n, number_errors):
        """Create a subroutine (func.FuncOp) operation for injecting physical noise mimic with Rot gates.
        The subroutine takes a physical codeblock, the indices of qubits and the corresponding rotation 
        parameters to be injected as inputs, and returns the noisy physical codeblock after error injection. 

        NOTE: The random rotation parameters and qubit indices are generated randomly
        from a `qnode` function.
        
        Args:
            k (int): The number of logical qubits of a physical codeblock.
            n (int): The number of physical qubits of a physical codeblock.
            number_errors (int): The number of errors to be injected.

        Returns:
            The corresponding subroutine (func.FuncOp).
        """
        # 1. Define the input and output types of the subroutine
        # The input types include: 1, a physical codeblock; 2, number of errors (rot operations); 3, a tensor containing rotation parameters for rot operations (errors). 
        codeblock_type = qecp.PhysicalCodeblockType(k, n)
        errors_indices_type = builtin.TensorType(element_type=builtin.IntegerType(64), shape=[number_errors,])
        rotation_params_type = builtin.TensorType(element_type=builtin.Float64Type(), shape=[number_errors, 3])
        input_types = (codeblock_type, errors_indices_type, rotation_params_type)

        # The output type is the noisy physical codeblock
        output_types = (codeblock_type,)

        block = Block(arg_types=input_types)

        with builder.ImplicitBuilder(block):
            # 1. Get the input arguments
            in_codeblock, errors_indices, rotation_params = block.args

            # 2. Define for loop bounds
            start = arith.ConstantOp(IndexType.get(), 0)
            stop = arith.ConstantOp.from_int_and_width(_tensor_shape_from_ssa(errors_indices)[0], 64)
            step = arith.ConstantOp(IndexType.get(), 1)

            zero = arith.ConstantOp.from_int_and_width(0, 64)
            one = arith.ConstantOp.from_int_and_width(1, 64)
            two = arith.ConstantOp.from_int_and_width(2, 64)
            
            loop_body = Block(arg_types=(IndexType(), codeblock_type))

            for_loop = scf.ForOp(start, stop, step, iter_args=(in_codeblock, ), body=loop_body)

            with builder.ImplicitBuilder(loop_body) as (index_var, current_codeblock):
                index_var_int = arith.IndexCastOp(index_var, IntegerType(64))

                # Get the qubit index for error injection from the input codeblock
                # Note that the qubit index is generated randomly from a `qnode` function and
                qubit_index = tensor.ExtractOp(errors_indices, indices=[index_var_int])
                # Get the rotation parameters for the current error to be injected
                phi = tensor.ExtractOp(rotation_params, indices=[index_var_int, zero])
                theta = tensor.ExtractOp(rotation_params, indices=[index_var_int, one])
                omega = tensor.ExtractOp(rotation_params, indices=[index_var_int, two])

                # Create the Rot operation for error injection
                rot_op = qecp.RotOp(current_codeblock, qubit_index, phi, theta, omega)

                # Yield the updated codeblock
                scf.YieldOp(rot_op.results[0])
            
            returned_codeblock = for_loop.results[0]

            func.ReturnOp(returned_codeblock)
        
        region = Region([block])
        # pylint: disable=line-too-long
        # Note that visibility is set as private to ensure the subroutines that are
        # not called (dead code) can be eliminated as the
        # ["symbol-dce"](https://github.com/PennyLaneAI/catalyst/blob/372c376eb821e830da778fdc8af423eeb487eab6/frontend/catalyst/pipelines.py#L248)_
        # pass was added to the pipeline.
        funcOp = func.FuncOp(
            "noise_injection_subroutine",
            (input_types, output_types),
            visibility="private",
            region=region,
        )
        # Add an attribute to the noise injection subroutine
        funcOp.attributes[f"{n}x{k}_code_noise"] = builtin.NoneAttr()
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