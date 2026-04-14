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

from xdsl import builder, context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin, func, scf, tensor
from xdsl.dialects.builtin import IndexType, IntegerType
from xdsl.ir import Block, Region
from xdsl.rewriter import InsertPoint

from catalyst.python_interface.dialects import qecp
from catalyst.python_interface.inspection.xdsl_conversion import _tensor_shape_from_ssa
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform

_NUM_ROT_PARAMS = 3


def _get_noise_subroutine_name(k, n):
    """Get the name of the noise injection subroutine for a given codeblock type."""
    return "noise_subroutine_code" + str(k) + "x" + str(n)


class ConvertNoiseOpToSubroutinePass(passes.ModulePass):
    """Pass that converts qecl.noise operations to subroutines in the qecp layer."""

    name = "convert-noiseop-to-subroutine"

    def __init__(self, **options):
        self._number_errors = options.get("number_errors", 1)

    def _create_noise_subroutine(self, k, n, number_errors):
        """Create a subroutine (func.FuncOp) operation for injecting physical noise mimic with Rot
        gates. The subroutine takes a physical codeblock, the qubit indices and the corresponding
        rotation parameters to be injected as inputs, and returns the noisy physical codeblock after
        error injection.

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
        # The input types include:
        # 1, a physical codeblock;
        # 2, number of errors (rot operations);
        # 3, a tensor containing rotation parameters for rot operations (errors).
        codeblock_type = qecp.PhysicalCodeblockType(k, n)
        errors_indices_type = builtin.TensorType(
            element_type=builtin.IntegerType(64),
            shape=[
                number_errors,
            ],
        )
        rotation_params_type = builtin.TensorType(
            element_type=builtin.Float64Type(), shape=[number_errors, _NUM_ROT_PARAMS]
        )
        input_types = (codeblock_type, errors_indices_type, rotation_params_type)

        # The output type is the noisy physical codeblock
        output_types = (codeblock_type,)

        block = Block(arg_types=input_types)

        with builder.ImplicitBuilder(block):
            # 1. Get the input arguments
            in_codeblock, errors_indices, rotation_params = block.args

            # 2. Define for loop bounds
            num_errors = arith.ConstantOp.from_int_and_width(number_errors, 64)

            zero = arith.ConstantOp.from_int_and_width(0, 64)
            one = arith.ConstantOp.from_int_and_width(1, 64)
            two = arith.ConstantOp.from_int_and_width(2, 64)

            loop_body = Block(arg_types=(IndexType(), codeblock_type))

            for_loop = scf.ForOp(
                lb=zero, ub=num_errors, step=one, iter_args=(in_codeblock,), body=loop_body
            )

            with builder.ImplicitBuilder(loop_body) as (index_var, current_codeblock):
                index_var_int = arith.IndexCastOp(index_var, IntegerType(64))

                # Get the qubit index for error injection from the input codeblock
                # Note that the qubit index is generated randomly from a `qnode` function and
                qubit_index = tensor.ExtractOp(
                    errors_indices,
                    indices=[index_var_int],
                    result_type=errors_indices.type.element_type,
                )
                # Get the rotation parameters for the current error to be injected
                phi = tensor.ExtractOp(
                    rotation_params,
                    indices=[index_var_int, zero],
                    result_type=rotation_params.type.element_type,
                )
                theta = tensor.ExtractOp(
                    rotation_params,
                    indices=[index_var_int, one],
                    result_type=rotation_params.type.element_type,
                )
                omega = tensor.ExtractOp(
                    rotation_params,
                    indices=[index_var_int, two],
                    result_type=rotation_params.type.element_type,
                )

                # Extract a phyiscal qubit from the current codeblock
                extracted_phyiscal_qubit = qecp.ExtractQubitOp(current_codeblock, qubit_index)
                # Create the Rot operation with the extracted qubit for error injection
                rot_op = qecp.RotOp(phi, theta, omega, extracted_phyiscal_qubit)

                # Insert the phyiscal qubit with noise back to the codeblock
                updated_codeblock = qecp.InsertQubitOp(
                    current_codeblock, qubit_index, rot_op.results[0]
                )

                # Yield the updated codeblock
                scf.YieldOp(updated_codeblock.results[0])

            returned_codeblock = for_loop.results[0]

            func.ReturnOp(returned_codeblock)

        region = Region([block])
        # pylint: disable=line-too-long
        # Note that visibility is set as private to ensure the subroutines that are
        # not called (dead code) can be eliminated as the
        # ["symbol-dce"](https://github.com/PennyLaneAI/catalyst/blob/372c376eb821e830da778fdc8af423eeb487eab6/frontend/catalyst/pipelines.py#L248)_
        # pass was added to the pipeline.
        symbol_name = _get_noise_subroutine_name(k, n)
        funcOp = func.FuncOp(
            symbol_name,
            (input_types, output_types),
            visibility="private",
            region=region,
        )
        # Add an attribute to the noise injection subroutine
        funcOp.attributes[_get_noise_subroutine_name(k, n)] = builtin.NoneAttr()
        return funcOp

    def apply(self, _ctx: context.Context, op: builtin.ModuleOp) -> None:
        """Apply the convert-noiseop-to-subroutine pass."""
        # pylint: disable=line-too-long
        # Insert subroutines for all gates in the MBQC gate set to the module.
        # Note that the visibility of those subroutines are set as private, which ensure the
        # ["symbol-dce"](https://github.com/PennyLaneAI/catalyst/blob/372c376eb821e830da778fdc8af423eeb487eab6/frontend/catalyst/pipelines.py#L248)_
        # pass could eliminate the unreferenced subroutines.
        # Collect different types of codeblocks in a module.
        codeblocks = set()
        noise_subroutine_dict = {}

        for op_ in op.walk():
            if isinstance(op_, func.FuncOp) and "quantum.node" in op_.attributes:
                for op_ in op_.walk():
                    if isinstance(op_, qecp.NoiseOp):
                        codeblocks.add(
                            (op_.in_codeblock.type.k.value.data, op_.in_codeblock.type.n.value.data)
                        )
                        break

        for k, n in codeblocks:
            noise_subroutine = self._create_noise_subroutine(k, n, self._number_errors)
            op.regions[0].blocks.first.add_op(noise_subroutine)
            noise_subroutine_dict[_get_noise_subroutine_name(k, n)] = noise_subroutine

        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier(
                [ConvertNoiseOpToSubroutinePattern(noise_subroutine_dict, self._number_errors)]
            ),
            apply_recursively=False,
        ).rewrite_module(op)


convert_noiseop_to_subroutine_pass = compiler_transform(ConvertNoiseOpToSubroutinePass)


class ConvertNoiseOpToSubroutinePattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for converting to the MBQC formalism."""

    def __init__(self, noise_subroutine_dict, number_errors):
        self.noise_subroutine_dict = noise_subroutine_dict
        self._number_errors = number_errors

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: qecp.NoiseOp,
        rewriter: pattern_rewriter.PatternRewriter,
        /,
    ):
        k = op.in_codeblock.type.k.value.data
        n = op.in_codeblock.type.n.value.data

        # Create random qubit indices and rotation parameters for error injection, which are
        # generated randomly from a `qnode` function.
        # NOTE: that the random qubit indices and rotation parameters are generated in the Python
        # layer and passed to the noise injection subroutine as inputs, which allows us to inject
        # different errors for different qecp.noise instances in the execution phase.
        qubit_indices = random.sample(range(n), self._number_errors)
        rotation_params = []
        for _ in range(self._number_errors * _NUM_ROT_PARAMS):
            rotation_params.append(random.uniform(0, 2 * math.pi))

        # Insert a tensor constant operation for the qubit indices and rotation parameters, which
        # will be passed to the noise injection subroutine as inputs.
        qubit_indices_constantop = arith.ConstantOp(
            builtin.DenseIntOrFPElementsAttr.from_list(
                type=builtin.TensorType(builtin.IntegerType(64), shape=(self._number_errors,)),
                data=qubit_indices,
            )
        )

        rotation_params_constantop = arith.ConstantOp(
            builtin.DenseIntOrFPElementsAttr.from_list(
                type=builtin.TensorType(
                    builtin.Float64Type(), shape=(self._number_errors, _NUM_ROT_PARAMS)
                ),
                data=rotation_params,
            )
        )

        # Insert qubit indices and rotation parameters tensor constants before the noise op
        rewriter.insert_op(qubit_indices_constantop, InsertPoint.before(op))
        rewriter.insert_op(rotation_params_constantop, InsertPoint.before(op))

        callee = builtin.SymbolRefAttr(_get_noise_subroutine_name(k, n))

        arguments = [
            op.in_codeblock,
            qubit_indices_constantop.results[0],
            rotation_params_constantop.results[0],
        ]

        return_types = self.noise_subroutine_dict[
            _get_noise_subroutine_name(k, n)
        ].function_type.outputs.data
        callOp = func.CallOp(callee, arguments, return_types)
        rewriter.insert_op(callOp, InsertPoint.before(op))
        rewriter.replace_all_uses_with(op.out_codeblock, callOp.results[0])
        rewriter.erase_op(op)
