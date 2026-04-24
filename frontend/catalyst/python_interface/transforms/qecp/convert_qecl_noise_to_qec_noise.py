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
This module contains the implementation of the xDSL convert-qecl-noise-to-qecp-noise pass.
"""

import math
import random
from dataclasses import dataclass

from xdsl import builder, context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin, func, scf, tensor
from xdsl.dialects.builtin import IndexType
from xdsl.ir import Block, Region
from xdsl.rewriter import InsertPoint
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass

from catalyst.python_interface.dialects import qecl, qecp
from catalyst.python_interface.pass_api.compiler_transform import compiler_transform

_NUM_ROT_PARAMS = 3


def _get_noise_subroutine_name(k, n, number_errors):
    """Get the symbol name of the noise injection subroutine for a given codeblock type.
    Args:
        k (int): The number of logical qubits of a physical codeblock.
        n (int): The number of physical data qubits of a physical codeblock.
        number_errors (int): The number of errors to be injected.
    Returns:
        str: The symbol name of the noise injection subroutine.
    """

    return f"noise_subroutine_code_{k}x{n}x{number_errors}"


class ConvertQECLNoiseOpToQECPNoisePattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for converting to qecl.noise operations to subroutines in the qecp layer."""

    def __init__(self, noise_subroutine, n, number_errors):
        self.noise_subroutine = noise_subroutine
        self._n = n
        self._number_errors = number_errors

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: qecl.NoiseOp,
        rewriter: pattern_rewriter.PatternRewriter,
        /,
    ):  # pylint: disable=missing-function-docstring
        k = op.in_codeblock.type.k.value.data

        in_block_cast = builtin.UnrealizedConversionCastOp.get(
            (op.in_codeblock,), (qecp.PhysicalCodeblockType(k, self._n),)
        )
        rewriter.insert_op(in_block_cast, InsertPoint.before(op))

        # Create random qubit indices and rotation parameters for error injection, which are
        # generated randomly from the python random module.
        # NOTE: that the random qubit indices and rotation parameters are generated in the Python
        # layer and passed to the noise injection subroutine as inputs, which allows us to inject
        # different errors for different qecp.noise instances in the execution phase.
        # NOTE: Another option: the logic below could be replaced with jax.random.uniform
        qubit_indices = random.sample(range(self._n), self._number_errors)
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

        callee = builtin.SymbolRefAttr(self.noise_subroutine.sym_name)

        arguments = [
            in_block_cast.results[0],
            qubit_indices_constantop.results[0],
            rotation_params_constantop.results[0],
        ]

        return_types = self.noise_subroutine.function_type.outputs.data
        callOp = func.CallOp(callee, arguments, return_types)
        rewriter.insert_op(callOp, InsertPoint.before(op))
        cast_op = builtin.UnrealizedConversionCastOp.get(
            (callOp.results[0],), (op.out_codeblock.type,)
        )
        rewriter.insert_op(cast_op, InsertPoint.before(op))
        rewriter.replace_all_uses_with(op.out_codeblock, cast_op.results[0])
        rewriter.erase_op(op)


@dataclass(frozen=True)
class ConvertQECLNoiseOpToQECPNoisePass(passes.ModulePass):
    """Pass that converts qecl.noise operations to subroutines in the qecp layer."""

    name = "convert-qecl-noise-to-qecp-noise"

    # the number of physical data qubits of per codeblocks. NOTE: this option is expected to
    # be specified in the `qecl-to-qecp` skeleton pass.
    n: int
    # the number of errors to be injected for each noise operation. Defaults to 1.
    number_errors: int = 1

    def _create_noise_subroutine(self, k):
        """Create a subroutine (func.FuncOp) operation for injecting physical noise mimic with Rot
        gates. The subroutine takes a physical codeblock, the qubit indices and the corresponding
        rotation parameters to be injected as inputs, and returns the noisy physical codeblock after
        error injection.

        NOTE: The random rotation parameters and qubit indices are generated randomly
        from a `qnode` function.

        Args:
            k (int): The number of logical qubits of a physical codeblock.

        Returns:
            The corresponding subroutine (func.FuncOp).
        """
        # 1. Define the input and output types of the subroutine
        # The input types include:
        # 1, a physical codeblock;
        # 2, number of errors (rot operations);
        # 3, a tensor containing rotation parameters for rot operations (errors).
        codeblock_type = qecp.PhysicalCodeblockType(k, self.n)
        errors_indices_type = builtin.TensorType(
            element_type=builtin.IntegerType(64),
            shape=[
                self.number_errors,
            ],
        )
        rotation_params_type = builtin.TensorType(
            element_type=builtin.Float64Type(), shape=[self.number_errors, _NUM_ROT_PARAMS]
        )
        input_types = (codeblock_type, errors_indices_type, rotation_params_type)

        # The output type is the noisy physical codeblock
        output_types = (codeblock_type,)

        block = Block(arg_types=input_types)

        with builder.ImplicitBuilder(block):
            # 1. Get the input arguments
            in_codeblock, errors_indices, rotation_params = block.args

            # 2. Define for loop bounds
            num_errors = arith.ConstantOp.from_int_and_width(self.number_errors, IndexType())

            zero = arith.ConstantOp.from_int_and_width(0, IndexType())
            one = arith.ConstantOp.from_int_and_width(1, IndexType())
            two = arith.ConstantOp.from_int_and_width(2, IndexType())

            loop_body = Block(arg_types=(IndexType(), codeblock_type))

            for_loop = scf.ForOp(
                lb=zero, ub=num_errors, step=one, iter_args=(in_codeblock,), body=loop_body
            )

            with builder.ImplicitBuilder(loop_body) as (index_var, current_codeblock):
                # Get the qubit index for error injection from the input codeblock
                # Note that the qubit index is generated randomly from a `qnode` function and
                qubit_index = tensor.ExtractOp(
                    errors_indices,
                    indices=[index_var],
                    result_type=errors_indices.type.element_type,
                )
                # Get the rotation parameters for the current error to be injected
                phi = tensor.ExtractOp(
                    rotation_params,
                    indices=[index_var, zero],
                    result_type=rotation_params.type.element_type,
                )
                theta = tensor.ExtractOp(
                    rotation_params,
                    indices=[index_var, one],
                    result_type=rotation_params.type.element_type,
                )
                omega = tensor.ExtractOp(
                    rotation_params,
                    indices=[index_var, two],
                    result_type=rotation_params.type.element_type,
                )

                # Extract a physical qubit from the current codeblock
                qubit_index_int = arith.IndexCastOp(qubit_index, IndexType())
                extracted_physical_qubit = qecp.ExtractQubitOp(current_codeblock, qubit_index_int)
                # Create the Rot operation with the extracted qubit for error injection
                rot_op = qecp.RotOp(phi, theta, omega, extracted_physical_qubit)

                # Insert the physical qubit with noise back to the codeblock
                updated_codeblock = qecp.InsertQubitOp(
                    current_codeblock, qubit_index_int, rot_op.results[0]
                )

                # Yield the updated codeblock
                scf.YieldOp(updated_codeblock.results[0])

            noisy_codeblock = for_loop.results[0]

            func.ReturnOp(noisy_codeblock)

        region = Region([block])
        # pylint: disable=line-too-long
        # Note that visibility is set as private to ensure the subroutines that are
        # not called (dead code) can be eliminated as the
        # ["symbol-dce"](https://github.com/PennyLaneAI/catalyst/blob/372c376eb821e830da778fdc8af423eeb487eab6/frontend/catalyst/pipelines.py#L248)_
        # pass was added to the pipeline.
        symbol_name = _get_noise_subroutine_name(k, self.n, self.number_errors)
        funcOp = func.FuncOp(
            symbol_name,
            (input_types, output_types),
            visibility="private",
            region=region,
        )
        # Add an attribute to the noise injection subroutine
        funcOp.attributes[_get_noise_subroutine_name(k, self.n, self.number_errors)] = (
            builtin.UnitAttr()
        )
        return funcOp

    def apply(self, _ctx: context.Context, op: builtin.ModuleOp) -> None:
        """Apply the convert-qecl-noise-to-qecp-noise pass."""
        # Collect different types of codeblocks in a module.
        k = None

        # Traverse the module to find the codeblock type (k).
        for op_ in op.walk():
            if isinstance(op_, func.FuncOp) and "quantum.node" in op_.attributes:
                for op_ in op_.walk():
                    if isinstance(op_, qecl.NoiseOp):
                        k = op_.in_codeblock.type.k.value.data
                        break
        # Skip the conversion if there is no noise operation in the module.
        if k is None:
            return

        # Insert a noise injection subroutine into the module.
        noise_subroutine = self._create_noise_subroutine(k)
        assert op.regions[0].blocks.first is not None
        op.regions[0].blocks.first.add_op(noise_subroutine)

        pattern_rewriter.PatternRewriteWalker(
            ConvertQECLNoiseOpToQECPNoisePattern(noise_subroutine, self.n, self.number_errors),
            apply_recursively=False,
        ).rewrite_module(op)

        # Pass to reconcile unrealized casts after the conversion.
        ReconcileUnrealizedCastsPass().apply(_ctx, op)


# TODOs: Add integration tests for the following line once the quantum-to-qecl pass is in.
convert_qecl_noise_to_qecp_noise_pass = compiler_transform(
    ConvertQECLNoiseOpToQECPNoisePass
)  # pragma: no cover
