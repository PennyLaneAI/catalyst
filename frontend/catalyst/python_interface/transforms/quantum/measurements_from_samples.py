# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the implementation of the measurements_from_samples transform,
written using xDSL.

Known Limitations
-----------------

  * The compilation pass assumes a static number of shots.
  * Usage patterns that are not yet supported with program capture are also not supported in the
    compilation pass. For example, operator arithmetic is not currently supported, such as
    qml.expval(qml.Y(0) @ qml.X(1)).
  * qml.counts() is not supported since the return type/shape is different in PennyLane and
    Catalyst. See
    https://docs.pennylane.ai/projects/catalyst/en/stable/dev/quick_start.html#measurements
    for more information.
"""

from abc import abstractmethod
from dataclasses import dataclass
from itertools import islice

import jax
import jax.numpy as jnp
from pennylane.exceptions import CompileError
from xdsl import context, ir, passes, pattern_rewriter
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.rewriter import InsertPoint

from catalyst.python_interface.conversion import xdsl_module
from catalyst.python_interface.dialects import quantum
from catalyst.python_interface.pass_api import compiler_transform
from catalyst.python_interface.transforms.quantum.diagonalize_measurements import (
    DiagonalizeFinalMeasurementsPass,
)
from catalyst.python_interface.transforms.quantum.wrap_qnode import (
    WrapQNodePass,
    get_call_op,
)
from catalyst.python_interface.utils import get_constant_from_ssa


@dataclass(frozen=True)
class MeasurementsFromSamplesPass(passes.ModulePass):
    """Pass that replaces all terminal measurements in a program with a single
    :func:`pennylane.sample` measurement, and adds postprocessing instructions to recover the
    original measurement. If observables are present in a basis other than Z, the pass
    diagonalizes them before conversion to samples in the computational basis.
    """

    name = "measurements-from-samples"

    def apply(self, _ctx: context.Context, op: builtin.ModuleOp) -> None:
        """Apply the measurements-from-samples pass."""

        # diagonalize measurements before converting to samples
        DiagonalizeFinalMeasurementsPass().apply(_ctx, op)

        # wrap the quantum.nodes in classical functions to store post-processing
        WrapQNodePass(pass_str="from_samples").apply(_ctx, op)

        # apply the patterns for measurements_from_samples
        greedy_applier = pattern_rewriter.GreedyRewritePatternApplier(
            [
                ExpvalAndVarPattern(),
                ProbsPattern(),
                CountsPattern(),
                StatePattern(),
            ]
        )
        walker = pattern_rewriter.PatternRewriteWalker(greedy_applier, apply_recursively=False)
        walker.rewrite_module(op)


measurements_from_samples_pass = compiler_transform(MeasurementsFromSamplesPass)


class MeasurementsFromSamplesPattern(RewritePattern):
    """Rewrite pattern base class for the ``measurements_from_samples`` transform, which replaces
    all terminal measurements in a program with a single :func:`pennylane.sample` measurement, and
    adds postprocessing instructions to recover the original measurement.

    Args:
        shots (int): The number of shots (e.g. as retrieved from the DeviceInitOp).
    """

    def __init__(self):
        super().__init__()

        self._shots = None
        self.qnode: func.FuncOp | None = None
        self.call_op: func.CallOp | None = None

    @abstractmethod
    def match_and_rewrite(self, op: ir.Operation, rewriter: PatternRewriter, /):
        """Abstract method for measurements-from-samples match-and-rewrite patterns."""

    @classmethod
    def get_observable_op(cls, op: quantum.ExpvalOp | quantum.VarianceOp) -> quantum.NamedObsOp:
        """Return the observable op (quantum.NamedObsOp) given as an input operand to `op`.

        We assume that `op` is either a quantum.ExpvalOp or quantum.VarianceOp, but this is not
        strictly enforced.

        Args:
            op (quantum.ExpvalOp | quantum.VarianceOp): The op that uses the observable op.

        Returns:
            quantum.NamedObsOp: The observable op.
        """
        observable_op = op.operands[0].owner
        cls._validate_observable_op(observable_op)

        return observable_op

    @staticmethod
    def _validate_observable_op(op: quantum.NamedObsOp):
        """Validate the observable op.

        Assert that the op is a quantum.NamedObsOp and check if it is supported in the current
        implementation of the measurements-from-samples transform.

        Raises:
            NotImplementedError: If the observable is anything but a PauliZ quantum.NamedObsOp.
        """
        assert isinstance(
            op, quantum.NamedObsOp
        ), f"Expected `op` to be a quantum.NamedObsOp, but got {type(op).__name__}"

        if op.type.data != "PauliZ":
            raise NotImplementedError(
                f"Observable '{op.type.data}' used as input to measurement operation is not "
                f"supported for the measurements_from_samples transform; currently only the "
                f"PauliZ observable is permitted"
            )

    @staticmethod
    def insert_compbasis_op(
        in_qubit: ir.SSAValue, ref_op: ir.Operation, rewriter: PatternRewriter
    ) -> quantum.ComputationalBasisOp:
        """Create and insert a computational-basis op (quantum.ComputationalBasisOp).

        The computation-basis op uses `in_qubit` as its input operand. It is inserted *before* the
        given reference operation, `ref_op`, using the supplied `rewriter`.

        Args:
            in_qubit (SSAValue): The SSA value used as input to the computational-basis op.
            ref_op (Operation): The reference op before which the quantum.ComputationalBasisOp is
                inserted.
            rewriter (PatternRewriter): The xDSL pattern rewriter.

        Returns:
            quantum.ComputationalBasisOp: The inserted computation-basis op.
        """
        assert isinstance(in_qubit, ir.SSAValue) and isinstance(in_qubit.type, quantum.QubitType), (
            f"Expected `in_qubit` to be an SSAValue with type quantum.QubitType, but got "
            f"{type(in_qubit).__name__}"
        )

        # The input operands are [[qubit, ...], qreg]
        compbasis_op = quantum.ComputationalBasisOp(
            operands=[in_qubit, None], result_types=[quantum.ObservableType()]
        )
        rewriter.insert_op(compbasis_op, insertion_point=InsertPoint.before(ref_op))

        return compbasis_op

    @staticmethod
    def insert_sample_op(
        compbasis_op: quantum.ComputationalBasisOp,
        shots: int,
        n_qubits: int,
        rewriter: PatternRewriter,
    ) -> quantum.SampleOp:
        """Create and insert a sample op (quantum.SampleOp).

        The type of the returned samples array is currently restricted to be static, with shape
        (shots, n_qubits).

        The sample op is inserted after the supplied `compbasis_op`.

        Args:
            compbasis_op (quantum.ComputationalBasisOp): The computational-basis op used as the
                input operand to the sample op.
            shots (int): Number of shots (to set the shape of the sample op returned array).
            n_qubits (int): Number of qubits (to set the shape of the sample op returned array).
            rewriter (PatternRewriter): The xDSL pattern rewriter.

        Returns:
            quantum.SampleOp: The inserted sample op.
        """
        assert isinstance(compbasis_op, quantum.ComputationalBasisOp), (
            f"Expected `compbasis_op` to be a quantum.ComputationalBasisOp, but got "
            f"{type(compbasis_op).__name__}"
        )

        sample_op = quantum.SampleOp(
            operands=[compbasis_op.results[0], None, None],
            result_types=[builtin.TensorType(builtin.Float64Type(), [shots, n_qubits])],
        )
        rewriter.insert_op(sample_op, insertion_point=InsertPoint.after(compbasis_op))

        return sample_op

    @staticmethod
    def get_postprocessing_func_op_from_block_by_name(
        block: ir.Block, name: str
    ) -> func.FuncOp | None:
        """Return the post-processing FuncOp from the given `block` with the given `name`.

        If the block does not contain a FuncOp with the matching name, returns None.

        Args:
            block (Block): The xDSL block to search.
            name (str): The name of the post-processing FuncOp.

        Returns:
            func.FuncOp: The FuncOp with matching name.
            None: If no match was found.
        """
        for op in block.ops:
            if isinstance(op, func.FuncOp) and op.sym_name.data == name:
                return op

        return None

    @classmethod
    def get_postprocessing_funcs_from_module_and_insert(
        cls,
        postprocessing_module: builtin.ModuleOp,
        matched_op: ir.Operation,
        name: str | None = None,
    ) -> func.FuncOp:
        """Get the post-processing FuncOp from `postprocessing_module` (and any helper functions
        also contained in `postprocessing_module`) and insert it (them) immediately after the FuncOp
        (in the same block) that contains `matched_op`.

        The post-processing function recovers the original measurement process result from the
        samples array. This post-postprocessing function is optionally renamed to `name`, if given.

        Args:
            postprocessing_module (builtin.ModuleOp): The MLIR module containing the post-processing
                FuncOp.
            matched_op (Operation): The reference op, the parent of which is used as the
                reference point when inserting the post-processing FuncOp. This is usually the op
                matched in the call to match_and_rewrite().
            name (str, optional): The name to assign to the post-processing FuncOp, if given.

        Returns:
            func.FuncOp: The inserted post-processing FuncOp.
        """
        parent_func_op = matched_op.parent_op()

        assert isinstance(parent_func_op, func.FuncOp), (
            f"Expected parent of matched op '{matched_op}' to be a func.FuncOp, but got "
            f"{type(parent_func_op).__name__}"
        )

        # This first op in `postprocessing_module` is the "main" post-processing function
        postprocessing_func_op = postprocessing_module.body.ops.first.clone()
        assert isinstance(postprocessing_func_op, func.FuncOp), (
            f"Expected the first operator of `postprocessing_module` to be a func.FuncOp but "
            f"got {type(postprocessing_func_op).__name__}"
        )

        # The function name from jax.jit is 'main'; rename it here
        if name is not None:
            postprocessing_func_op.sym_name = builtin.StringAttr(data=name)

        parent_block = parent_func_op.parent
        parent_block.insert_op_after(postprocessing_func_op, parent_func_op)

        # Get and insert helper functions, if any
        if len(postprocessing_module.body.ops) > 1:
            prev_op = postprocessing_func_op
            for _op in islice(postprocessing_module.body.ops, 1, None):
                helper_op = _op.clone()
                parent_block.insert_op_after(helper_op, prev_op)
                prev_op = helper_op

        return postprocessing_func_op

    @staticmethod
    def insert_constant_int_op(
        value: int,
        insert_point: InsertPoint,
        rewriter: PatternRewriter,
        value_type: int = 64,
    ) -> arith.ConstantOp:
        """Create and insert a constant op with the given integer value.

        The integer value is contained within a rankless, dense tensor.

        Args:
            value (int): The integer value.
            insert_point (InsertPoint): The insertion point for the constant op.
            rewriter (PatternRewriter): The xDSL pattern rewriter.
            value_type (int, optional): The integer value type (i.e. number of bits).
                Defaults to 64.

        Returns:
            arith.ConstantOp: The created constant op.
        """
        constant_int_op = arith.ConstantOp(
            builtin.DenseIntOrFPElementsAttr.from_list(
                type=builtin.TensorType(builtin.IntegerType(value_type), shape=()), data=(value,)
            )
        )
        rewriter.insert_op(constant_int_op, insertion_point=insert_point)

        return constant_int_op

    @staticmethod
    def get_n_qubits_from_qreg(qreg: ir.SSAValue):
        """Get the number of qubits from a qreg SSA value.

        This method walks back through the SSA graph from the qreg until it reaches its root
        quantum.alloc op, or alloc-like op (with possibly zero or more quantum.insert ops
        in-between), from which the number of qubits is extracted.

        An op is "alloc-like" if it has an 'nqubits_attr' attribute.

        Args:
            qreg (SSAValue): The qreg SSA value.
        """
        assert isinstance(qreg, ir.SSAValue) and isinstance(qreg.type, quantum.QuregType), (
            f"Expected `qreg` to be an SSAValue with type quantum.QuregType, but got "
            f"{type(qreg).__name__}"
        )

        def _walk_back_to_alloc_op(
            insert_or_alloc_op: quantum.AllocOp | quantum.InsertOp,
        ) -> quantum.AllocOp | None:
            """Recursively walk back from a quantum.insert op to its root quantum.alloc op or
            alloc-like op.

            Once found, return the quantum.alloc op.
            """
            if (
                isinstance(insert_or_alloc_op, quantum.AllocOp)
                or insert_or_alloc_op.properties.get("nqubits_attr") is not None
            ):
                return insert_or_alloc_op

            if isinstance(insert_or_alloc_op, quantum.InsertOp):
                return _walk_back_to_alloc_op(insert_or_alloc_op.operands[0].owner)

            return None

        alloc_op = _walk_back_to_alloc_op(qreg.owner)
        assert alloc_op is not None, "Unable to walk back from qreg to alloc op"

        nqubits_attr = alloc_op.properties.get("nqubits_attr")
        assert (
            nqubits_attr is not None
        ), "Unable to determine number of qubits from alloc op; missing property 'nqubits_attr'"

        n_qubits = nqubits_attr.value.data
        assert n_qubits is not None, "Unable to determine number of qubits from qreg SSA value"

        return n_qubits

    def update_returns(
        self,
        mp_index: int,
        sample_op: quantum.SampleOp,
        postprocessing_func_call_op: func.CallOp,
        rewriter: PatternRewriter,
    ):
        """Update the return structure so that the qnode returns the output of sample
        instead of the original output, and the outer function returns the output of
        post-processing instead of directly returning the output of calling the qnode.

        Also update result shapes in the CallOp for the QNode call to reflect this change,
        since this is not updated when modifying the ReturnOps.

        This function updates the self.qnode and self.call_op attributes on the Pattern.

        Args:
            mp_index (int): The index of the measurement process to be updated. The
                relevant function returns will be updated at this index.
            sample_op (quantum.SampleOp): The operation whose results the Qnode should return.
            postprocessing_func_call_op (func.CallOp): The postprocessing CallOp that accepts
                results of the QNode CallOp. The outer function will be updated to return the
                results of this operation instead of returning the QNode CallOp results directly.
            rewriter (PatternRewriter): The xDSL pattern rewriter.
        """
        assert self.qnode is not None
        assert self.call_op is not None

        # update the qnode to return the result of the SampleOp directly
        return_op = self.qnode.get_return_op()
        assert return_op is not None, "QNode has no return op"
        return_op.operands[mp_index] = sample_op.results[0]
        rewriter.notify_op_modified(return_op)
        self.qnode.update_function_type()

        # update the qnode CallOp in the outer function to reflect the new shape for returned data
        result_types = self.qnode.function_type.outputs.data
        new_call_op = func.CallOp(self.call_op.callee, self.call_op.arguments, result_types)
        rewriter.replace_op(self.call_op, new_call_op)
        self.call_op = new_call_op

        # update the outer function to return postprocessed data instead of the raw data
        final_return = [
            use.operation
            for use in list(self.call_op.results[mp_index].uses)
            if isinstance(use.operation, func.ReturnOp)
        ][0]
        final_return.operands[mp_index] = postprocessing_func_call_op.results[0]
        rewriter.notify_op_modified(final_return)


class ExpvalAndVarPattern(MeasurementsFromSamplesPattern):
    """A rewrite pattern for the ``measurements_from_samples`` transform that matches and rewrites
    ``qml.expval()`` and ``qml.var()`` operations.

    Note: only single-wire observables are currently supported

    Args:
        shots (int): The number of shots (e.g. as retrieved from the DeviceInitOp).
    """

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter, /):
        """Match and rewrite for quantum.ExpvalOp and quantum.VarianceOp."""

        if "quantum.node" not in func_op.attributes:
            return

        self._shots = get_shots(func_op)
        self.qnode = func_op
        self.call_op = get_call_op(func_op)

        measurement_processes = [
            op
            for op in self.qnode.body.walk()
            if isinstance(op, (quantum.VarianceOp, quantum.ExpvalOp))
        ]

        for matched_op in measurement_processes:

            observable_op = self.get_observable_op(matched_op)
            in_qubit = observable_op.operands[0]
            compbasis_op = self.insert_compbasis_op(in_qubit, observable_op, rewriter)
            sample_op = self.insert_sample_op(compbasis_op, self._shots, 1, rewriter)

            # Insert the post-processing function into current module or get handle to it if already
            # inserted
            match matched_op:
                case quantum.ExpvalOp():
                    postprocessing_func_name = f"expval_from_samples.tensor.{self._shots}x1xf64"
                    postprocessing_jit_func = _postprocessing_expval
                case quantum.VarianceOp():
                    postprocessing_func_name = f"var_from_samples.tensor.{self._shots}x1xf64"
                    postprocessing_jit_func = _postprocessing_var
                case _:
                    assert False, (
                        f"Expected a quantum.ExpvalOp or quantum.VarianceOp, but got "
                        f"{type(matched_op).__name__}"
                    )

            postprocessing_func_op = self.get_postprocessing_func_op_from_block_by_name(
                matched_op.parent_op().parent, postprocessing_func_name
            )

            if postprocessing_func_op is None:
                # TODO: Do we have to set the shape of the samples array statically here? Or can the
                # shape (shots, wire) be dynamic and given as SSA values?
                # Same goes for the column/wire indices (the second argument).
                postprocessing_module = postprocessing_jit_func(
                    jax.core.ShapedArray([self._shots, 1], float)
                )

                postprocessing_func_op = self.get_postprocessing_funcs_from_module_and_insert(
                    postprocessing_module, matched_op, postprocessing_func_name
                )

            # get the from_elements_op the original MP result is passed to
            # from its uses get the index this result is returned at
            from_elements_op = list(matched_op.results[0].uses)[0].operation
            assert isinstance(
                from_elements_op, tensor.FromElementsOp
            ), f"Expected a tensor.from_elements op, but got {type(from_elements_op).__name__}"
            mp_index = list(from_elements_op.results[0].uses)[0].index

            # Insert the call to the post-processing function
            postprocessing_func_call_op = func.CallOp(
                callee=builtin.SymbolRefAttr(postprocessing_func_op.sym_name),
                arguments=[self.call_op.results[mp_index]],
                return_types=[builtin.TensorType(builtin.Float64Type(), shape=())],
            )
            rewriter.insert_op(
                postprocessing_func_call_op, insertion_point=InsertPoint.after(self.call_op)
            )

            # update the returns of the QNode (to return the raw samples) and the
            # outer function (to return the post-processed values)
            self.update_returns(mp_index, sample_op, postprocessing_func_call_op, rewriter)

            # delete now unused obs --> mp --> tensor chain
            rewriter.erase_op(from_elements_op)
            rewriter.erase_op(matched_op)
            rewriter.erase_op(observable_op)


class ProbsPattern(MeasurementsFromSamplesPattern):
    """A rewrite pattern for the ``measurements_from_samples`` transform that matches and rewrites
    ``qml.probs()`` operations.

    Args:
        shots (int): The number of shots (e.g. as retrieved from the DeviceInitOp).
    """

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter, /):
        """Match and rewrite for quantum.ProbsOp."""

        if "quantum.node" not in func_op.attributes:
            return

        self._shots = get_shots(func_op)
        self.qnode = func_op
        self.call_op = get_call_op(func_op)

        probs_ops = [op for op in self.qnode.body.walk() if isinstance(op, quantum.ProbsOp)]

        for probs_op in probs_ops:

            compbasis_op = probs_op.operands[0].owner
            assert isinstance(compbasis_op, quantum.ComputationalBasisOp)

            n_qubits = None
            if compbasis_op.qreg is not None:
                n_qubits = self.get_n_qubits_from_qreg(compbasis_op.qreg)

            elif compbasis_op.qubits != ():
                n_qubits = len(compbasis_op.qubits)

            assert (
                n_qubits is not None
            ), "Unable to determine number of qubits from quantum.compbasis op"

            sample_op = self.insert_sample_op(compbasis_op, self._shots, n_qubits, rewriter)

            # Insert the post-processing function into current module or
            # get handle to it if already inserted
            postprocessing_func_name = f"probs_from_samples.tensor.{self._shots}x{n_qubits}xf64"

            postprocessing_func_op = self.get_postprocessing_func_op_from_block_by_name(
                probs_op.parent_op().parent, postprocessing_func_name
            )

            if postprocessing_func_op is None:
                # TODO: Do we have to set the shape of the samples array statically here? Or can the
                # shape (shots, wire) be dynamic and given as SSA values?
                # Same goes for the column/wire indices (the second argument).
                postprocessing_module = _postprocessing_probs(
                    jax.core.ShapedArray([self._shots, n_qubits], float)
                )

                postprocessing_func_op = self.get_postprocessing_funcs_from_module_and_insert(
                    postprocessing_module, probs_op, postprocessing_func_name
                )

            # Insert the call to the post-processing function
            postprocessing_func_call_op = func.CallOp(
                callee=builtin.SymbolRefAttr(postprocessing_func_op.sym_name),
                arguments=[sample_op.results[0]],
                return_types=[builtin.TensorType(builtin.Float64Type(), shape=(2**n_qubits,))],
            )

            # get the index the probs MP result is returned at
            result_index = list(probs_op.results[0].uses)[0].index

            # Insert the call to the post-processing function
            postprocessing_func_call_op = func.CallOp(
                callee=builtin.SymbolRefAttr(postprocessing_func_op.sym_name),
                arguments=[self.call_op.results[result_index]],
                return_types=[builtin.TensorType(builtin.Float64Type(), shape=(2**n_qubits,))],
            )
            rewriter.insert_op(
                postprocessing_func_call_op, insertion_point=InsertPoint.after(self.call_op)
            )

            # update the returns of the QNode (to return raw samples) and the
            # outer function (to return post-processed values)
            self.update_returns(result_index, sample_op, postprocessing_func_call_op, rewriter)

            # delete now unused probs_op
            rewriter.erase_op(probs_op)


class CountsPattern(MeasurementsFromSamplesPattern):
    """A rewrite pattern for the ``measurements_from_samples`` transform that matches and rewrites
    ``qml.counts()`` operations.

    Currently there is no plan to support ``qml.counts()`` for this transform. It is included for
    completeness and to notify users that workloads containing ``counts`` measurement processes are
    not supported with the measurements-from-samples transform.

    Args:
        shots (int): The number of shots (e.g. as retrieved from the DeviceInitOp).
    """

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, counts_op: quantum.CountsOp, rewriter: PatternRewriter, /):
        """Match and rewrite for quantum.CountsOp."""
        raise NotImplementedError("qml.counts() operations are not supported.")


class StatePattern(MeasurementsFromSamplesPattern):
    """A rewrite pattern for the ``measurements_from_samples`` transform that matches and rewrites
    ``qml.state()`` operations.

    It is not possible to recover a quantum state from samples; this pattern is included for
    completeness and to notify users that workloads containing ``state`` measurement processes are
    not supported with the measurements-from-samples transform.

    Args:
        shots (int): The number of shots (e.g. as retrieved from the DeviceInitOp).
    """

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, state_op: quantum.StateOp, rewriter: PatternRewriter, /):
        """Match and rewrite for quantum.StateOp."""
        raise NotImplementedError("qml.state() operations are not supported.")


def get_shots(quantum_node: func.FuncOp) -> int:
    """Get the shots for a quantum.node. Extracts shots from the device and validates that shots
    are static, and a non-zero integer.

    This function is meant to act on a FuncOp with the `quantum.node` attribute, which should only
    contain a single quantum.DeviceInitOp op.

        Args:
        quantum_node (func.FuncOp): The quantum.node FuncOp containing the quantum.DeviceInitOp.

    Returns:
        int: The number of shots.

    Raises:
        CompileError: If `quantum_node` does not contain a quantum.DeviceInitOp.
        CompileError: If its not possible to extract a static constant from the
            SSAValue for the shots
        ValueError: if the extracted shots are zero

    """
    device_op = None

    for op in quantum_node.body.walk():
        if isinstance(op, quantum.DeviceInitOp):
            device_op = op
            break

    if device_op is None:
        raise CompileError(
            "Cannot get number of shots; the module does not contain a quantum.DeviceInitOp"
        )

    # The number of shots is passed as an SSA value operand to the DeviceInitOp
    shots_operand = device_op.shots
    shots = get_constant_from_ssa(shots_operand)
    if shots is None:
        raise CompileError(
            "Cannot get number of shots. Note that using a dynamic number of shots is not "
            "supported with measurements-from-samples."
        )

    assert isinstance(shots, int), "Expected `shots` to be an integer"
    if shots == 0:
        raise ValueError("The measurements_from_samples pass requires non-zero shots")
    return shots


@xdsl_module
@jax.jit
def _postprocessing_expval(samples):
    """Post-processing to recover the expectation value from the given `samples` array for each
    requested `column` in the array.

    This function assumes that the samples are in the computational basis (0s and 1s) and that the
    observable operand of the expectation value has eigenvalues +1 and -1.

    Args:
        samples (jax.core.ShapedArray): Array of samples, with shape (shots, wires).
        column (int, jax.core.ShapedArray): Column index (or indices) of the `samples` array over
            which the expectation value is computed.

    Returns:
        jax.core.ShapedArray: The expectation value for each requested column.
    """
    return jnp.mean(1.0 - 2.0 * samples[:, 0], axis=0)


@xdsl_module
@jax.jit
def _postprocessing_var(samples):
    """Post-processing to recover the variance from the given `samples` array for each requested
    `column` in the array.

    This function assumes that the samples are in the computational basis (0s and 1s) and that the
    observable operand of the variance has eigenvalues +1 and -1.

    Args:
        samples (jax.core.ShapedArray): Array of samples, with shape (shots, wires).
        column (int, jax.core.ShapedArray): Column index (or indices) of the `samples` array over
            which the variance is computed.

    Returns:
        jax.core.ShapedArray: The variance for each requested column.
    """
    return jnp.var(1.0 - 2.0 * samples[:, 0], axis=0)


@xdsl_module
@jax.jit
def _postprocessing_probs(samples):
    """Post-processing to recover the probability values from the given `samples` array.

    This function assumes that the samples are in the computational basis (0s and 1s).

    Args:
        samples (jax.core.ShapedArray): Array of samples, with shape (shots, wires).
    """
    n_samples = samples.shape[0]
    n_wires = samples.shape[1]

    # Convert samples from a list of 0, 1 integers to base 10 representation
    powers_of_two = 2 ** jnp.arange(n_wires)[::-1]
    indices = samples @ powers_of_two
    dim = 2**n_wires

    # This block is effectively equivalent to `jnp.bincount(indices.astype(int), length=dim)`.
    # However, we are currently not able to use jnp.bincount with Catalyst because after lowering,
    # it contains a stablehlo.scatter op with <indices_are_sorted = false, unique_indices = false>,
    # which we currently do not support.
    # If Catalyst PR https://github.com/PennyLaneAI/catalyst/pull/1849 is merged, then we should be
    # able to use bincount.
    init_counts = jnp.zeros(dim, dtype=int)

    def body_fun(i, counts):
        idx = indices[i].astype(int)
        return counts.at[idx].add(1)

    counts = jax.lax.fori_loop(0, n_samples, body_fun, init_counts)

    return counts / n_samples
