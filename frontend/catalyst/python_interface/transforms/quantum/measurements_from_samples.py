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

  * Only measurements in the computational basis (or where the observable is a Pauli Z op) are
    currently supported; for arbitrary observables we require an equivalent compilation pass of the
    diagonalize_measurements transform.
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
from pennylane import math
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



MEASUREMENT_PROCESS_TYPES = (
    quantum.CountsOp,
    quantum.ExpvalOp, 
    quantum.ProbsOp, 
    quantum.SampleOp, 
    quantum.StateOp,
    quantum.VarianceOp, 
    )

@dataclass(frozen=True)
class MeasurementsFromSamplesPass(passes.ModulePass):
    """Pass that replaces all terminal measurements in a program with a single
    :func:`pennylane.sample` measurement, and adds postprocessing instructions to recover the
    original measurement.
    """

    name = "measurements-from-samples"

    def apply(self, _ctx: context.Context, op: builtin.ModuleOp) -> None:
        """Apply the measurements-from-samples pass."""
        shots = _get_static_shots_value_from_first_device_op(op)

        # diagonalize measurements before converting to samples
        DiagonalizeFinalMeasurementsPass().apply(_ctx, op)

        greedy_applier = pattern_rewriter.GreedyRewritePatternApplier(
            [
                # ExpvalAndVarPattern(shots),
                # ProbsPattern(shots),
                # CountsPattern(shots),
                # StatePattern(shots),
                NewMeasurementsFromSamplesPattern(shots)
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

    def __init__(self, shots: int):
        super().__init__()
        assert isinstance(
            shots, int
        ), f"Expected `shots` to be an integer value but got {type(shots).__name__}"
        if shots == 0:
            raise ValueError("The measurements_from_samples pass requires non-zero shots")
        self._shots = shots

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

        if isinstance(observable_op, quantum.HamiltonianOp):
            raise CompileError("Encountered a quantum.HamiltonianOp while applying `measurements_from_samples`. This is not supported with Catalyst. " \
            "Please apply `qml.transforms.split_non_commuting` prior to `measurements_from_samples` to split the Hamiltonian into separate terms.")

        supported_obs_types = (
            quantum.NamedObsOp,
            quantum.TensorOp,
            quantum.MCMObsOp,
        )
        if not isinstance(observable_op, supported_obs_types):
            raise NotImplementedError(
                "Supported observable types for measurements-from-samples are quantum.NamedObsOp, "
                "quantum.TensorOp, and quantum.MCMObsOp, but received "
                f"{type(op).__name__}"
            )

        return observable_op

    @classmethod
    def _get_observable_qubits(
        cls, op: quantum.NamedObsOp | quantum.TensorOp | quantum.HamiltonianOp | quantum.MCMObsOp
    ):
        """Validate the observable op. It must be a NambedObsOp in the Z basis, a composite of 
        NamedObsOps in the Z basis, or an MCMObsOp"""
        
        if isinstance(op, quantum.NamedObsOp):
            if op.type.data != "PauliZ":
                raise NotImplementedError(
                    "Expected all observables to be diagonalized before application of rewrite" 
                    f"pattern, but received '{op.type.data}'"
                )
            
            in_qubits = [
                op.operands[0],
            ]

        elif isinstance(op, quantum.TensorOp):
            all_obs = [obs.owner for obs in op.operands]

            for obs in all_obs:
                if obs.type.data != "PauliZ":
                    raise NotImplementedError(
                        "Expected all observables to be diagonalized before application of" 
                        f"rewrite pattern, but received '{op.type.data}'"
                    )

            in_qubits = [obs.operands[0] for obs in all_obs]

        elif isinstance(op, quantum.MCMObsOp):
            in_qubits = [
                op.operands[0],
            ]

        else:
            raise NotImplementedError(
                f"Supported observable types for measurements-from-samples are quantum.NamedObsOp, quantum.TensorOp, and quantum.MCMObsOp, but received {type(op).__name__}"
            )
        
        return in_qubits

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

    @staticmethod
    def create_compbasis_op(
        in_qubits: ir.SSAValue, ref_op: ir.Operation, rewriter: PatternRewriter
    ) -> quantum.ComputationalBasisOp:
        """Create and insert a computational-basis op (quantum.ComputationalBasisOp).

        The computation-basis op uses `in_qubit` as its input operand. It is inserted *before* the
        given reference operation, `ref_op`, using the supplied `rewriter`.

        Args:
            in_qubits (List[SSAValue]): The SSA value used as input to the computational-basis op.
            ref_op (Operation): The reference op before which the quantum.ComputationalBasisOp is
                inserted.
            rewriter (PatternRewriter): The xDSL pattern rewriter.

        Returns:
            quantum.ComputationalBasisOp: The inserted computation-basis op.
        """
        for qubit in in_qubits:
            assert isinstance(qubit, ir.SSAValue) and isinstance(qubit.type, quantum.QubitType), (
                f"Expected `in_qubits` to be a list of SSAValue with type quantum.QubitType, but got "
                f"{type(qubit).__name__}"
            )

        # The input operands are [[qubit, ...], qreg]
        compbasis_op = quantum.ComputationalBasisOp(
            operands=[in_qubits, None], result_types=[quantum.ObservableType()]
        )

        return compbasis_op

    @staticmethod
    def create_sample_op(
        compbasis_op: quantum.ComputationalBasisOp,
        shots: int,
        sample_dim: int,
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
            sample_dim (int): Dimension of each sample (to set the shape of the sample op returned 
                array). 1 for MCMs, otherwise equal to the number of qubits on the observable.
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
            result_types=[builtin.TensorType(builtin.Float64Type(), [shots, sample_dim])],
        )

        return sample_op


class NewMeasurementsFromSamplesPattern(MeasurementsFromSamplesPattern):
    """RewritePattern for making each Qnode intro a classical function that 
    calls the QNode and performs post-processing."""

    def __init__(self, shots):
        super().__init__(shots)

        self.qnode: func.FuncOp = None
        self.original_qnode: func.FuncOp = None
        self.call_op: func.CallOp = None

    def match_and_rewrite(
        self, func_op: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter, /
    ):
        """Transform a quantum function (qnode) to ..."""
        if "qnode" not in func_op.attributes:
            return
        
        self.original_qnode = func_op
        self._wrap_qnode_in_classical_post_processing(rewriter)

        measurement_processes = [op for op in self.qnode.body.walk() if isinstance(op, MEASUREMENT_PROCESS_TYPES)]
        for mp in measurement_processes:
            if isinstance(mp, quantum.ExpvalOp | quantum.VarianceOp):
                self.expval_and_var_to_samples(mp, rewriter)
            elif isinstance(mp, quantum.ProbsOp):
                self.probs_to_samples(mp, rewriter)
            elif isinstance(mp, quantum.SampleOp):
                pass     
            elif isinstance(mp, quantum.CountsOp):
                raise NotImplementedError("qml.counts() is not implemented with measurements_from_samples.")
            elif isinstance(mp, quantum.StateOp):
                raise ValueError("qml.state() is incompatible with shots and is not supported supported with measurements_from_samples.")  

    def _get_parent_module(self, op: func.FuncOp) -> builtin.ModuleOp:
        """Get the first ancestral builtin.ModuleOp op of a given func.func op."""
        while (op := op.parent_op()) and not isinstance(op, builtin.ModuleOp):
            pass  # pragma: no cover
        if op is None:
            raise RuntimeError(  # pragma: no cover
                "The given qnode func is not nested within a builtin.module. Please ensure the "
                "qnode func is defined in a builtin.module."
            )

        return op

    def _wrap_qnode_in_classical_post_processing(self, rewriter):
        """Clones some qnode FuncOp (circ_name) as circ_name.from_samples 
        and inserts it, then replaces circ_name with a FuncOp of the same name 
        that calls circ_name.from_samples."""

        # this seems circuitous, why not just rename the QNode and then insert an 
        # empty function right before it? Something to come back to when tidying up

        # add a clone of the "qnode" FuncOp named original_name.from_samples to the module
        # we keep the original qnode unmodified for generating post-processing functions later
        # ToDo: if we don't need to do that after all, we could just rename the QNode circ.from_samples and insert an empty FuncOp named circ instead
        module = self._get_parent_module(self.original_qnode)
        qnode = self.original_qnode.clone()
        qnode_name = qnode.sym_name.data + ".from_samples"
        qnode.sym_name = builtin.StringAttr(qnode_name)
        rewriter.insert_op(qnode, InsertPoint.at_end(module.body.block))

        # create an empty function and replace the original "qnode" FuncOp
        func_type = self.original_qnode.function_type
        outer_fn = func.FuncOp(
            name=self.original_qnode.sym_name.data, 
            function_type=func_type, 
            visibility="public"
        )
        rewriter.replace_op(self.original_qnode, outer_fn)

        # call the cloned QNode inside the new FuncOp
        call_args = outer_fn.body.block.args
        result_types = func_type.outputs.data
        call_op = func.CallOp(qnode_name, call_args, result_types)
        outer_fn.body.block.add_op(call_op)

        # add a return op in the new FuncOp returning QNode results
        qnode_call_results = call_op.results
        return_op = func.ReturnOp(*qnode_call_results)
        outer_fn.body.block.add_op(return_op)

        self.qnode = qnode
        self.call_op = call_op

    # ToDo: clean up this mess!!
    def expval_and_var_to_samples(
        self, mp: quantum.ExpvalOp | quantum.VarianceOp, rewriter: PatternRewriter, /
    ):
        """Match and rewrite for quantum.ExpvalOp and quantum.VarOp."""

        observable_op = self.get_observable_op(mp)
        if isinstance(observable_op, quantum.MCMObsOp):

            sample_dim = 1

            # this is the part you need to figure out how to do correctly tomorrow
            sample_op = quantum.SampleOp(
                operands=[observable_op, None, None],
                result_types=[builtin.TensorType(builtin.Float64Type(), [self._shots, sample_dim])],
                )
            rewriter.insert_op(sample_op, insertion_point=InsertPoint.after(observable_op))
        
        else:
            in_qubits = self._get_observable_qubits(observable_op)
        
            # insert compbasis_op and sample op
            compbasis_op = self.create_compbasis_op(in_qubits, observable_op, rewriter)
            rewriter.insert_op(compbasis_op, insertion_point=InsertPoint.before(observable_op))

            sample_dim = len(in_qubits)
            sample_op = self.create_sample_op(compbasis_op, self._shots, sample_dim, rewriter)
            rewriter.insert_op(sample_op, insertion_point=InsertPoint.after(compbasis_op))

        postprocessing_func_op = self.insert_postprocessing(mp, sample_dim, observable_op)

        # get the tensor_op the original MP result is passed to
        # from its uses get the index this result is returned at
        tensor_op = list(mp.results[0].uses)[0].operation
        assert isinstance(
            tensor_op, tensor.FromElementsOp
        ), f"Expected to find a tensor.from_elements op, but got {type(tensor_op).__name__}"
        mp_index = list(tensor_op.results[0].uses)[0].index

        # Insert the call to the post-processing function
        postprocessing_func_call_op = func.CallOp(
            callee=builtin.FlatSymbolRefAttr(postprocessing_func_op.sym_name),
            arguments=[self.call_op.results[mp_index]],
            return_types=[builtin.TensorType(builtin.Float64Type(), shape=())],
        )
        rewriter.insert_op(postprocessing_func_call_op, insertion_point=InsertPoint.after(self.call_op))

        # update the returns of the QNode (to return the raw samples) and the 
        # outer function (to return the post-processed values)
        self.update_returns(mp_index, sample_op, postprocessing_func_call_op, rewriter)

        # delete now ununsed obs --> mp --> tensor chain
        rewriter.erase_op(tensor_op)
        rewriter.erase_op(mp)
        if not isinstance(observable_op, quantum.MCMObsOp):
            rewriter.erase_op(observable_op)

    def probs_to_samples(self, probs_op: quantum.ProbsOp, rewriter: PatternRewriter):
        """Match and rewrite for quantum.ProbsOp."""
        compbasis_op = probs_op.operands[0].owner

        n_qubits = None
        if compbasis_op.qreg is not None:
            n_qubits = self.get_n_qubits_from_qreg(compbasis_op.qreg)

        elif not compbasis_op.qubits == ():
            n_qubits = len(compbasis_op.qubits)

        assert (
            n_qubits is not None
        ), "Unable to determine number of qubits from quantum.compbasis op"

        sample_op = self.create_sample_op(compbasis_op, self._shots, n_qubits, rewriter)
        rewriter.insert_op(sample_op, insertion_point=InsertPoint.after(compbasis_op))

        postprocessing_func_op = self.insert_postprocessing(probs_op, n_qubits, None)

        # get the index the probs MP result is returned at
        result_index = list(probs_op.results[0].uses)[0].index

        # Insert the call to the post-processing function
        postprocessing_func_call_op = func.CallOp(
            callee=builtin.FlatSymbolRefAttr(postprocessing_func_op.sym_name),
            arguments=[self.call_op.results[result_index]],
            return_types=[builtin.TensorType(builtin.Float64Type(), shape=(2**n_qubits,))],
        )
        rewriter.insert_op(postprocessing_func_call_op, insertion_point=InsertPoint.after(self.call_op))

        # update the returns of the QNode (to return raw samples) and the 
        # outer function (to return post-processed values)
        self.update_returns(result_index, sample_op, postprocessing_func_call_op, rewriter)

        # delete now ununsed probs_op
        rewriter.erase_op(probs_op)
        
    def update_returns(self, mp_index, sample_op, postprocessing_func_call_op, rewriter):
        """Update the return structure so that the qnode returns the output of sample 
        instead of the original output, and the outer function returns the output of 
        post-processing instead of directly returning the output of calling the qnode.

        Also update result shapes in the CallOp for the Qnode call to reflect this change, 
        since this is not updated when modifying the ReturnOps.

        This function updates the self.qnode and self.call_op attributes on the Pattern.

        Args:
            mp_index (int): The index of the measurement process to be updated. The 
                relevant function returns will be updated at this index.
            sample_op (quantum.SampleOp): The operation whose results the Qnode should return.
            postprocessing_func_call_op (func.CallOp): The postprocessing CallOp whose results 
                the outer function should return.
            rewriter (PatternRewriter): The xDSL pattern rewriter.
        """
        # update the qnode to return the result of the SampleOp directly
        return_op = self.qnode.get_return_op()
        return_op.operands[mp_index] = sample_op.results[0]
        rewriter.notify_op_modified(return_op)

        # update the outer function return to return postprocessed data instead of the raw data
        final_return = [use.operation for use in list(self.call_op.results[mp_index].uses) if isinstance(use.operation, func.ReturnOp)][0]
        final_return.operands[mp_index] = postprocessing_func_call_op.results[0]
        rewriter.notify_op_modified(final_return)

        # update the call op to correctly reflect the new shape of the returned data
        self.qnode.update_function_type()
        result_types = self.qnode.function_type.outputs.data
        new_call_op = func.CallOp(self.call_op.callee, self.call_op.arguments, result_types)
        rewriter.replace_op(self.call_op, new_call_op)
        self.call_op = new_call_op

    def insert_postprocessing(self, mp, n_qubits, observable_op):
        # Insert the post-processing function into current module or get handle to it if already
        # inserted
        match mp:
            case quantum.ExpvalOp():
                postprocessing_func_name = (
                    f"expval_from_samples.tensor.{self._shots}x{n_qubits}xf64"
                )
                postprocessing_jit_func = create_postprocessing_obs(observable_op, n_qubits, jnp.mean)
            case quantum.VarianceOp():
                postprocessing_func_name = (
                    f"var_from_samples.tensor.{self._shots}x{n_qubits}xf64"
                )
                postprocessing_jit_func = create_postprocessing_obs(observable_op, n_qubits, jnp.var)
            case quantum.ProbsOp():
                postprocessing_func_name = (
                    f"probs_from_samples.tensor.{self._shots}x{n_qubits}xf64"
                )
                postprocessing_jit_func = _postprocessing_probs
            case _:
                assert False, (
                    f"Expected a quantum.ExpvalOp or quantum.VarianceOp, but got "
                    f"{type(mp).__name__}"
                )

        postprocessing_func_op = self.get_postprocessing_func_op_from_block_by_name(
            mp.parent_op().parent, postprocessing_func_name
        )

        if postprocessing_func_op is None:
            # TODO: Do we have to set the shape of the samples array statically here? Or can the
            # shape (shots, wire) be dynamic and given as SSA values?
            postprocessing_module = postprocessing_jit_func(
                jax.core.ShapedArray([self._shots, n_qubits], float)
            )

            postprocessing_func_op = self.get_postprocessing_funcs_from_module_and_insert(
                postprocessing_module, mp, postprocessing_func_name
            )       

        return postprocessing_func_op 

def create_postprocessing_obs(obs, num_wires, math_op):
    
    if isinstance(obs, quantum.NamedObsOp):
        eigvals = jnp.array([1, -1])
        
    elif isinstance(obs, quantum.TensorOp):
        eigvals = []
        for op in range(num_wires):
            eigvals.append(
                math.expand_vector(jnp.array([1, -1]), [op], range(num_wires))
            )
        eigvals = jnp.prod(jnp.asarray(eigvals), axis=0)

    elif isinstance(obs, quantum.MCMObsOp):
        eigvals = jnp.array([0, 1])
    
    else:
        raise CompileError(f"Tried to get eigenvalues function but encountered unknown observable {obs}")


    @xdsl_module
    @jax.jit
    def _postprocessing(samples):
        """Post-processing to recover the expectation value from the given `samples` array.

        This function assumes that the samples are in the computational basis (0s and 1s) and that the
        observable operand of the expectation value has eigenvalues +1 and -1.

        Args:
            samples (jax.core.ShapedArray): Array of samples, with shape (shots, wires).
            column (int, jax.core.ShapedArray): Column index (or indices) of the `samples` array over
                which the expectation value is computed.

        Returns:
            jax.core.ShapedArray: The expectation value for each requested column.
        """
        powers_of_two = 2 ** jnp.arange(num_wires)[::-1]
        indices = samples @ powers_of_two
        eigval_samples = jnp.take(eigvals, indices.astype(int))

        return math_op(eigval_samples, axis=0)

    return _postprocessing
        
# ToDo: we should probably get shots one "qnode" at a time instead of getting them from 
# the module and then passing them to the pattern. Right now there's no implementation
# that adds different shots to different "qnode"s, but its more "we haven't added one yet"
# than its "against the rules"
def _get_static_shots_value_from_first_device_op(module: builtin.ModuleOp) -> int:
    """Returns the number of shots as a static (i.e. known at compile time) integer value from the
    first instance of a device-initialization op (quantum.DeviceInitOp) found in `module`.

    If `module` contains multiple quantum.DeviceInitOp ops, only the number of shots from the
    *first* instance is used, and the others are ignored.

    This function expects the number of shots to be an SSA value given as an operand to the
    quantum.DeviceInitOp op. It also assumes that the number of shots is static, retrieving it from
    the 'value' attribute of its corresponding constant op.

    Args:
        module (builtin.ModuleOp): The MLIR module containing the quantum.DeviceInitOp.

    Returns:
        int: The number of shots.

    Raises:
        CompileError: If `module` does not contain a quantum.DeviceInitOp.
    """
    device_op = None

    for op in module.body.walk():
        if isinstance(op, quantum.DeviceInitOp):
            device_op = op
            break

    if device_op is None:
        raise CompileError(
            "Cannot get number of shots; the module does not contain a quantum.DeviceInitOp"
        )

    # The number of shots is passed as an SSA value operand to the DeviceInitOp
    shots_operand = device_op.shots
    shots_extract_op = shots_operand.owner

    if isinstance(shots_extract_op, tensor.ExtractOp):
        shots_constant_op = shots_extract_op.operands[0].owner
        shots_value_attribute: builtin.DenseIntOrFPElementsAttr = shots_constant_op.properties.get(
            "value"
        )
        if shots_value_attribute is None:
            raise ValueError("Cannot get number of shots; the constant op has no 'value' attribute")

        shots_int_values = shots_value_attribute.get_values()
        if len(shots_int_values) != 1:
            raise ValueError(f"Expected a single shots value, got {len(shots_int_values)}")

        return shots_int_values[0]

    if isinstance(shots_extract_op, arith.ConstantOp):
        shots_value_attribute: builtin.IntAttr = shots_extract_op.properties.get("value")
        return shots_value_attribute.value.data

    raise ValueError(
        f"Expected owner of shots operand to be a tensor.ExtractOp or arith.ConstantOp but got "
        f"{type(shots_extract_op).__name__}"
    )


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
    counts = jnp.zeros(dim, dtype=int)
    for i in indices.astype(int):
        counts = counts.at[i].add(1)

    return counts / n_samples
