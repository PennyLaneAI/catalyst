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

"""Implementation of the Tree-Traversal MCM simulation method as an xDSL transform in Catalyst."""

from dataclasses import dataclass, field
from itertools import chain
from typing import Type, TypeVar

import jax
import pennylane as qml
from pennylane.compiler.python_compiler import compiler_transform
from pennylane.compiler.python_compiler.dialects import quantum
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf, tensor
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import BlockInsertPoint, InsertPoint

from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

T = TypeVar("T")


def get_parent_of_type(op: Operation, kind: Type[T]) -> T | None:
    """Walk up the parent tree until an op of the specified type is found."""

    while (op := op.parent_op()) and not isinstance(op, kind):
        pass

    return op


@dataclass
class ProgramSegment:
    """A program segment and associated data."""

    ops: list[Operation] = field(default_factory=list)
    mcm: quantum.MeasureOp = None
    reg_in: SSAValue = None
    reg_out: SSAValue = None
    inputs: set[SSAValue] = None
    outputs: set[SSAValue] = None
    fun: func.FuncOp = None


class TreeTraversal(RewritePattern):

    def __init__(self):
        self.module: builtin.ModuleOp = None
        self.ttOp: func.FuncOp = None
        self.quantum_segments: list[ProgramSegment] = []

    @op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: PatternRewriter):
        """Transform a quantum function (qnode) to perform tree-traversal simulation."""
        if not "qnode" in funcOp.attributes:
            return

        self.module = get_parent_of_type(funcOp, builtin.ModuleOp)
        assert self.module is not None, "got orphaned qnode function"

        # Start with creating a new QNode function that will perform the tree traversal simulation.
        # We prep the original QNode by ensuring measure boundaries are also register boundaries.
        cloned_func = self.simplify_quantum_io(funcOp, rewriter)

        self.setup_traversal_function(cloned_func, rewriter)

        self.split_traversal_segments(cloned_func, rewriter)

        self.initialize_data_structures(rewriter)

        self.generate_traversal_code(rewriter)

        self.finalize_traversal_function(rewriter)

    def simplify_quantum_io(self, funcOp: func.FuncOp, rewriter: PatternRewriter) -> func.FuncOp:
        """In order to facilitate quantum value handling, we will reinsert all extracted qubits
        into the register at the end of each segment, and only allow the register as quantum
        input and output of segments.

        This pass guarantees that each measure op is preceded by exactly 1 ExtractOp, and whose
        input is a fully "reassembled" register ready to be passed across difficult program
        boundaries (e.g. control flow, function calls).
        """
        cloned_fun = funcOp.clone()
        cloned_fun.sym_name = builtin.StringAttr(funcOp.sym_name.data + ".simple_io")
        rewriter.insert_op(cloned_fun, InsertPoint.after(funcOp))

        current_reg = None
        qubit_to_reg_idx = {}
        for op in cloned_fun.body.ops:
            match op:
                case quantum.AllocOp():
                    current_reg = op.qreg
                case quantum.ExtractOp():
                    qubit_to_reg_idx[op.qubit] = op.idx if op.idx else op.idx_attr
                    # update register since it might have changed
                    op.operands = (current_reg, op.idx)
                case quantum.CustomOp():
                    for i, qb in enumerate(chain(op.in_qubits, op.in_ctrl_qubits)):
                        qubit_to_reg_idx[op.results[i]] = qubit_to_reg_idx[qb]
                        del qubit_to_reg_idx[qb]
                case quantum.InsertOp():
                    assert qubit_to_reg_idx[op.qubit] is op.idx_attr if op.idx_attr else True
                    del qubit_to_reg_idx[op.qubit]
                    # update register since it might have changed
                    op.operands = (current_reg, op.idx, op.qubit)
                    current_reg = op.out_qreg
                case quantum.MeasureOp():
                    # find the qubit to be measured and its index
                    mcm_qubit, mcm_idx = next(
                        ((qb, idx) for qb, idx in qubit_to_reg_idx.items() if qb == op.in_qubit),
                        (None, None),
                    )

                    if mcm_qubit is None:
                        raise RuntimeError(
                            f"Could not find qubit {op.in_qubit} in register mapping"
                        )

                    # create a register boundary before the measure
                    rewriter.insertion_point = InsertPoint.before(op)
                    for qb, idx in qubit_to_reg_idx.items():
                        insertOp = quantum.InsertOp(current_reg, idx, qb)
                        rewriter.insert(insertOp)
                        current_reg = insertOp.out_qreg

                    # extract the qubit that will be measured from the register
                    extractOp = quantum.ExtractOp(current_reg, mcm_idx)

                    # mark the extract op as a measure boundary
                    extractOp.attributes["meas_boundary"] = builtin.IntegerAttr(
                        1, value_type=builtin.IntegerType(1)
                    )
                    rewriter.insert(extractOp)

                    # update the measure operation to use the extracted qubit
                    op.operands = (extractOp.qubit,)

                    # update the map to process the measure qubit - remove the old qubit
                    del qubit_to_reg_idx[mcm_qubit]

                    # restore qubit values from before the register boundary
                    rewriter.insertion_point = InsertPoint.after(op)
                    for qb, idx in list(qubit_to_reg_idx.items()):
                        extractOp = quantum.ExtractOp(current_reg, idx)
                        rewriter.insert(extractOp)
                        rewriter.replace_all_uses_with(qb, extractOp.qubit)
                        qubit_to_reg_idx[extractOp.qubit] = idx
                        del qubit_to_reg_idx[qb]

                    qubit_to_reg_idx[op.out_qubit] = mcm_idx

                    # insert an extract op as a way to store the mcm qubit index for later
                    # extractOp = quantum.ExtractOp(current_reg, mcm_idx)
                    # rewriter.insert_op(extractOp, InsertPoint.before(op))
                    # op.operands = (extractOp.qubit,)
                    # insertOp = quantum.InsertOp(current_reg, mcm_idx, op.out_qubit)

                    # current_reg.replace_by_if(
                    #     insertOp.out_qreg, lambda use: use.operation not in (extractOp, insertOp)
                    # )
                    # current_reg = insertOp.out_qreg

                    # rewriter.insert_op(insertOp, InsertPoint.after(op))
                case _:
                    # TODO: handle other ops which might use qubits
                    pass

        return cloned_fun

    def setup_traversal_function(self, funcOp: func.FuncOp, rewriter: RewritePattern):
        """Setup a clone of the original QNode function, which will instead perform TT."""

        ttOp = funcOp.clone_without_regions()
        ttOp.sym_name = builtin.StringAttr(funcOp.sym_name.data + ".tree_traversal")
        rewriter.create_block(BlockInsertPoint.at_start(ttOp.body), ttOp.function_type.inputs)
        rewriter.insert_op(ttOp, InsertPoint.at_end(self.module.body.block))

        self.ttOp = ttOp

    def finalize_traversal_function(self, rewriter: RewritePattern):
        """Complete the function and ensure it's correctly formed, e.g. returning proper results."""
        rewriter.insertion_point = InsertPoint.at_end(self.ttOp.body.block)

        # For now we return empty values from the traversal function to satisfy the verifier.
        # TODO: Return the original function results computed via TT.
        result_vals = []
        for resType in self.ttOp.function_type.outputs:
            assert isinstance(resType, builtin.TensorType)
            result = tensor.EmptyOp((), tensor_type=resType)
            result_vals.append(rewriter.insert(result))

        rewriter.insert(func.ReturnOp(*result_vals))

    def split_traversal_segments(self, funcOp: func.FuncOp, rewriter: PatternRewriter):
        """Split the quantum function into segments separated by measure operations.

        Due to the pre-processing of the QNode, we can assume the register is the only
        quantum value going between segments.
        """
        rewriter.insertion_point = InsertPoint.at_start(self.ttOp.body.block)

        # Ideally try to iterate over the function only once.
        op_iter = funcOp.body.walk()

        # Skip to the start of the first simulation segment.
        value_mapper = {}
        while (op := next(op_iter, None)) and not isinstance(op, quantum.AllocOp):
            rewriter.insert(op.clone(value_mapper))
        assert op is not None, "didn't find an alloc op"
        cloned_alloc = rewriter.insert(op.clone(value_mapper))

        # Split ops into segments divided by measurements.
        quantum_segments = [ProgramSegment(reg_in=op.qreg)]
        while (op := next(op_iter, None)) and not isinstance(op, quantum.DeallocOp):
            if hasattr(op, "attributes") and "meas_boundary" in op.attributes:
                del op.attributes["meas_boundary"]
                last_op = quantum_segments[-1].ops[-1]
                assert isinstance(last_op, quantum.InsertOp)
                quantum_segments[-1].reg_out = last_op.out_qreg
                quantum_segments.append(ProgramSegment(reg_in=last_op.out_qreg))
            elif isinstance(op, quantum.MeasureOp):
                quantum_segments[-1].mcm = op

            quantum_segments[-1].ops.append(op)
        assert op is not None, "didn't find a dealloc op"
        quantum_segments[-1].reg_out = op.qreg
        self.quantum_segments = quantum_segments

        # Go through the rest of the function to initialize the missing input values set.
        terminal_segment = ProgramSegment(ops=[op])  # dealloc op
        while op := next(op_iter, None):
            terminal_segment.ops.append(op)
        # TODO: do something with the terminal program segment (copy into traversal function?)

        # Generate new functions for each segment separated by a measure op.
        # We traverse them bottom up first to correctly determine the I/O of each segment.
        missing_input_values = set()
        self.populate_segment_io(terminal_segment, missing_input_values)
        for segment in reversed(quantum_segments):
            self.populate_segment_io(segment, missing_input_values)  # inplace

        # contains the inputs to the first segment + all MCM results
        all_segment_io = [quantum_segments[0].reg_in, *missing_input_values]
        values_as_io_index = {v: k for k, v in enumerate(all_segment_io)}
        for idx, segment in enumerate(quantum_segments):
            segment.fun = self.clone_ops_into_func(segment, idx, rewriter)
            values_as_io_index.update(
                (x, i + len(all_segment_io)) for i, x in enumerate(segment.outputs)
            )
            all_segment_io.extend(segment.outputs)

        self.generate_function_table(all_segment_io, values_as_io_index, rewriter)

        # store some useful values for later
        self.alloc_op = cloned_alloc
        self.all_segment_io = all_segment_io
        self.values_as_io_index = values_as_io_index

    def generate_function_table(
        self,
        all_segment_io: list[SSAValue],
        values_as_io_index: dict[SSAValue, int],
        rewriter: PatternRewriter,
    ):
        """Create a program segment dispatcher via a large function table switch statement.

        The dispatcher needs the entire segment IO as arguments/results in order to properly
        wire the input & output of each segment together, since the dispatcher is invoked
        dynamically but call arguments & results are static SSA values. An alternative would be
        to pass around segment IO via memory, but this requires additional IR operations & types
        not available in builtin dialects.
        """

        # function op
        all_io_types = [val.type for val in all_segment_io]
        fun_type = builtin.FunctionType.from_lists(
            # function id, branch type, io types
            [builtin.IndexType(), builtin.IndexType(), *all_io_types],
            all_io_types,
        )
        funcTableOp = func.FuncOp("segment_table", fun_type)
        rewriter.insert_op(funcTableOp, InsertPoint.at_end(self.module.body.block))

        # function body
        fun_index = funcTableOp.args[0]
        branch_type = funcTableOp.args[1]
        io_args = funcTableOp.args[2:]

        cases = builtin.DenseArrayBase.from_list(builtin.i64, range(len(self.quantum_segments)))
        switchOp = scf.IndexSwitchOp(
            fun_index,
            cases,
            Region(Block()),
            [Region(Block()) for _ in range(len(cases))],
            all_io_types,
        )
        returnOp = func.ReturnOp(*switchOp.results)

        for op in (switchOp, returnOp):
            rewriter.insert_op(op, InsertPoint.at_end(funcTableOp.body.block))

        # switch op base case
        rewriter.insert_op(scf.YieldOp(*io_args), InsertPoint.at_end(switchOp.default_region.block))

        # switch op match cases
        for case, segment in enumerate(self.quantum_segments):
            args = (
                [branch_type]  # branch_type for postselect logic
                + [io_args[0]]  # quantum register is always first
                + [io_args[values_as_io_index[value]] for value in segment.inputs]
            )
            res_types = [quantum.QuregType()] + [res.type for res in segment.outputs]
            callOp = func.CallOp(self.quantum_segments[case].fun.sym_name.data, args, res_types)

            updated_results = list(io_args)
            updated_results[0] = callOp.results[0]
            for new_res, ref in zip(callOp.results[1:], segment.outputs):
                updated_results[values_as_io_index[ref]] = new_res

            rewriter.insert_op(callOp, InsertPoint.at_end(switchOp.case_regions[case].block))

            # perform a state projection for the branch we're about to take
            # TODO: The postselect value has to be static in our dialect, so we need an if statement
            # here to convert it to static information. An alternative would be to create the
            # MCM inside the case statement deciding wether to walk left, right, or up, but there
            # we don't have access to the right qubit value of the MCM which is tied to the current
            # segment.
            # if segment.mcm:
            #     mcm_qubit_idx = list(segment.outputs).index(segment.mcm.in_qubit)
            #     mcm_qubit_in = callOp.results[mcm_qubit_idx]
            #     measureOp = quantum.MeasureOp(mcm_qubit_in, postselect=0)

            #     updated_results[values_as_io_index[segment.mcm.mres]] = measureOp.mres
            #     updated_results[values_as_io_index[segment.mcm.out_qubit]] = measureOp.out_qubit

            #     rewriter.insert_op(measureOp, InsertPoint.at_end(switchOp.case_regions[case].block))

            yieldOp = scf.YieldOp(*updated_results)
            rewriter.insert_op(yieldOp, InsertPoint.at_end(switchOp.case_regions[case].block))

    def clone_ops_into_func(self, segment: ProgramSegment, counter: int, rewriter: PatternRewriter):
        """Clone a set of ops into a new function."""
        op_list, input_vals, output_vals = segment.ops, segment.inputs, segment.outputs
        input_vals = [segment.reg_in] + input_vals
        output_vals = [segment.reg_out] + output_vals
        if not op_list:
            return

        fun_type = builtin.FunctionType.from_lists(
            [builtin.IndexType()] + [arg.type for arg in input_vals],
            [res.type for res in output_vals],
        )
        new_func = func.FuncOp(f"quantum_segment_{counter}", fun_type)

        # branch_type is args[0], actual inputs start from args[1]
        branch_type = new_func.args[0]
        value_mapper = dict(zip(input_vals, new_func.args[1:]))
        for op in op_list:
            if isinstance(op, quantum.MeasureOp):
                self.clone_measure_op_with_postselect(
                    op, branch_type, value_mapper, new_func, rewriter
                )
            else:
                new_op = op.clone(value_mapper)
                rewriter.insert_op(new_op, InsertPoint.at_end(new_func.body.block))

        returnOp = func.ReturnOp(*(value_mapper[res] for res in output_vals))
        rewriter.insert_op(returnOp, InsertPoint.at_end(new_func.body.block))

        rewriter.insert_op(new_func, InsertPoint.at_end(self.module.body.block))
        return new_func

    def clone_measure_op_with_postselect(
        self,
        measure_op: quantum.MeasureOp,
        branch_type: SSAValue,
        value_mapper: dict,
        new_func: func.FuncOp,
        rewriter: PatternRewriter,
    ):
        """Clone a MeasureOp with postselect based on branch_type."""

        c0_branch = arith.ConstantOp.from_int_and_width(0, branch_type.type)
        is_left_branch = arith.CmpiOp(branch_type, c0_branch, "eq") # branch_type == 0

        mapped_in_qubit = value_mapper[measure_op.in_qubit]
        result_types = [measure_op.mres.type, measure_op.out_qubit.type]
        if_op = scf.IfOp(is_left_branch, result_types, Region(Block()), Region(Block()))

        # True branch: postselect = 0 (left branch)
        true_block = if_op.true_region.block
        measure_op_left = quantum.MeasureOp(mapped_in_qubit, postselect=0)
        rewriter.insert_op(measure_op_left, InsertPoint.at_end(true_block))
        rewriter.insert_op(
            scf.YieldOp(measure_op_left.mres, measure_op_left.out_qubit),
            InsertPoint.at_end(true_block),
        )

        # False branch: postselect = 1 (right branch)
        false_block = if_op.false_region.block
        measure_op_right = quantum.MeasureOp(mapped_in_qubit, postselect=1)
        rewriter.insert_op(measure_op_right, InsertPoint.at_end(false_block))
        rewriter.insert_op(
            scf.YieldOp(measure_op_right.mres, measure_op_right.out_qubit),
            InsertPoint.at_end(false_block),
        )

        for op in (c0_branch, is_left_branch, if_op):
            rewriter.insert_op(op, InsertPoint.at_end(new_func.body.blocks[0]))

        value_mapper[measure_op.mres] = if_op.results[0]
        value_mapper[measure_op.out_qubit] = if_op.results[1]

    @staticmethod
    def populate_segment_io(
        segment: ProgramSegment, missing_inputs: set[SSAValue]
    ) -> tuple[set[SSAValue], set[SSAValue]]:
        """Gather SSA values that need to be passed in and out of the segment to be outlined."""
        inputs = set()
        outputs = set()

        # The segment only needs to return values produced here (i.e. in all op.results) and
        # required by segments further down (i.e. in missing_inputs).
        # The inputs are determined straightforwardly by all operands not defined in this segment.
        # TODO: We might need to be more careful with qubit/register values in the future.
        for op in reversed(segment.ops):
            inputs.update(op.operands)
            inputs.difference_update(op.results)

            outputs.update(op.results)
        outputs.intersection_update(missing_inputs)

        # Update the information used in subsequent calls.
        segment.inputs = [inp for inp in inputs if not isinstance(inp.type, quantum.QuregType)]
        segment.outputs = [out for out in outputs if not isinstance(out.type, quantum.QuregType)]

        missing_inputs.difference_update(segment.outputs)
        missing_inputs.update(segment.inputs)

    def initialize_data_structures(self, rewriter: PatternRewriter):
        """Create data structures in the IR required for the dynamic tree traversal."""
        rewriter.insertion_point = InsertPoint.at_end(self.ttOp.body.block)

        # get the qubit count
        if self.alloc_op.nqubits:
            qubit_count = arith.IndexCastOp(self.alloc_op.nqubits, builtin.IndexType())
        else:
            qubit_count = arith.ConstantOp.from_int_and_width(
                self.alloc_op.nqubits_attr.value, builtin.IndexType()
            )

        # get the tree depth (for now just the segment count)
        tree_depth = arith.ConstantOp.from_int_and_width(
            len(self.quantum_segments), builtin.IndexType()
        )

        # initialize stack variables #

        # statevector storage to allow for rollback
        c1 = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
        statevec_size = arith.ShLIOp(c1, qubit_count)
        statevec_stack_type = builtin.MemRefType(
            builtin.ComplexType(builtin.f64), (builtin.DYNAMIC_INDEX, builtin.DYNAMIC_INDEX)
        )
        statevec_stack = memref.AllocOp((tree_depth, statevec_size), (), statevec_stack_type)

        # probabilities for each branch are tracked here
        probs_stack_type = builtin.MemRefType(builtin.f64, (builtin.DYNAMIC_INDEX,))
        probs_stack = memref.AllocOp((tree_depth,), (), probs_stack_type)

        # For the current path, we track whether a node is:
        #  - unvisited: 0
        #  - visited down the left branch: 1
        #  - visited down the right branch: 2
        visited_stack_type = builtin.MemRefType(builtin.i8, (builtin.DYNAMIC_INDEX,))
        visited_stack = memref.AllocOp((tree_depth,), (), visited_stack_type)

        for op in (
            qubit_count,
            tree_depth,
            c1,
            statevec_size,
            statevec_stack,
            probs_stack,
            visited_stack,
        ):
            rewriter.insert(op)

        # store some useful values for later
        self.tree_depth = tree_depth.result
        self.statevec_size = statevec_size.result
        self.statevec_stack = statevec_stack.results[0]
        self.probs_stack = probs_stack.results[0]
        self.visited_stack = visited_stack.results[0]

    def generate_traversal_code(self, rewriter: PatternRewriter):
        """Create the traversal code of the quantum simulation tree."""
        rewriter.insertion_point = InsertPoint.at_end(self.ttOp.body.block)

        # loop instruction
        depth_init = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
        keep_traversal_init = arith.ConstantOp.from_int_and_width(1, 1)

        segment_io_types = [val.type for val in self.all_segment_io]
        segment_io_inits = self.initialize_values_from_types(segment_io_types)
        iter_arg_types = [
            depth_init.result.type,
            keep_traversal_init.result.type,
            *segment_io_types,
        ]
        iter_arg_inits = [depth_init, keep_traversal_init, *segment_io_inits]
        conditionBlock = Block(arg_types=iter_arg_types)
        bodyBlock = Block(arg_types=iter_arg_types)
        traversalOp = scf.WhileOp(iter_arg_inits, iter_arg_types, (conditionBlock,), (bodyBlock,))

        for op in (depth_init, keep_traversal_init, *segment_io_inits, traversalOp):
            if isinstance(op, Operation):  # bypass "ops" that are already SSA values
                rewriter.insert(op)

        # condition block of the while loop
        current_depth, keep_traversal = conditionBlock.args[:2]
        current_depth, keep_traversal = self.check_if_leaf(current_depth, keep_traversal, rewriter)
        segment_iter_args = conditionBlock.args[2:]

        condOp = scf.ConditionOp(keep_traversal, current_depth, keep_traversal, *segment_iter_args)

        rewriter.insert_op(condOp, InsertPoint.at_end(conditionBlock))

        # body block of the while
        current_depth, keep_traversal = bodyBlock.args[:2]
        segment_iter_args = bodyBlock.args[2:]

        node_status = memref.LoadOp.get(self.visited_stack, (current_depth,))
        casted_status = arith.IndexCastOp(node_status, builtin.IndexType())

        cases = builtin.DenseArrayBase.from_list(builtin.i64, [0, 1, 2])
        switchOp = scf.IndexSwitchOp(
            casted_status,
            cases,
            Region(Block()),
            [Region(Block()) for _ in range(len(cases))],
            (current_depth.type, keep_traversal.type, *segment_io_types),
        )

        self.process_node(
            switchOp, current_depth, casted_status, keep_traversal, segment_iter_args, rewriter
        )

        yieldOp = scf.YieldOp(*switchOp.results)

        for op in (node_status, casted_status, switchOp, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(bodyBlock))

    def initialize_values_from_types(self, types):
        """Generate dummy values for the provided types. Quantum types are treated specially and
        will make use of the quantum.AllocOp reference collected at an earlier stage."""

        # TODO: handling quantum dummy values can be tricky, let's try this for now
        qureg_stub = self.alloc_op.results[0]
        qubit_stub = quantum.ExtractOp(qureg_stub, 0)
        qubit_stub_used = False

        ops = []
        for ty in types:
            match ty:
                case builtin.IndexType() | builtin.IntegerType():
                    ops.append(arith.ConstantOp(builtin.IntegerAttr(0, ty)))
                case builtin._FloatType():
                    ops.append(arith.ConstantOp(builtin.FloatAttr(0.0, ty)))
                case builtin.ComplexType():
                    assert False, "Complex type unsupported"
                case builtin.TensorType():
                    ops.append(tensor.EmptyOp((), ty))  # assume no dynamic dim
                case builtin.MemRefType():
                    ops.append(memref.AllocaOp.get(ty))  # assume this is not called in a loop
                case quantum.QubitType():
                    if qubit_stub_used:
                        ops.append(qubit_stub.results[0])
                    else:
                        ops.append(qubit_stub)
                        qubit_stub_used = True
                case quantum.QuregType():
                    ops.append(self.alloc_op.results[0])

        return ops

    def check_if_leaf(
        self, current_depth: SSAValue, keep_traversal: SSAValue, rewriter: PatternRewriter
    ) -> tuple[SSAValue, SSAValue]:
        """Verify whether we've hit the bottom of the tree, and perform update actions."""
        assert isinstance(current_depth.owner, Block)
        assert current_depth.owner == keep_traversal.owner
        ip_backup = rewriter.insertion_point
        rewriter.insertion_point = InsertPoint.at_start(current_depth.owner)

        # if instruction
        hit_leaf = arith.CmpiOp(current_depth, self.tree_depth, "eq")
        trueBlock, falseBlock = Block(), Block()
        ifOp = scf.IfOp(
            hit_leaf, (current_depth.type, keep_traversal.type), (trueBlock,), (falseBlock,)
        )

        for op in (hit_leaf, ifOp):
            rewriter.insert(op)

        # true branch body - hit leaf, just go back up
        c1 = arith.ConstantOp.from_int_and_width(1, current_depth.type)
        updated_depth = arith.SubiOp(current_depth, c1)

        yieldOp = scf.YieldOp(updated_depth, keep_traversal)

        for op in (c1, updated_depth, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(trueBlock))

        # false branch body
        yieldOp = scf.YieldOp(current_depth, keep_traversal)
        rewriter.insert_op(yieldOp, InsertPoint.at_end(falseBlock))

        rewriter.insertion_point = ip_backup

        return ifOp.results

    def process_node(
        self,
        switchOp: scf.IndexSwitchOp,
        current_depth: SSAValue,
        current_branch: SSAValue,
        keep_traversal: SSAValue,
        segment_iter_args: list[SSAValue],
        rewriter: PatternRewriter,
    ):
        """Update data structures and effect transition from one node to the next."""
        defaultBlock = switchOp.default_region.block
        unvisitedBlock, leftVisitedBlock, rightVisitedBlock = (
            reg.block for reg in switchOp.case_regions
        )

        # handle unvisited region: need to go left
        c1 = arith.ConstantOp.from_int_and_width(1, self.visited_stack.type.element_type)
        storeOp = memref.StoreOp.get(c1, self.visited_stack, (current_depth,))

        # update depth
        c1_ = arith.ConstantOp.from_int_and_width(1, current_depth.type)
        updated_depth = arith.AddiOp(current_depth, c1_)

        # run a simulation segment
        sim_ops, callOp = self.simulate(current_depth, current_branch, segment_iter_args)

        yieldOp = scf.YieldOp(updated_depth, keep_traversal, *callOp.results)

        for op in (c1, c1_, storeOp, updated_depth, *sim_ops, callOp, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(unvisitedBlock))

        # handle left visited region: need to go right
        # First check if we're at root - if so, terminate immediately
        c0_depth = arith.ConstantOp.from_int_and_width(0, current_depth.type)
        is_at_root = arith.CmpiOp(current_depth, c0_depth, "eq")  # current_depth == 0
        false = arith.ConstantOp.from_int_and_width(0, 1)

        terminate_condition = scf.IfOp(
            is_at_root,
            (current_depth.type, keep_traversal.type, *[arg.type for arg in segment_iter_args]),
            Region(Block()),
            Region(Block()),
        )

        # If at root, yield current values with keep_traversal = false
        rewriter.insert_op(
            scf.YieldOp(current_depth, false, *segment_iter_args),
            InsertPoint.at_end(terminate_condition.true_region.block),
        )

        # Otherwise, continue to right branch
        else_block = terminate_condition.false_region.block
        c2 = arith.ConstantOp.from_int_and_width(2, self.visited_stack.type.element_type)
        storeOp = memref.StoreOp.get(c2, self.visited_stack, (current_depth,))

        # Always restore when going right (status == 1)
        self.handle_restore(current_depth, rewriter, else_block)

        # run a simulation segment
        sim_ops, callOp = self.simulate(current_depth, current_branch, segment_iter_args)

        c1 = arith.ConstantOp.from_int_and_width(1, current_depth.type)
        updated_depth = arith.AddiOp(current_depth, c1)

        for op in (c2, storeOp, c1, updated_depth, *sim_ops, callOp):
            rewriter.insert_op(op, InsertPoint.at_end(else_block))
        rewriter.insert_op(
            scf.YieldOp(updated_depth, keep_traversal, *callOp.results),
            InsertPoint.at_end(else_block),
        )

        yieldOp = scf.YieldOp(*terminate_condition.results)

        # Only insert the condition check and final yield into leftVisitedBlock
        for op in (c0_depth, is_at_root, false, terminate_condition, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(leftVisitedBlock))

        # handle right visited region: need to go back up
        c0 = arith.ConstantOp.from_int_and_width(0, self.visited_stack.type.element_type)
        storeOp = memref.StoreOp.get(c0, self.visited_stack, (current_depth,))  # erase tracks

        c1 = arith.ConstantOp.from_int_and_width(1, current_depth.type)
        updated_depth = arith.SubiOp(current_depth, c1)

        # Simple backtrack - no termination check needed (handled in case 1)
        yieldOp = scf.YieldOp(updated_depth, keep_traversal, *segment_iter_args)

        for op in (c0, storeOp, c1, updated_depth, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(rightVisitedBlock))

        # handle default region, TODO: ideally should raise a runtime exception here
        cm1 = arith.ConstantOp.from_int_and_width(-1, current_depth.type)  # will end traversal
        yieldOp = scf.YieldOp(cm1, keep_traversal, *segment_iter_args)

        for op in (cm1, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(defaultBlock))

    def simulate(
        self, current_depth: SSAValue, current_branch: SSAValue, segment_iter_args: SSAValue
    ):
        """This function is called when going down the tree (left or right), which requires
        backing up the statevector, storing mcm probabilities, projecting the state, and running
        a simulation segment."""

        # TODO: we need the refilled register here
        # quantum.ComputationalBasisOp()
        # quantum.StateOp()

        # quantum.ComputationalBasisOp()
        # quantum.ProbsOp()

        # quantum.MeasureOp(..., postselect=)

        callOp = func.CallOp(
            "segment_table",
            [current_depth, current_branch, *segment_iter_args],
            [val.type for val in segment_iter_args],
        )

        return (), callOp

    def handle_restore(
        self,
        current_depth: SSAValue,
        rewriter: PatternRewriter,
        insert_block: Block,
    ):
        """Restore statevector to previous state when entering a new branch in the tree."""

        # true branch, restore statevector from the stack
        targetType = builtin.MemRefType(
            builtin.ComplexType(builtin.f64),
            (builtin.DYNAMIC_INDEX,),  # rank-reduce: get rid of leading size-1 dimension
            builtin.StridedLayoutAttr([1], None),  # needed due to dynamic indexing
        )
        statevec = memref.SubviewOp.get(
            self.statevec_stack, (current_depth, 0), (1, self.statevec_size), (1, 1), targetType
        )

        rewriter.insert_op(statevec, InsertPoint.at_end(insert_block))

    @staticmethod
    def sum_array(arg: SSAValue, rewriter: PatternRewriter, insert_into: Block) -> SSAValue:
        """Generate a sum reduction using the scf dialect. Produces an integer sum of index type."""
        assert isinstance(arg.type, builtin.MemRefType), "expected memref value to sum"
        assert len(arg.type.shape) == 1, "expected 1D memref to sum"
        assert isinstance(arg.type.element_type, builtin.IntegerType), "expected int memref to sum"
        ip_backup = rewriter.insertion_point
        rewriter.insertion_point = InsertPoint.at_end(insert_into)

        # TODO: reduce not available in either linalg or stablehlo dialects

        # loop instruction
        c0 = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
        c1 = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
        l = memref.DimOp.from_source_and_index(arg, c0)

        sum_init = arith.ConstantOp.from_int_and_width(0, builtin.i64)

        bodyBlock = Block(arg_types=(builtin.IndexType(), sum_init.result.type))
        reduced_sum = scf.ForOp(c0, l, c1, iter_args=(sum_init,), body=bodyBlock)
        casted_sum = arith.IndexCastOp(reduced_sum, builtin.IndexType())

        for op in (c0, c1, l, sum_init, reduced_sum, casted_sum):
            rewriter.insert(op)

        # loop body
        iter_index, last_sum = bodyBlock.args

        val = memref.LoadOp.get(arg, (iter_index,))
        casted_val = arith.ExtSIOp(val, last_sum.type)
        curr_sum = arith.AddiOp(casted_val, last_sum)
        yieldOp = scf.YieldOp(curr_sum)

        for op in (val, casted_val, curr_sum, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(bodyBlock))

        rewriter.insertion_point = ip_backup
        return casted_sum


@compiler_transform
class TTPass(ModulePass):
    name = "tree-traversal"

    def apply(self, ctx: Context, module: builtin.ModuleOp) -> None:

        # # Fixed-point iteration with pattern application, not suited for all kinds of transforms.
        # pattern_list = [TreeTraversal()]
        # greedy_rewriter = GreedyRewritePatternApplier(pattern_list)
        # rewrite_walker = PatternRewriteWalker(greedy_rewriter)
        # rewrite_walker.rewrite_module(module)

        self.apply_on_qnode(module, TreeTraversal())

        print(module)
        module.verify()

    def apply_on_qnode(self, module: builtin.ModuleOp, pattern: RewritePattern):
        """Apply given pattern once to the QNode function in this module."""
        rewriter = PatternRewriter(module)

        qnode = None
        for op in module.ops:
            if isinstance(op, func.FuncOp) and "qnode" in op.attributes:
                qnode = op
                break
        assert qnode is not None, "expected QNode in module"

        pattern.match_and_rewrite(qnode, rewriter)


###########
# Example #
###########

if __name__ == "__main__":

    qml.capture.enable()

    @jax.jit
    def add(a, b):
        return a + b

    @TTPass
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def captured_circuit(x: float):

        qml.Hadamard(wires=0)
        m = qml.measure(0)
        qml.RX(x, wires=0)
        m = qml.measure(0)
        qml.RY(add(x, x), wires=0)
        m = qml.measure(0)
        qml.cond(m, lambda: qml.X(0))

        return qml.state()

    @qml.qjit(keep_intermediate=False, pass_plugins=[getXDSLPluginAbsolutePath()])
    def main(x: float):
        return captured_circuit(x)

    print(main(1.0))
