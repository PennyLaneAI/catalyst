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

from typing import Type, TypeVar

import jax
import pennylane as qml
import pennylane.compiler.python_compiler.dialects.quantum as quantum
# import pennylane.compiler.python_compiler.quantum_dialect as quantum
from pennylane.compiler.python_compiler.transforms.api import compiler_transform
from xdsl import ir, passes
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf, tensor
from xdsl.pattern_rewriter import *
from xdsl.rewriter import InsertPoint

from xdsl.printer import Printer

from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

#############
# Transform #
#############

T = TypeVar("T")


def get_parent_of_type(op: ir.Operation, kind: Type[T]) -> T | None:
    """Walk up the parent tree until an op of the specified type is found."""

    while (op := op.parent_op()) and not isinstance(op, kind):
        pass

    return op

printer = Printer()
class TT_ctrl_flow_for_loop(RewritePattern):
    """Rewrite pattern to handle the control flow for loops in the tree traversal."""

    def __init__(self):
        self.ttOp: func.FuncOp = None
        self.segment_functions: list[func.FuncOp] = []


    @op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: PatternRewriter):
        """Rewrite the scf.for operation to handle tree traversal."""

        if not "qnode" in funcOp.attributes:
            return

        module = get_parent_of_type(funcOp, builtin.ModuleOp)
        assert module is not None, "got orphaned qnode function"
        
        
        self.simple_peeling(funcOp, module, rewriter)
        # self.extract_first_loop_iteration(funcOp, module, rewriter)
        # self.swap_dominants(funcOp, module, rewriter)
        # self.copy_paste(funcOp, module, rewriter)
    
    def simple_peeling(self, funcOp: func.FuncOp, module: builtin.ModuleOp, rewriter: PatternRewriter):
        
        op_iter = funcOp.body.walk()
        
        last_quantum_op_before_loop = None
        
        for op in op_iter:
        
            if isinstance(op, quantum.CustomOp):
                # Store the last quantum operation before the loop
                last_quantum_op_before_loop = op
                continue
                
            if isinstance(op, scf.ForOp):
                # Extract the entire loop block. 
                for_loop = op
                break
        
        for_loop_block = for_loop.body.clone()

        # Create a list to hold the operations in the loop body
        fop_list = []
        for_loop_op_iter = for_loop_block.walk()
        for fop in for_loop_op_iter:
            if isinstance(fop, scf.YieldOp):
                # Stop at the YieldOp
                break

            if not isinstance(fop, quantum.CustomOp):
                continue

            fop_list.append(fop)
            
        # Copy the operations in the loop body before the loop
        for for_op_A in fop_list:
            # Insert the operation before the loop
            value_mapper = {
                in_qubit: last_quantum_op_before_loop.out_qubits[0] for in_qubit in for_op_A.in_qubits
            }
            clone_for_op_A = for_op_A.clone(value_mapper=value_mapper)
            rewriter.insert_op(clone_for_op_A, InsertPoint.after(last_quantum_op_before_loop))


        # For loop step
        step = for_loop.step
        
        # Increment the loop by one step
        lower_bound = for_loop.lb
        incremented_lower_bound = arith.AddiOp(lower_bound, step)
        rewriter.insert_op(incremented_lower_bound, InsertPoint.before(for_loop))
        
        # Decrement the upper bound by one step
        upper_bound = for_loop.ub
        decremented_upper_bound = arith.SubiOp(upper_bound, step)
        
        rewriter.insert_op(decremented_upper_bound, InsertPoint.before(for_loop))


        # Create a new for loop with the new bounds
        
        new_for_loop = scf.ForOp(
            incremented_lower_bound.result,  # new lower bound
            decremented_upper_bound.result,  # new upper bound
            step,                            # same step
            iter_args=for_loop.iter_args,    # same iteration arguments
            body=for_loop_block              # the same body block
        )
    
        # Replace the old for loop with the new one
        rewriter.replace_op(op, new_for_loop)
            
        
    def extract_first_loop_iteration(self, funcOp: func.FuncOp, module: builtin.ModuleOp, rewriter: PatternRewriter):
        
        op_iter = funcOp.body.walk()
        
        last_quantum_op_before_loop = None
        list_op_in_loop_before_mcm = []
        
        # Walk thought the body until find scf.for but storing the previous quantum operations
        for op in op_iter:
            
            if isinstance(op, quantum.CustomOp):
                last_quantum_op_before_loop = op
                continue
            
            if not isinstance(op, scf.ForOp):
                # get block of the operation
                continue
                        
            # Extract the entire loop block. 
            for_loop_block = op.body 
            
            for_loop_op_iter = for_loop_block.walk()
            
            for loop_op in for_loop_op_iter:
                if isinstance(loop_op, quantum.CustomOp):
                    list_op_in_loop_before_mcm.append(loop_op)
                    continue
                
                if isinstance(loop_op, quantum.MeasureOp):
                    break

            # Copy the op in loop before mcm before the loop
                
                
    def copy_op_after_target_test(self, target_op: Type[ir.Operation], op_list: list[ir.Operation], rewriter: PatternRewriter):
        """Copy the operations in the list after the target operation."""
        if not op_list:
            return

        module = get_parent_of_type(target_op, builtin.ModuleOp)
        assert module is not None, "got orphaned operation"
        
        inputs, outputs = self.gather_segment_io(op_list)
        
        printer.print("Inputs: \n")
        for input in inputs:
            printer.print(input.op)
        printer.print("Outputs: \n")
        for output in outputs:
            printer.print(output.op) 
        
        
        op_prev = target_op.prev_op
        
        # Create an array to hold the op that feeds the operations in the list.
        # op_dependencies = [op_prev]
        op_dependencies = [target_op.in_qubits[0].op]
        
        # input_original : input_new
        op_equivalent = {target_op: target_op.in_qubits[0].op}
                
        for insert_op in op_list:
            
            printer.print("^"*100)
            printer.print("\n\nPrevious op: ", op_prev)
            printer.print("\n\nInsert op: ", insert_op)
            printer.print("\n\nCurrent op: ", target_op)
            
            
            # Insert operation after the previous operation of target_op

            # Define the values to mapping  
            value_in_insert = insert_op.in_qubits[0]
            new_value_in_insert = target_op.in_qubits[0]

            # new_value_in_insert = op_equivalent.get(value_in_insert.op, None)
            
            # if new_value_in_insert is None:
            #     op_equivalent[value_in_insert.op] = value_in_insert.op
            
            # new_value_in_insert = new_value_in_insert.out_qubits[0]
            
            # if not new_value_in_insert.op in op_dependencies:
            #     op_dependencies.append(value_in_insert.op)
            #     new_value_in_insert = insert_op.in_qubits[0]
            
            value_mapper = {value_in_insert: new_value_in_insert}
            clone_insert = insert_op.clone(value_mapper=value_mapper)
            
            clone_insert.attributes["FDX_id"] = builtin.StringAttr("new_op")
            
            rewriter.insert_op(clone_insert, InsertPoint.after(op_prev))

            #  The op before target has to be replaced by the new operation
            op_prev = target_op.prev_op
            
            
            # breakpoint()

            
            # Update the target operation to take input from the new operation
            value_out_prev = op_prev.out_qubits[0]
            value_in_copy_op = target_op.in_qubits[0]
            
            value_mapper = {value_in_copy_op: value_out_prev}
            new_op = target_op.clone(value_mapper=value_mapper)
            
            rewriter.replace_op(target_op, new_op)
            rewriter.notify_op_modified(target_op)
            rewriter.notify_op_modified(new_op)

            # #  replace op_prev in op_dependencies with the new operation
            # if op_prev.in_qubits[0].op in op_dependencies:
            #     index = op_dependencies.index(op_prev.in_qubits[0].op)
            #     op_dependencies[index] = new_op.prev_op
            
            # Update the previous operation to be the new target operation and the target operation            
            op_prev = new_op.prev_op
            target_op = new_op
            
            # op_equivalent[insert_op] = op_prev
            
            # print(module)


    def copy_op_after_target(self, target_op: Type[ir.Operation], op_list: list[ir.Operation], rewriter: PatternRewriter):
        """Copy the operations in the list after the target operation."""
        if not op_list:
            return

        module = get_parent_of_type(target_op, builtin.ModuleOp)
        assert module is not None, "got orphaned operation"
        
        op_prev = target_op.prev_op
                        
        for insert_op in op_list:
            
            printer.print("^"*100)
            printer.print("\n\nPrevious op: ", op_prev)
            printer.print("\n\nInsert op: ", insert_op)
            printer.print("\n\nCurrent op: ", target_op)
            
            
            # Insert operation after the previous operation of target_op

            # Define the values to mapping  
            value_in_insert = insert_op.in_qubits[0]
            new_value_in_insert = target_op.in_qubits[0]
            
            value_mapper = {value_in_insert: new_value_in_insert}
            clone_insert = insert_op.clone(value_mapper=value_mapper)
            
            clone_insert.attributes["FDX_id"] = builtin.StringAttr("new_op")
            
            rewriter.insert_op(clone_insert, InsertPoint.after(op_prev))

            #  The op before target has to be replaced by the new operation
            op_prev = target_op.prev_op
            
            
            # breakpoint()

            
            # Update the target operation to take input from the new operation
            value_out_prev = op_prev.out_qubits[0]
            value_in_copy_op = target_op.in_qubits[0]
            
            value_mapper = {value_in_copy_op: value_out_prev}
            new_op = target_op.clone(value_mapper=value_mapper)
            
            rewriter.replace_op(target_op, new_op)
            rewriter.notify_op_modified(target_op)
            rewriter.notify_op_modified(new_op)
            
            # Update the previous operation to be the new target operation and the target operation            
            op_prev = new_op.prev_op
            target_op = new_op
            
            print(module)


    def copy_paste(self, funcOp: func.FuncOp, module: builtin.ModuleOp, rewriter: PatternRewriter):
        
        op_iter = funcOp.body.walk()
        
        op_list = []
        record = False
        for op in op_iter:
            
            if isinstance(op, quantum.CustomOp) and op.gate_name.data == "Identity":
                op.attributes["FDX_id"] = builtin.StringAttr("op_IDENTITIY")
                record = True
                continue

            if isinstance(op, quantum.CustomOp) and op.gate_name.data == "PauliZ":
                op.attributes["FDX_id"] = builtin.StringAttr("op_PAULI_Z")
                record = False
                continue
            
            if record:
                #op_list.append(op.clone())
                op_list.append(op)
                
        op_iter = funcOp.body.walk()
        # for op in op_iter:
            # if isinstance(op, quantum.CustomOp) and op.gate_name.data == "Identity":
                # op_prev = op.prev_op
                # op_curr = op

                # self.copy_op_after_target(op_curr, op_list, rewriter)                
                # self.copy_op_after_target_test(op_curr, op_list, rewriter)
                
        identities = [op for op in funcOp.body.walk() if isinstance(op, quantum.CustomOp) and op.gate_name.data == "Identity"]
        assert len(identities) == 1
        identity = identities[0]
        
        print(funcOp)

        while identity.next_op in {*op_list}:

            current_op = identity.next_op
            
            # Here, we know that all the operands
            # of current_op are either defined by identity
            # or above identity.
            
            acts_on_same_wire: bool = current_op.in_qubits[0] == identity.out_qubits[0]
            if not acts_on_same_wire:
                # Then we know we can safely move it.
                current_op.detach()
                rewriter.insert_op(current_op, InsertPoint.before(identity))
            if acts_on_same_wire:
                identity_definition = identity.in_qubits[0].owner
                rewriter.replace_all_uses_with(identity.out_qubits[0], identity_definition.out_qubits[0])
                #rewriter.erase_op(identity)
                identity.detach()
                # current_op is now where it is supposed to be
                # we need to insert a new identity after current_op
                def just_for_test(use):
                    return use.operation == identity
                identity.in_qubits[0].replace_by_if(current_op.out_qubits[0], just_for_test)
            
                rewriter.insert_op(identity, InsertPoint.after(current_op))
                print(funcOp)

                def just_for_test(use):
                    return use.operation != identity
                current_op.out_qubits[0].replace_by_if(identity.out_qubits[0], just_for_test)
                print(funcOp)
                
                
            #    print(identity.next_op)
            #    breakpoint()     
                
                

    def swap_dominants(self, funcOp: func.FuncOp, module: builtin.ModuleOp, rewriter: PatternRewriter):
        """Swap the dominants of the operations in the loop body."""
        op_iter = funcOp.body.walk()
        
        counter = 0
        clone = None
        for op in op_iter:
            
            if counter == 7:
                op_A = op
                op_A.attributes["FDX_id"] = builtin.StringAttr("op_A")
            
                op_after_A = op.next_op
                op_after_A.attributes["FDX_id"] = builtin.StringAttr("op_after_A")
                
            if counter == 10:
                op_B = op                
                op_B.attributes["FDX_id"] = builtin.StringAttr("op_B")
                
                
                # Insert the new operation after op_A
                value_from_A = op_A.out_qubits[0]
                value_from_B = op_B.in_qubits[0]
                
                value_mapper = {value_from_B: value_from_A}
                clone = op_B.clone(value_mapper=value_mapper)
                clone.attributes["FDX_id"] = builtin.StringAttr("clone_of_op_B")

                rewriter.insert_op(clone, InsertPoint.after(op_A))
                
                
                # Update the operation after op_A to take input from clone
                value_from_after_A = op_after_A.in_qubits[0]
                value_from_clone = clone.out_qubits[0]
                
                value_mapper = {value_from_after_A: value_from_clone}
                op_after_A_new = op_after_A.clone(value_mapper=value_mapper)
                op_after_A_new.attributes["FDX_id"] = builtin.StringAttr("new_op_after_A")
                                
                rewriter.replace_op(op_after_A, op_after_A_new)
                
                
        # # modify dominants after move 
        # hit_point = False
        # for op in funcOp.body.walk():
        #     if op == clone:      
        #         hit_point = True
        #         continue
        #     if hit_point:
        #         pass
                
                
                # # Swap the dominants
                # rewriter.replace_by(op_A, op_B.clone())
                # rewriter.replace_by(op_11, op_10.clone())
                # break
            
            
            counter += 1


    def extract_first_loop_iteration(self, funcOp: func.FuncOp, module: builtin.ModuleOp, rewriter: PatternRewriter):
        """Extract the first loop iteration of the quantum function."""
        
        # Steps:
        
        # * Find the scf.for operation 
        # * Extract the loop body block 'for_loop_block'.
        # * Walk thought 'for_loop_block' and stop at the first quantum measure
        # * 
        
        # * Insert 'A_op'  before the 'for_loop_block' with 
        #   'rewriter.insert_op(A_op, InsertPoint.before(for_loop_block))'
        # * Walk through 'for_loop_block' until the and get the last operation 'B_op'
        # * Insert 'B_op' after the 'for_loop_block' with
        #   'rewriter.insert_op(B_op, InsertPoint.after(for_loop_block))'
        # * Change the scf.for operation to have the correct bounds:
        #   - lower bound: +1
        #   - upper bound: -1
        
        # Problems:
        
        # Dependencies between gates operants 
        # How transfer the scf.for arguments 
        
        
        # take care of the arg before the loop  and the quantum rgister
        # avoid using func 
        
        
        

        # Ideally try to iterate over the function only once.
        op_iter = funcOp.body.walk()
        
        # Walk thought the body until find scf.for 
        for op in op_iter:

            if not isinstance(op, scf.ForOp):
                continue
            
            # Extract the entire loop block. 
            for_loop_block = op.body 

            # Extract lower, upper bounds and step of the loop
            lower_bound = op.lb
            upper_bound = op.ub
            step = op.step

            # Create a list to hold the operations in the loop body            
            fop_list = []
            
            for_loop_op_iter = for_loop_block.walk()
            for fop in for_loop_op_iter:
                # Check if the operation is a MeasureOp or YieldOp
                if isinstance(fop, quantum.MeasureOp):

                    # David option                     
                    # self.clone_ops_into_func(fop_list, op, rewriter, segment_tile="segment_for_loop_")


                    #### Straightforward option
                    
                    # Update args in the first operation by the lower bound
                    
                    # first_iter = arith.ConstantOp.from_int_and_width(lower_bound.value, 64)
                    # rewriter.insert_op(first_iter, InsertPoint.before(fop))
                    
                    # Insert the list of operations before the loop
                    for for_op_A in fop_list[1:]:
                        breakpoint()
                        rewriter.insert_op(for_op_A.clone(), InsertPoint.before(op))
                        
                    # Update the dominants for each operation in the list
                    # ????
                        
                        
                    fop_list = []
                    continue # avoid including measure
                elif isinstance(fop, scf.YieldOp):
                    break

                # If it's not a MeasureOp or YieldOp, add it the list
                fop_list.append(fop) 

            # Update the lower bound of the loop to be +step
            # insert must be before use
            operation = arith.AddiOp(lower_bound, step)
            value = operation.result
            rewriter.insert(operation)
            # except the user by the AddiOp above.
            # maybe use replace_by?
            rewriter.replace_by_if(lower_bound, value, lambda : value != operation)
            # lower_bound = arith.AddiOp(lower_bound, arith.ConstantOp.from_int_and_width(1, lower_bound.type))
            # rewriter.insert_op(lower_bound, InsertPoint.before(op))
                            
            
        
        # Change the loop arguments: lower bound + 1
        # Might better to delete the loop and insert a new one with the correct bounds.
    def clone_ops_into_func(self, op_list: list[ir.Operation], trg_op: Type[ir.Operation], rewriter: PatternRewriter, segment_tile: str = "quantum_segment_"):
        """Clone a set of ops into a new function."""
        if not op_list:
            return

        module = get_parent_of_type(op_list[0], builtin.ModuleOp)
        assert module is not None, "got orphaned operation"

        input_vals, output_vals = self.gather_segment_io(op_list)

        fun_type = builtin.FunctionType.from_lists(
            [arg.type for arg in input_vals], [res.type for res in output_vals]
        )
        new_func = func.FuncOp(f"{segment_tile}{len(self.segment_functions)}", fun_type)
        self.segment_functions.append(new_func)

        rewriter.insertion_point = InsertPoint.at_start(new_func.body.block)

        value_mapper = dict(zip(input_vals, new_func.args))
        for op in op_list:
            new_op = op.clone(value_mapper)
            rewriter.insert(new_op)

        returnOp = func.ReturnOp(*(value_mapper[res] for res in output_vals))
        rewriter.insert(returnOp)

        rewriter.insert_op(new_func, InsertPoint.at_end(module.body.block))
        
        
        rewriter.inline_block(new_func.body.block, InsertPoint.before(trg_op))

    @staticmethod
    def gather_segment_io(op_list: list[ir.Operation]) -> tuple[set[ir.SSAValue], set[ir.SSAValue]]:
        """Gather SSA values that need to be passed in and out of the segment to be outlined."""
        inputs = set()
        outputs = set()

        # TODO: The outputs are not quite right, since just because an op result has been used
        #       doesn't mean it won't be needed in a subsequent region.
        #       The inputs should be correct though.
        #       Might also have to be more careful with the quantum input/output values.
        for op in op_list:
            inputs.update(op.operands)
            inputs.difference_update(op.results)

            outputs.update(op.results)
            outputs.difference_update(op.operands)

        # Remove op that don't have a unique input 
        
        remove_mid_inputs = set()
        for inp in inputs:
            for a in  inp.op.operands:
                if a in inputs:
                    remove_mid_inputs.add(inp)

        inputs.difference_update(remove_mid_inputs)



        return inputs, outputs



class TreeTraversal(RewritePattern):

    def __init__(self):
        self.ttOp: func.FuncOp = None
        self.segment_functions: list[func.FuncOp] = []

    @op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: PatternRewriter):
        """Transform a quantum function (qnode) to perform tree-traversal simulation."""
        if not "qnode" in funcOp.attributes:
            return

        module = get_parent_of_type(funcOp, builtin.ModuleOp)
        assert module is not None, "got orphaned qnode function"

        # Start with creating a new QNode function that will perform the tree traversal simulation.
        self.setup_traversal_function(funcOp, module, rewriter)
        
        # self.extract_first_loop_iteration(funcOp, module, rewriter)

        # self.split_traversal_segments(funcOp, rewriter)

        # self.initialize_data_structures(rewriter)

        # self.generate_traversal_code(rewriter)

        # self.finalize_traversal_function(rewriter)

    def setup_traversal_function(
        self, funcOp: func.FuncOp, module: builtin.ModuleOp, rewriter: RewritePattern
    ):
        """Setup a clone of the original QNode function, which will instead perform TT."""

        ttOp = funcOp.clone_without_regions() # Alfredo: Why we copy without regions?
        ttOp.sym_name = builtin.StringAttr(funcOp.sym_name.data + ".tree_traversal")
        rewriter.create_block(BlockInsertPoint.at_start(ttOp.body), ttOp.function_type.inputs)
        rewriter.insert_op(ttOp, InsertPoint.at_end(module.body.block))

        self.ttOp = ttOp

    def finalize_traversal_function(self, rewriter: RewritePattern):
        """Complete the function and ensure it's correctly formed, e.g. returning proper results."""
        rewriter.insertion_point = InsertPoint.at_end(self.ttOp.body.block)

        # For now return nothing from the Tree-Traversal function to satisfy the verifier.
        result_vals = []
        for resType in self.ttOp.function_type.outputs:
            assert isinstance(resType, builtin.TensorType)
            result = tensor.EmptyOp((), tensor_type=resType)
            result_vals.append(rewriter.insert(result))

        rewriter.insert(func.ReturnOp(*result_vals))
        
        
    def extract_first_loop_iteration(self, funcOp: func.FuncOp, module: builtin.ModuleOp, rewriter: PatternRewriter):
        """Extract the first loop iteration of the quantum function."""
        
        # Steps:
        
        # * Find the scf.for operation 
        # * Extract the loop body block 'for_loop_block'.
        # * Walk thought 'for_loop_block' and stop at the first quantum measure
        # * 
        
        # * Insert 'A_op'  before the 'for_loop_block' with 
        #   'rewriter.insert_op(A_op, InsertPoint.before(for_loop_block))'
        # * Walk through 'for_loop_block' until the and get the last operation 'B_op'
        # * Insert 'B_op' after the 'for_loop_block' with
        #   'rewriter.insert_op(B_op, InsertPoint.after(for_loop_block))'
        # * Change the scf.for operation to have the correct bounds:
        #   - lower bound: +1
        #   - upper bound: -1
        
        # Problems:
        
        # Dependencies between gates operants 
        # How transfer the scf.for arguments 
        
        
        # take care of the arg before the loop  and the quantum rgister
        
        #  create a new pattern only for the loop peelling 
        
        # avoid using func 
        
        
        

        # Ideally try to iterate over the function only once.
        op_iter = funcOp.body.walk()
        
        # Walk thought the body until find scf.for 
        for op in op_iter:

            if not isinstance(op, scf.ForOp):
                # get block of the operation
                
                continue
            
            # Extract the entire loop block. 
            for_loop_block = op.body 
            
            for_loop_op_iter = for_loop_block.walk()
            
            fop_list = []
            after_measure = False
            for fop in for_loop_op_iter:
                if isinstance(fop, quantum.MeasureOp):
                    
                    # Found the first quantum operation in the loop body
                    # Insert it before the loop block
                    # rewriter.insert_op(fop.clone(), InsertPoint.before(op))
                    self.clone_ops_into_func(fop_list, rewriter, segment_tile="segment_for_loop_")
                    fop_list = []
                    continue # avoid including measure
                elif isinstance(fop, scf.YieldOp):
                    break
                                        
                fop_list.append(fop) 
            self.clone_ops_into_func(fop_list, rewriter, segment_tile="segment_for_loop_")
            
        
        # Change the loop arguments: lower bound + 1
        # Might better to delete the loop and insert a new one with the correct bounds.

    def split_traversal_segments(self, funcOp: func.FuncOp, rewriter: PatternRewriter):
        """Split the quantum function into segments separated by measure operations."""
        rewriter.insertion_point = InsertPoint.at_start(self.ttOp.body.block)

        # Ideally try to iterate over the function only once.
        op_iter = funcOp.body.walk()

        # Skip to the start of the first simulation segment.
        value_mapper = {}
        while (op := next(op_iter, None)) and not isinstance(op, quantum.AllocOp):
            rewriter.insert(op.clone(value_mapper))
        assert op is not None, "didn't find an alloc op"
        self.alloc_op = rewriter.insert(op.clone(value_mapper))
        
        # I dont understand this section  :( 

        # Generate new functions for each segment separated by a measure op.
        op_list = []
        while op := next(op_iter, None):
            if isinstance(op, quantum.MeasureOp):
                self.clone_ops_into_func(op_list, rewriter)
                op_list = []
            elif isinstance(op, quantum.DeallocOp):
                self.clone_ops_into_func(op_list, rewriter)
                break
            else:
                op_list.append(op)
        assert op is not None, "didn't find a dealloc op"

    def clone_ops_into_func(self, op_list: list[ir.Operation], rewriter: PatternRewriter, segment_tile: str = "quantum_segment_"):
        """Clone a set of ops into a new function."""
        if not op_list:
            return

        module = get_parent_of_type(op_list[0], builtin.ModuleOp)
        assert module is not None, "got orphaned operation"

        input_vals, output_vals = self.gather_segment_io(op_list)

        fun_type = builtin.FunctionType.from_lists(
            [arg.type for arg in input_vals], [res.type for res in output_vals]
        )
        new_func = func.FuncOp(f"{segment_tile}{len(self.segment_functions)}", fun_type)
        self.segment_functions.append(new_func)

        rewriter.insertion_point = InsertPoint.at_start(new_func.body.block)

        value_mapper = dict(zip(input_vals, new_func.args))
        for op in op_list:
            new_op = op.clone(value_mapper)
            rewriter.insert(new_op)

        returnOp = func.ReturnOp(*(value_mapper[res] for res in output_vals))
        rewriter.insert(returnOp)

        rewriter.insert_op(new_func, InsertPoint.at_end(module.body.block))

    @staticmethod
    def gather_segment_io(op_list: list[ir.Operation]) -> tuple[set[ir.SSAValue], set[ir.SSAValue]]:
        """Gather SSA values that need to be passed in and out of the segment to be outlined."""
        inputs = set()
        outputs = set()

        # TODO: The outputs are not quite right, since just because an op result has been used
        #       doesn't mean it won't be needed in a subsequent region.
        #       The inputs should be correct though.
        #       Might also have to be more careful with the quantum input/output values.
        for op in op_list:
            inputs.update(op.operands)
            inputs.difference_update(op.results)

            outputs.update(op.results)
            outputs.difference_update(op.operands)

        return inputs, outputs

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
            len(self.segment_functions), builtin.IndexType()
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
        restore_init = arith.ConstantOp.from_int_and_width(0, 1)

        conditionBlock = builtin.Block(arg_types=(depth_init.result.type, restore_init.result.type))
        bodyBlock = builtin.Block(arg_types=conditionBlock.arg_types)
        traversalOp = scf.WhileOp(
            (depth_init, restore_init), conditionBlock.arg_types, (conditionBlock,), (bodyBlock,)
        )

        for op in (depth_init, restore_init, traversalOp):
            rewriter.insert(op)

        # condition block of the while loop
        current_depth, needs_restore = self.check_if_leaf(*conditionBlock.args, rewriter)

        c0 = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
        c2 = arith.ConstantOp.from_int_and_width(2, builtin.IndexType())

        positive_depth = arith.CmpiOp(current_depth, c0, "sge")

        visited_sum = self.sum_array(self.visited_stack, rewriter, conditionBlock)
        max_sum = arith.MuliOp(c2, self.tree_depth)
        not_fully_traversed = arith.CmpiOp(visited_sum, max_sum, "ne")

        final_condition = arith.AndIOp(positive_depth, not_fully_traversed)
        condOp = scf.ConditionOp(final_condition, current_depth, needs_restore)

        for op in (c0, c2, positive_depth, max_sum, not_fully_traversed, final_condition, condOp):
            rewriter.insert_op(op, InsertPoint.at_end(conditionBlock))

        # body block of the while
        current_depth, needs_restore = bodyBlock.args

        node_status = memref.LoadOp.get(self.visited_stack, (current_depth,))
        casted_status = arith.IndexCastOp(node_status, builtin.IndexType())

        cases = builtin.DenseArrayBase.from_list(builtin.i64, [0, 1, 2])
        switchOp = scf.IndexSwitchOp(
            casted_status,
            cases,
            builtin.Region(builtin.Block()),
            [builtin.Region(builtin.Block()) for _ in range(3)],
            (current_depth.type, needs_restore.type),
        )

        self.process_node(switchOp, current_depth, needs_restore, rewriter)

        yieldOp = scf.YieldOp(*switchOp.results)

        for op in (node_status, casted_status, switchOp, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(bodyBlock))

    def check_if_leaf(
        self, current_depth: SSAValue, needs_restore: SSAValue, rewriter: PatternRewriter
    ) -> tuple[SSAValue, SSAValue]:
        """Verify whether we've hit the bottom of the tree, and perform update actions."""
        assert isinstance(current_depth.owner, builtin.Block)
        assert current_depth.owner == needs_restore.owner
        ip_backup = rewriter.insertion_point
        rewriter.insertion_point = InsertPoint.at_start(current_depth.owner)

        # if instruction
        hit_leaf = arith.CmpiOp(current_depth, self.tree_depth, "eq")
        trueBlock, falseBlock = builtin.Block(), builtin.Block()
        ifOp = scf.IfOp(
            hit_leaf, (current_depth.type, needs_restore.type), (trueBlock,), (falseBlock,)
        )

        for op in (hit_leaf, ifOp):
            rewriter.insert(op)

        # true branch body
        c1 = arith.ConstantOp.from_int_and_width(1, current_depth.type)
        updated_depth = arith.SubiOp(current_depth, c1)
        true = arith.ConstantOp.from_int_and_width(1, 1)
        yieldOp = scf.YieldOp(updated_depth, true)

        for op in (c1, updated_depth, true, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(trueBlock))

        # false branch body
        yieldOp = scf.YieldOp(current_depth, needs_restore)
        rewriter.insert_op(yieldOp, InsertPoint.at_end(falseBlock))

        rewriter.insertion_point = ip_backup

        return ifOp.results

    def process_node(
        self,
        switchOp: scf.IndexSwitchOp,
        current_depth: SSAValue,
        needs_restore: SSAValue,
        rewriter: PatternRewriter,
    ):
        """Update data structures and effect transition from one node to the next."""
        defaultBlock = switchOp.default_region.block
        unvisitedBlock, leftVisitedBlock, rightVisitedBlock = (
            reg.block for reg in switchOp.case_regions
        )

        # handle unvisited region: need to ge left
        c1 = arith.ConstantOp.from_int_and_width(1, self.visited_stack.type.element_type)
        storeOp = memref.StoreOp.get(c1, self.visited_stack, (current_depth,))

        # TODO: add simulation segment

        c1_ = arith.ConstantOp.from_int_and_width(1, current_depth.type)
        updated_depth = arith.AddiOp(current_depth, c1_)

        yieldOp = scf.YieldOp(updated_depth, needs_restore)

        for op in (c1, c1_, storeOp, updated_depth, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(unvisitedBlock))

        # handle left visited region: need to go right
        c2 = arith.ConstantOp.from_int_and_width(2, self.visited_stack.type.element_type)
        storeOp = memref.StoreOp.get(c2, self.visited_stack, (current_depth,))

        trueBlock, falseBlock = builtin.Block(), builtin.Block()
        updated_restore = scf.IfOp(
            needs_restore, (needs_restore.type,), (trueBlock,), (falseBlock,)
        )
        self.handle_restore(needs_restore, current_depth, trueBlock, falseBlock, rewriter)

        # TODO: add simulation segment

        c1 = arith.ConstantOp.from_int_and_width(1, current_depth.type)
        updated_depth = arith.AddiOp(current_depth, c1)

        yieldOp = scf.YieldOp(updated_depth, updated_restore)

        for op in (c1, c2, storeOp, updated_restore, updated_depth, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(leftVisitedBlock))

        # handle right visited region: need to go back up
        c0 = arith.ConstantOp.from_int_and_width(0, self.visited_stack.type.element_type)
        storeOp = memref.StoreOp.get(c0, self.visited_stack, (current_depth,))  # erase tracks

        c1 = arith.ConstantOp.from_int_and_width(1, current_depth.type)
        updated_depth = arith.SubiOp(current_depth, c1)

        yieldOp = scf.YieldOp(updated_depth, needs_restore)

        for op in (c0, c1, storeOp, updated_depth, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(rightVisitedBlock))

        # handle default region, TODO: ideally should raise a runtime exception here
        cm1 = arith.ConstantOp.from_int_and_width(-1, current_depth.type)  # will end traversal
        yieldOp = scf.YieldOp(cm1, needs_restore)

        for op in (cm1, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(defaultBlock))

    def handle_restore(
        self,
        needs_restore: SSAValue,
        current_depth: SSAValue,
        trueBlock: Block,
        falseBlock: Block,
        rewriter: PatternRewriter,
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
        # TODO: do something with the statevector (quantum op)

        false = arith.ConstantOp.from_int_and_width(0, 1)
        yieldOp = scf.YieldOp(false)

        for op in (false, statevec, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(trueBlock))

        # false branch, no restore needed
        rewriter.insert_op(scf.YieldOp(needs_restore), InsertPoint.at_end(falseBlock))

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

        bodyBlock = builtin.Block(arg_types=(builtin.IndexType(), sum_init.result.type))
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


# @xdsl_transform
@compiler_transform 
class TTPass(passes.ModulePass):
    name = "tree-traversal"

    def apply(self, ctx: Context, module: builtin.ModuleOp) -> None:

        # # Fixed-point iteration with pattern application, not suited for all kinds of transforms.
        # pattern_list = [TreeTraversal()]
        # greedy_rewriter = GreedyRewritePatternApplier(pattern_list)
        # rewrite_walker = PatternRewriteWalker(greedy_rewriter)
        # rewrite_walker.rewrite_module(module)


        print("%"*120)
        print("Before--"*10)
        print("")
        print(module)
        print("*"*120)
        print("%"*120)


        # self.apply_on_qnode(module, TreeTraversal())
        # breakpoint()
        self.apply_on_qnode(module, TT_ctrl_flow_for_loop())
        
        # Why do we not use ctx here?

        # module.verify()
        print("%"*120)
        print("*"*120)
        print(module)
        print("*"*120)
        print("%"*120)

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
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def captured_circuit(x: float):


        qml.X(0)
        for i in range(5):
            qml.H(0)
        
        # qml.Hadamard(wires=0)
        # qml.Hadamard(wires=1)
        
        # for i in range(3):
        #     qml.X(0)
        #     qml.Z(1)
            
        #     m = qml.measure(0)
        #     qml.Y(0)
        #     qml.Z(1)


        # -------------------------------------------------
        # qml.Hadamard(wires=0)
        # qml.S(wires=1)
        
        # qml.Identity(wires=0)

        # qml.X(wires=0)
        # qml.Y(wires=0)
        
        # qml.X(wires=1)
        # qml.Y(wires=1)
        

        # qml.Z(wires=0)
        # -------------------------------------------------
        
        # for i in range(3):
        #     qml.Hadamard(wires=0)
        #     qml.PauliX(wires=0)
        #     qml.PauliY(wires=0)
        #     m = qml.measure(0)
        #     qml.Hadamard(wires=0)
        #     qml.PauliX(wires=0)
        #     qml.PauliY(wires=0)


        # qml.Hadamard(wires=0)
        # m = qml.measure(0)
        # qml.RX(x, wires=0)
        # m = qml.measure(0)
        # qml.RY(add(x, x), wires=0)
        # m = qml.measure(0)
        # qml.cond(m, lambda: qml.X(0))

        return qml.state()

    @qml.qjit(keep_intermediate=False, pass_plugins=[getXDSLPluginAbsolutePath()], autograph=True)
    def main(x: float):
        return captured_circuit(x)

    print(main(1.0))
