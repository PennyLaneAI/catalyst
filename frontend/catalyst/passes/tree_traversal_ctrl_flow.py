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
from pennylane.compiler.python_compiler.transforms.api import compiler_transform

from xdsl import context, passes, pattern_rewriter, ir
from xdsl.dialects import arith, builtin, func
from xdsl.dialects.scf import ForOp, IfOp, WhileOp, YieldOp
from xdsl.rewriter import InsertPoint
from xdsl.printer import Printer

from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

def print_module(module: builtin.ModuleOp, section: str = ""):

    length_line = 100

    if True:
        # printer = Printer()
        print("%"*length_line)
        print("")
        print(f"---{section}---"*4)
        print("")
        Printer().print_op(module)
        print("")
        # print("%"*length_line)
        # print("")

T = TypeVar("T")

def get_parent_of_type(op: ir.Operation, kind: Type[T]) -> T | None:
    """Walk up the parent tree until an op of the specified type is found."""

    while (op := op.parent_op()) and not isinstance(op, kind):
        pass

    return op

CASE_PASS = "erick_help"
CASE_PASS = "example_02"
CASE_PASS = "example_03"
CASE_PASS = "example_04"

class TT_ctrl_flow_for_loop(pattern_rewriter.RewritePattern):
    """RewritePattern for combining all :class:`~pennylane.GlobalPhase` gates within the same region
    at the last global phase gate."""

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self, root: func.FuncOp | IfOp | ForOp | WhileOp, rewriter: pattern_rewriter.PatternRewriter
    ):  
        """Rewrite the scf.for operation to handle tree traversal."""

        if not "qnode" in root.attributes:
            return

        module = get_parent_of_type(root, builtin.ModuleOp)
        assert module is not None, "got orphaned qnode function"

        match CASE_PASS:
            case "erick_help":
                self.copy_paste(root, module, rewriter)
            case "example_02":
                self.simple_peeling(root, module, rewriter)
            case "example_03":
                self.select_peeling(root, module, rewriter)
            case "example_04":
                self.tt_loop_peeling(root, module, rewriter)

    def tt_loop_peeling(self, funcOp: func.FuncOp, module: builtin.ModuleOp, rewriter: pattern_rewriter.PatternRewriter):

        op_iter = funcOp.body.walk()
        
        last_quantum_op_before_loop = None
        first_quantum_op_after_loop = None
        
        record_last_quantum_op_before_loop = True
        record_first_quantum_op_after_loop = False

        for op in op_iter:
        
            if isinstance(op, quantum.CustomOp) and record_last_quantum_op_before_loop:
                # Store the last quantum operation before the loop
                last_quantum_op_before_loop = op
                continue

            if isinstance(op, quantum.CustomOp) and record_first_quantum_op_after_loop:
                # Store the last quantum operation before the loop
                first_quantum_op_after_loop = op
                record_first_quantum_op_after_loop = False
                continue
                
            if isinstance(op, ForOp):
                # Extract the entire loop block. 
                record_last_quantum_op_before_loop = False
                for_loop = op

            if isinstance(op, YieldOp):
                # Extract the entire loop block. 
                record_first_quantum_op_after_loop = True
                continue

        
        for_loop_block = for_loop.body.clone()

        # Create a list to hold the operations in the loop body
        fop_list_section_A = []
        fop_list_section_B = []
        fop_split = None
        
        record_section = "A"
        
        for_loop_op_iter = for_loop_block.walk()
        for fop in for_loop_op_iter:

            if isinstance(fop, quantum.CustomOp) and fop.gate_name.data == "PauliY":
                record_section = "B"
                fop_split = fop
                continue

            if isinstance(fop, quantum.CustomOp):
                if record_section == "A":
                    fop_list_section_A.append(fop)
                elif record_section == "B":
                    fop_list_section_B.append(fop)

        # Copy the operations in the loop body before the loop
        for for_op_A in fop_list_section_A[::-1]:
            # # Insert the operation before the loop
            
            value_mapper = { for_op_A.in_qubits[0]: last_quantum_op_before_loop.out_qubits[0] }

            clone_for_op_A = for_op_A.clone(value_mapper=value_mapper)
            clone_for_op_A.attributes["FDX_id"] = builtin.StringAttr("op_BEFORE_LOOP")
            
            rewriter.insert_op(clone_for_op_A, InsertPoint.after(last_quantum_op_before_loop))

            print_module(funcOp, "insert_op")

            output_last_op = last_quantum_op_before_loop.out_qubits[0]
            output_last_op.replace_by_if(clone_for_op_A.out_qubits[0], lambda use: use.operation != clone_for_op_A)
            
            print_module(funcOp, "replace_if")


        # Insert the loop split operation
        
        fop_list_section_B.insert(0, fop_split)

        # Add the operation after the loop body

        for for_op_B in fop_list_section_B:

            value_mapper = {
                for_op_B.in_qubits[0]: first_quantum_op_after_loop.in_qubits[0]
            }
            clone_for_op_B = for_op_B.clone(value_mapper=value_mapper)
            clone_for_op_B.attributes["FDX_id"] = builtin.StringAttr("op_AFTER_LOOP")

            rewriter.insert_op(clone_for_op_B, InsertPoint.before(first_quantum_op_after_loop))

            print_module(funcOp, "insert first op after loop")
            
            input_first_op_after_loop = first_quantum_op_after_loop.in_qubits[0]
            input_first_op_after_loop.replace_by_if(clone_for_op_B.out_qubits[0], lambda use: use.operation == first_quantum_op_after_loop)

            print_module(funcOp, "replace if after loop")
            

        # For loop step
        step = for_loop.step
        
        # Increment the loop by one step
        # lower_bound = for_loop.lb
        # incremented_lower_bound = arith.AddiOp(lower_bound, step)
        # rewriter.insert_op(incremented_lower_bound, InsertPoint.before(for_loop))
        
        # Decrement the upper bound by one step
        upper_bound = for_loop.ub
        decremented_upper_bound = arith.SubiOp(upper_bound, step)
        
        rewriter.insert_op(decremented_upper_bound, InsertPoint.before(for_loop))

        # Modify the order in the for loop 
        # for_loop_op_iter = for_loop_block.walk()
        # for fop in for_loop_op_iter:
        #     if isinstance(fop, quantum.CustomOp):
        #         test = fop

        # Rearrange the operations to put Y at top and move S to last
        print_module(for_loop, "For Loop before rearrange")
        
        
        last_B = fop_list_section_B[-1]
        first_A = fop_list_section_A[0]

        fop_split.in_qubits[0].replace_by_if(first_A.in_qubits[0], lambda use: use.operation == fop_split)

        first_A.in_qubits[0].replace_by_if(last_B.out_qubits[0], lambda use: use.operation == first_A)
        last_B.out_qubits[0].replace_by_if(first_A.out_qubits[0], lambda use: use.operation != first_A)

        first_A.detach()
        rewriter.insert_op(first_A, InsertPoint.after(last_B))




        # Create a new for loop with the new bounds
        
        new_for_loop = ForOp(
            # incremented_lower_bound.result,  # new lower bound
            for_loop.lb,  # new lower bound
            decremented_upper_bound.result,  # new upper bound
            step,                            # same step
            iter_args=for_loop.iter_args,    # same iteration arguments
            body=for_loop_block              # the same body block
        )
        print_module(new_for_loop, "For Loop rearrange")

        # Replace the old for loop with the new one
        rewriter.replace_op(for_loop, new_for_loop)

    def select_peeling(self, funcOp: func.FuncOp, module: builtin.ModuleOp, rewriter: pattern_rewriter.PatternRewriter):

        op_iter = funcOp.body.walk()
        
        last_quantum_op_before_loop = None
        first_quantum_op_after_loop = None
        
        record_last_quantum_op_before_loop = True
        record_first_quantum_op_after_loop = False

        for op in op_iter:
        
            if isinstance(op, quantum.CustomOp) and record_last_quantum_op_before_loop:
                # Store the last quantum operation before the loop
                last_quantum_op_before_loop = op
                record_last_quantum_op_before_loop = False
                continue

            if isinstance(op, quantum.CustomOp) and record_first_quantum_op_after_loop:
                # Store the last quantum operation before the loop
                first_quantum_op_after_loop = op
                record_first_quantum_op_after_loop = False
                continue
                
            if isinstance(op, ForOp):
                # Extract the entire loop block. 
                for_loop = op

            if isinstance(op, YieldOp):
                # Extract the entire loop block. 
                record_first_quantum_op_after_loop = True
                continue

        
        for_loop_block = for_loop.body.clone()

        # Create a list to hold the operations in the loop body
        fop_list_section_A = []
        fop_list_section_B = []
        
        record_section = "A"
        
        for_loop_op_iter = for_loop_block.walk()
        for fop in for_loop_op_iter:

            if isinstance(fop, quantum.CustomOp) and fop.gate_name.data == "PauliY":
                record_section = "B"
                continue

            if isinstance(fop, quantum.CustomOp):
                if record_section == "A":
                    fop_list_section_A.append(fop)
                elif record_section == "B":
                    fop_list_section_B.append(fop)



            
        # Copy the operations in the loop body before the loop
        for for_op_A in fop_list_section_A[::-1]:
            # # Insert the operation before the loop
            
            value_mapper = { for_op_A.in_qubits[0]: last_quantum_op_before_loop.out_qubits[0] }

            clone_for_op_A = for_op_A.clone(value_mapper=value_mapper)
            clone_for_op_A.attributes["FDX_id"] = builtin.StringAttr("op_BEFORE_LOOP")
            
            rewriter.insert_op(clone_for_op_A, InsertPoint.after(last_quantum_op_before_loop))

            print_module(funcOp, "insert_op")

            output_last_op = last_quantum_op_before_loop.out_qubits[0]
            output_last_op.replace_by_if(clone_for_op_A.out_qubits[0], lambda use: use.operation != clone_for_op_A)
            
            print_module(funcOp, "replace_if")


        # Add the operation after the loop body

        for for_op_B in fop_list_section_B:

            value_mapper = {
                for_op_B.in_qubits[0]: first_quantum_op_after_loop.in_qubits[0]
            }
            clone_for_op_B = for_op_B.clone(value_mapper=value_mapper)
            clone_for_op_B.attributes["FDX_id"] = builtin.StringAttr("op_AFTER_LOOP")

            rewriter.insert_op(clone_for_op_B, InsertPoint.before(first_quantum_op_after_loop))

            print_module(funcOp, "insert first op after loop")
            
            input_first_op_after_loop = first_quantum_op_after_loop.in_qubits[0]
            input_first_op_after_loop.replace_by_if(clone_for_op_B.out_qubits[0], lambda use: use.operation == first_quantum_op_after_loop)

            print_module(funcOp, "replace if after loop")

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
        
        new_for_loop = ForOp(
            incremented_lower_bound.result,  # new lower bound
            decremented_upper_bound.result,  # new upper bound
            step,                            # same step
            iter_args=for_loop.iter_args,    # same iteration arguments
            body=for_loop_block              # the same body block
        )

        # Replace the old for loop with the new one
        rewriter.replace_op(for_loop, new_for_loop)

    def simple_peeling(self, funcOp: func.FuncOp, module: builtin.ModuleOp, rewriter: pattern_rewriter.PatternRewriter):

        op_iter = funcOp.body.walk()
        
        last_quantum_op_before_loop = None
        
        for op in op_iter:
        
            if isinstance(op, quantum.CustomOp):
                # Store the last quantum operation before the loop
                last_quantum_op_before_loop = op
                continue
                
            if isinstance(op, ForOp):
                # Extract the entire loop block. 
                for_loop = op
                break
        
        for_loop_block = for_loop.body.clone()

        # Create a list to hold the operations in the loop body
        fop_list = []
        for_loop_op_iter = for_loop_block.walk()
        for fop in for_loop_op_iter:
            if isinstance(fop, YieldOp):
                # Stop at the YieldOp
                break

            if not isinstance(fop, quantum.CustomOp):
                continue

            fop_list.append(fop)
            
        # Copy the operations in the loop body before the loop
        for for_op_A in fop_list[::-1]:
            # # Insert the operation before the loop
            
            value_mapper = { for_op_A.in_qubits[0]: last_quantum_op_before_loop.out_qubits[0] }

            clone_for_op_A = for_op_A.clone(value_mapper=value_mapper)
            clone_for_op_A.attributes["FDX_id"] = builtin.StringAttr("op_BEFORE_LOOP")
            
            rewriter.insert_op(clone_for_op_A, InsertPoint.after(last_quantum_op_before_loop))

            print_module(funcOp, "insert_op")

            output_last_op = last_quantum_op_before_loop.out_qubits[0]
            output_last_op.replace_by_if(clone_for_op_A.out_qubits[0], lambda use: use.operation != clone_for_op_A)
            
            print_module(funcOp, "replace_if")


        op_iter = funcOp.body.walk()
        
        first_quantum_op_after_loop = False
        
        for op in op_iter:
        
                
            if isinstance(op, YieldOp):
                # Extract the entire loop block. 
                first_quantum_op_after_loop = True
                continue

            if isinstance(op, quantum.CustomOp) and first_quantum_op_after_loop:
                # Store the last quantum operation before the loop
                first_quantum_op_after_loop = op
                break
        
        # Add the operation after the loop body
        
        for for_op_A in fop_list:
    
            value_mapper = {
                for_op_A.in_qubits[0]: first_quantum_op_after_loop.in_qubits[0]
            }
            clone_for_op_A = for_op_A.clone(value_mapper=value_mapper)
            clone_for_op_A.attributes["FDX_id"] = builtin.StringAttr("op_BEFORE_LOOP")

            rewriter.insert_op(clone_for_op_A, InsertPoint.before(first_quantum_op_after_loop))

            print_module(funcOp, "insert first op after loop")
            
            input_first_op_after_loop = first_quantum_op_after_loop.in_qubits[0]
            input_first_op_after_loop.replace_by_if(clone_for_op_A.out_qubits[0], lambda use: use.operation == first_quantum_op_after_loop)

            print_module(funcOp, "replace if after loop")

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
        
        new_for_loop = ForOp(
            incremented_lower_bound.result,  # new lower bound
            decremented_upper_bound.result,  # new upper bound
            step,                            # same step
            iter_args=for_loop.iter_args,    # same iteration arguments
            body=for_loop_block              # the same body block
        )

        # Replace the old for loop with the new one
        rewriter.replace_op(for_loop, new_for_loop)    

    def copy_paste(self, funcOp: func.FuncOp, module: builtin.ModuleOp, rewriter: pattern_rewriter.PatternRewriter):
        
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
        
        print_module(funcOp, "Annotations")

        while identity.next_op in {*op_list}:

            current_op = identity.next_op
            
            print(current_op)
            
            # Here, we know that all the operands
            # of current_op are either defined by identity
            # or above identity.
            
            acts_on_same_wire: bool = current_op.in_qubits[0] == identity.out_qubits[0]
            if not acts_on_same_wire:
                # Then we know we can safely move it.
                current_op.detach()
                rewriter.insert_op(current_op, InsertPoint.before(identity))
                
                print_module(funcOp, "not acts_on_same_wire")
                
            if acts_on_same_wire:
                identity_definition = identity.in_qubits[0].owner
                print_module(funcOp, "acts_on_same_wire - Identity def")
                rewriter.replace_all_uses_with(identity.out_qubits[0], identity_definition.out_qubits[0])
                print_module(funcOp, "acts_on_same_wire - replace all uses")
                
                #rewriter.erase_op(identity)
                identity.detach()
                # current_op is now where it is supposed to be
                # we need to insert a new identity after current_op
                def just_for_test(use):
                    return use.operation == identity
                identity.in_qubits[0].replace_by_if(current_op.out_qubits[0], just_for_test)
            
                
                print_module(funcOp, "acts_on_same_wire - replace by if ==")
            
                rewriter.insert_op(identity, InsertPoint.after(current_op))
                
                print_module(funcOp, "acts_on_same_wire - insert")
                

                def just_for_test(use):
                    return use.operation != identity
                current_op.out_qubits[0].replace_by_if(identity.out_qubits[0], just_for_test)
                
                print_module(funcOp, "acts_on_same_wire")
                
                
            #    print(identity.next_op)
            #    breakpoint()     


@compiler_transform 
class TT_for_loop_Pass(passes.ModulePass):
    name = "tree-traversal"
    def apply(self, ctx: context.Context, module: builtin.ModuleOp) -> None:
        
        print_module(module, "Initial")
        self.apply_on_qnode(module, TT_ctrl_flow_for_loop())
        print_module(module, "After first pass")
        

    def apply_on_qnode(self, module: builtin.ModuleOp, pattern: pattern_rewriter.RewritePattern):
        rewriter = pattern_rewriter.PatternRewriter(module)

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

    match CASE_PASS:
        
        case "erick_help":
            @TT_for_loop_Pass
            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def captured_circuit(x: float):
                qml.Hadamard(wires=0)
                qml.S(wires=1)
                
                qml.Identity(wires=0)

                qml.X(wires=0)
                qml.Y(wires=0)
                
                qml.X(wires=1)
                qml.Y(wires=1)
                

                qml.Z(wires=0)
                return qml.state()
            
            @qml.qjit(keep_intermediate=False, pass_plugins=[getXDSLPluginAbsolutePath()], autograph=True)
            def main(x: float):
                return captured_circuit(x)

            print(main(1.0))

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def text_01():
                qml.Hadamard(wires=0)
                qml.S(wires=1)
                
                qml.X(wires=0)
                qml.Y(wires=0)
                
                qml.X(wires=1)
                qml.Y(wires=1)
                
                qml.Identity(wires=0)

                qml.Z(wires=0)
                return qml.state()

            print(text_01())
            
        case "example_02": 

            @TT_for_loop_Pass
            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def captured_circuit(x: float):

                # -------------------------------------------------
                # Example 01 and 02 
                qml.X(0)
                for i in range(5):
                    qml.H(0)
                    qml.S(0)
                qml.Z(0)        
                return qml.state()


            @qml.qjit(keep_intermediate=False, pass_plugins=[getXDSLPluginAbsolutePath()], autograph=True)
            def main(x: float):
                return captured_circuit(x)

            print(main(1.0))

            # After transform 

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def text_01():
                # qml.I(1)
                qml.X(0); qml.H(0); qml.S(0)

                for i in range(1,4):
                    qml.H(0)
                    qml.S(0)

                qml.H(0); qml.S(0); qml.Z(0)
                return qml.state()
            
            print(text_01())

        case "example_03": 
            @TT_for_loop_Pass
            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def captured_circuit(x: float):

                # -------------------------------------------------
                # Example 03 
                qml.X(0)
                for i in range(5):
                    qml.S(0)
                    qml.H(0)
                    qml.Y(0)
                    qml.H(0)
                    qml.T(0)
                qml.Z(0)        
                return qml.state()


            @qml.qjit(keep_intermediate=False, pass_plugins=[getXDSLPluginAbsolutePath()], autograph=True)
            def main(x: float):
                return captured_circuit(x)

            print(main(1.0))

            # After transform 

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def text_01():
                # qml.I(1)
                qml.X(0); qml.S(0); qml.H(0)

                for i in range(1,4):
                    qml.S(0)
                    qml.H(0)
                    
                    qml.Y(0)
                    
                    qml.H(0)
                    qml.T(0)

                qml.H(0); qml.T(0); qml.Z(0)
                return qml.state()
            
            print(text_01())
            
        case "example_04": 
            @TT_for_loop_Pass
            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def captured_circuit(x: float):

                # -------------------------------------------------
                # Example 04
                # qml.H(0); qml.H(1)
                qml.X(0)
                for i in range(5):
                    qml.S(0)
                    qml.H(0)
                    qml.Y(0)
                    qml.H(0)
                    qml.T(0)
                qml.Z(0)        
                return qml.state()


            @qml.qjit(keep_intermediate=False, pass_plugins=[getXDSLPluginAbsolutePath()], autograph=True)
            def main(x: float):
                return captured_circuit(x)

            print(main(1.0))

            # After transform 

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def text_01():
                # qml.H(0); qml.H(1)
                qml.X(0); qml.S(0); qml.H(0)

                for i in range(0,4):
                    qml.Y(0)                    
                    qml.H(0)
                    qml.T(0)
                    qml.S(0)
                    qml.H(0)

                qml.Y(0); qml.H(0);  qml.T(0); qml.Z(0)
                return qml.state()
            
            print(text_01())
            
            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def text_01():
                # qml.H(0); qml.H(1)
                qml.X(0)
                for i in range(5):
                    qml.S(0)
                    qml.Y(0)
                    qml.T(0)
                qml.Z(0)        
                return qml.state()
            
            print(text_01())

