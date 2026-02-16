
# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import qiskit
import qiskit
try:
    # Use mlir_quantum for Quantum dialect (custom)
    # from mlir_quantum.dialects import quantum
    quantum = None # Mock or unused
    # Try using standard MLIR package first
    try:
        from mlir.dialects import scf, func, arith
        from mlir.ir import Context, Module, Location, InsertionPoint, StringAttr, SymbolTable, Type, IntegerType, IndexType, IntegerAttr, Operation
    except ImportError:
        # Use jaxlib for standard dialects to match the runtime patched by catalyst
        from jaxlib.mlir.dialects import scf, func, arith
        # Use jaxlib for IR components
        from jaxlib.mlir.ir import Context, Module, Location, InsertionPoint, StringAttr, SymbolTable, Type, IntegerType, IndexType, IntegerAttr, Operation
except ImportError:
    # Fallback/Error handling
    try:
        from jaxlib.mlir.ir import Context, Module, Location, InsertionPoint, StringAttr, SymbolTable
        from jaxlib.mlir.dialects import scf, func, arith
        # construct quantum dialect manually? No, quantum dialect is in mlir_quantum.
        # Check if we can proceed without quantum dialect for some reason?
        pass 
    except ImportError:
        raise

print(f"DEBUG: mlir.ir imported from {Context.__module__}")
import sys
if 'mlir.ir' in sys.modules:
    print(f"DEBUG: sys.modules['mlir.ir'] = {sys.modules['mlir.ir']}")
if 'jaxlib.mlir.ir' in sys.modules:
    print(f"DEBUG: sys.modules['jaxlib.mlir.ir'] = {sys.modules['jaxlib.mlir.ir']}")

from catalyst.utils.exceptions import CompileError

class QiskitToCatalystImporter:
    """
    Imports a Qiskit QuantumCircuit into a Catalyst MLIR module.
    """
    def __init__(self, circuit: qiskit.QuantumCircuit):
        self.circuit = circuit
        self.ctx = Context()
        # self.ctx.load_all_dialects() # Not available in jaxlib < 0.4.x?
        self.ctx.allow_unregistered_dialects = True
        try:
             self.ctx.load_all_available_dialects()
        except AttributeError:
             pass 
        
        # Try explicit registration for quantum if needed
        try:
             # quantum.register_dialect(self.ctx)
             pass
        except AttributeError:
             pass

        self.module = Module.create(Location.unknown(self.ctx))
        self.qubit_map = {} # Maps qiskit.Qubit -> ir.Value
        self.global_qubit_reg = None

    def _emit_quantum_op(self, name, operands, result_types, loc, attrs=None):
        return Operation.create(name, results=result_types, operands=operands, attributes=attrs, loc=loc)

    def convert(self):
        """
        Converts the stored Qiskit circuit to an MLIR module.
        """
        with InsertionPoint(self.module.body):
            # Define main function with no arguments for now, assuming global register
            # In a real compiler, we might pass qubits as arguments.
            # For simplicity matching the instruction, we'll create a main function.
            
            # Note: Catalyst usually expects a function annotated with @qjit.
            # Here we are building the IR directly.
            
            # We need a location for operations
            loc = Location.unknown(self.ctx)
            
            # Create the main function
            func_type = func.FunctionType.get([], [], context=self.ctx)
            func_op = func.FuncOp("main", func_type, loc=loc)
            
            with InsertionPoint(func_op.add_entry_block()):
                # Allocate qubits or assume they act on a register. 
                # Catalyst quantum dialect typically uses `quantum.alloc` or pass arguments.
                # Let's allocate qubits based on the circuit width.
                
                num_qubits = self.circuit.num_qubits
                # We will map each qiskit qubit to an allocated qubit in MLIR
                
                for i, qubit in enumerate(self.circuit.qubits):
                    # Allocate a single qubit
                    # quantum.alloc returns a !quantum.reg, but we need !quantum.bit for operations
                    # Actually, standard catalyst usage often allocs a register.
                    # Let's use `quantum.alloc(n_qubits)` -> !quantum.reg
                    # and then `quantum.extract(reg, i)` -> !quantum.bit
                    pass

                # BETTER APPROACH for "no-cloning consistency":
                # In Catalyst/MLIR quantum dialect, gates populate SSA values.
                # `val_out = gate(val_in)`
                
                # So we need an initial source of qubits.
                # Let's emit `quantum.alloc` for each qubit for simplicity, 
                # or one alloc and extracts. One alloc is cleaner.
                
                # r_type = quantum.RegType.get(self.ctx)
                # Use Type.parse if specific class binding is missing
                from jaxlib.mlir.ir import Type, IntegerType, IndexType, IntegerAttr
                r_type = Type.parse("!quantum.reg", context=self.ctx)
                # idx_type = arith.IntegerType.get(self.ctx, 64) # Error
                idx_type = IntegerType.get_signless(64, self.ctx) # i64
                
                # Alloc constant for number of qubits
                # n_qubits_val = arith.ConstantOp(idx_type, num_qubits, loc=loc).result
                val_attr = IntegerAttr.get(idx_type, num_qubits)
                n_qubits_val = arith.ConstantOp(idx_type, val_attr, loc=loc).result
                
                # Allocate register
                # reg = quantum.AllocOp(r_type, n_qubits_val, n_attr=None, loc=loc).result
                reg = self._emit_quantum_op("quantum.alloc", [n_qubits_val], [r_type], loc).result
                
                # Extract all qubits and store in map
                for i, qubit in enumerate(self.circuit.qubits):
                    idx_attr = IntegerAttr.get(idx_type, i)
                    idx_val = arith.ConstantOp(idx_type, idx_attr, loc=loc).result
                    # ExtractOp(reg, idx) -> bit
                    # We need !quantum.bit type
                    bit_type = Type.parse("!quantum.bit", context=self.ctx)
                    # q_bit = quantum.ExtractOp(bit_type, reg, idx_val, loc=loc).result
                    q_bit = self._emit_quantum_op("quantum.extract", [reg, idx_val], [bit_type], loc).result
                    self.qubit_map[qubit] = q_bit
                
                # Process instructions
                self._process_instructions(self.circuit.data, loc)
                
                # Deallocate? In Catalyst, we might return them or dealloc. 
                # For now, let's just end the function.
                # We need to deallocate to be valid? 
                # Or re-assemble into register?
                # For this task, we focus on the operations.
                
                func.ReturnOp([], loc=loc)
                
        return self.module

    def _process_instructions(self, instructions, loc):
        for instruction in instructions:
            # instruction is a CircuitInstruction (op, qubits, clbits)
            op = instruction.operation
            qubits = instruction.qubits
            
            if op.name == "h":
                self._emit_gate("h", qubits, [], loc)
            elif op.name == "cx":
                self._emit_gate("cnot", qubits, [], loc)
            elif op.name == "for_loop":
                # Handling for_loop
                # op.params is [index_parameter, loop_parameter, body] 
                # wait, qiskit for_loop is complex. 
                # Usually: for_loop(index, start, stop, step)
                
                # Qiskit for_loop: (indexset, loop_parameter, body) or (start, stop, step, ...)
                # Assuming typical range usage.
                
                # We need to dig into the ControlFlowOp
                self._emit_for_loop(op, qubits, loc)
            elif op.name == "measure":
                self._emit_measure(qubits, loc)
                
            else:
                 # Provide a default for other gates as quantum.custom
                 self._emit_gate(op.name, qubits, op.params, loc)

    def _emit_measure(self, qubits, loc):
        for q in qubits:
            val_in = self.qubit_map[q]
            # quantum.measure(q) -> (bit, q_out)
            bit_type = Type.parse("!quantum.bit", context=self.ctx)
            i1_type = IntegerType.get_signless(1, self.ctx)
            
            op = self._emit_quantum_op("quantum.measure", [val_in], [i1_type, bit_type], loc)
            self.qubit_map[q] = op.results[1]

    def _emit_gate(self, gate_name, qubits, params, loc):
        # Retrieve current SSA values for qubits
        in_qubits = [self.qubit_map[q] for q in qubits]
        bit_type = Type.parse("!quantum.bit", context=self.ctx)
        result_types = [bit_type] * len(qubits)
        
        # Construct attributes for params if needed?
        # quantum.custom expects attributes? Or params as operand?
        # Catalyst uses `quantum.custom` "name"(operands) : types
        # But if it has params, they are usually attributes or operands.
        # For simplicity, we ignore params (or assume empty) for now as requested.
        
        # We need to use "quantum.custom" op name.
        # Attributes: gate_name string.
        
        # Wait, quantum.custom syntax: 
        # %0 = quantum.custom "h"() %in : !quantum.bit
        
        attrs = {"gate_name": StringAttr.get(gate_name, self.ctx)}
        
        op = self._emit_quantum_op("quantum.custom", in_qubits, result_types, loc, attrs)
        
        results = op.results
        for i, q in enumerate(qubits):
            self.qubit_map[q] = results[i]

    def _emit_for_loop(self, op, qubits, loc):
        # op is ForLoopOp
        # params: indexset, loop_parameter, body
        # indexset is usually (start, stop, step) or an iterable
        
        indexset, loop_param, body = op.params
        
        if len(indexset) != 3:
             # Handle list or other iterables? 
             # Task says: "Convert scf.for to for <var> in [start:step:stop]"
             # So we expect (start, stop, step)
             # Let's assume indexset is (start, stop, step)
             start, stop, step = 0, 1, 1 # default
             pass
        
        start, stop, step = indexset
        
        # Create constants
        idx_type = arith.IntegerType.get(self.ctx, 64)
        lb = arith.ConstantOp(idx_type, start, loc=loc).result
        ub = arith.ConstantOp(idx_type, stop, loc=loc).result
        st = arith.ConstantOp(idx_type, step, loc=loc).result
        
        # scf.for structure:
        # scf.ForOp(lb, ub, step, iter_args, loc)
        
        # We need to pass ALL qubits that might be modified as iter_args
        # This is tricky in SSA. We ideally find the "used" qubits in the body.
        # For simplicity, we pass ALL known qubits in the map? Expensive.
        # Or just the ones in `qubits` argument? 
        # The `qubits` argument to for_loop usually indicates the qubits the loop acts on.
        
        # Let's map the `qubits` to iter_args.
        iter_args_init = [self.qubit_map[q] for q in qubits]
        
        for_op = scf.ForOp(lb, ub, st, iter_args_init, loc=loc)
        
        # Enter body
        with InsertionPoint(for_op.body):
            # Argument 0 is induction var
            # Arguments 1..N are the current qubit states
            ind_var = for_op.induction_var
            block_args = for_op.inner_iter_args
            
            # Update qubit map for the body duration
            # We need to save old map to restore/reconcile?
            # Actually, inside the loop, we use the block_args.
            
            # Shadow the map
            old_map = self.qubit_map.copy()
            for i, q in enumerate(qubits):
                self.qubit_map[q] = block_args[i]
                
            # Process body
            self._process_instructions(body.data, loc)
            
            # Yield the final states of these qubits
            yield_args = [self.qubit_map[q] for q in qubits]
            scf.YieldOp(yield_args, loc=loc)
            
            # Restore map for outer scope (but we will update with results)
            self.qubit_map = old_map
            
        # Update map with loop results
        results = for_op.results
        for i, q in enumerate(qubits):
            self.qubit_map[q] = results[i]

