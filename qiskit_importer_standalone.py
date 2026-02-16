
# Standalone Qiskit Importer for Verification
# Bypasses catalyst package initialization issues

import qiskit
try:
    # Use standard MLIR package
    from mlir.dialects import scf, func, arith
    from mlir.ir import Context, Module, Location, InsertionPoint, StringAttr, SymbolTable, Type, IntegerType, IndexType, IntegerAttr, Operation, DenseI32ArrayAttr
except ImportError:
    # Error out if mlir not found
    raise ImportError("Could not import mlir package. Ensure PYTHONPATH includes mlir_core.")

class CompileError(Exception):
    pass

class QiskitToCatalystImporter:
    """
    Imports a Qiskit QuantumCircuit into a Catalyst MLIR module.
    """
    def __init__(self, circuit: qiskit.QuantumCircuit):
        self.circuit = circuit
        self.ctx = Context()
        self.ctx.allow_unregistered_dialects = True
        try:
             self.ctx.load_all_available_dialects()
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
            loc = Location.unknown(self.ctx)
            func_type = func.FunctionType.get([], [], context=self.ctx)
            func_op = func.FuncOp("main", func_type, loc=loc)
            
            with InsertionPoint(func_op.add_entry_block()):
                num_qubits = self.circuit.num_qubits
                
                # r_type = quantum.RegType.get(self.ctx)
                r_type = Type.parse("!quantum.reg", context=self.ctx)
                idx_type = IntegerType.get_signless(64, self.ctx) # i64
                
                val_attr = IntegerAttr.get(idx_type, num_qubits)
                n_qubits_val = arith.ConstantOp(idx_type, val_attr, loc=loc).result
                
                # Allocate register
                reg = self._emit_quantum_op("quantum.alloc", [n_qubits_val], [r_type], loc).result
                
                # Extract all qubits and store in map
                for i, qubit in enumerate(self.circuit.qubits):
                    idx_attr = IntegerAttr.get(idx_type, i)
                    idx_val = arith.ConstantOp(idx_type, idx_attr, loc=loc).result
                    bit_type = Type.parse("!quantum.bit", context=self.ctx)
                    q_bit = self._emit_quantum_op("quantum.extract", [reg, idx_val], [bit_type], loc).result
                    self.qubit_map[qubit] = q_bit
                
                # Process instructions
                self._process_instructions(self.circuit.data, loc)
                
                func.ReturnOp([], loc=loc)
                
        return self.module

    def _process_instructions(self, instructions, loc):
        for instruction in instructions:
            op = instruction.operation
            qubits = instruction.qubits
            
            if op.name == "h":
                self._emit_gate("h", qubits, [], loc)
            elif op.name == "cx":
                self._emit_gate("cnot", qubits, [], loc)
            elif op.name == "for_loop":
                self._emit_for_loop(op, qubits, loc)
            elif op.name == "measure":
                self._emit_measure(qubits, loc)
            else:
                 self._emit_gate(op.name, qubits, op.params, loc)

    def _emit_measure(self, qubits, loc):
        for q in qubits:
            val_in = self.qubit_map[q]
            bit_type = Type.parse("!quantum.bit", context=self.ctx)
            i1_type = IntegerType.get_signless(1, self.ctx)
            
            op = self._emit_quantum_op("quantum.measure", [val_in], [i1_type, bit_type], loc)
            self.qubit_map[q] = op.results[1]

    def _emit_gate(self, gate_name, qubits, params, loc):
        in_qubits = [self.qubit_map[q] for q in qubits]
        bit_type = Type.parse("!quantum.bit", context=self.ctx)
        result_types = [bit_type] * len(qubits)
        
        # quantum.custom arguments:
        # Variadic<F64>:$params,
        # Variadic<QubitType>:$in_qubits,
        # StrAttr:$gate_name,
        # UnitAttr:$adjoint,
        # Variadic<QubitType>:$in_ctrl_qubits,
        # Variadic<I1>:$in_ctrl_values
        
        # We need to provide operands for ALL variadic segments.
        # operands = params + in_qubits + ctrls + ctrlvals
        
        # Params are currently ignored/empty in this simplified imported
        # If we had params, they should be F64 values.
        op_params = [] 
        op_ctrls = []
        op_ctrlvals = []
        
        all_operands = op_params + in_qubits + op_ctrls + op_ctrlvals
        
        # Calculate sizes
        sizes = [len(op_params), len(in_qubits), len(op_ctrls), len(op_ctrlvals)]
        result_sizes = [len(result_types), 0] # 0 for out_ctrl_qubits

        attrs = {
            "gate_name": StringAttr.get(gate_name, self.ctx),
            "operandSegmentSizes": DenseI32ArrayAttr.get(sizes, context=self.ctx),
            "resultSegmentSizes": DenseI32ArrayAttr.get(result_sizes, context=self.ctx)
        }
        
        op = self._emit_quantum_op("quantum.custom", all_operands, result_types, loc, attrs)
        
        results = op.results
        for i, q in enumerate(qubits):
            self.qubit_map[q] = results[i]

    def _emit_for_loop(self, op, qubits, loc):
        indexset, loop_param, body = op.params
        
        if len(indexset) != 3:
             start, stop, step = 0, 1, 1 
        else:
             start, stop, step = indexset
        
        idx_type = arith.IntegerType.get(self.ctx, 64)
        lb = arith.ConstantOp(idx_type, start, loc=loc).result
        ub = arith.ConstantOp(idx_type, stop, loc=loc).result
        st = arith.ConstantOp(idx_type, step, loc=loc).result
        
        iter_args_init = [self.qubit_map[q] for q in qubits]
        
        for_op = scf.ForOp(lb, ub, st, iter_args_init, loc=loc)
        
        with InsertionPoint(for_op.body):
            ind_var = for_op.induction_var
            block_args = for_op.inner_iter_args
            
            old_map = self.qubit_map.copy()
            for i, q in enumerate(qubits):
                self.qubit_map[q] = block_args[i]
                
            self._process_instructions(body.data, loc)
            
            yield_args = [self.qubit_map[q] for q in qubits]
            scf.YieldOp(yield_args, loc=loc)
            
            self.qubit_map = old_map
            
        results = for_op.results
        for i, q in enumerate(qubits):
            self.qubit_map[q] = results[i]
