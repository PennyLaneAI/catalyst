# Standalone Qiskit Importer for Verification
# Bypasses catalyst package initialization issues

import qiskit

try:
    # Use standard MLIR package
    from mlir.dialects import scf, func, arith
    from mlir.ir import (
        Context,
        Module,
        Location,
        InsertionPoint,
        StringAttr,
        SymbolTable,
        Type,
        IntegerType,
        IndexType,
        IntegerAttr,
        FloatAttr,
        Operation,
        DenseI32ArrayAttr,
    )
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
        self.qubit_map = {}  # Maps qiskit.Qubit -> ir.Value
        self.clbit_map = {}  # Maps qiskit.Clbit -> ir.Value (i1)
        self.global_qubit_reg = None

    def _emit_quantum_op(self, name, operands, result_types, loc, attrs=None):
        return Operation.create(
            name, results=result_types, operands=operands, attributes=attrs, loc=loc
        )

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
                idx_type = IntegerType.get_signless(64, self.ctx)  # i64

                val_attr = IntegerAttr.get(idx_type, num_qubits)
                n_qubits_val = arith.ConstantOp(idx_type, val_attr, loc=loc).result

                # Allocate register
                reg = self._emit_quantum_op("quantum.alloc", [n_qubits_val], [r_type], loc).result

                # Extract all qubits and store in map
                for i, qubit in enumerate(self.circuit.qubits):
                    idx_attr = IntegerAttr.get(idx_type, i)
                    idx_val = arith.ConstantOp(idx_type, idx_attr, loc=loc).result
                    bit_type = Type.parse("!quantum.bit", context=self.ctx)
                    q_bit = self._emit_quantum_op(
                        "quantum.extract", [reg, idx_val], [bit_type], loc
                    ).result
                    self.qubit_map[qubit] = q_bit

                # Process instructions
                self._process_instructions(self.circuit.data, loc)

                func.ReturnOp([], loc=loc)

        return self.module

    def _process_instructions(self, instructions, loc):
        for instruction in instructions:
            op = instruction.operation
            qubits = instruction.qubits

            # Check for legacy QASM2 classical condition FIRST so that named
            # gates like "h" or "cx" with op.condition don't bypass it.
            if getattr(op, "condition", None) is not None:
                self._emit_conditional_gate(op, qubits, loc)
            elif op.name == "h":
                self._emit_gate("h", qubits, [], loc)
            elif op.name == "cx":
                self._emit_gate("cnot", qubits, [], loc)
            elif op.name == "for_loop":
                self._emit_for_loop(op, qubits, loc)
            elif op.name == "measure":
                self._emit_measure(qubits, instruction.clbits, loc)
            elif op.name == "if_else":
                self._emit_if_else(op, qubits, instruction.clbits, loc)
            elif type(op).__name__ == "IfElseOp":
                # Some Qiskit versions denote it directly by type, with condition in properties
                condition = getattr(op, "condition", None)
                if condition:
                    # Usually (clbit, value)
                    clbits = [condition[0]]
                elif instruction.clbits:
                    clbits = instruction.clbits
                else:
                    clbits = []
                # In Qiskit, if_else touches all qubits in its body potentially,
                # but instruction.qubits will contain them.
                self._emit_if_else(op, qubits, clbits, loc)
            else:
                self._emit_gate(op.name, qubits, op.params, loc)

    def _emit_measure(self, qubits, clbits, loc):
        for i, q in enumerate(qubits):
            val_in = self.qubit_map[q]
            bit_type = Type.parse("!quantum.bit", context=self.ctx)
            i1_type = IntegerType.get_signless(1, self.ctx)

            op = self._emit_quantum_op("quantum.measure", [val_in], [i1_type, bit_type], loc)
            self.qubit_map[q] = op.results[1]
            if i < len(clbits):
                self.clbit_map[clbits[i]] = op.results[0]

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
        f64_type = Type.parse("f64", context=self.ctx)

        with loc:
            for p in params:
                # Assume p is a number (float/int)
                try:
                    val = float(p)
                except (ValueError, TypeError):
                    # Skip complex params or expressions for now
                    continue

                val_attr = FloatAttr.get(f64_type, val)
                val_op = arith.ConstantOp(f64_type, val_attr, loc=loc)
                op_params.append(val_op.result)

        op_ctrls = []
        op_ctrlvals = []

        all_operands = op_params + in_qubits + op_ctrls + op_ctrlvals

        # Calculate sizes
        sizes = [len(op_params), len(in_qubits), len(op_ctrls), len(op_ctrlvals)]
        result_sizes = [len(result_types), 0]  # 0 for out_ctrl_qubits

        attrs = {
            "gate_name": StringAttr.get(gate_name, self.ctx),
            "operandSegmentSizes": DenseI32ArrayAttr.get(sizes, context=self.ctx),
            "resultSegmentSizes": DenseI32ArrayAttr.get(result_sizes, context=self.ctx),
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

    def _emit_conditional_gate(self, op, qubits, loc):
        """Wrap a gate with op.condition=(ClassicalRegister, value) in nested scf.if blocks."""
        cond_reg, cond_val = op.condition
        i1_type = IntegerType.get_signless(1, self.ctx)
        bit_type = Type.parse("!quantum.bit", context=self.ctx)
        result_types = [bit_type] * len(qubits)

        # Build one condition per measured bit in the register.
        # XOrIOp encodes expected==0 so the C++ translator can emit "name == false".
        conditions = []
        for bit_idx, clbit in enumerate(cond_reg):
            if clbit not in self.clbit_map:
                continue
            expected = (cond_val >> bit_idx) & 1
            measured = self.clbit_map[clbit]
            if expected == 0:
                one = arith.ConstantOp(i1_type, IntegerAttr.get(i1_type, 1), loc=loc).result
                measured = arith.XOrIOp(measured, one, loc=loc).result
            conditions.append(measured)

        if not conditions:
            self._emit_gate(op.name, qubits, op.params, loc)
            return

        # Generate nested scf.if blocks — one level per bit condition.
        # The innermost level emits the actual gate; each outer level passes
        # qubits through its else branch unchanged.
        def emit_nested(depth):
            if depth == len(conditions):
                self._emit_gate(op.name, qubits, op.params, loc)
                return

            passthrough = [self.qubit_map[q] for q in qubits]
            if_op = scf.IfOp(conditions[depth], result_types, hasElse=True, loc=loc)

            with InsertionPoint(if_op.then_block):
                old_map = self.qubit_map.copy()
                emit_nested(depth + 1)
                scf.YieldOp([self.qubit_map[q] for q in qubits], loc=loc)
                self.qubit_map = old_map

            with InsertionPoint(if_op.else_block):
                scf.YieldOp(passthrough, loc=loc)

            for i, q in enumerate(qubits):
                self.qubit_map[q] = if_op.results[i]

        emit_nested(0)

    def _emit_if_else(self, op, qubits, clbits, loc):
        # op.params = [true_body, false_body]
        true_body = op.params[0]
        false_body = op.params[1]

        if not clbits:
            raise CompileError("if_else instruction requires classical bits for condition")

        cond_bit = clbits[0]
        if cond_bit not in self.clbit_map:
            raise CompileError(
                f"Classical bit {cond_bit} used in if_else but not measured previously"
            )

        cond_val = self.clbit_map[cond_bit]  # i1

        # Determine qubits involved in the if_else block
        # The 'qubits' argument to this function *should* contain all qubits touched by the body.

        # Ensure hasElse=True because we must yield results (qubits) from both branches
        if_op = scf.IfOp(cond_val, [self.qubit_map[q].type for q in qubits], hasElse=True, loc=loc)

        # True Body
        with InsertionPoint(if_op.then_block):
            old_map = self.qubit_map.copy()
            self._process_instructions(true_body.data, loc)
            yield_results = [self.qubit_map[q] for q in qubits]
            scf.YieldOp(yield_results, loc=loc)
            self.qubit_map = old_map

        # False Body
        with InsertionPoint(if_op.else_block):
            if false_body is not None:
                old_map = self.qubit_map.copy()
                self._process_instructions(false_body.data, loc)
                yield_results = [self.qubit_map[q] for q in qubits]
                scf.YieldOp(yield_results, loc=loc)
                self.qubit_map = old_map
            else:
                # Default pass-through
                yield_results = [self.qubit_map[q] for q in qubits]
                scf.YieldOp(yield_results, loc=loc)

        # Update mappings with if_op results
        results = if_op.results
        for i, q in enumerate(qubits):
            self.qubit_map[q] = results[i]
