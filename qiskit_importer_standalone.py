# Standalone Qiskit Importer for Verification
# Bypasses catalyst package initialization issues

import qiskit
from qiskit.circuit import Clbit

try:
    # Use standard MLIR package
    from mlir.dialects import arith, func, scf
    from mlir.ir import (
        Context,
        DenseI32ArrayAttr,
        FloatAttr,
        IndexType,
        InsertionPoint,
        IntegerAttr,
        IntegerType,
        Location,
        Module,
        Operation,
        StringAttr,
        SymbolTable,
        Type,
        UnitAttr,
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

    # Control-flow operations carry their own op.condition; they must not be
    # routed through the legacy single-gate conditional path.
    _CONTROL_FLOW_OPS = {"if_else", "while_loop", "for_loop", "switch_case"}

    # Gates the QASM3 emitter can express directly (stdgates.inc, the builtin
    # U, gates with hard-coded defs, and the reset/barrier markers). Anything
    # else with a Qiskit definition gets inlined into these.
    _KNOWN_GATES = {
        "h", "x", "y", "z", "s", "t", "sdg", "tdg", "sx", "sxdg",
        "p", "phase", "id", "u", "u1", "u2", "u3",
        "rx", "ry", "rz",
        "cx", "cnot", "cy", "cz", "ch", "cp", "cu", "cu1", "cu3",
        "crx", "cry", "crz",
        "swap", "ccx", "cswap",
        "rzz", "rxx", "ryy", "rccx",
        "reset", "barrier",
    }

    def _process_instructions(self, instructions, loc):
        for instruction in instructions:
            op = instruction.operation
            qubits = instruction.qubits

            # Check for legacy QASM2 classical condition FIRST so that named
            # gates like "h" or "cx" with op.condition don't bypass it.
            if (
                op.name not in self._CONTROL_FLOW_OPS
                and getattr(op, "condition", None) is not None
            ):
                self._emit_conditional_gate(op, qubits, loc)
            elif op.name == "h":
                self._emit_gate("h", qubits, [], loc)
            elif op.name == "cx":
                self._emit_gate("cnot", qubits, [], loc)
            elif op.name == "reset":
                self._emit_gate("reset", qubits, [], loc)
            elif op.name == "barrier":
                self._emit_gate("barrier", qubits, [], loc)
            elif op.name == "for_loop":
                self._emit_for_loop(op, qubits, loc)
            elif op.name == "while_loop":
                self._emit_while_loop(op, qubits, instruction.clbits, loc)
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
                self._emit_gate_or_inline(op, qubits, loc)

    def _emit_gate_or_inline(self, op, qubits, loc):
        """Emit a gate directly if the QASM3 emitter knows it; otherwise
        inline its Qiskit definition."""
        if op.name not in self._KNOWN_GATES and getattr(op, "definition", None) is not None:
            self._inline_gate_definition(op, qubits, loc)
        else:
            self._emit_gate(op.name, qubits, op.params, loc)

    def _inline_gate_definition(self, op, qubits, loc):
        """Inline a user-defined gate by emitting its (parameter-bound)
        definition body on the instruction's qubits. An empty definition is
        the identity and emits nothing (e.g. `gate post q { }`)."""
        body = op.definition
        for bq, oq in zip(body.qubits, qubits):
            self.qubit_map[bq] = self.qubit_map[oq]

        self._process_instructions(body.data, loc)

        for bq, oq in zip(body.qubits, qubits):
            self.qubit_map[oq] = self.qubit_map[bq]

    def _creg_attrs(self, clbit):
        """Locate the classical register owning `clbit` and return MLIR attrs
        (creg_name/creg_idx/creg_size) describing it, or None for loose bits.

        The translator uses these to emit `bit[n] c;` declarations and
        `c[i] = measure q[j];` assignments instead of anonymous bits — which
        is also what makes while-loop conditions re-assignable in QASM3.
        """
        try:
            registers = self.circuit.find_bit(clbit).registers
        except Exception:
            return None
        if not registers:
            return None
        reg, idx = registers[0]
        i64_type = IntegerType.get_signless(64, self.ctx)
        return {
            "creg_name": StringAttr.get(reg.name, self.ctx),
            "creg_idx": IntegerAttr.get(i64_type, idx),
            "creg_size": IntegerAttr.get(i64_type, reg.size),
        }

    def _emit_measure(self, qubits, clbits, loc):
        for i, q in enumerate(qubits):
            val_in = self.qubit_map[q]
            bit_type = Type.parse("!quantum.bit", context=self.ctx)
            i1_type = IntegerType.get_signless(1, self.ctx)

            attrs = self._creg_attrs(clbits[i]) if i < len(clbits) else None
            op = self._emit_quantum_op(
                "quantum.measure", [val_in], [i1_type, bit_type], loc, attrs
            )
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

    def _build_condition_value(self, target, value, loc, skip_unmeasured=False):
        """Fold a Qiskit condition (Clbit or ClassicalRegister, expected int)
        into a single i1 SSA value: AND of per-bit tests, with arith.xori for
        expected-0 bits. The translator renders it as e.g. `c[0] && !c[1]`.

        Returns None if no condition bits are available. Raises CompileError
        for unmeasured bits unless skip_unmeasured is set (legacy behavior).
        """
        i1_type = IntegerType.get_signless(1, self.ctx)
        bits = [target] if isinstance(target, Clbit) else list(target)

        conditions = []
        for bit_idx, clbit in enumerate(bits):
            expected = (value >> bit_idx) & 1
            if clbit not in self.clbit_map:
                if skip_unmeasured:
                    continue
                # QASM3 bits default to 0 until measured: an expected-0 test
                # on an unmeasured bit is trivially true (skip); an expected-1
                # test makes the whole conjunction constant false.
                if expected == 0:
                    continue
                return arith.ConstantOp(
                    i1_type, IntegerAttr.get(i1_type, 0), loc=loc
                ).result
            measured = self.clbit_map[clbit]
            if expected == 0:
                one = arith.ConstantOp(i1_type, IntegerAttr.get(i1_type, 1), loc=loc).result
                measured = arith.XOrIOp(measured, one, loc=loc).result
            conditions.append(measured)

        if not conditions:
            return None

        cond = conditions[0]
        for c in conditions[1:]:
            cond = arith.AndIOp(cond, c, loc=loc).result
        if len(conditions) > 1:
            # Mark this conjunction as a register-equality fold so the QASM3
            # translator may soundly reconstruct `reg == value` from it.
            # Generic logic ANDs (e.g. c[0] && c[1] on a wider register) must
            # NOT carry this tag — they don't constrain unmentioned bits.
            cond.owner.attributes["qasm3_creg_eq"] = UnitAttr.get(self.ctx)
        return cond

    def _emit_conditional_gate(self, op, qubits, loc):
        """Wrap a gate with op.condition=(ClassicalRegister|Clbit, value) in a
        single scf.if whose condition AND-folds all register bits."""
        cond_reg, cond_val = op.condition
        bit_type = Type.parse("!quantum.bit", context=self.ctx)
        result_types = [bit_type] * len(qubits)

        cond = self._build_condition_value(cond_reg, cond_val, loc, skip_unmeasured=True)
        if cond is None:
            self._emit_gate_or_inline(op, qubits, loc)
            return

        passthrough = [self.qubit_map[q] for q in qubits]
        if_op = scf.IfOp(cond, result_types, hasElse=True, loc=loc)

        with InsertionPoint(if_op.then_block):
            old_map = self.qubit_map.copy()
            self._emit_gate_or_inline(op, qubits, loc)
            scf.YieldOp([self.qubit_map[q] for q in qubits], loc=loc)
            self.qubit_map = old_map

        with InsertionPoint(if_op.else_block):
            scf.YieldOp(passthrough, loc=loc)

        for i, q in enumerate(qubits):
            self.qubit_map[q] = if_op.results[i]

    def _resolve_condition(self, condition, clbits, loc):
        """Turn a control-flow op condition into a single i1 SSA value.

        Accepts Qiskit's (Clbit|ClassicalRegister, value) tuples; falls back
        to the first associated clbit when no condition is attached. Classical
        expressions (qiskit.circuit.classical.expr) are not supported (P2).
        """
        if condition is not None:
            return self._build_condition(condition, loc)

        if not clbits:
            raise CompileError("if_else instruction requires classical bits for condition")

        cond_bit = clbits[0]
        if cond_bit not in self.clbit_map:
            raise CompileError(
                f"Classical bit {cond_bit} used in if_else but not measured previously"
            )
        return self.clbit_map[cond_bit]  # i1

    @staticmethod
    def _unpack_condition(condition):
        try:
            target, value = condition
        except (TypeError, ValueError):
            # Not a (target, value) tuple — e.g. a classical expr.Expr node.
            raise CompileError(
                f"Unsupported condition type {type(condition).__name__}: only "
                "(Clbit|ClassicalRegister, value) conditions are supported"
            )
        return target, value

    def _build_condition(self, condition, loc):
        """Lower a control-flow condition — (Clbit|ClassicalRegister, value)
        tuple or qiskit classical expr tree — to a single i1 SSA value."""
        if isinstance(condition, tuple):
            target, value = self._unpack_condition(condition)
            cond = self._build_condition_value(target, value, loc)
            if cond is None:
                raise CompileError("Control-flow condition has no classical bits")
            return cond
        return self._condition_from_expr(condition, loc)

    def _condition_clbits(self, condition):
        """Ordered, deduplicated clbits a condition depends on."""
        bits = []
        if isinstance(condition, tuple):
            target, _ = self._unpack_condition(condition)
            bits = [target] if isinstance(target, Clbit) else list(target)
        else:
            from qiskit.circuit.classical import expr as qexpr

            def walk(n):
                if isinstance(n, qexpr.Var):
                    t = n.var
                    bits.extend([t] if isinstance(t, Clbit) else list(t))
                elif isinstance(n, qexpr.Unary):
                    walk(n.operand)
                elif isinstance(n, qexpr.Binary):
                    walk(n.left)
                    walk(n.right)

            walk(condition)
        seen, ordered = set(), []
        for b in bits:
            if b not in seen:
                seen.add(b)
                ordered.append(b)
        return ordered

    def _condition_from_expr(self, node, loc):
        """Lower a qiskit classical expr tree (qiskit.circuit.classical.expr)
        to a single i1 SSA value. Supports Var(clbit/creg), Value, unary
        BIT_NOT/LOGIC_NOT, and binary EQUAL/NOT_EQUAL/LOGIC_AND/LOGIC_OR."""
        from qiskit.circuit.classical import expr as qexpr

        i1_type = IntegerType.get_signless(1, self.ctx)

        def negate(val):
            one = arith.ConstantOp(i1_type, IntegerAttr.get(i1_type, 1), loc=loc).result
            return arith.XOrIOp(val, one, loc=loc).result

        def lower(n):
            if isinstance(n, qexpr.Var):
                target = n.var
                if isinstance(target, Clbit):
                    if target not in self.clbit_map:
                        raise CompileError(
                            f"Classical bit {target} used in condition but not "
                            "measured previously"
                        )
                    return ("bit", self.clbit_map[target])
                return ("reg", target)
            if isinstance(n, qexpr.Value):
                return ("const", int(n.value))
            if isinstance(n, qexpr.Unary):
                if n.op in (qexpr.Unary.Op.BIT_NOT, qexpr.Unary.Op.LOGIC_NOT):
                    kind, operand = lower(n.operand)
                    if kind != "bit":
                        raise CompileError("NOT on non-bit expr condition")
                    return ("bit", negate(operand))
                raise CompileError(f"Unsupported unary expr op: {n.op}")
            if isinstance(n, qexpr.Binary):
                op = n.op
                if op in (qexpr.Binary.Op.EQUAL, qexpr.Binary.Op.NOT_EQUAL):
                    lhs, rhs = lower(n.left), lower(n.right)
                    if lhs[0] == "const":
                        lhs, rhs = rhs, lhs
                    if rhs[0] != "const":
                        raise CompileError(
                            "expr comparison requires one constant side"
                        )
                    value = rhs[1]
                    if lhs[0] == "reg":
                        cond = self._build_condition_value(lhs[1], value, loc)
                        if cond is None:
                            raise CompileError("expr condition has no bits")
                    else:
                        cond = lhs[1]
                        if value == 0:
                            cond = negate(cond)
                    if op == qexpr.Binary.Op.NOT_EQUAL:
                        cond = negate(cond)
                    return ("bit", cond)
                if op in (qexpr.Binary.Op.LOGIC_AND, qexpr.Binary.Op.LOGIC_OR):
                    lk, lv = lower(n.left)
                    rk, rv = lower(n.right)
                    if lk != "bit" or rk != "bit":
                        raise CompileError("logic expr requires bit operands")
                    if op == qexpr.Binary.Op.LOGIC_AND:
                        return ("bit", arith.AndIOp(lv, rv, loc=loc).result)
                    return ("bit", arith.OrIOp(lv, rv, loc=loc).result)
                raise CompileError(f"Unsupported binary expr op: {op}")
            raise CompileError(f"Unsupported expr node: {type(n).__name__}")

        kind, val = lower(node)
        if kind == "reg":
            # Bare register truthiness: reg != 0
            cond = self._build_condition_value(val, 0, loc)
            if cond is None:
                raise CompileError("expr condition has no bits")
            return negate(cond)
        if kind == "const":
            raise CompileError("Constant-only expr condition is not supported")
        return val

    def _emit_if_else(self, op, qubits, clbits, loc):
        # op.params = [true_body, false_body]
        true_body = op.params[0]
        false_body = op.params[1]

        cond_val = self._resolve_condition(getattr(op, "condition", None), clbits, loc)

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

    def _emit_while_loop(self, op, qubits, clbits, loc):
        """Lower Qiskit while_loop to scf.while.

        Loop-carried values are [condition clbits...] + [qubits...]. The
        before region recomputes the boolean from its block arguments and
        forwards all of them via scf.condition; the body re-measures the
        condition clbits and yields the new values. The QASM3 translator maps
        this back to `while (c[i]) { ... }` where the in-body measurement
        re-assigns the same named register bit.
        """
        condition = getattr(op, "condition", None)
        if condition is None:
            raise CompileError("while_loop instruction requires a condition")
        cond_clbits = self._condition_clbits(condition)

        for cb in cond_clbits:
            if cb not in self.clbit_map:
                raise CompileError(
                    f"Classical bit {cb} used in while_loop condition but not "
                    "measured previously"
                )

        body = op.params[0]
        i1_type = IntegerType.get_signless(1, self.ctx)

        # Carry ALL clbits associated with the loop (condition bits first, so
        # the before-region indices line up), not just the condition bits —
        # otherwise a non-condition bit measured inside the body would be
        # invisible to code after the loop. Bits never measured before the
        # loop start as constant 0, matching QASM3's default bit value.
        carried_clbits = list(cond_clbits)
        for cb in clbits:
            if cb not in carried_clbits:
                carried_clbits.append(cb)

        init_clbit_vals = []
        for cb in carried_clbits:
            if cb in self.clbit_map:
                init_clbit_vals.append(self.clbit_map[cb])
            else:
                zero = arith.ConstantOp(i1_type, IntegerAttr.get(i1_type, 0), loc=loc).result
                init_clbit_vals.append(zero)

        inits = init_clbit_vals + [self.qubit_map[q] for q in qubits]
        arg_types = [v.type for v in inits]

        with loc:
            while_op = scf.WhileOp(arg_types, inits, loc=loc)
            while_op.before.blocks.append(*arg_types)
            while_op.after.blocks.append(*arg_types)
        before_block = while_op.before.blocks[0]
        after_block = while_op.after.blocks[0]

        # Before region: rebuild the boolean from the loop-carried clbits and
        # forward ALL arguments to the after region. The clbit_map is
        # temporarily rebound to the block arguments so the generic condition
        # builder (tuple or expr form) reads loop-carried values.
        with InsertionPoint(before_block):
            before_args = list(before_block.arguments)
            saved_clbit_map = self.clbit_map.copy()
            for i, cb in enumerate(cond_clbits):
                self.clbit_map[cb] = before_args[i]
            cond = self._build_condition(condition, loc)
            self.clbit_map = saved_clbit_map
            scf.ConditionOp(cond, before_args, loc=loc)

        # After region (loop body): rebind maps to block arguments, process the
        # body, then yield the updated clbit/qubit values.
        with InsertionPoint(after_block):
            after_args = list(after_block.arguments)
            old_qubit_map = self.qubit_map.copy()
            old_clbit_map = self.clbit_map.copy()

            for i, cb in enumerate(carried_clbits):
                self.clbit_map[cb] = after_args[i]
            for i, q in enumerate(qubits):
                self.qubit_map[q] = after_args[len(carried_clbits) + i]

            # Builder-style bodies reuse the outer Qubit/Clbit objects; the
            # explicit while_loop(...) form may use body-local bits, which
            # correspond positionally to instruction.qubits/clbits.
            for bq, oq in zip(body.qubits, qubits):
                if bq not in self.qubit_map:
                    self.qubit_map[bq] = self.qubit_map[oq]
            for bc, oc in zip(body.clbits, clbits):
                if bc not in self.clbit_map and oc in self.clbit_map:
                    self.clbit_map[bc] = self.clbit_map[oc]

            self._process_instructions(body.data, loc)

            # Propagate body-local bit updates back to the outer bit objects.
            for bq, oq in zip(body.qubits, qubits):
                if bq is not oq:
                    self.qubit_map[oq] = self.qubit_map[bq]
            for bc, oc in zip(body.clbits, clbits):
                if bc is not oc and bc in self.clbit_map:
                    self.clbit_map[oc] = self.clbit_map[bc]

            yield_vals = [self.clbit_map[cb] for cb in carried_clbits] + [
                self.qubit_map[q] for q in qubits
            ]
            scf.YieldOp(yield_vals, loc=loc)

            self.qubit_map = old_qubit_map
            self.clbit_map = old_clbit_map

        # Rebind outer maps to the loop results.
        results = while_op.results
        for i, cb in enumerate(carried_clbits):
            self.clbit_map[cb] = results[i]
        for i, q in enumerate(qubits):
            self.qubit_map[q] = results[len(carried_clbits) + i]
