"""Hybrid OpenQASM 3 frontend for the QCC pipeline.

``load_qasm3()`` first tries qiskit's QASM3 importer (fast path, best dynamic-
circuit support). When qiskit cannot parse the program, it falls back to a
partial evaluator built on the reference ``openqasm3`` parser (full grammar
coverage) that lowers static OpenQASM 3 constructs qiskit does not understand:

- const declarations and compile-time classical types (int/uint/float/angle/bool)
- classical expressions, casts and math functions evaluated at compile time
- def subroutines (inlined; measured-bit returns bind to the assignment target)
- user gate definitions (built as qiskit gates) and gate modifiers
  (ctrl @ / negctrl @ / inv @ / pow(k) @)
- for loops (unrolled), let aliases, register broadcasting
- runtime feedforward: if/while on measurement bits, including
  ``int[n](creg) == k`` casts and ``creg != k`` (via qiskit classical expr)

Constructs with no circuit semantics raise ``QASM3FrontendError`` with a clear
reason: extern functions, defcal/calibration, timing (delay/stretch/duration/
box), and runtime classical arithmetic feeding gate parameters.
"""

import math
import re
from pathlib import Path

import openqasm3
from openqasm3 import ast

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Clbit, Qubit, Gate
from qiskit.circuit.classical import expr as qexpr


class QASM3FrontendError(Exception):
    """Raised when a program uses constructs with no circuit representation."""


class _Return(Exception):
    """Internal unwind for subroutine return statements."""

    def __init__(self, value):
        self.value = value


_CONSTANTS = {
    "pi": math.pi, "π": math.pi,
    "tau": math.tau, "τ": math.tau,
    "euler": math.e, "ℇ": math.e,
}

_MATH_FUNCS = {
    "arccos": math.acos, "arcsin": math.asin, "arctan": math.atan,
    "cos": math.cos, "sin": math.sin, "tan": math.tan,
    "exp": math.exp, "log": math.log, "ln": math.log,
    "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil,
    "mod": lambda a, b: a % b, "abs": abs,
    "popcount": lambda v: bin(int(v)).count("1"),
}

# stdgates.inc / builtin gate name -> (qiskit method name, n_params, n_qubits)
_STD_GATES = {
    "U": ("u", 3, 1), "CX": ("cx", 0, 2), "gphase": (None, 1, 0),
    "p": ("p", 1, 1), "phase": ("p", 1, 1), "x": ("x", 0, 1), "y": ("y", 0, 1),
    "z": ("z", 0, 1), "h": ("h", 0, 1), "s": ("s", 0, 1), "sdg": ("sdg", 0, 1),
    "t": ("t", 0, 1), "tdg": ("tdg", 0, 1), "sx": ("sx", 0, 1),
    "rx": ("rx", 1, 1), "ry": ("ry", 1, 1), "rz": ("rz", 1, 1),
    "cx": ("cx", 0, 2), "cy": ("cy", 0, 2), "cz": ("cz", 0, 2),
    "cp": ("cp", 1, 2), "cphase": ("cp", 1, 2), "crx": ("crx", 1, 2),
    "cry": ("cry", 1, 2), "crz": ("crz", 1, 2), "ch": ("ch", 0, 2),
    "swap": ("swap", 0, 2), "ccx": ("ccx", 0, 3), "cswap": ("cswap", 0, 3),
    "cu": ("cu", 4, 2), "id": ("id", 0, 1),
    "u1": ("p", 1, 1), "u2": ("u", 2, 1), "u3": ("u", 3, 1),
}


class _BitRef:
    """A runtime classical value: an ordered list of Clbits (LSB first),
    optionally the whole underlying register."""

    def __init__(self, clbits, register=None, initial=None):
        self.clbits = list(clbits)
        self.register = register
        # Known compile-time initial value (before any measurement), or None.
        self.initial = initial

    def __len__(self):
        return len(self.clbits)


class _QubitRef:
    """An ordered list of Qubits."""

    def __init__(self, qubits):
        self.qubits = list(qubits)

    def __len__(self):
        return len(self.qubits)


class _GateDef:
    def __init__(self, node):
        self.node = node


class _SubroutineDef:
    def __init__(self, node):
        self.node = node


class AstFrontend:
    """Evaluates an openqasm3 AST into a qiskit QuantumCircuit."""

    MAX_UNROLL = 10000

    def __init__(self, inputs=None):
        self.inputs = dict(inputs or {})
        self.qc = None
        self.scopes = [{}]
        self._name_counter = {}
        self._return_targets = []

    # ------------------------------------------------------------- scopes
    def _lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise QASM3FrontendError(f"Identifier '{name}' is not defined")

    def _declare(self, name, value):
        self.scopes[-1][name] = value

    def _assign_existing(self, name, value):
        for scope in reversed(self.scopes):
            if name in scope:
                scope[name] = value
                return
        self.scopes[-1][name] = value

    def _unique_name(self, base):
        n = self._name_counter.get(base, 0)
        self._name_counter[base] = n + 1
        return base if n == 0 else f"{base}__{n}"

    # -------------------------------------------------------------- entry
    def build(self, source):
        program = openqasm3.parse(source)
        self.qc = QuantumCircuit()
        self._create_implicit_registers(program)
        for stmt in program.statements:
            self._exec_stmt(stmt)
        return self.qc

    def _create_implicit_registers(self, program):
        """Spec examples sometimes use qubit registers without declaring them
        (e.g. cphase.qasm). Pre-scan for undeclared qubit operands and create
        registers sized to the largest literal index seen."""
        declared = set()
        implicit = {}

        def scan_operand(node):
            tname = type(node).__name__
            if tname == "Identifier":
                implicit.setdefault(node.name, 0)
            elif tname == "IndexedIdentifier":
                if type(node.name).__name__ != "Identifier":
                    return
                for group in node.indices:
                    items = group if isinstance(group, list) else [group]
                    for idx in items:
                        if type(idx).__name__ == "IntegerLiteral":
                            current = implicit.get(node.name.name, 0)
                            implicit[node.name.name] = max(current, idx.value)

        def scan(statements):
            for stmt in statements:
                tname = type(stmt).__name__
                if tname == "QubitDeclaration":
                    declared.add(stmt.qubit.name)
                elif tname == "AliasStatement":
                    declared.add(stmt.target.name)
                elif tname == "QuantumGate":
                    for q in stmt.qubits:
                        scan_operand(q)
                elif tname in ("QuantumReset",):
                    scan_operand(stmt.qubits)
                elif tname == "QuantumMeasurementStatement":
                    scan_operand(stmt.measure.qubit)
                elif tname == "BranchingStatement":
                    scan(stmt.if_block or [])
                    scan(stmt.else_block or [])
                elif tname in ("WhileLoop", "ForInLoop"):
                    scan(stmt.block)

        scan(program.statements)
        for name, max_idx in implicit.items():
            if name not in declared:
                reg = QuantumRegister(max_idx + 1, self._unique_name(name))
                self.qc.add_register(reg)
                self._declare(name, _QubitRef(list(reg)))

    # --------------------------------------------------------- statements
    def _exec_block(self, statements):
        for stmt in statements:
            self._exec_stmt(stmt)

    def _exec_stmt(self, stmt):
        name = type(stmt).__name__
        handler = getattr(self, f"_stmt_{name}", None)
        if handler is None:
            raise QASM3FrontendError(f"Unsupported statement: {name}")
        handler(stmt)

    def _stmt_Include(self, stmt):
        if stmt.filename not in ("stdgates.inc",):
            raise QASM3FrontendError(f"Unsupported include: {stmt.filename}")

    def _stmt_Pragma(self, stmt):
        pass

    def _stmt_QubitDeclaration(self, stmt):
        size = 1 if stmt.size is None else self._eval_int(stmt.size)
        reg = QuantumRegister(size, self._unique_name(stmt.qubit.name))
        self.qc.add_register(reg)
        self._declare(stmt.qubit.name, _QubitRef(list(reg)))

    def _make_bit_value(self, name, size, init_value):
        reg = ClassicalRegister(size, self._unique_name(name))
        self.qc.add_register(reg)
        return _BitRef(list(reg), register=reg, initial=init_value)

    def _stmt_ClassicalDeclaration(self, stmt):
        tname = type(stmt.type).__name__
        if tname == "BitType":
            # A subroutine's return variable may be pre-bound to the caller's
            # assignment target (see _call_subroutine); keep that binding so
            # measurements land in the caller's register.
            existing = self.scopes[-1].get(stmt.identifier.name)
            if isinstance(existing, _BitRef):
                return
            size = 1 if stmt.type.size is None else self._eval_int(stmt.type.size)
            init = 0
            measure_init = None
            if stmt.init_expression is not None:
                if type(stmt.init_expression).__name__ == "QuantumMeasurement":
                    measure_init = stmt.init_expression
                else:
                    init = self._eval_classical(stmt.init_expression)
            bit_value = self._make_bit_value(stmt.identifier.name, size, int(init))
            self._declare(stmt.identifier.name, bit_value)
            if measure_init is not None:
                qubits = self._eval_qubits(measure_init.qubit).qubits
                if len(qubits) != len(bit_value.clbits):
                    raise QASM3FrontendError("measure size mismatch in declaration")
                for q, c in zip(qubits, bit_value.clbits):
                    self.qc.measure(q, c)
            return
        if tname in ("IntType", "UintType", "FloatType", "AngleType", "BoolType"):
            value = 0
            if stmt.init_expression is not None:
                value = self._eval_classical(stmt.init_expression)
            self._declare(stmt.identifier.name, value)
            return
        if tname == "ArrayType":
            if stmt.init_expression is None:
                dims = [self._eval_int(d) for d in stmt.type.dimensions]
                value = self._zero_array(dims)
            else:
                import copy

                # QASM3 array assignment copies data.
                value = copy.deepcopy(self._eval_classical(stmt.init_expression))
            self._declare(stmt.identifier.name, value)
            return
        if tname in ("DurationType", "StretchType"):
            raise QASM3FrontendError(
                f"Timing type '{tname}' has no circuit representation"
            )
        raise QASM3FrontendError(f"Unsupported classical type: {tname}")

    def _zero_array(self, dims):
        if len(dims) == 1:
            return [0] * dims[0]
        return [self._zero_array(dims[1:]) for _ in range(dims[0])]

    def _stmt_ConstantDeclaration(self, stmt):
        self._declare(stmt.identifier.name, self._eval_classical(stmt.init_expression))

    def _stmt_IODeclaration(self, stmt):
        if stmt.io_identifier == ast.IOKeyword.input:
            name = stmt.identifier.name
            if name not in self.inputs:
                raise QASM3FrontendError(
                    f"Program requires input '{name}'; pass it via "
                    f"load_qasm3(..., inputs={{'{name}': value}})"
                )
            self._declare(name, self.inputs[name])
        else:  # output
            tname = type(stmt.type).__name__
            if tname == "BitType":
                size = 1 if stmt.type.size is None else self._eval_int(stmt.type.size)
                self._declare(
                    stmt.identifier.name,
                    self._make_bit_value(stmt.identifier.name, size, 0),
                )
            else:
                self._declare(stmt.identifier.name, 0)

    def _stmt_QuantumGateDefinition(self, stmt):
        self._declare(stmt.name.name, _GateDef(stmt))

    def _stmt_SubroutineDefinition(self, stmt):
        self._declare(stmt.name.name, _SubroutineDef(stmt))

    def _stmt_ExternDeclaration(self, stmt):
        raise QASM3FrontendError(
            f"extern function '{stmt.name.name}' cannot be evaluated: extern "
            "requires a runtime implementation"
        )

    def _stmt_CalibrationGrammarDeclaration(self, stmt):
        raise QASM3FrontendError("defcal/calibration grammar is not supported")

    _stmt_CalibrationStatement = _stmt_CalibrationGrammarDeclaration
    _stmt_CalibrationDefinition = _stmt_CalibrationGrammarDeclaration

    def _stmt_DelayInstruction(self, stmt):
        raise QASM3FrontendError("Timing construct 'delay' is not supported")

    def _stmt_Box(self, stmt):
        raise QASM3FrontendError("Timing construct 'box' is not supported")

    def _stmt_QuantumReset(self, stmt):
        for q in self._eval_qubits(stmt.qubits).qubits:
            self.qc.reset(q)

    def _stmt_QuantumBarrier(self, stmt):
        if not stmt.qubits:
            self.qc.barrier()
            return
        qubits = []
        for operand in stmt.qubits:
            qubits.extend(self._eval_qubits(operand).qubits)
        self.qc.barrier(*qubits)

    # ------------------------------------------------------- measurements
    def _stmt_QuantumMeasurementStatement(self, stmt):
        qubits = self._eval_qubits(stmt.measure.qubit).qubits
        if stmt.target is None:
            # Bare measure: discard into a hidden register.
            hidden = self._make_bit_value("_discard", len(qubits), 0)
            targets = hidden.clbits
        else:
            targets = self._eval_bits(stmt.target).clbits
        if len(qubits) != len(targets):
            raise QASM3FrontendError(
                f"measure size mismatch: {len(qubits)} qubits -> {len(targets)} bits"
            )
        for q, c in zip(qubits, targets):
            self.qc.measure(q, c)

    # --------------------------------------------------------- assignment
    def _stmt_ClassicalAssignment(self, stmt):
        op = stmt.op.name  # e.g. '=', '+=', '<<='
        rtype = type(stmt.rvalue).__name__

        if rtype == "QuantumMeasurement":
            qubits = self._eval_qubits(stmt.rvalue.qubit).qubits
            targets = self._eval_bits(stmt.lvalue).clbits
            if len(qubits) != len(targets):
                raise QASM3FrontendError("measure size mismatch in assignment")
            for q, c in zip(qubits, targets):
                self.qc.measure(q, c)
            return

        if rtype == "FunctionCall":
            fname = stmt.rvalue.name.name
            target = self._lookup(fname)
            if isinstance(target, _SubroutineDef):
                lval = self._resolve_ref(stmt.lvalue)
                if isinstance(lval, _BitRef):
                    self._call_subroutine(target, stmt.rvalue.arguments,
                                          return_target=lval)
                else:
                    value = self._call_subroutine(target, stmt.rvalue.arguments)
                    self._store_classical(stmt.lvalue, op, value)
                return

        lval = self._resolve_ref(stmt.lvalue, allow_missing=True)
        if isinstance(lval, _BitRef):
            raise QASM3FrontendError(
                "Runtime assignment to measured bits from a classical "
                f"expression ('{op}') is not supported"
            )
        value = self._eval_classical(stmt.rvalue)
        self._store_classical(stmt.lvalue, op, value)

    def _store_classical(self, lvalue, op, value):
        ltype = type(lvalue).__name__
        if ltype in ("IndexedIdentifier", "IndexExpression"):
            if op != "=":
                raise QASM3FrontendError(
                    "Compound assignment to indexed values is not supported"
                )
            base_node = (lvalue.name if ltype == "IndexedIdentifier"
                         else lvalue.collection)
            base = self._resolve_ref(base_node)
            if not isinstance(base, list):
                raise QASM3FrontendError(
                    "Indexed assignment target must be a classical array"
                )
            indices = (lvalue.indices if ltype == "IndexedIdentifier"
                       else [lvalue.index])
            flat = []
            for index_group in indices:
                group = index_group if isinstance(index_group, list) else [index_group]
                flat.extend(group)
            container = base
            for idx_node in flat[:-1]:
                container = container[self._eval_int(idx_node)]
            last = flat[-1]
            last_type = type(last).__name__
            if last_type in ("RangeDefinition", "DiscreteSet"):
                items = self._eval_loop_set(last) if last_type == "RangeDefinition" \
                    else [self._eval_classical(v) for v in last.values]
                if len(items) != len(value):
                    raise QASM3FrontendError("Slice assignment shape mismatch")
                for i, v in zip(items, list(value)):
                    container[i] = v
            else:
                container[self._eval_int(last)] = value
            return
        if ltype != "Identifier":
            raise QASM3FrontendError(f"Unsupported assignment target: {ltype}")
        name = lvalue.name
        if op == "=":
            self._assign_existing(name, value)
            return
        current = self._lookup(name)
        ops = {
            "+=": lambda a, b: a + b, "-=": lambda a, b: a - b,
            "*=": lambda a, b: a * b, "/=": lambda a, b: a / b,
            "<<=": lambda a, b: int(a) << int(b),
            ">>=": lambda a, b: int(a) >> int(b),
            "&=": lambda a, b: int(a) & int(b), "|=": lambda a, b: int(a) | int(b),
            "^=": lambda a, b: int(a) ^ int(b), "%=": lambda a, b: a % b,
        }
        if op not in ops:
            raise QASM3FrontendError(f"Unsupported assignment operator: {op}")
        if isinstance(current, _BitRef):
            raise QASM3FrontendError(
                f"Runtime classical arithmetic on measured bits ('{op}') is "
                "not supported"
            )
        self._assign_existing(name, ops[op](current, value))

    def _stmt_AliasStatement(self, stmt):
        self._assign_existing(stmt.target.name, self._resolve_ref(stmt.value))

    def _stmt_ExpressionStatement(self, stmt):
        etype = type(stmt.expression).__name__
        if etype == "FunctionCall":
            target = self._lookup(stmt.expression.name.name)
            if isinstance(target, _SubroutineDef):
                self._call_subroutine(target, stmt.expression.arguments)
                return
        if etype == "QuantumMeasurement":
            qubits = self._eval_qubits(stmt.expression.qubit).qubits
            hidden = self._make_bit_value("_discard", len(qubits), 0)
            for q, c in zip(qubits, hidden.clbits):
                self.qc.measure(q, c)
            return
        raise QASM3FrontendError(f"Unsupported expression statement: {etype}")

    def _stmt_ReturnStatement(self, stmt):
        value = None
        if stmt.expression is not None:
            if type(stmt.expression).__name__ == "QuantumMeasurement":
                # `return measure q;` — measure straight into the caller's
                # assignment target when one was provided.
                qubits = self._eval_qubits(stmt.expression.qubit).qubits
                target = self._return_targets[-1] if self._return_targets else None
                if target is None:
                    target = self._make_bit_value("_ret", len(qubits), 0)
                if len(qubits) != len(target.clbits):
                    raise QASM3FrontendError("measure size mismatch in return")
                for q, c in zip(qubits, target.clbits):
                    self.qc.measure(q, c)
                raise _Return(target)
            value = self._eval_any(stmt.expression)
        raise _Return(value)

    # ------------------------------------------------------------- gates
    def _stmt_QuantumGate(self, stmt):
        # `mysub q0, q1;` — a subroutine invoked with gate-call syntax.
        try:
            target = self._lookup(stmt.name.name)
        except QASM3FrontendError:
            target = None
        if isinstance(target, _SubroutineDef):
            if stmt.modifiers:
                raise QASM3FrontendError(
                    "Gate modifiers on subroutine calls are not supported"
                )
            self._call_subroutine(target, list(stmt.arguments) + list(stmt.qubits))
            return

        params = [self._eval_classical(a) for a in stmt.arguments]
        operand_refs = [self._eval_qubits(q) for q in stmt.qubits]

        # gphase has no qubit operands
        if stmt.name.name == "gphase" and not operand_refs:
            self.qc.global_phase += params[0]
            return

        gate = self._build_gate(stmt.name.name, params)
        for mod in reversed(stmt.modifiers):
            gate = self._apply_modifier(gate, mod)

        n_expected = gate.num_qubits
        # Broadcast: registers of the same size > 1 iterate together;
        # single qubits repeat.
        sizes = {len(r) for r in operand_refs if len(r) > 1}
        if len(sizes) > 1:
            raise QASM3FrontendError("Mismatched register sizes in gate broadcast")
        reps = sizes.pop() if sizes else 1
        for i in range(reps):
            qubits = [r.qubits[i] if len(r) > 1 else r.qubits[0]
                      for r in operand_refs]
            if len(qubits) != n_expected:
                raise QASM3FrontendError(
                    f"Gate '{stmt.name.name}' expects {n_expected} qubits, "
                    f"got {len(qubits)}"
                )
            self.qc.append(gate, qubits)

    def _stmt_QuantumPhase(self, stmt):
        params = [self._eval_classical(a) for a in stmt.arguments]
        self.qc.global_phase += params[0]

    def _build_gate(self, name, params):
        # User-defined gate?
        try:
            target = self._lookup(name)
        except QASM3FrontendError:
            target = None
        if isinstance(target, _GateDef):
            return self._build_user_gate(target.node, params)

        if name not in _STD_GATES:
            raise QASM3FrontendError(f"Gate '{name}' is not defined")
        method, n_params, n_qubits = _STD_GATES[name]
        if len(params) != n_params:
            raise QASM3FrontendError(
                f"Gate '{name}' expects {n_params} parameters, got {len(params)}"
            )
        proto = QuantumCircuit(max(n_qubits, 1))
        getattr(proto, method)(*params, *range(n_qubits))
        return proto.data[0].operation

    def _build_user_gate(self, node, params):
        sub = QuantumCircuit(len(node.qubits))
        scope = {}
        for arg_node, value in zip(node.arguments, params):
            scope[arg_node.name] = value
        for i, q_node in enumerate(node.qubits):
            scope[q_node.name] = _QubitRef([sub.qubits[i]])

        saved_qc, saved_scopes = self.qc, self.scopes
        self.qc = sub
        self.scopes = self.scopes + [scope]
        try:
            self._exec_block(node.body)
        finally:
            self.qc, self.scopes = saved_qc, saved_scopes
        gate = sub.to_gate(label=None)
        gate.name = node.name.name
        return gate

    def _apply_modifier(self, gate, mod):
        kind = mod.modifier
        if kind == ast.GateModifierName.inv:
            return gate.inverse()
        if kind == ast.GateModifierName.pow:
            exponent = self._eval_classical(mod.argument)
            if isinstance(gate, Gate):
                return gate.power(exponent)
            raise QASM3FrontendError("pow @ modifier on non-gate operation")
        if kind in (ast.GateModifierName.ctrl, ast.GateModifierName.negctrl):
            n = 1 if mod.argument is None else self._eval_int(mod.argument)
            state = None if kind == ast.GateModifierName.ctrl else 0
            return gate.control(n, ctrl_state=state)
        raise QASM3FrontendError(f"Unsupported gate modifier: {kind}")

    # -------------------------------------------------------- subroutines
    def _call_subroutine(self, subdef, arg_nodes, return_target=None):
        node = subdef.node
        scope = {}
        for arg_node, expr_node in zip(node.arguments, arg_nodes):
            if type(arg_node).__name__ == "QuantumArgument":
                scope[arg_node.name.name] = self._eval_qubits(expr_node)
            else:
                value = self._eval_any(expr_node)
                scope[arg_node.name.name] = value

        # If the body ends with `return <identifier>` of a locally declared
        # bit register and the caller assigns the result to a bit register,
        # bind that local name directly to the caller's bits so measurements
        # write into the right register.
        if return_target is not None:
            ret_name = self._simple_return_identifier(node.body)
            if ret_name is not None:
                scope[ret_name] = return_target

        self.scopes = self.scopes + [scope]
        self._return_targets.append(return_target)
        result = None
        try:
            self._exec_block(node.body)
        except _Return as ret:
            result = ret.value
        finally:
            self.scopes = self.scopes[:-1]
            self._return_targets.pop()
        if (return_target is not None and isinstance(result, _BitRef)
                and result.clbits != return_target.clbits):
            raise QASM3FrontendError(
                "Subroutine returns measured bits that cannot be bound to the "
                "assignment target (runtime bit copies are not supported)"
            )
        return result

    @staticmethod
    def _simple_return_identifier(body):
        for stmt in body:
            if (type(stmt).__name__ == "ReturnStatement"
                    and stmt.expression is not None
                    and type(stmt.expression).__name__ == "Identifier"):
                return stmt.expression.name
        return None

    # ------------------------------------------------------- control flow
    def _stmt_BranchingStatement(self, stmt):
        cond = self._try_static_bool(stmt.condition)
        if cond is not None:
            self._exec_scoped(stmt.if_block if cond else stmt.else_block)
            return

        target, value, invert = self._runtime_condition(stmt.condition)
        if not invert:
            with self.qc.if_test((target, value)) as else_:
                self._exec_scoped(stmt.if_block)
            if stmt.else_block:
                with else_:
                    self._exec_scoped(stmt.else_block)
        else:
            # `!=` condition: swap branches around an equality test.
            with self.qc.if_test((target, value)) as else_:
                self._exec_scoped(stmt.else_block or [])
            with else_:
                self._exec_scoped(stmt.if_block)

    def _stmt_WhileLoop(self, stmt):
        cond = self._try_static_bool(stmt.while_condition)
        if cond is not None and not self._condition_uses_bits(stmt.while_condition):
            # Pure classical loop: unroll.
            count = 0
            while self._try_static_bool(stmt.while_condition):
                self._exec_scoped(stmt.block)
                count += 1
                if count > self.MAX_UNROLL:
                    raise QASM3FrontendError("while loop exceeds unroll limit")
            return

        target, value, invert = self._runtime_condition(stmt.while_condition)

        # Qiskit clbits start at 0. If the condition bits carry a nonzero
        # declared initial value that makes the condition true, peel one
        # body execution so the loop semantics survive the lost initializer.
        initial = self._condition_initial_value(stmt.while_condition)
        if initial is not None:
            initially_true = (initial != value) if invert else (initial == value)
            if initially_true and initial != 0:
                self._exec_scoped(stmt.block)

        if invert:
            condition = qexpr.not_equal(target, value)
        else:
            condition = (target, value)
        with self.qc.while_loop(condition):
            self._exec_scoped(stmt.block)

    def _stmt_ForInLoop(self, stmt):
        values = self._eval_loop_set(stmt.set_declaration)
        if len(values) > self.MAX_UNROLL:
            raise QASM3FrontendError("for loop exceeds unroll limit")
        for v in values:
            self.scopes.append({stmt.identifier.name: v})
            try:
                self._exec_block(stmt.block)
            finally:
                self.scopes.pop()

    def _stmt_BreakStatement(self, stmt):
        raise QASM3FrontendError("break is not supported")

    def _stmt_ContinueStatement(self, stmt):
        raise QASM3FrontendError("continue is not supported")

    def _stmt_EndStatement(self, stmt):
        pass

    def _exec_scoped(self, statements):
        self.scopes.append({})
        try:
            self._exec_block(statements or [])
        finally:
            self.scopes.pop()

    def _eval_loop_set(self, node):
        tname = type(node).__name__
        if tname == "RangeDefinition":
            start = 0 if node.start is None else self._eval_int(node.start)
            end = self._eval_int(node.end)
            step = 1 if node.step is None else self._eval_int(node.step)
            return list(range(start, end + (1 if step > 0 else -1), step))
        if tname == "DiscreteSet":
            return [self._eval_classical(v) for v in node.values]
        return self._eval_classical(node)

    # -------------------------------------------------- runtime conditions
    def _condition_uses_bits(self, node):
        try:
            ref = self._peel_condition_operand(node)
        except QASM3FrontendError:
            return False
        return ref is not None

    def _peel_condition_operand(self, node):
        """Return the _BitRef a condition tests, if any."""
        tname = type(node).__name__
        if tname == "BinaryExpression":
            for side in (node.lhs, node.rhs):
                ref = self._peel_condition_operand(side)
                if ref is not None:
                    return ref
            return None
        if tname == "UnaryExpression":
            return self._peel_condition_operand(node.expression)
        if tname == "Cast":
            return self._peel_condition_operand(node.argument)
        if tname in ("Identifier", "IndexedIdentifier", "IndexExpression"):
            try:
                ref = self._resolve_ref(node)
            except QASM3FrontendError:
                return None
            return ref if isinstance(ref, _BitRef) else None
        return None

    def _runtime_condition(self, node):
        """Lower a condition on measured bits to (clbit_or_creg, value, invert)."""
        tname = type(node).__name__
        if tname == "BinaryExpression" and node.op.name in ("==", "!="):
            invert = node.op.name == "!="
            lhs_ref = self._peel_condition_operand(node.lhs)
            if lhs_ref is not None:
                value = self._eval_int(node.rhs)
                return self._condition_target(lhs_ref), value, invert
            rhs_ref = self._peel_condition_operand(node.rhs)
            if rhs_ref is not None:
                value = self._eval_int(node.lhs)
                return self._condition_target(rhs_ref), value, invert
        if tname == "UnaryExpression" and node.op.name in ("!", "~"):
            ref = self._peel_condition_operand(node.expression)
            if ref is not None and len(ref) == 1:
                return ref.clbits[0], 0, False
        ref = self._peel_condition_operand(node)
        if ref is not None:
            if len(ref) == 1:
                return ref.clbits[0], 1, False
            # Bare register truthiness: creg != 0
            return self._condition_target(ref), 0, True
        raise QASM3FrontendError(
            f"Unsupported runtime condition: {tname}"
        )

    @staticmethod
    def _condition_target(ref):
        if len(ref) == 1:
            return ref.clbits[0]
        if ref.register is not None and len(ref.register) == len(ref.clbits):
            return ref.register
        raise QASM3FrontendError(
            "Conditions on bit slices (not whole registers) are not supported"
        )

    def _condition_initial_value(self, node):
        ref = self._peel_condition_operand(node)
        return None if ref is None else ref.initial

    def _try_static_bool(self, node):
        if self._condition_uses_bits(node):
            return None
        try:
            return bool(self._eval_classical(node))
        except QASM3FrontendError:
            return None

    # -------------------------------------------------------- expressions
    def _eval_any(self, node):
        """Evaluate to either a compile-time value or a Qubit/Bit reference."""
        try:
            ref = self._resolve_ref(node)
            if isinstance(ref, (_BitRef, _QubitRef)):
                return ref
        except QASM3FrontendError:
            pass
        return self._eval_classical(node)

    def _resolve_ref(self, node, allow_missing=False):
        tname = type(node).__name__
        if tname == "Identifier":
            try:
                return self._lookup(node.name)
            except QASM3FrontendError:
                if allow_missing:
                    return None
                raise
        if tname in ("IndexedIdentifier", "IndexExpression"):
            base_node = node.name if tname == "IndexedIdentifier" else node.collection
            base = self._resolve_ref(base_node, allow_missing=allow_missing)
            if base is None:
                return None
            indices = node.indices if tname == "IndexedIdentifier" else [node.index]
            for index_group in indices:
                base = self._index_ref(base, index_group)
            return base
        if allow_missing:
            return None
        raise QASM3FrontendError(f"Cannot resolve reference: {tname}")

    def _index_ref(self, base, index_group):
        try:
            return self._index_ref_impl(base, index_group)
        except IndexError:
            raise QASM3FrontendError("Index out of bounds")

    def _index_ref_impl(self, base, index_group):
        if not isinstance(index_group, list):
            index_group = [index_group]

        def eval_index(idx_node):
            """Returns (is_slice, indices)."""
            idx_type = type(idx_node).__name__
            if idx_type == "RangeDefinition":
                return True, self._eval_loop_set(idx_node)
            if idx_type == "DiscreteSet":
                return True, [self._eval_classical(v) for v in idx_node.values]
            return False, [self._eval_int(idx_node)]

        if isinstance(base, (_QubitRef, _BitRef)):
            picked = []
            for idx_node in index_group:
                _, items = eval_index(idx_node)
                picked.extend(items)
            if isinstance(base, _QubitRef):
                return _QubitRef([base.qubits[i] for i in picked])
            sliced = _BitRef([base.clbits[i] for i in picked])
            # Only an identity-ordered full slice is still "the register";
            # reordered or duplicated picks must not be treated as one.
            if picked == list(range(len(base))):
                sliced.register = base.register
            return sliced

        if isinstance(base, list):
            # Comma-separated scalars chain into dimensions; a slice selects
            # a sub-list of the current dimension.
            value = base
            for idx_node in index_group:
                is_slice, items = eval_index(idx_node)
                if is_slice:
                    value = [value[i] for i in items]
                else:
                    value = value[items[0]]
            return value
        if isinstance(base, (int, bool)):
            # Bit-indexing into a classical integer: a_in[i] is bit i.
            picked = []
            for idx_node in index_group:
                _, items = eval_index(idx_node)
                picked.extend(items)
            if len(picked) == 1:
                return (int(base) >> picked[0]) & 1
            return sum(((int(base) >> i) & 1) << pos
                       for pos, i in enumerate(picked))
        raise QASM3FrontendError("Cannot index into this value")

    def _eval_qubits(self, node):
        ref = self._resolve_ref(node, allow_missing=True)
        if isinstance(ref, _QubitRef):
            return ref
        # Implicit qubit register (spec examples use undeclared registers).
        tname = type(node).__name__
        if tname in ("IndexedIdentifier", "IndexExpression"):
            base_node = node.name if tname == "IndexedIdentifier" else node.collection
            if type(base_node).__name__ == "Identifier" and ref is None:
                indices = node.indices if tname == "IndexedIdentifier" else [node.index]
                flat = indices[0] if isinstance(indices[0], list) else [indices[0]]
                max_idx = max(self._eval_int(i) for i in flat)
                reg = QuantumRegister(max_idx + 1, base_node.name)
                self.qc.add_register(reg)
                self._declare(base_node.name, _QubitRef(list(reg)))
                return self._resolve_ref(node)
        if isinstance(ref, _BitRef):
            raise QASM3FrontendError("Expected qubits, found classical bits")
        raise QASM3FrontendError(
            f"Cannot resolve qubit operand ({tname})"
        )

    def _eval_bits(self, node):
        ref = self._resolve_ref(node)
        if not isinstance(ref, _BitRef):
            raise QASM3FrontendError(
                "Measurement target must be a bit or bit register; measuring "
                "into other classical types (e.g. angle) implies runtime "
                "classical arithmetic, which has no circuit representation"
            )
        return ref

    def _eval_int(self, node):
        return int(self._eval_classical(node))

    def _eval_classical(self, node):
        tname = type(node).__name__
        if tname in ("IntegerLiteral", "FloatLiteral", "BooleanLiteral"):
            return node.value
        if tname == "BitstringLiteral":
            return node.value
        if tname == "ImaginaryLiteral":
            return complex(0, node.value)
        if tname == "DurationLiteral":
            raise QASM3FrontendError("Timing literal (duration) is not supported")
        if tname == "Identifier":
            if node.name in _CONSTANTS:
                return _CONSTANTS[node.name]
            value = self._lookup(node.name)
            if isinstance(value, _BitRef):
                if value.initial is not None and not self._bits_measured(value):
                    return value.initial
                raise QASM3FrontendError(
                    f"'{node.name}' holds runtime measurement results; "
                    "compile-time evaluation is impossible"
                )
            if isinstance(value, (_QubitRef, _GateDef, _SubroutineDef)):
                raise QASM3FrontendError(
                    f"'{node.name}' is not a classical value"
                )
            return value
        if tname == "UnaryExpression":
            v = self._eval_classical(node.expression)
            op = node.op.name
            if op == "-":
                return -v
            if op == "!":
                return not v
            if op == "~":
                return ~int(v)
            raise QASM3FrontendError(f"Unsupported unary operator: {op}")
        if tname == "BinaryExpression":
            a = self._eval_classical(node.lhs)
            b = self._eval_classical(node.rhs)
            return self._binary_op(node.op.name, a, b)
        if tname == "Cast":
            v = self._eval_classical(node.argument)
            return self._cast_value(node.type, v)
        if tname == "FunctionCall":
            fname = node.name.name
            if fname == "sizeof":
                arr = self._eval_classical(node.arguments[0])
                dim = 0
                if len(node.arguments) > 1:
                    dim = self._eval_int(node.arguments[1])
                for _ in range(dim):
                    arr = arr[0]
                return len(arr)
            if fname in _MATH_FUNCS:
                args = [self._eval_classical(a) for a in node.arguments]
                return _MATH_FUNCS[fname](*args)
            target = self._lookup(fname)
            if isinstance(target, _SubroutineDef):
                result = self._call_subroutine(target, node.arguments)
                if isinstance(result, _BitRef):
                    raise QASM3FrontendError(
                        "Subroutine returning measured bits used inside an "
                        "expression is not supported"
                    )
                return result
            raise QASM3FrontendError(f"Unknown function: {fname}")
        if tname in ("IndexExpression", "IndexedIdentifier"):
            ref = self._resolve_ref(node)
            if isinstance(ref, _BitRef):
                raise QASM3FrontendError(
                    "Runtime bit used in compile-time expression"
                )
            return ref
        if tname == "ArrayLiteral":
            return [self._eval_classical(v) for v in node.values]
        raise QASM3FrontendError(f"Unsupported expression: {tname}")

    def _bits_measured(self, ref):
        measured = set()
        for inst in self.qc.data:
            if inst.operation.name == "measure":
                measured.update(inst.clbits)
        return any(c in measured for c in ref.clbits)

    @staticmethod
    def _binary_op(op, a, b):
        ops = {
            "+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b,
            "/": lambda: a / b if isinstance(a, float) or isinstance(b, float)
                 or (isinstance(a, int) and isinstance(b, int) and a % b != 0)
                 else a // b,
            "%": lambda: a % b, "**": lambda: a ** b,
            "<<": lambda: int(a) << int(b), ">>": lambda: int(a) >> int(b),
            "==": lambda: a == b, "!=": lambda: a != b,
            "<": lambda: a < b, ">": lambda: a > b,
            "<=": lambda: a <= b, ">=": lambda: a >= b,
            "&&": lambda: bool(a) and bool(b), "||": lambda: bool(a) or bool(b),
            "&": lambda: int(a) & int(b), "|": lambda: int(a) | int(b),
            "^": lambda: int(a) ^ int(b),
        }
        if op not in ops:
            raise QASM3FrontendError(f"Unsupported binary operator: {op}")
        return ops[op]()

    @staticmethod
    def _cast_value(type_node, value):
        tname = type(type_node).__name__
        if tname in ("IntType", "UintType"):
            return int(value)
        if tname == "FloatType" or tname == "AngleType":
            return float(value)
        if tname == "BoolType":
            return bool(value)
        if tname == "BitType":
            return int(value)
        raise QASM3FrontendError(f"Unsupported cast to {tname}")


def load_qasm3(source, inputs=None):
    """Load an OpenQASM 3 program into a qiskit QuantumCircuit.

    Tries qiskit's importer first; falls back to the openqasm3-AST partial
    evaluator for programs qiskit cannot parse.

    Args:
        source: QASM3 source text, or a path to a .qasm file.
        inputs: dict of values for the program's `input` declarations.

    Raises:
        QASM3FrontendError: for constructs with no circuit representation
            (extern, defcal, timing) — the message names the blocker.
    """
    path = None
    text = source
    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and source.endswith(".qasm")
    ):
        path = Path(source)
        text = path.read_text()

    # qiskit importer quirk: single-bit '== 1' comparisons must be '== true'.
    qiskit_text = re.sub(r"==\s*1\b", "==true", text)
    try:
        import qiskit.qasm3

        return qiskit.qasm3.loads(qiskit_text)
    except Exception:
        pass

    return AstFrontend(inputs=inputs).build(text)
