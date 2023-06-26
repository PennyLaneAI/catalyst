from _ast import BinOp, Compare, Constant, Expr, If, Return, Tuple
from _ast import FunctionDef, Module, Call, Name, Attribute
from typing import Any
import ast, inspect, textwrap
import numpy as np
import pennylane

import mlir_quantum.ir as ir
import mlir_quantum.dialects.arith as arith
import mlir_quantum.dialects.quantum as quantum
import mlir_quantum.dialects.func as func
import mlir_quantum.dialects.scf as scf


class MLIRGenerator(ast.NodeVisitor):
    def __init__(
        self,
        frame_info: inspect.FrameInfo,
        module: ir.Module,
        called_argtypes: dict[str, tuple[type]],
    ) -> None:
        self.frame_info = frame_info
        self.module = module
        self.called_argtypes = called_argtypes
        self._env = {}

        # TODO: Would be nice to have this natively represented
        self.qureg_type = ir.OpaqueType.get("quantum", "reg")
        self.qubit_type = ir.OpaqueType.get("quantum", "bit")
        self.obs_type = ir.OpaqueType.get("quantum", "obs")
        self.qstate = None
        super().__init__()

    def _lookup(self, name: str):
        if name in self._env:
            return self._env[name]

        env = self.frame_info.frame.f_locals
        if name in env:
            return env[name]
        env = self.frame_info.frame.f_globals
        if name not in env:
            raise ValueError(f"MLIR Generator: name not found: {name}")
        return env[name]

    def _lookup_attr(self, attr: Attribute):
        module = self._lookup(attr.value.id)
        if module == pennylane and hasattr(pennylane, attr.attr):
            # May get messy to distinguish a namedobservable vs a gate
            return getattr(pennylane, attr.attr)
        return None

    def _type_to_mlir_type(self, typ: type):
        if typ == float:
            return ir.F64Type.get()
        if typ == int:
            return ir.IntegerType.get_signless(64)
        assert False, f"Unhandled type {typ}"

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        qnode = (
            self._lookup(node.name).fn
            if isinstance(self._lookup(node.name).fn, pennylane.QNode)
            else None
        )

        called_types = self.called_argtypes[node.name]
        if len(called_types) != len(node.args.args):
            raise TypeError(
                f"Function {node.name} expects {len(node.args.args)} types but was called with {len(called_types)}"
            )
        # TODO: definitely possible that there's a type mismatch here
        fn_type = ir.FunctionType.get(
            [self._type_to_mlir_type(typ) for typ in called_types], [ir.F64Type.get()]
        )
        func_op = func.FuncOp(node.name, fn_type)
        entry_block = func_op.add_entry_block()
        self._env = {}
        for argname, blockarg in zip(node.args.args, entry_block.arguments):
            # TODO: this ignores annotations on the function arguments
            self._env[argname.arg] = blockarg

        with ir.InsertionPoint(entry_block):
            if qnode:
                self.qstate = quantum.AllocOp(self.qureg_type, nqubits_attr=qnode.device.num_wires)

            for stmt in node.body:
                self.visit(stmt)

    def visit_Expr(self, node: Expr) -> Any:
        # Not sure this is required
        return self.visit(node.value)

    def get_qubits_for_wires(self, wires: tuple[int]):
        def emit_extract(wire: int):
            return quantum.ExtractOp(self.qubit_type, self.qstate, idx_attr=wire).result

        return tuple(emit_extract(wire) for wire in wires)

    def update_qubits(self, wires: tuple[int], qubits):
        for wire, qubit in zip(wires, qubits):
            self.qstate = quantum.InsertOp(self.qureg_type, self.qstate, qubit, idx_attr=wire)

    def visit_NamedObs(self, node: Call) -> Any:
        callee = self._lookup_attr(node.func)
        assert len(node.args) == 1, "Expected named observable to have one argument (wires)"
        wire = self.visit(node.args[0])
        qubit = self.get_qubits_for_wires((wire,))[0]
        kind = callee.__name__
        obs_type = ir.OpaqueAttr.get(
            "quantum", ("named_observable " + kind).encode("utf-8"), ir.NoneType.get()
        )
        return quantum.NamedObsOp(self.obs_type, qubit, obs_type).results

    def visit_If(self, node: If) -> Any:
        pred = self.visit(node.test)
        # Naively carry the entire state through the op
        result_types = [self.qureg_type]
        if_op = scf.IfOp(pred, result_types, hasElse=True)
        then_block = if_op.then_block

        saved_qstate = self.qstate
        with ir.InsertionPoint(then_block):
            for stmt in node.body:
                self.visit(stmt)
                scf.YieldOp((self.qstate,))
        else_block = if_op.else_block
        self.qstate = saved_qstate
        with ir.InsertionPoint(else_block):
            for stmt in node.orelse:
                self.visit(stmt)
                scf.YieldOp((self.qstate,))

        self.qstate = if_op.results[0]

    def visit_Return(self, node: Return) -> Any:
        return_val = self.visit(node.value)
        func.ReturnOp(return_val)

    def materialize_constant(self, value, force_float=False) -> ir.Value:
        if isinstance(value, int) and force_float:
            value = float(value)
        if isinstance(value, float) or isinstance(value, int):
            return arith.ConstantOp(self._type_to_mlir_type(type(value)), value).result
        return value

    def get_wires(self, operation: pennylane.operation.Operation, node: Call):
        # First check if the wires are defined explicitly, then check the positional arg at index num_params
        wire_kws = [kw for kw in node.keywords if kw.arg == "wires"]
        if len(wire_kws) != 0:
            return self.visit(wire_kws[0].value)

        if len(node.args) < operation.num_params + 1:
            raise TypeError(
                f"{operation.__name__} expected {operation.num_params + 1} arguments but got {len(node.args)}"
            )

        return self.visit(node.args[operation.num_params])

    def visit_Call(self, node: Call) -> Any:
        callee = self._lookup_attr(node.func)
        is_operation = inspect.isclass(callee) and issubclass(callee, pennylane.operation.Operation)
        if callee:
            if callee == pennylane.expval:
                assert len(node.args) == 1, "expected 1 argument"
                assert isinstance(node.args[0], Call), "expected expval to have Call arg"
                namedobs = self.visit_NamedObs(node.args[0])

                return quantum.ExpvalOp(ir.F64Type.get(), namedobs).results

            elif is_operation:
                if len(node.args) < callee.num_params:
                    raise TypeError(
                        f"{callee.__name__} expected at least {callee.num_params} but got {len(node.args)}"
                    )
                wires = self.get_wires(callee, node)
                params = tuple(
                    self.materialize_constant(self.visit(node.args[i]), force_float=True)
                    for i in range(callee.num_params)
                )

                # Ensure we're dealing with a sequence
                if isinstance(wires, str) or isinstance(wires, int):
                    wires = (wires,)

                assert (
                    len(wires) == callee.num_wires
                ), f"{node.func.attr} called with incorrect number of wires"

                qubits = self.get_qubits_for_wires(wires)
                qubits = quantum.CustomOp(
                    [self.qubit_type] * len(wires), params, qubits, node.func.attr
                ).results
                self.update_qubits(wires, qubits)

    def visit_Name(self, node: Name) -> Any:
        return self._lookup(node.id)

    def cast_to_float(self, val: ir.Value):
        def casted_to_float(typ):
            # TODO: handle tensor case
            return ir.F64Type.get()

        return (
            arith.SIToFPOp(casted_to_float(val.type), val).result
            if self.isIntOrIntTensor(val.type)
            else val
        )

    def isIntOrIntTensor(self, typ):
        # TODO: tensor check not implemented
        return ir.IntegerType.isinstance(typ)

    def visit_BinOp(self, node: BinOp) -> Any:
        lhs = self.materialize_constant(self.visit(node.left))
        rhs = self.materialize_constant(self.visit(node.right))

        both_int = self.isIntOrIntTensor(lhs.type) and self.isIntOrIntTensor(rhs.type)
        int_op_map = {
            ast.Mod: arith.RemSIOp,
            ast.Mult: arith.MulIOp,
            ast.Sub: arith.SubIOp,
        }
        float_op_map = {
            ast.Mod: arith.RemFOp,
            ast.Mult: arith.MulFOp,
            ast.Sub: arith.SubFOp,
        }

        Op = int_op_map[type(node.op)] if both_int else float_op_map[type(node.op)]
        if not both_int:
            lhs = self.cast_to_float(lhs)
            rhs = self.cast_to_float(rhs)
        return Op(lhs, rhs).result

    def visit_Compare(self, node: Compare) -> Any:
        lhs = self.materialize_constant(self.visit(node.left))
        comparators = tuple(self.materialize_constant(self.visit(c)) for c in node.comparators)
        assert len(comparators) == 1, "more than one comparator not yet supported"
        all_int = self.isIntOrIntTensor(lhs.type) and all(
            self.isIntOrIntTensor(c.type) for c in comparators
        )
        Op = arith.CmpIOp if all_int else arith.CmpFOp
        if not all_int:
            lhs = self.cast_to_float(lhs)
            comparators = [self.cast_to_float(c) for c in comparators]

        #   I64EnumAttrCase<"eq", 0>,
        #   I64EnumAttrCase<"ne", 1>,
        #   I64EnumAttrCase<"slt", 2>,
        #   I64EnumAttrCase<"sle", 3>,
        #   I64EnumAttrCase<"sgt", 4>,
        #   I64EnumAttrCase<"sge", 5>,
        #   I64EnumAttrCase<"ult", 6>,
        #   I64EnumAttrCase<"ule", 7>,
        #   I64EnumAttrCase<"ugt", 8>,
        #   I64EnumAttrCase<"uge", 9>,
        int_pred_map = {
            ast.Eq: 0,
        }
        float_pred_map = {ast.Eq: 1, ast.Gt: 2}
        pred = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64),
            int_pred_map[type(node.ops[0])] if all_int else float_pred_map[type(node.ops[0])],
        )
        return Op(pred, lhs, comparators[0]).result

    # Is there an easier way to get constants back into native Python land?
    def visit_Tuple(self, node: Tuple) -> tuple:
        return tuple(self.visit(val) for val in node.elts)

    def visit_Constant(self, node: Constant) -> Any:
        return node.value


"""
Goals for today: June 26th:
Focus on parsing. Need support for nested function calls
Need a general way to think about "I see a name - what does this refer to?"
Basically an environment

Lots to think about w.r.t. nested modules, traversing other imported modules

Numba doesn't use an AST approach, it operates on the bytecote. May be too low level for our needs
"""


class AST_QJIT:
    def __init__(self, fn) -> None:
        self.fn = fn
        self._mlir = ""

    def __call__(self, *args, **kwargs):
        # Get argument types and shapes (at least ranks in the case of dynamic types)
        called_argtypes = {}
        called_argtypes[self.fn.__name__] = tuple(type(arg) for arg in args)
        frame_info = inspect.stack()[1]

        mod_ast = ast.parse(textwrap.dedent(inspect.getsource(self.fn)), type_comments=True)
        with ir.Context() as ctx, ir.Location.file(
            frame_info.filename, line=frame_info.lineno, col=0
        ):
            quantum.register_dialect(ctx)
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                gen = MLIRGenerator(frame_info, module, called_argtypes)
                gen.visit(mod_ast)

            self._mlir = module.operation.get_asm(print_generic_op_form=False)
            self._mlir = quantum.mlir_run_pipeline(self._mlir, "canonicalize")

    @property
    def mlir(self):
        return self._mlir


def qjit_ast(fn=None):
    """Just-in-time compiles the function by parsing it in to an abstract syntax tree."""
    if fn is not None:
        return AST_QJIT(fn)

    return lambda fn: AST_QJIT(fn)
