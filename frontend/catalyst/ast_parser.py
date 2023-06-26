from _ast import Constant, Expr, Return, Tuple
from ast import FunctionDef, Module, Call, Name, Attribute
from typing import Any
import ast, inspect, textwrap
import numpy as np
import pennylane

import mlir_quantum.ir as ir
import mlir_quantum.dialects.arith as arith
import mlir_quantum.dialects.quantum as quantum
import mlir_quantum.dialects.func as func


class MLIRGenerator(ast.NodeVisitor):
    KNOWN_GATES = {"CNOT"}

    def __init__(self, frame_info: inspect.FrameInfo, module: ir.Module) -> None:
        self.frame_info = frame_info
        self.module = module

        # TODO: Would be nice to have this natively represented
        self.qureg_type = ir.OpaqueType.get("quantum", "reg")
        self.qubit_type = ir.OpaqueType.get("quantum", "bit")
        self.obs_type = ir.OpaqueType.get("quantum", "obs")
        self.qstate = None
        super().__init__()

    def _lookup(self, name: str):
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

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        qnode = (
            self._lookup(node.name).fn
            if isinstance(self._lookup(node.name).fn, pennylane.QNode)
            else None
        )
        # TODO: parse function type
        fn_type = ir.FunctionType.get([], [ir.F64Type.get()])
        func_op = func.FuncOp(node.name, fn_type)
        entry_block = func_op.add_entry_block()
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

    def visit_Return(self, node: Return) -> Any:
        return_val = self.visit(node.value)
        func.ReturnOp(return_val)

    def materialize_constant(self, value, force_float=False) -> ir.Value:
        if isinstance(value, int) and force_float:
            value = float(value)
        if isinstance(value, float):
            return arith.ConstantOp(ir.F64Type.get(), value)
        elif isinstance(value, int):
            return arith.ConstantOp(ir.IntegerType.get_signless(64), value)
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
        # print(ast.dump(node))
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

        return 42

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
        # for arg in args:
        #     if isinstance(arg, np.ndarray):
        #         print(arg.shape, arg.dtype == np.float64)
        frame_info = inspect.stack()[1]

        fn_ast = ast.parse(textwrap.dedent(inspect.getsource(self.fn)), type_comments=True)
        with ir.Context() as ctx, ir.Location.file(
            frame_info.filename, line=frame_info.lineno, col=0
        ):
            quantum.register_dialect(ctx)
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                gen = MLIRGenerator(frame_info, module)
                gen.visit(fn_ast)
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
