from _ast import BinOp, Compare, Constant, Expr, If, For, Return, Tuple
from _ast import (
    FunctionDef,
    Module,
    Call,
    Name,
    Attribute,
    UnaryOp,
    List,
    ListComp,
    Assign,
    Subscript,
)
from typing import Any, Callable
import ast, inspect, textwrap
from catalyst.libloader import SharedLibraryManager
from catalyst.compiler import Compiler, CompileOptions, CompilerDriver
from catalyst.compilation_pipelines import CompiledFunction

import os

# import numpy as np
import pennylane
import pennylane.numpy as np

import mlir_quantum.ir as ir
import mlir_quantum.dialects.arith as arith
import mlir_quantum.dialects.func as func
import mlir_quantum.dialects.quantum as quantum
import mlir_quantum.dialects.scf as scf
import mlir_quantum.dialects.tensor as tensor

from dataclasses import dataclass


@dataclass
class MethodCall:
    this: ir.Value
    method: Callable


@dataclass
class Range:
    start: int
    stop: int
    step: int


@dataclass
class Enumerate:
    value: ir.Value
    start: int = 0


class MLIRGenerator(ast.NodeVisitor):
    def __init__(
        self,
        frame_info: inspect.FrameInfo,
        module: ir.Module,
        fn,
        called_argtypes: dict[str, tuple[type]],
    ) -> None:
        self.frame_info = frame_info
        self.module = module
        self.function_cache = {}
        self.fn = fn
        self.called_argtypes = called_argtypes
        # Environments are searched in reverse order
        self._envs = [
            frame_info.frame.f_builtins,
            frame_info.frame.f_globals,
            frame_info.frame.f_locals,
        ]
        self._len_base_envs = len(self._envs)

        self.qureg_type = quantum.QuregType.get()
        self.qubit_type = quantum.QubitType.get()
        self.obs_type = quantum.ObservableType.get()
        self.qstate = None
        super().__init__()

    def _lookup(self, name: str | Attribute | Name):
        if isinstance(name, Attribute):
            return self._lookup_attr(name)
        if isinstance(name, Name):
            return self._lookup_str(name.id)
        return self._lookup_str(name)

    def _push_env(self):
        self._envs.append({})

    def _pop_env(self):
        assert len(self._envs) > self._len_base_envs, "_pop_env on empty environment"
        self._envs.pop()

    def _declare(self, name: str, value):
        assert len(self._envs) > self._len_base_envs, "_declare on empty environment"
        self._envs[-1][name] = value

    def _lookup_str(self, name: str):
        for env in self._envs[::-1]:
            if name in env:
                return env[name]

        raise ValueError(f"MLIR Generator: name not found: {name}")

    def _lookup_attr(self, attr: Attribute):
        value_id = self._lookup_str(attr.value.id)
        if inspect.ismodule(value_id):
            if value_id == pennylane and hasattr(pennylane, attr.attr):
                return getattr(pennylane, attr.attr)
            if value_id == np and hasattr(np, attr.attr):
                return getattr(np, attr.attr)
        if isinstance(value_id, ir.Value):
            # This is a method call, try to infer the method from the MLIR type
            if ir.RankedTensorType.isinstance(value_id.type):
                return MethodCall(value_id, getattr(np, attr.attr))
        return None

    def visit_Module(self, node: Module):
        for stmt in node.body:
            self.visit(stmt)

        # Add setup and teardown functions
        void_func_type = ir.FunctionType.get([], [])
        setup_func = func.FuncOp("setup", void_func_type)
        entry_block = setup_func.add_entry_block()
        with ir.InsertionPoint(entry_block):
            quantum.InitializeOp()
            func.ReturnOp([])
        teardown_func = func.FuncOp("teardown", void_func_type)
        entry_block = teardown_func.add_entry_block()
        with ir.InsertionPoint(entry_block):
            quantum.FinalizeOp()
            func.ReturnOp([])

        # Add the public entry point
        func_op = ir.SymbolTable(self.module.operation)[self.fn.__name__]

        with ir.InsertionPoint.at_block_begin(self.module.body):
            public_fn = func.FuncOp(f"jit_{self.fn.__name__}", func_op.type)
            entry_block = public_fn.add_entry_block()
            with ir.InsertionPoint(entry_block):
                results = func.CallOp(func_op, list(entry_block.arguments)).results
                func.ReturnOp(results)
        return public_fn

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        fn = self._lookup_str(node.name)

        is_qnode = isinstance(fn, AST_QJIT) and isinstance(fn.fn, pennylane.QNode)
        qnode = self._lookup_str(node.name).fn if is_qnode else None

        called_types = self.called_argtypes[node.name]
        has_qstate = len(called_types) > 0 and self.is_qureg(called_types[-1])
        if not (
            len(called_types) == len(node.args.args)
            or (has_qstate and len(called_types) - 1 == len(node.args.args))
        ):
            raise TypeError(
                f"Function {node.name} expects {len(node.args.args)} types but was called with {len(called_types) - int(has_qstate)}"
            )
        # TODO: definitely possible that there's a type mismatch here
        arg_types = [_type_to_mlir_type(typ) for typ in called_types]
        fn_type = ir.FunctionType.get(arg_types, [])
        func_op = func.FuncOp(node.name, fn_type)
        func_op.attributes["sym_visibility"] = ir.StringAttr.get("private")
        entry_block = func_op.add_entry_block()
        self._push_env()
        for argname, blockarg in zip(node.args.args, entry_block.arguments):
            # TODO: this ignores annotations on the function arguments
            self._declare(argname.arg, blockarg)

        result_types = [self.qureg_type] if has_qstate else []
        with ir.InsertionPoint(entry_block):
            if has_qstate:
                self.qstate = entry_block.arguments[-1]
            elif qnode:
                name_attr = ir.StringAttr.get(qnode.device.short_name)
                quantum.DeviceOp(specs=ir.ArrayAttr.get([ir.StringAttr.get("backend"), name_attr]))
                self.qstate = quantum.AllocOp(
                    self.qureg_type, nqubits_attr=qnode.device.num_wires
                ).result

            for stmt in node.body:
                result = self.visit(stmt)
                if isinstance(stmt, Return):
                    result_types = [r.type for r in result]

        # Patch the return type based on the returned statement
        fn_type = ir.FunctionType.get(arg_types, result_types)
        func_op.attributes["function_type"] = ir.TypeAttr.get(fn_type)
        self._pop_env()

        self.function_cache[node.name] = func_op

        return func_op

    def visit_Expr(self, node: Expr) -> Any:
        # Not sure this is required
        return self.visit(node.value)

    def get_qubits_for_wires(self, wires: tuple[int | ir.Value]):
        def emit_extract(wire: int | ir.Value):
            kwargs = {
                "idx_attr": wire if isinstance(wire, int) else None,
                "idx": self.cast_index_to_int(wire) if isinstance(wire, ir.Value) else None,
            }
            return quantum.ExtractOp(self.qubit_type, self.qstate, **kwargs).result

        return tuple(emit_extract(wire) for wire in wires)

    def update_qubits(self, wires: tuple[int | ir.Value], qubits):
        for wire, qubit in zip(wires, qubits):
            kwargs = {
                "idx_attr": wire if isinstance(wire, int) else None,
                "idx": self.cast_index_to_int(wire) if isinstance(wire, ir.Value) else None,
            }
            self.qstate = quantum.InsertOp(self.qureg_type, self.qstate, qubit, **kwargs)

    def visit_NamedObs(self, node: Call) -> Any:
        callee = self._lookup_attr(node.func)
        assert len(node.args) == 1, "Expected named observable to have one argument (wires)"
        wire = self.visit(node.args[0])
        qubit = self.get_qubits_for_wires((wire,))[0]
        obs_type = quantum.NamedObservableAttr.get(callee.__name__)
        return quantum.NamedObsOp(self.obs_type, qubit, obs_type).results

    def visit_Assign(self, node: Assign):
        assert len(node.targets) == 1, "destructuring not yet supported"
        targets = [self.visit(target) for target in node.targets]
        self._declare(targets[0], self.materialize_constant(self.visit(node.value)))

    def visit_ListComp(self, node: ListComp):
        self._push_env()
        # TODO: should reuse code with for code
        assert len(node.generators) == 1, "Nested list comprehension generators not supported"
        for idx, generator in enumerate(node.generators):
            targets = self.visit(generator.target)
            iteration_space = self.visit(generator.iter)
            is_tensor = lambda value: isinstance(
                value, ir.Value
            ) and ir.RankedTensorType.isinstance(value.type)

            is_enumerate = isinstance(iteration_space, Enumerate)

            assert isinstance(generator.target, Name) or (
                is_enumerate and isinstance(generator.target, Tuple)
            ), "Expected for loop iv to be a name"
            is_iterating_over_tensor = is_tensor(iteration_space) or (
                is_enumerate and is_tensor(iteration_space.value)
            )

            if isinstance(iteration_space, Range):
                start, stop, step = (
                    iteration_space.start,
                    iteration_space.stop,
                    iteration_space.step,
                )
                num_iters = arith.CeilDivSIOp(arith.SubIOp(stop, start), step).result
            elif is_iterating_over_tensor:
                iter_val = iteration_space.value if is_enumerate else iteration_space
                tensor_type = ir.RankedTensorType(iter_val.type)

                if len(tensor_type.shape) == 0:
                    raise ValueError("Can't iterate over a Rank-0 tensor")
                get_index = lambda idx: self.cast_int_to_index(self.materialize_constant(idx))
                c0, c1 = get_index(0), get_index(1)
                start = c0
                stop = tensor.DimOp(iter_val, c0)
                step = c1
            else:
                raise TypeError(f"Unknown loop iteration space: {iteration_space}")

            # Assume for now that the result of the list comprehension is an expval (f64)
            output = tensor.EmptyOp([num_iters], ir.F64Type.get())

            for_op = scf.ForOp(start, stop, step, iter_args=(output, self.qstate))
            body_block = for_op.body
            with ir.InsertionPoint(body_block):
                # iv = body_block.arguments[0]
                iv, output, self.qstate = body_block.arguments
                # self.qstate = body_block.arguments[-1]
                if isinstance(iteration_space, Range):
                    self._declare(targets, iv)
                elif is_iterating_over_tensor:
                    # Iterate over slices of the outermost dimension. Iterate over the elements themselves if the tensor is rank 1.
                    if len(tensor_type.shape) == 1:
                        # TODO: If we want to support tensor mutation, the tensor needs to be passed through the iter_args
                        values = tensor.ExtractOp(iter_val, (iv,))
                    else:
                        # TODO: slicing is wip/untested
                        slice_rank = tensor_type.rank - 1
                        slice_type = ir.RankedTensorType.get(
                            tensor_type.shape[1:], tensor_type.element_type
                        )
                        values = tensor.ExtractSliceOp(
                            slice_type,
                            iter_val,
                            [iv],  # offsets
                            [],  # sizes
                            [],  # strides
                            [ir.ShapedType.get_dynamic_size()] + [0] * slice_rank,  # static offsets
                            tensor_type.shape[1:],  # static sizes
                            [1] * slice_rank,  # static strides
                        )
                    if not enumerate:
                        values = (values,)
                    for name, val in zip(targets, [iv, values]):
                        self._declare(name, val)

                # for stmt in node.body:
                #     self.visit(stmt)
                if idx == len(node.generators) - 1:
                    output = tensor.InsertOp(self.visit(node.elt), output, [iv])

                scf.YieldOp((output, self.qstate))

            self.qstate = for_op.results[-1]

        self._pop_env()
        return for_op.results[0]

    def visit_For(self, node: For):
        if len(node.orelse) != 0:

            class ShameError(TypeError):
                """You should be ashamed of yourself"""

                pass

            raise ShameError("Why in gods name would you use an else block after a for loop")

        targets = self.visit(node.target)
        iteration_space = self.visit(node.iter)
        is_tensor = lambda value: isinstance(value, ir.Value) and ir.RankedTensorType.isinstance(
            value.type
        )

        is_enumerate = isinstance(iteration_space, Enumerate)

        assert isinstance(node.target, Name) or (
            is_enumerate and isinstance(node.target, Tuple)
        ), "Expected for loop iv to be a name"
        is_iterating_over_tensor = is_tensor(iteration_space) or (
            is_enumerate and is_tensor(iteration_space.value)
        )

        if isinstance(iteration_space, Range):
            start, stop, step = iteration_space.start, iteration_space.stop, iteration_space.step
        elif is_iterating_over_tensor:
            iter_val = iteration_space.value if is_enumerate else iteration_space
            tensor_type = ir.RankedTensorType(iter_val.type)

            if len(tensor_type.shape) == 0:
                raise ValueError("Can't iterate over a Rank-0 tensor")
            get_index = lambda idx: self.cast_int_to_index(self.materialize_constant(idx))
            c0, c1 = get_index(0), get_index(1)
            start = c0
            stop = tensor.DimOp(iter_val, c0)
            step = c1
        else:
            raise TypeError(f"Unknown loop iteration space: {iteration_space}")

        for_op = scf.ForOp(start, stop, step, iter_args=(self.qstate,))
        body_block = for_op.body
        with ir.InsertionPoint(body_block):
            iv = body_block.arguments[0]
            self.qstate = body_block.arguments[-1]
            if isinstance(iteration_space, Range):
                self._declare(targets, iv)
            elif is_iterating_over_tensor:
                # Iterate over slices of the outermost dimension. Iterate over the elements themselves if the tensor is rank 1.
                if len(tensor_type.shape) == 1:
                    # TODO: If we want to support tensor mutation, the tensor needs to be passed through the iter_args
                    values = tensor.ExtractOp(iter_val, (iv,))
                else:
                    # TODO: slicing is wip/untested
                    slice_rank = tensor_type.rank - 1
                    slice_type = ir.RankedTensorType.get(
                        tensor_type.shape[1:], tensor_type.element_type
                    )
                    values = tensor.ExtractSliceOp(
                        slice_type,
                        iter_val,
                        [iv],  # offsets
                        [],  # sizes
                        [],  # strides
                        [ir.ShapedType.get_dynamic_size()] + [0] * slice_rank,  # static offsets
                        tensor_type.shape[1:],  # static sizes
                        [1] * slice_rank,  # static strides
                    )
                if not enumerate:
                    values = (values,)
                for name, val in zip(targets, [iv, values]):
                    self._declare(name, val)

            for stmt in node.body:
                self.visit(stmt)
            scf.YieldOp((self.qstate,))

        self.qstate = for_op.results[0]

    def visit_If(self, node: If) -> Any:
        pred = self.materialize_constant(self.visit(node.test))
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
        # TODO: Need an ensure_sequence method?
        if isinstance(return_val, ir.Value):
            return_val = (return_val,)

        func.ReturnOp(return_val)
        return return_val

    def materialize_constant(self, value, force_float=False) -> ir.Value:
        if isinstance(value, int) and force_float:
            value = float(value)
        if isinstance(value, float) or isinstance(value, int) or isinstance(value, bool):
            return arith.ConstantOp(_type_to_mlir_type(type(value)), value).result
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
        callee = self._lookup(node.func)
        is_operation = inspect.isclass(callee) and issubclass(callee, pennylane.operation.Operation)

        def visit_range(node):
            args = tuple(self.visit(arg) for arg in node.args)
            if len(args) == 1:
                bounds = (0, args[0], 1)
            else:
                bounds = (args[0], args[1], args[2] if len(args) >= 3 else 1)
            return Range(
                *tuple(self.cast_int_to_index(self.materialize_constant(bound)) for bound in bounds)
            )

        def visit_tuple_func(node: Call):
            assert len(node.args) == 1, "tuple function expected to have one argument"
            args = tuple(self.visit(arg) for arg in node.args)

            if ir.RankedTensorType.isinstance(args[0].type):
                return args[0]
            assert False, "Unsupported type for tuple constructor"

        builtin_functions = {
            enumerate: lambda node: Enumerate(*(self.visit(arg) for arg in node.args)),
            range: visit_range,
            tuple: visit_tuple_func,
        }
        if callee:
            if isinstance(callee, MethodCall):
                # We don't support classes, so assume method calls are to builtin (probably numpy) functions.
                if callee.method == np.reshape:
                    args = [self.visit(node.args[i]) for i in range(len(node.args))]
                    result_shape = [
                        arg if isinstance(arg, int) else ir.ShapedType.get_dynamic_size()
                        for arg in args
                    ]
                    result_type = ir.RankedTensorType.get(
                        result_shape, _get_element_type(callee.this.type)
                    )
                    args = tuple(self.materialize_constant(args[i]) for i in range(len(args)))
                    new_shape_tensor = tensor.FromElementsOp(
                        ir.RankedTensorType.get((len(args),), ir.IntegerType.get_signless(64)), args
                    )
                    return tensor.ReshapeOp(result_type, callee.this, new_shape_tensor).result
                assert False, "unimplemented method call"
            elif callee == np.zeros:
                # TODO: code duplication between numpy functions
                # TODO: This won't work with dynamic shapes
                args = [self.visit(node.args[i]) for i in range(len(node.args))]
                result_shape = [
                    arg if isinstance(arg, int) else ir.ShapedType.get_dynamic_size()
                    for arg in args
                ]
                result_type = ir.RankedTensorType.get(result_shape, ir.F64Type.get())
                zero = self.materialize_constant(0.0)
                result = tensor.EmptyOp(result_shape, ir.F64Type.get()).result
                return result
            elif callee == pennylane.expval:
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
                if isinstance(wires, ir.Value):
                    wires = (self.cast_index_to_int(wires),)

                assert (
                    len(wires) == callee.num_wires
                ), f"{node.func.attr} called with incorrect number of wires"

                qubits = self.get_qubits_for_wires(wires)
                qubits = quantum.CustomOp(
                    [self.qubit_type] * len(wires), params, qubits, node.func.attr
                ).results
                self.update_qubits(wires, qubits)
            elif callee in builtin_functions:
                return builtin_functions[callee](node)
            else:
                # If it's an unknown function, look for the source
                mod_ast = ast.parse(textwrap.dedent(inspect.getsource(callee)))
                # Implicitly pass in the quantum state as the last argument
                args = [
                    self.materialize_constant(self.visit(node.args[i]))
                    for i in range(len(node.args))
                ] + [self.qstate]

                # Use dynamically sized tensors by default
                arg_types = []
                for i, arg in enumerate(args):
                    if ir.RankedTensorType.isinstance(arg.type):
                        tensor_type = ir.RankedTensorType(arg.type)
                        dynamic_type = ir.RankedTensorType.get(
                            [ir.ShapedType.get_dynamic_size()] * tensor_type.rank,
                            tensor_type.element_type,
                        )
                        arg_types.append(dynamic_type)
                        args[i] = tensor.CastOp(dynamic_type, arg)
                    else:
                        arg_types.append(arg.type)

                self.called_argtypes[callee.__name__] = tuple(arg_types)

                if callee.__name__ in self.function_cache:
                    func_op = self.function_cache[callee.__name__]
                else:
                    with ir.InsertionPoint(self.module.body):
                        func_op = self.visit(mod_ast.body[0])
                    func_op.attributes["sym_visibility"] = ir.StringAttr.get("private")
                    # If there's no return, insert an implicit return of the quantum state.
                    # Otherwise, augment the existing return.
                    oplist = func_op.body.blocks[0].operations
                    # OperationList in the MLIR bindings doesn't support [-1] indexing
                    if not isinstance(oplist[len(oplist) - 1], func.ReturnOp):
                        func_op.body.blocks[0].append(func.ReturnOp((self.qstate,)))
                    else:
                        assert False, "is it possible to append an argument to a return op?"

                # The bindings need the args to be a list and not a tuple
                results = func.CallOp(func_op, list(args)).results
                self.qstate = results[-1]
        else:
            assert False, f"unhandled callee: {ast.dump(node.func)}"

    def visit_Name(self, node: Name) -> Any:
        if isinstance(node.ctx, ast.Load):
            return self._lookup_str(node.id)
        # The ctx is store, meaning a variable shouldn't exist
        return node.id

    #
    # Casting
    #
    def cast_int_to_index(self, val: ir.Value):
        return (
            arith.IndexCastOp(ir.IndexType.get(), val)
            if ir.IntegerType.isinstance(val.type)
            else val
        )

    def cast_index_to_int(self, val: ir.Value):
        # TODO: handle tensor case
        is_index_type = ir.IndexType.isinstance(val.type)
        return (
            arith.IndexCastOp(ir.IntegerType.get_signless(64), val).result if is_index_type else val
        )

    def cast_to_float(self, val: ir.Value):
        def casted_to_float(typ):
            # TODO: handle tensor case
            return ir.F64Type.get()

        val = self.cast_index_to_int(val)
        return (
            arith.SIToFPOp(casted_to_float(val.type), val).result
            if self.is_int_or_int_tensor(val.type)
            else val
        )

    def is_qureg(self, typ):
        return isinstance(typ, ir.Type) and quantum.QuregType.isinstance(typ)

    def is_float_or_float_tensor(self, typ):
        return ir.F64Type.isinstance(typ)

    def is_int_or_index_or_tensor(self, typ):
        return ir.IndexType.isinstance(typ) or self.is_int_or_int_tensor(typ)

    def is_int_or_int_tensor(self, typ):
        # TODO: tensor check not implemented
        return ir.IntegerType.isinstance(typ)

    def visit_UnaryOp(self, node: UnaryOp):
        assert isinstance(node.op, ast.USub), f"Unsupported unary op: {node.op}"
        value = self.materialize_constant(self.visit(node.operand))
        if self.is_float_or_float_tensor(value.type):
            return arith.NegFOp(value).result

        if not self.is_int_or_int_tensor(value.type):
            raise TypeError(f"Expected float or int type, got: {value.type}")

        c0 = self.materialize_constant(0)
        return arith.SubIOp(c0, value).result

    def visit_BinOp(self, node: BinOp) -> Any:
        lhs = self.materialize_constant(self.visit(node.left))
        rhs = self.materialize_constant(self.visit(node.right))
        both_int = self.is_int_or_index_or_tensor(lhs.type) and self.is_int_or_index_or_tensor(
            rhs.type
        )
        int_op_map = {
            ast.Mod: arith.RemSIOp,
            ast.Mult: arith.MulIOp,
            ast.Sub: arith.SubIOp,
            ast.Add: arith.AddIOp,
        }
        float_op_map = {
            ast.Mod: arith.RemFOp,
            ast.Mult: arith.MulFOp,
            ast.Sub: arith.SubFOp,
            ast.Add: arith.AddFOp,
        }

        Op = int_op_map[type(node.op)] if both_int else float_op_map[type(node.op)]
        if both_int:
            lhs = self.cast_index_to_int(lhs)
            rhs = self.cast_index_to_int(rhs)
        else:
            lhs = self.cast_to_float(lhs)
            rhs = self.cast_to_float(rhs)
        return Op(lhs, rhs).result

    def visit_Compare(self, node: Compare) -> Any:
        lhs = self.materialize_constant(self.visit(node.left))
        comparators = tuple(self.materialize_constant(self.visit(c)) for c in node.comparators)
        assert len(comparators) == 1, "more than one comparator not yet supported"
        all_int = self.is_int_or_int_tensor(lhs.type) and all(
            self.is_int_or_int_tensor(c.type) for c in comparators
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
            ast.Lt: 2,
            ast.LtE: 3,
            ast.Gt: 4,
            ast.GtE: 5,
        }
        float_pred_map = {ast.Eq: 1, ast.Gt: 2}
        pred = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64),
            int_pred_map[type(node.ops[0])] if all_int else float_pred_map[type(node.ops[0])],
        )
        return Op(pred, lhs, comparators[0]).result

    def visit_Subscript(self, node: Subscript):
        value = self.visit(node.value)
        _slice = self.visit(node.slice)
        if not ir.RankedTensorType.isinstance(value.type):
            raise TypeError("Cannot subscript a non-tensor")

        if not isinstance(_slice, tuple):
            _slice = (_slice,)

        assert all(indexer is not None for indexer in _slice), "Unhandled indexing"

        # Assume no Slice() indexers for now
        tensor_type = ir.RankedTensorType(value.type)
        if len(_slice) == tensor_type.rank:
            offsets = [self.cast_int_to_index(self.materialize_constant(offs)) for offs in _slice]
            return tensor.ExtractOp(value, offsets).result
        if len(_slice) < tensor_type.rank:
            result_type = ir.RankedTensorType.get(
                tensor_type.shape[len(_slice) :], tensor_type.element_type
            )
            reduced_rank = len(_slice)
            dyn_offsets = [
                self.cast_int_to_index(offs) for offs in _slice if isinstance(offs, ir.Value)
            ]
            static_offsets = [
                (ir.ShapedType.get_dynamic_size() if isinstance(offs, ir.Value) else offs)
                for offs in _slice
            ]
            return tensor.ExtractSliceOp(
                result_type,
                value,
                dyn_offsets,
                [],  # dyn_sizes
                [],  # dyn_strides
                static_offsets + [0] * result_type.rank,
                [1] * reduced_rank + result_type.shape,
                [1] * tensor_type.rank,
            ).result

        assert False, "unhandled subscript"

    # Is there an easier way to get constants back into native Python land?
    def visit_Tuple(self, node: Tuple) -> tuple:
        return tuple(self.visit(val) for val in node.elts)

    def visit_List(self, node: List):
        return [self.visit(val) for val in node.elts]

    def visit_Constant(self, node: Constant) -> Any:
        return node.value


def _get_element_type(typ: ir.Type):
    if ir.ShapedType.isinstance(typ):
        return ir.ShapedType(typ).element_type
    return None


def _type_to_mlir_type(typ: type):
    if isinstance(typ, ir.Type):
        return typ
    if typ == float:
        return ir.F64Type.get()
    if typ == int:
        return ir.IntegerType.get_signless(64)
    if typ == bool:
        return ir.IntegerType.get_signless(1)

    assert False, f"Unhandled type {typ}"


def get_argument_types(args):
    def get_argument_type(arg):
        if isinstance(arg, np.ndarray):
            return ir.RankedTensorType.get(arg.shape, _type_to_mlir_type(arg.dtype))
        return type(arg)

    return [get_argument_type(arg) for arg in args]


class AST_QJIT:
    def __init__(self, fn, canonicalize) -> None:
        self.fn = fn
        self._mlir = ""
        self._module = None
        self._func_name = None
        self._func_type = None
        self.canonicalize = canonicalize

    def generate_mlir(self, *args, **kwargs):
        # Get argument types and shapes (at least ranks in the case of dynamic types)
        called_argtypes = {}
        # The indexing [2] here is super brittle
        frame_info = inspect.stack()[2]

        mod_ast = ast.parse(textwrap.dedent(inspect.getsource(self.fn)), type_comments=True)
        with ir.Context() as ctx, ir.Location.file(
            frame_info.filename, line=frame_info.lineno, col=0
        ):
            quantum.register_dialect(ctx)
            called_argtypes[self.fn.__name__] = tuple(get_argument_types(args))
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                gen = MLIRGenerator(frame_info, module, self.fn, called_argtypes)
                func_op = gen.visit(mod_ast)
                self._func_name = func_op.name
                self._func_type = func_op.type

            self._module = module
            self._mlir = module.operation.get_asm(print_generic_op_form=False)
            if self.canonicalize:
                self._mlir = quantum.mlir_run_pipeline(self._mlir, "canonicalize")

    def __call__(self, *args, **kwargs):
        self.generate_mlir(*args, **kwargs)
        public_func_name = str(self._func_name).replace('"', "")
        compiler = Compiler()
        filename = f"{compiler.workspace.name}/{self.fn.__name__}.o"
        quantum.compile(self._module, filename)
        shared_object = os.path.abspath(CompilerDriver.run(filename))
        manager = SharedLibraryManager(shared_object, public_func_name, self._func_type)
        return manager(*args, **kwargs)

    @property
    def mlir(self):
        return self._mlir


def qjit_ast(fn=None, canonicalize=True):
    """Just-in-time compiles the function by parsing it in to an abstract syntax tree."""
    if fn is not None:
        return AST_QJIT(fn, canonicalize=canonicalize)

    return lambda fn: AST_QJIT(fn, canonicalize=canonicalize)
