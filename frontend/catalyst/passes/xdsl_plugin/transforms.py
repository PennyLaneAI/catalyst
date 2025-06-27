import jax
import pennylane as qml
from pennylane.compiler.python_compiler import compiler_transform
from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin, func, llvm
from xdsl.rewriter import InsertPoint, Rewriter


def add_call_to_add_new_frame(f: func.FuncOp, module):
    rewriter = Rewriter()
    name = f.sym_name.data
    encoded = name.encode() + b"\x00"
    t_type = builtin.TensorType(builtin.i8, [len(encoded)])
    value = builtin.DenseIntOrFPElementsAttr([t_type, builtin.BytesAttr(encoded)])
    _global = llvm.GlobalOp(
        llvm.LLVMArrayType.from_size_and_type(len(encoded), builtin.i8),
        name + "_name",
        llvm.LinkageAttr("internal"),
        constant=True,
        value=value,
    )
    rewriter.insert_op(_global, InsertPoint.before(module.body.blocks[0].first_op))

    addressof_op = llvm.AddressOfOp(_global.sym_name, llvm.LLVMPointerType.opaque())
    first = f.body.blocks[0].first_op
    rewriter.insert_op(addressof_op, InsertPoint.before(first))
    call_op = llvm.CallOp("add_new_frame", addressof_op.result, variadic_args=0)
    call_op.var_callee_type = None
    call_op2 = llvm.CallOp("rm_frame", variadic_args=0)
    call_op2.var_callee_type = None
    last = f.body.blocks[0].last_op
    rewriter.insert_op(call_op, InsertPoint.before(first))
    rewriter.insert_op(call_op2, InsertPoint.before(last))


@compiler_transform
class ProfileMemory(passes.ModulePass):
    name = "profile-memory"

    def apply(self, ctx: context.Context, module: builtin.ModuleOp) -> None:

        rewriter = Rewriter()
        func_decl = llvm.FuncOp(
            "add_new_frame",
            llvm.LLVMFunctionType([llvm.LLVMPointerType.opaque()]),
            linkage=llvm.LinkageAttr("external"),
        )
        func_decl2 = llvm.FuncOp(
            "rm_frame", llvm.LLVMFunctionType([]), linkage=llvm.LinkageAttr("external")
        )
        for op in module.walk():
            if isinstance(op, func.FuncOp):
                add_call_to_add_new_frame(op, module)
        rewriter.insert_op(func_decl, InsertPoint.before(module.body.blocks[0].first_op))
        rewriter.insert_op(func_decl2, InsertPoint.before(module.body.blocks[0].first_op))
