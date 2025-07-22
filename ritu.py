import pennylane as qml
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath
from catalyst import pure_callback
from xdsl import passes
from xdsl import context
from xdsl.dialects import builtin
from xdsl.traits import SymbolTable
from xdsl.rewriter import Rewriter, InsertPoint
from pennylane.compiler.python_compiler import compiler_transform, QuantumParser
from pennylane.compiler.python_compiler.dialects import catalyst
import jax

@compiler_transform
class jitting(passes.ModulePass):

    name = "jitting-through-callback"
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:

        qml.capture.disable()

        @qml.qjit
        def foo():
            @pure_callback
            def callback() -> jax.core.ShapedArray([2], float):
                return jax.numpy.array([0.0, 1.0])
            return callback()

        generic = foo.mlir_module.operation.get_asm(binary=False, print_generic_op_form=True, assume_verified=True)
        ctx = context.Context(allow_unregistered=True)
        xdsl_module = QuantumParser(ctx, generic).parse_module()
        for operation in xdsl_module.walk():
            if isinstance(operation, catalyst.CallbackOp):
                callback = operation
        func = SymbolTable.lookup_symbol(xdsl_module, "jit_foo")
        current_function = SymbolTable.lookup_symbol(module, "foo")
        rewriter = Rewriter()
        func.detach()
        func.sym_name = current_function.sym_name
        rewriter.replace_op(current_function, func)
        callback.detach()
        rewriter.insert_op(callback, InsertPoint.before(func))

qml.capture.enable()

@qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
@jitting
@qml.qnode(qml.device("null.qubit", wires=1))
def foo():
    return qml.probs()

print(foo())
