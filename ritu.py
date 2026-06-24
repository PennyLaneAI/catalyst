import pennylane as qml
import pathlib
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath
from catalyst import pure_callback
from catalyst.compiler import _catalyst, LinkerDriver
from xdsl import passes
from xdsl import context
from xdsl.dialects import builtin
from xdsl.traits import SymbolTable
from xdsl.rewriter import Rewriter, InsertPoint
from pennylane.compiler.python_compiler import compiler_transform, QuantumParser
from pennylane.compiler.python_compiler.dialects import catalyst
import jax
import subprocess
import ctypes
from jax.interpreters import mlir

@compiler_transform
class jitting(passes.ModulePass):

    name = "jitting-through-callback"

    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:

        qml.capture.disable()

        @qml.qjit
        def runtime_function(wire: int):

            @pure_callback
            def callback(wire: int) -> jax.core.ShapedArray([4], float):
                # INSIDE HERE WE ARE RUNNING...

                # TODO:
                # * Add arguments to callback
                # * Make this arguments literals into the program below
                # * Obtain the program from str(module) but change the function to have no parameters
                # * automatically update the return values
                # * Better names
                # * automatically make callback arguments same as the ones from the original function.
                program = f"""
                    func.func public @foobar() -> tensor<4xf64> attributes {{diff_method = "parameter-shift", llvm.emit_c_interface, qnode}} {{
                        // SPECIALIZING THE WIRE AT RUNTIME
                        %dyn_wire = arith.constant {wire} : i64
                        %c0_i64 = arith.constant 0 : i64
                        quantum.device shots(%c0_i64) ["/home/ubuntu/Code/env2/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}}"]
                        %0 = quantum.alloc( 2) : !quantum.reg
                        %1 = quantum.extract %0[ %dyn_wire] : !quantum.reg -> !quantum.bit
                        %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
                        %2 = quantum.insert %0[ %dyn_wire], %out_qubits : !quantum.reg, !quantum.bit
                        %3 = quantum.compbasis qreg %2 : !quantum.obs
                        %4 = quantum.probs %3 : tensor<4xf64>
                        quantum.dealloc %2 : !quantum.reg
                        quantum.device_release
                        return %4 : tensor<4xf64>
                }}
                """
                import os
                dir_path = os.path.dirname(os.path.realpath(__file__))
                inp = dir_path + "/ritu_dir/input.mlir"
                with open(inp, "w") as file:
                    file.write(program)
                out1 = _catalyst(("--tool=all"), ("--workspace", str(dir_path) + "/ritu_dir"), ("--keep-intermediate",), ("--verbose",), ("-o", dir_path + "/jit.so"), (inp,))
                shared_object = LinkerDriver.run(pathlib.Path(dir_path + "/ritu_dir/catalyst_module.o").absolute())
                output_object_name = str(pathlib.Path(shared_object).absolute())
                with mlir.ir.Context(), mlir.ir.Location.unknown():
                    f64 = mlir.ir.F64Type.get()
                    from catalyst.compiled_functions import CompiledFunction
                    compiled_function = CompiledFunction(output_object_name, "foobar", [mlir.ir.RankedTensorType.get((4,), f64)], None, None)
                return compiled_function()

            return callback(wire)

        generic = runtime_function.mlir_module.operation.get_asm(binary=False, print_generic_op_form=True, assume_verified=True)
        ctx = context.Context(allow_unregistered=True)
        xdsl_module = QuantumParser(ctx, generic).parse_module()
        for operation in xdsl_module.walk():
            if isinstance(operation, catalyst.CallbackOp):
                callback = operation

        func = SymbolTable.lookup_symbol(xdsl_module, "jit_runtime_function")
        current_function = SymbolTable.lookup_symbol(module, "foo")
        rewriter = Rewriter()
        func.detach()
        func.sym_name = current_function.sym_name
        rewriter.replace_op(current_function, func)
        callback.detach()
        rewriter.insert_op(callback, InsertPoint.before(func))

qml.capture.enable()

#@qml.qjit
@qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
@jitting
@qml.qnode(qml.device("null.qubit", wires=2))
def foo(wire: int):
    return qml.probs()

print(foo(0))
