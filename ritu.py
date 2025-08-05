import ctypes
import functools
import pathlib
import subprocess
from pathlib import Path

import jax
import pennylane as qml
from jax.interpreters import mlir

from catalyst import pure_callback
from catalyst.compiler import LinkerDriver, _catalyst


def jitting(qnode):
    assert isinstance(qnode, qml.QNode)

    @functools.wraps(qnode)
    def tracing_function(*args, **kwargs):

        @pure_callback
        def runtime_function(wire: int) -> jax.core.ShapedArray([4], float):
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
                    quantum.device shots(%c0_i64) ["{qnode.device.get_c_interface()[1]}", "{qnode.device.get_c_interface()[0]}", ""]
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

            dir_path = os.path.dirname(os.path.realpath(__file__)) + "/ritu_dir"
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            inp = dir_path + "/input.mlir"
            with open(inp, "w") as file:
                file.write(program)
            out1 = _catalyst(
                ("--tool=all"),
                ("--workspace", str(dir_path)),
                ("--keep-intermediate",),
                ("--verbose",),
                ("-o", dir_path + "/jit.so"),
                (inp,),
            )
            shared_object = LinkerDriver.run(
                pathlib.Path(dir_path + "/catalyst_module.o").absolute()
            )
            output_object_name = str(pathlib.Path(shared_object).absolute())
            with mlir.ir.Context(), mlir.ir.Location.unknown():
                f64 = mlir.ir.F64Type.get()
                from catalyst.compiled_functions import CompiledFunction

                compiled_function = CompiledFunction(
                    output_object_name,
                    "foobar",
                    [mlir.ir.RankedTensorType.get((4,), f64)],
                    None,
                    None,
                )

            return compiled_function()

        return runtime_function(args[0])

    return tracing_function


@qml.qjit
@jitting
@qml.qnode(qml.device("lightning.qubit", wires=2))
def foo(wire: int):
    return qml.probs()


print(foo(0))
