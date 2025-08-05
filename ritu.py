import functools
import os
from pathlib import Path

import numpy as np
import pennylane as qml
from jax.interpreters import mlir

from catalyst import QJIT, CompileOptions, pure_callback
from catalyst.compiled_functions import CompiledFunction
from catalyst.compiler import LinkerDriver, _catalyst


def jitting(qnode):
    assert isinstance(qnode, qml.QNode)

    @functools.wraps(qnode)
    def tracing_function(*args, **kwargs):

        # trace the code once to get the i/o signature
        # to avoid overheads, we create the QJIT object manually and invoke specific stages
        class Basic_QJIT:
            def __init__(self, qnode):
                self.__name__ = "jit_" + qnode.__name__
                self.user_function = qnode
                self.compile_options = CompileOptions()

        jit_obj = Basic_QJIT(qnode)
        jaxpr, _, treedef, _ = QJIT.capture(jit_obj, args, **kwargs)

        @pure_callback(result_type=treedef.unflatten(jaxpr.out_avals))
        def runtime_function(*concrete_args):
            # INSIDE HERE WE ARE RUNNING...

            # TODO:
            # * Better names
            # * static args handling currently only works for scalars

            # ensure we are using static arguments for tracing this time
            # this is actually tricky because arrays cannot be used as static args
            # we might need a different way of removing arguments from the function
            jit_obj.compile_options.static_argnums = tuple(range(len(concrete_args)))

            # get the specialized jaxpr+mlir this time
            #   (we could probably simplify further and just use `catalyst.qjit` directly)
            jit_obj.jaxpr, jit_obj.out_type, jit_obj.treedef, jit_obj.dynamic_sig = QJIT.capture(
                jit_obj, tuple(arg.item() for arg in concrete_args)
            )
            mlir_module = QJIT.generate_ir(jit_obj)

            # eliminate setup/teardown functions which will mess with the runtime
            for _ in range(2):
                mlir_module.body.operations[-1].erase()

            # get some properties of the entry point function
            entry_point = mlir_module.body.operations[0]
            entry_symbol = str(entry_point.name).replace('"', "")
            entry_result_types = entry_point.type.results

            dir_path = os.path.dirname(os.path.realpath(__file__)) + "/ritu_dir"
            Path(dir_path).mkdir(parents=True, exist_ok=True)

            inp = dir_path + "/input.mlir"
            with open(inp, "w+") as file:
                file.write(str(mlir_module))

            out1 = _catalyst(
                ("--tool=all"),
                ("--workspace", str(dir_path)),
                ("--keep-intermediate",),
                ("--verbose",),
                ("-o", dir_path + "/jit.so"),
                (inp,),
            )

            shared_object = LinkerDriver.run(Path(dir_path + "/catalyst_module.o").absolute())
            output_object_name = str(Path(shared_object).absolute())

            with mlir.ir.Context(), mlir.ir.Location.unknown():
                f64 = mlir.ir.F64Type.get()

                compiled_function = CompiledFunction(
                    output_object_name, entry_symbol, entry_result_types, None, None
                )

            return compiled_function()

        return runtime_function(*args)

    return tracing_function


@qml.qjit
@jitting
@qml.qnode(qml.device("lightning.qubit", wires=2))
def foo(w1, w2):
    qml.Hadamard(w1)
    qml.CNOT([w1, w2])
    return qml.probs()


print(foo(0, 1))
