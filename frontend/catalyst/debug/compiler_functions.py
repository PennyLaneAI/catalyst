# Copyright 2023-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains debug functions to interact with the compiler and compiled functions.
"""
import logging
import os
import shutil

from jax.interpreters import mlir

import catalyst
from catalyst.compiled_functions import CompiledFunction
from catalyst.compiler import Compiler, LinkerDriver
from catalyst.logging import debug_logger
from catalyst.tracing.contexts import EvaluationContext
from catalyst.tracing.type_signatures import filter_static_args, promote_arguments
from catalyst.utils.filesystem import WorkspaceManager

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@debug_logger
def print_compilation_stage(fn, stage):
    """Print one of the recorded compilation stages for a JIT-compiled function.

    The stages are indexed by their Catalyst compilation pipeline name, which are either provided
    by the user as a compilation option, or predefined in ``catalyst.compiler``.

    Requires ``keep_intermediate=True``.

    Args:
        fn (QJIT): a qjit-decorated function
        stage (str): string corresponding with the name of the stage to be printed

    .. seealso:: :doc:`/dev/debugging`

    **Example**

    .. code-block:: python

        @qjit(keep_intermediate=True)
        def func(x: float):
            return x

    >>> debug.print_compilation_stage(func, "HLOLoweringPass")
    module @func {
      func.func public @jit_func(%arg0: tensor<f64>)
      -> tensor<f64> attributes {llvm.emit_c_interface} {
        return %arg0 : tensor<f64>
      }
      func.func @setup() {
        quantum.init
        return
      }
      func.func @teardown() {
        quantum.finalize
        return
      }
    }
    """
    EvaluationContext.check_is_not_tracing("C interface cannot be generated from tracing context.")

    if not isinstance(fn, catalyst.QJIT):
        raise TypeError(f"First argument needs to be a 'QJIT' object, got a {type(fn)}.")

    print(fn.compiler.get_output_of(stage))


@debug_logger
def get_cmain(fn, *args):
    """Return a C program that calls a jitted function with the provided arguments.

    Args:
        fn (QJIT): a qjit-decorated function
        *args: argument values to use in the C program when invoking ``fn``

    Returns:
        str: A C program that can be compiled and linked with the current shared object.
    """
    EvaluationContext.check_is_not_tracing("C interface cannot be generated from tracing context.")

    if not isinstance(fn, catalyst.QJIT):
        raise TypeError(f"First argument needs to be a 'QJIT' object, got a {type(fn)}.")

    requires_promotion = fn.jit_compile(args)

    if requires_promotion:
        dynamic_args = filter_static_args(args, fn.compile_options.static_argnums)
        args = promote_arguments(fn.c_sig, dynamic_args)

    return fn.compiled_function.get_cmain(*args)


# pylint: disable=line-too-long
@debug_logger
def compile_from_mlir(ir, compiler=None, compile_options=None):
    """Compile a Catalyst function to binary code from the provided MLIR.

    Args:
        ir (str): the MLIR to compile in string form
        compile_options: options to use during compilation

    Returns:
        CompiledFunction: A callable that manages the compiled shared library and its invocation.

    **Example**

    The main entry point of the program is required to start with ``catalyst.entry_point``, and
    the program is required to contain ``setup`` and ``teardown`` functions.

    .. code-block:: python

        ir = r\"""
            module @workflow {
                func.func public @catalyst.entry_point(%arg0: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
                    return %arg0 : tensor<f64>
                }
                func.func @setup() {
                    quantum.init
                    return
                }
                func.func @teardown() {
                    quantum.finalize
                    return
                }
            }
        \"""

        compiled_function = debug.compile_from_mlir(ir)

    >>> compiled_function(0.1)
    [0.1]
    """
    EvaluationContext.check_is_not_tracing("Cannot compile from IR in tracing context.")

    if compiler is None:
        compiler = Compiler(compile_options)

    module_name = "debug_module"
    workspace_dir = os.getcwd() if compiler.options.keep_intermediate else None
    workspace = WorkspaceManager.get_or_create_workspace("debug_workspace", workspace_dir)
    shared_object, _llvm_ir, func_data = compiler.run_from_ir(ir, module_name, workspace)

    # Parse inferred function data, like name and return types.
    qfunc_name = func_data[0]
    with mlir.ir.Context():
        result_types = [mlir.ir.RankedTensorType.parse(rt) for rt in func_data[1].split(",")]

    return CompiledFunction(shared_object, qfunc_name, result_types, None, compiler.options)


@debug_logger
def compile_executable(fn, *args):
    """Generate and compile a C program that calls a jitted function with the provided arguments.

    Args:
        fn (QJIT): a qjit-decorated function
        *args: argument values to use in the C program when invoking ``fn``

    Returns:
        (str): the paths that should be included in LD_LIBRARY_PATH.
        (str): the path of output binary.
    """
    f_name = str(fn.__name__)
    workspace = str(fn.workspace)
    main_c_file = workspace + "/main.c"
    output_file = workspace + "/" + f_name + ".out"
    shared_object_file = workspace + "/" + f_name + ".so"
    options = fn.compiler.options
    with open(main_c_file, "w", encoding="utf-8") as file:
        file.write(get_cmain(fn, *args))

    # configure flags
    default_flags = LinkerDriver.get_default_flags(options)
    no_shared_flags = [fs for fs in default_flags if fs != "-shared"]
    link_so_flags = no_shared_flags + [
        "-Wl,-rpath," + workspace,
        shared_object_file,
    ]
    LinkerDriver.run(main_c_file, outfile=output_file, flags=link_so_flags, options=options)

    # generate ld library paths
    lib_strings = [s[2:] for s in link_so_flags if s.startswith("-L")]
    ld_env = "$LD_LIBRARY_PATH:" + ":".join(lib_strings)
    return ld_env, output_file
