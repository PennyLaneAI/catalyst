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
import platform
import re
import shutil
import subprocess
import sys
import sysconfig

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
def get_compilation_stage(fn, stage):
    """Print one of the recorded compilation stages for a JIT-compiled function.

    The stages are indexed by their Catalyst compilation pipeline name, which are either provided
    by the user as a compilation option, or predefined in ``catalyst.compiler``.

    All the available stages are:

    - MILR: mlir, HLOLoweringPass, QuantumCompilationPass, BufferizationPass, and MLIRToLLVMDialect

    - LLVM: llvm_ir, CoroOpt, O2Opt, Enzyme, and last.

    Note that `CoroOpt` (Coroutine lowering), `O2Opt` (O2 optimization), and `Enzyme` (Automatic
    differentiation) passes do not always happen. `last` denotes the stage right before object file
    generation.

    Requires ``keep_intermediate=True``.

    Args:
        fn (QJIT): a qjit-decorated function
        stage (str): string corresponding with the name of the stage to be printed

    Returns:
        str: output ir from the target compiler stage

    .. seealso:: :doc:`/dev/debugging`

    **Example**

    .. code-block:: python

        @qjit(keep_intermediate=True)
        def func(x: float):
            return x

    >>> print(debug.get_compilation_stage(func, "HLOLoweringPass"))
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

    if stage == "last":
        return fn.compiler.last_compiler_output.get_output_ir()
    return fn.compiler.get_output_of(stage)


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
def replace_ir(fn, stage, new_ir):
    """Replace the IR at any compilation stage that will be used the next time the function runs.

    It is important that the function signature (inputs & outputs) for the next execution matches
    that of the provided IR, or else the behaviour is undefined.

    All the available stages are:

    - MILR: mlir, HLOLoweringPass, QuantumCompilationPass, BufferizationPass, and MLIRToLLVMDialect.

    - LLVM: llvm_ir, CoroOpt, O2Opt, Enzyme, and last.

    Note that `CoroOpt` (Coroutine lowering), `O2Opt` (O2 optimization), and `Enzyme` (Automatic
    differentiation) passes do not always happen. `last` denotes the stage right before object file
    generation.

    Args:
        fn (QJIT): a qjit-decorated function
        stage (str): Recompilation picks up after this stage.
        new_ir (str): The replacement IR to use for recompilation.
    """
    fn.overwrite_ir = new_ir
    fn.compiler.options.checkpoint_stage = stage
    fn.fn_cache.clear()


@debug_logger
def compile_executable(fn, *args):
    """Generate and compile a C program that calls a jitted function with the provided arguments.


    Args:
        fn (QJIT): a qjit-decorated function
        *args: argument values to use in the C program when invoking ``fn``

    Returns:
        (str): the paths that should be included in LD_LIBRARY_PATH.
        (str): the path of output binary.

    **Example**

    The following example is a square function.
    Here we are using ``debug.print_memref`` to print the information of the result from ``y``.

    .. code-block:: python

        @qjit
        def f(x):
            y = x*x
            debug.print_memref(y)
            return y

    >>> f(5)
    MemRef: base@ = 0x64fc9dd5ffc0 rank = 0 offset = 0 sizes = [] strides = [] data =
    25

    The executable will be saved in the directory for intermediate results if ``keep_intermediate=True``.
    Otherwise, the executable will appear in the Catalyst project root.

    .. code-block:: python

        from catalyst.debug import compile_executable
        binary = compile_executable(f, 1)

    >>> print(binary)
    /path/to/executable

    .. code-block:: shell

        $ /path/to/executable
        MemRef: base@ = 0x64fc9dd5ffc0 rank = 0 offset = 0 sizes = [] strides = [] data =
        25

    """
    # if fn is not compiled, compile it first.
    if not fn.compiled_function:
        fn(*args)

    # get python version
    python_lib_dir_path = sysconfig.get_config_var("LIBDIR")
    version_info = sys.version_info

    # If libpython3.so exists, link to that instead of libpython3.x.so
    if os.path.isfile(python_lib_dir_path + f"/libpython{version_info.major}.so"):
        version_str = f"{version_info.major}"
    else:
        version_str = f"{version_info.major}.{version_info.minor}"

    lib_path_flags = [
        f"-Wl,-rpath,{python_lib_dir_path}",
        f"-L{python_lib_dir_path}",
        "-lpython" + version_str,
    ]

    # Linker in macOS might use @rpath/Python3.framework/Versions/3.x/Python3.
    if platform.system() == "Darwin":  # pragma: nocover
        python_lib_dir_rpath = python_lib_dir_path.split("Python3.framework")[0]
        lib_path_flags.insert(1, f"-Wl,-rpath,{python_lib_dir_rpath}")

    f_name = str(fn.__name__)
    workspace = str(fn.workspace) if fn.compile_options.keep_intermediate else os.getcwd()
    main_c_file = workspace + "/main.c"
    output_file = workspace + "/" + f_name + ".out"
    shared_object_file = workspace + "/" + f_name + ".so"

    # copy shared object to current directory
    if not fn.compile_options.keep_intermediate:
        original_shared_object_file = str(fn.workspace) + "/" + f_name + ".so"
        shutil.copy(original_shared_object_file, shared_object_file)

    options = fn.compiler.options
    with open(main_c_file, "w", encoding="utf-8") as file:
        file.write(get_cmain(fn, *args))

    # Set search path mainly for gfortran and quadmath, which are located in the same
    # directory as openblas from scipy.
    if platform.system() == "Linux":
        object_directory = "$ORIGIN"
    else:  # pragma: nocover
        object_directory = "@loader_path"

    # configure flags
    link_so_flags = [
        "-Wl,-rpath," + workspace,
        shared_object_file,
        f"-Wl,-rpath,{object_directory}",
    ] + lib_path_flags
    LinkerDriver.run(main_c_file, outfile=output_file, flags=link_so_flags, options=options)

    # Patch DLC prefix related to openblas
    if platform.system() == "Darwin":  # pragma: nocover
        otool_path = shutil.which("otool")
        install_name_tool_path = shutil.which("install_name_tool")
        otool_result = subprocess.run(
            [otool_path, "-l", shared_object_file], capture_output=True, text=True, check=True
        )

        dlc_pattern = r"/DLC[^)]+\.dylib"
        dlc_matches = re.findall(dlc_pattern, otool_result.stdout)
        for entry in dlc_matches:
            dylib_pattern = r"/([^/]+\.dylib)$"
            dylib_file_name = re.findall(dylib_pattern, entry)[-1]
            new_entry = f"@rpath/{dylib_file_name}"
            subprocess.run(
                [install_name_tool_path, "-change", entry, new_entry, shared_object_file],
                capture_output=True,
                text=True,
                check=True,
            )

        # Update the path of shared library if copy happens.
        if not fn.compile_options.keep_intermediate:
            original_shared_object_file = str(fn.workspace) + "/" + f_name + ".so"
            subprocess.run(
                [
                    install_name_tool_path,
                    "-change",
                    original_shared_object_file,
                    shared_object_file,
                    output_file,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

    return output_file
