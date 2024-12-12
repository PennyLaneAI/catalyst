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

import catalyst
from catalyst.compiler import LinkerDriver
from catalyst.logging import debug_logger
from catalyst.tracing.contexts import EvaluationContext
from catalyst.tracing.type_signatures import filter_static_args, promote_arguments

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@debug_logger
def get_compilation_stage(fn, stage):
    """Returns the intermediate representation of one of the recorded compilation
    stages for a JIT-compiled function.

    The stages are indexed by their Catalyst compilation pipeline name, which are either provided
    by the user as a compilation option, or predefined in ``catalyst.compiler``.

    All the available stages are:

    - MILR: ``mlir``, ``HLOLoweringPass``, ``QuantumCompilationPass``, ``BufferizationPass``,
      and ``MLIRToLLVMDialect``.

    - LLVM: ``llvm_ir``, ``CoroOpt``, ``O2Opt``, ``Enzyme``, and ``last``.

    Note that ``CoroOpt`` (Coroutine lowering), ``O2Opt`` (O2 optimization), and ``Enzyme``
    (automatic differentiation) passes do not always happen. ``last`` denotes the stage
    right before object file generation.

    .. note::

        In order to use this function, ``keep_intermediate=True`` must be
        set in the :func:`~.qjit` decorator of the input function.

    Args:
        fn (QJIT): a qjit-decorated function
        stage (str): string corresponding with the name of the stage to be printed

    Returns:
        str: output ir from the target compiler stage

    .. seealso:: :doc:`/dev/debugging`, :func:`~.replace_ir`.

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

    return fn.compiler.get_output_of(stage, fn.workspace)


@debug_logger
def get_compilation_stages_groups(options):
    """Returns a list of tuples. The tuples correspond to the name
    of the compilation stage and the list of passes within that stage.
    """
    return options.get_stages()


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


@debug_logger
def replace_ir(fn, stage, new_ir):
    r"""Replace the IR at any compilation stage that will be used the next time the function runs.

    It is important that the function signature (inputs and outputs) for the next execution matches
    that of the provided IR, or else the behaviour is undefined.

    Available stages include:

    - MILR: ``mlir``, ``HLOLoweringPass``, ``QuantumCompilationPass``, ``BufferizationPass``,
      and ``MLIRToLLVMDialect``.

    - LLVM: ``llvm_ir``, ``CoroOpt``, ``O2Opt``, ``Enzyme``, and ``last``.

    Note that ``CoroOpt`` (Coroutine lowering), ``O2Opt`` (O2 optimization), and ``Enzyme``
    (automatic differentiation) passes do not always happen. ``last`` denotes the stage
    right before object file generation.

    Args:
        fn (QJIT): a qjit-decorated function
        stage (str): Recompilation picks up after this stage.
        new_ir (str): The replacement IR to use for recompilation.

    .. seealso:: :doc:`/dev/debugging`, :func:`~.get_compilation_stage`.

    **Example**

    >>> from catalyst.debug import get_compilation_stage, replace_ir
    >>> @qjit(keep_intermediate=True)
    >>> def f(x):
    ...     return x**2
    >>> f(2.0)  # just-in-time compile the function
    4.0

    Here we modify ``%2 = arith.mulf %in, %in_0 : f64`` to turn the square function into a cubic one:

    >>> old_ir = get_compilation_stage(f, "HLOLoweringPass")
    >>> new_ir = old_ir.replace(
    ...   "%2 = arith.mulf %in, %in_0 : f64\n",
    ...   "%t = arith.mulf %in, %in_0 : f64\n    %2 = arith.mulf %t, %in_0 : f64\n"
    ... )

    The recompilation starts after the given checkpoint stage:

    >>> replace_ir(f, "HLOLoweringPass", new_ir)
    >>> f(2.0)
    8.0
    """
    fn.overwrite_ir = new_ir
    fn.compiler.options.checkpoint_stage = stage
    fn.fn_cache.clear()


@debug_logger
def compile_executable(fn, *args):
    """Generate an executable binary for the native host architecture from a
    :func:`~.qjit` decorated function with provided arguments.

    Args:
        fn (QJIT): a qjit-decorated function
        *args: argument values to use in the C program when invoking ``fn``

    Returns:
        str: the path of output binary

    **Example**

    For example, considering the following function where we are
    using :func:`~.print_memref` to print (at runtime) information about
    variable ``y``:

    .. code-block:: python

        @qjit
        def f(x):
            y = x * x
            debug.print_memref(y)
            return y

    >>> f(5)
    MemRef: base@ = 0x64fc9dd5ffc0 rank = 0 offset = 0 sizes = [] strides = [] data =
    25
    Array(25, dtype=int64)

    We can now use ``compile_executable`` to compile this function to a binary.

    The executable will be saved in the directory for intermediate results if
    ``keep_intermediate=True``. Otherwise, the executable will appear in the Catalyst project
    root.

    >>> from catalyst.debug import compile_executable
    >>> binary = compile_executable(f, 5)
    >>> print(binary)
    /path/to/executable

    Executing this function from a shell environment:

    .. code-block:: shell

        $ /path/to/executable
        MemRef: base@ = 0x64fc9dd5ffc0 rank = 0 offset = 0 sizes = [] strides = [] data =
        25

    """

    # if fn is not compiled, compile it first.
    if not fn.compiled_function:
        fn(*args)

    f_name = str(fn.__name__)
    workspace = str(fn.workspace) if fn.compile_options.keep_intermediate else os.getcwd()
    main_c_file = workspace + "/main.c"
    output_file = workspace + "/" + f_name + ".out"
    shared_object_file = workspace + "/" + f_name + ".so"

    # copy shared object to current directory
    original_shared_object_file = str(fn.workspace) + "/" + f_name + ".so"
    if not fn.compile_options.keep_intermediate:
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
    ]
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
