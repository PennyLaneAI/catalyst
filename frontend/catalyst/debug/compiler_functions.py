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

from jax.interpreters import mlir

import catalyst
from catalyst.compiled_functions import CompiledFunction
from catalyst.compiler import Compiler
from catalyst.logging import debug_logger
from catalyst.tracing.contexts import EvaluationContext
from catalyst.tracing.type_signatures import filter_static_args, promote_arguments
from catalyst.utils.filesystem import WorkspaceManager

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

    .. seealso:: :doc:`/dev/debugging`, :func:`~.replace_ir`, :func:`~.compile_from_mlir`.

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
    r"""Compile a Catalyst function to binary code from the provided MLIR.

    Args:
        ir (str): the MLIR to compile in string form
        compile_options: options to use during compilation

    Returns:
        CompiledFunction: A callable that manages the compiled shared library and its invocation.

    .. seealso:: :doc:`/dev/debugging`, :func:`~.get_compilation_stage`, :func:`~.replace_ir`.

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

    .. seealso:: :doc:`/dev/debugging`, :func:`~.get_compilation_stage`, :func:`~.compile_from_mlir`.

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
