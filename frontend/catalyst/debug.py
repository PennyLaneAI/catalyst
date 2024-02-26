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

"""Catalyst's debug module contains functions useful for user program debugging."""

import builtins
import os

import jax
from jax.interpreters import mlir

import catalyst
from catalyst.compiled_functions import CompiledFunction
from catalyst.compiler import Compiler
from catalyst.jax_primitives import print_p
from catalyst.tracing.contexts import EvaluationContext
from catalyst.tracing.type_signatures import filter_static_args, promote_arguments
from catalyst.utils.filesystem import WorkspaceManager


# pylint: disable=redefined-builtin
def print(x, memref=False):
    """A :func:`qjit` compatible print function for printing values at runtime.

    Enables printing of numeric values at runtime. Can also print objects or strings as constants.

    Args:
        x (jax.Array, Any): A single jax array whose numeric values are printed at runtime, or any
            object whose string representation will be treated as a constant and printed at runtime.
        memref (Optional[bool]): When set to ``True``, additional information about how the array is
            stored in memory is printed, via the so-called "memref" descriptor. This includes the
            base memory address of the data buffer, as well as the rank of the array, the size of
            each dimension, and the strides between elements.

    **Example**

    .. code-block:: python

        @qjit
        def func(x: float):
            debug.print(x, memref=True)
            debug.print("exit")

    >>> func(jnp.array(0.43))
    Unranked Memref base@ = 0x5629ff2b6680 rank = 0 offset = 0 sizes = [] strides = [] data =
    [0.43]
    exit

    Outside a :func:`qjit` compiled function the operation falls back to the Python print statement.

    .. note::

        Python f-strings will not work as expected since they will be treated as Python objects.
        This means that array values embeded in them will have their compile-time representation
        printed, instead of actual data.
    """
    if EvaluationContext.is_tracing():
        if isinstance(x, jax.core.Tracer):
            print_p.bind(x, memref=memref)
        else:
            print_p.bind(string=str(x))
    else:
        # Dispatch to Python print outside a qjit context.
        builtins.print(x)


def print_compilation_stage(fn, stage):
    """Print one of the recorded compilation stages for a JIT-compiled function.

    The stages are indexed by their Catalyst compilation pipeline name, which are either provided
    by the user as a compilation option, or predefined in ``catalyst.compiler``.

    Requires ``keep_intermediate=True``.

    Args:
        fn (QJIT): a qjit-decorated function
        stage (str): string corresponding with the name of the stage to be printed

    **Example**

    .. code-block:: python

        @qjit(keep_intermediate=True)
        def func(x: float):
            return x

        debug.print_compilation_stage(func, "HLOLoweringPass")
    """
    EvaluationContext.check_is_not_tracing("C interface cannot be generated from tracing context.")

    if not isinstance(fn, catalyst.QJIT):
        raise TypeError(f"First argument needs to be a 'QJIT' object, got a {type(fn)}.")

    print(fn.compiler.get_output_of(stage))


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

    return CompiledFunction(shared_object, qfunc_name, result_types, compiler.options)
