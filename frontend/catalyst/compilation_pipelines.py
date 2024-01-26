# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains classes and decorators for just-in-time and ahead-of-time
compiling of hybrid quantum-classical functions using Catalyst.
"""

# pylint: disable=too-many-lines

import ctypes
import functools
import inspect
import pathlib
import typing
import warnings
from copy import deepcopy
from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
from jax._src.interpreters.partial_eval import infer_lambda_input_type
from jax._src.pjit import _flat_axes_specs
from jax.interpreters.mlir import ir
from jax.tree_util import tree_flatten, tree_unflatten
from mlir_quantum.runtime import (
    as_ctype,
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
    make_zero_d_memref_descriptor,
)

import catalyst
from catalyst.ag_utils import run_autograph
from catalyst.compiler import CompileOptions, Compiler
from catalyst.jax_tracer import trace_to_mlir
from catalyst.pennylane_extensions import QFunc
from catalyst.utils import wrapper  # pylint: disable=no-name-in-module
from catalyst.utils.c_template import get_template, mlir_type_to_numpy_type
from catalyst.utils.contexts import EvaluationContext
from catalyst.utils.filesystem import WorkspaceManager
from catalyst.utils.gen_mlir import inject_functions
from catalyst.utils.jax_extras import get_aval2, get_implicit_and_explicit_flat_args
from catalyst.utils.patching import Patcher

# Required for JAX tracer objects as PennyLane wires.
# pylint: disable=unnecessary-lambda
setattr(jax.interpreters.partial_eval.DynamicJaxprTracer, "__hash__", lambda x: id(x))

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_dynamic_shapes", True)


def are_params_annotated(f: typing.Callable):
    """Return true if all parameters are typed-annotated."""
    signature = inspect.signature(f)
    parameters = signature.parameters
    return all(p.annotation is not inspect.Parameter.empty for p in parameters.values())


def get_type_annotations(func: typing.Callable):
    """Get all type annotations if all parameters are typed-annotated."""
    params_are_annotated = are_params_annotated(func)
    if params_are_annotated:
        return getattr(func, "__annotations__", {}).values()

    return None


class SharedObjectManager:
    """Shared object manager.

    Manages the life time of the shared object. When is it loaded, when to close it.

    Args:
        shared_object_file: path to shared object containing compiled function
        func_name: name of compiled function
    """

    def __init__(self, shared_object_file, func_name):
        self.shared_object = ctypes.CDLL(shared_object_file)
        self.function, self.setup, self.teardown, self.mem_transfer = self.load_symbols(func_name)

    def close(self):
        """Close the shared object"""
        self.function = None
        self.setup = None
        self.teardown = None
        self.mem_transfer = None
        dlclose = ctypes.CDLL(None).dlclose
        dlclose.argtypes = [ctypes.c_void_p]
        # pylint: disable=protected-access
        dlclose(self.shared_object._handle)

    def load_symbols(self, func_name):
        """Load symbols necessary for for execution of the compiled function.

        Args:
            func_name: name of compiled function to be executed

        Returns:
            function: function handle
            setup: handle to the setup function, which initializes the device
            teardown: handle to the teardown function, which tears down the device
            mem_transfer: memory transfer shared object
        """

        setup = self.shared_object.setup
        setup.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        setup.restypes = ctypes.c_int

        teardown = self.shared_object.teardown
        teardown.argtypes = None
        teardown.restypes = None

        # We are calling the c-interface
        function = self.shared_object["_catalyst_pyface_" + func_name]
        # Guaranteed from _mlir_ciface specification
        function.restypes = None
        # Not needed, computed from the arguments.
        # function.argyptes

        mem_transfer = self.shared_object["_mlir_memory_transfer"]

        return function, setup, teardown, mem_transfer

    def __enter__(self):
        params_to_setup = [b"jitted-function"]
        argc = len(params_to_setup)
        array_of_char_ptrs = (ctypes.c_char_p * len(params_to_setup))()
        array_of_char_ptrs[:] = params_to_setup
        self.setup(ctypes.c_int(argc), array_of_char_ptrs)
        return self

    def __exit__(self, _type, _value, _traceback):
        self.teardown()


class TypeCompatibility(Enum):
    """Enum class for state machine.

    The state represent the transition between states.
    """

    UNKNOWN = 0
    CAN_SKIP_PROMOTION = 1
    NEEDS_PROMOTION = 2
    NEEDS_COMPILATION = 3


class CompiledFunction:
    """CompiledFunction, represents a Compiled Function.

    Args:
        shared_object_file: path to shared object containing compiled function
        func_name: name of compiled function
        restype: list of MLIR tensor types representing the result of the compiled function
    """

    def __init__(
        self,
        shared_object_file,
        func_name,
        restype,
        compile_options,
    ):
        self.shared_object = SharedObjectManager(shared_object_file, func_name)
        self.return_type_c_abi = None
        self.func_name = func_name
        self.restype = restype
        self.compile_options = compile_options

    @staticmethod
    def typecheck(abstracted_axes, compiled_signature, runtime_signature):
        """Whether arguments can be promoted.

        Args:
            compiled_signature: user supplied signature, obtain from either an annotation or a
                                previously compiled implementation of the compiled function
            runtime_signature: runtime signature

        Returns:
            bool.
        """
        compiled_data, compiled_shape = tree_flatten(compiled_signature)
        runtime_data, runtime_shape = tree_flatten(runtime_signature)
        with Patcher(
            # pylint: disable=protected-access
            (jax._src.interpreters.partial_eval, "get_aval", get_aval2),
        ):
            axes_specs_compile = _flat_axes_specs(abstracted_axes, *compiled_signature, {})
            axes_specs_runtime = _flat_axes_specs(abstracted_axes, *runtime_signature, {})
            in_type_compiled = infer_lambda_input_type(axes_specs_compile, compiled_data)
            in_type_runtime = infer_lambda_input_type(axes_specs_runtime, runtime_data)

            if in_type_compiled == in_type_runtime:
                return TypeCompatibility.CAN_SKIP_PROMOTION

        if compiled_shape != runtime_shape:
            return TypeCompatibility.NEEDS_COMPILATION

        best_case = TypeCompatibility.CAN_SKIP_PROMOTION
        for c_param, r_param in zip(compiled_data, runtime_data):
            if c_param.dtype != r_param.dtype:
                best_case = TypeCompatibility.NEEDS_PROMOTION

            if c_param.shape != r_param.shape:
                return TypeCompatibility.NEEDS_COMPILATION

            promote_to = jax.numpy.promote_types(r_param.dtype, c_param.dtype)
            if c_param.dtype != promote_to:
                return TypeCompatibility.NEEDS_COMPILATION

        return best_case

    @staticmethod
    def promote_arguments(compiled_signature, *args):
        """Promote arguments from the type specified in args to the type specified by
           compiled_signature.

        Args:
            compiled_signature: user supplied signature, obtain from either an annotation or a
                                previously compiled implementation of the compiled function
            *args: actual arguments to the function

        Returns:
            promoted_args: Arguments after promotion.
        """
        compiled_data, compiled_shape = tree_flatten(compiled_signature)
        runtime_data, runtime_shape = tree_flatten(args)
        assert (
            compiled_shape == runtime_shape
        ), "Compiled function incompatible runtime arguments' shape"

        promoted_args = []
        for c_param, r_param in zip(compiled_data, runtime_data):
            assert isinstance(c_param, jax.core.ShapedArray)
            r_param = jax.numpy.asarray(r_param)
            arg_dtype = r_param.dtype
            promote_to = jax.numpy.promote_types(arg_dtype, c_param.dtype)
            promoted_arg = jax.numpy.asarray(r_param, dtype=promote_to)
            promoted_args.append(promoted_arg)
        return tree_unflatten(compiled_shape, promoted_args)

    @staticmethod
    def get_runtime_signature(*args):
        """Get signature from arguments.

        Args:
            *args: arguments to the compiled function

        Returns:
            a list of JAX shaped arrays
        """
        args_data, args_shape = tree_flatten(args)

        try:
            r_sig = []
            for arg in args_data:
                r_sig.append(jax.api_util.shaped_abstractify(arg))
            # Unflatten JAX abstracted args to preserve the shape
            return tree_unflatten(args_shape, r_sig)
        except Exception as exc:
            arg_type = type(arg)
            raise TypeError(f"Unsupported argument type: {arg_type}") from exc

    @staticmethod
    def _exec(shared_object, has_return, numpy_dict, *args):
        """Execute the compiled function with arguments ``*args``.

        Args:
            lib: Shared object
            has_return: whether the function returns a value or not
            numpy_dict: dictionary of numpy arrays of buffers from the runtime
            *args: arguments to the function

        Returns:
            retval: the value computed by the function or None if the function has no return value
        """

        with shared_object as lib:
            result_desc = type(args[0].contents) if has_return else None
            retval = wrapper.wrap(lib.function, args, result_desc, lib.mem_transfer, numpy_dict)

        return retval

    @staticmethod
    def get_ranked_memref_descriptor_from_mlir_tensor_type(mlir_tensor_type):
        """Convert an MLIR tensor type to a memref descriptor.

        Args:
            mlir_tensor_type: an MLIR tensor type
        Returns:
            a memref descriptor with empty data
        """
        assert mlir_tensor_type
        assert mlir_tensor_type is not tuple
        shape = ir.RankedTensorType(mlir_tensor_type).shape
        mlir_element_type = ir.RankedTensorType(mlir_tensor_type).element_type
        numpy_element_type = mlir_type_to_numpy_type(mlir_element_type)
        ctp = as_ctype(numpy_element_type)
        if shape:
            memref_descriptor = make_nd_memref_descriptor(len(shape), ctp)()
        else:
            memref_descriptor = make_zero_d_memref_descriptor(ctp)()

        return memref_descriptor

    @staticmethod
    def get_etypes(mlir_tensor_type):
        """Get element type for an MLIR tensor type."""
        mlir_element_type = ir.RankedTensorType(mlir_tensor_type).element_type
        return mlir_type_to_numpy_type(mlir_element_type)

    @staticmethod
    def get_sizes(mlir_tensor_type):
        """Get element type size for an MLIR tensor type."""
        mlir_element_type = ir.RankedTensorType(mlir_tensor_type).element_type
        numpy_type = mlir_type_to_numpy_type(mlir_element_type)
        dtype = np.dtype(numpy_type)
        return dtype.itemsize

    @staticmethod
    def get_ranks(mlir_tensor_type):
        """Get rank for an MLIR tensor type."""
        shape = ir.RankedTensorType(mlir_tensor_type).shape
        return len(shape) if shape else 0

    def getCompiledReturnValueType(self, mlir_tensor_types):
        """Compute the type for the return value and memoize it

        This type does not need to be recomputed as it is generated once per compiled function.
        Args:
            mlir_tensor_types: a list of MLIR tensor types which match the expected return type
        Returns:
            a pointer to a CompiledFunctionReturnValue, which corresponds to a structure in which
            fields match the expected return types
        """

        if self.return_type_c_abi is not None:
            return self.return_type_c_abi

        error_msg = """This function must be called with a non-zero length list as an argument."""
        assert mlir_tensor_types, error_msg
        _get_rmd = CompiledFunction.get_ranked_memref_descriptor_from_mlir_tensor_type
        return_fields_types = [_get_rmd(mlir_tensor_type) for mlir_tensor_type in mlir_tensor_types]
        ranks = [
            CompiledFunction.get_ranks(mlir_tensor_type) for mlir_tensor_type in mlir_tensor_types
        ]

        etypes = [
            CompiledFunction.get_etypes(mlir_tensor_type) for mlir_tensor_type in mlir_tensor_types
        ]

        sizes = [
            CompiledFunction.get_sizes(mlir_tensor_type) for mlir_tensor_type in mlir_tensor_types
        ]

        class CompiledFunctionReturnValue(ctypes.Structure):
            """Programmatically create a structure which holds tensors of varying base types."""

            _fields_ = [("f" + str(i), type(t)) for i, t in enumerate(return_fields_types)]
            _ranks_ = ranks
            _etypes_ = etypes
            _sizes_ = sizes

        return_value = CompiledFunctionReturnValue()
        return_value_pointer = ctypes.pointer(return_value)
        self.return_type_c_abi = return_value_pointer
        return self.return_type_c_abi

    def restype_to_memref_descs(self, mlir_tensor_types):
        """Converts the return type to a compatible type for the expected ABI.

        Args:
            mlir_tensor_types: a list of MLIR tensor types which match the expected return type
        Returns:
            a pointer to a CompiledFunctionReturnValue, which corresponds to a structure in which
            fields match the expected return types
        """
        return self.getCompiledReturnValueType(mlir_tensor_types)

    def args_to_memref_descs(self, restype, args):
        """Convert ``args`` to memref descriptors.

        Besides converting the arguments to memrefs, it also prepares the return value. To respect
        the ABI, the return value is changed to a pointer and passed as the first parameter.

        Args:
            restype: the type of restype is a ``CompiledFunctionReturnValue``
            args: the JAX arrays to be used as arguments to the function

        Returns:
            c_abi_args: a list of memref descriptor pointers to return values and parameters
            numpy_arg_buffer: A list to the return values. It must be kept around until the function
                finishes execution as the memref descriptors will point to memory locations inside
                numpy arrays.

        """
        numpy_arg_buffer = []
        return_value_pointer = ctypes.POINTER(ctypes.c_int)()  # This is the null pointer

        if restype:
            return_value_pointer = self.restype_to_memref_descs(restype)

        c_abi_args = []

        args_data, args_shape = tree_flatten(args)

        for arg in args_data:
            numpy_arg = np.asarray(arg)
            numpy_arg_buffer.append(numpy_arg)
            c_abi_ptr = ctypes.pointer(get_ranked_memref_descriptor(numpy_arg))
            c_abi_args.append(c_abi_ptr)

        args = tree_unflatten(args_shape, c_abi_args)

        class CompiledFunctionArgValue(ctypes.Structure):
            """Programmatically create a structure which holds tensors of varying base types."""

            _fields_ = [("f" + str(i), type(t)) for i, t in enumerate(c_abi_args)]

            def __init__(self, c_abi_args):
                for ft_tuple, c_abi_arg in zip(CompiledFunctionArgValue._fields_, c_abi_args):
                    f = ft_tuple[0]
                    setattr(self, f, c_abi_arg)

        arg_value_pointer = ctypes.POINTER(ctypes.c_int)()

        if len(args) > 0:
            arg_value = CompiledFunctionArgValue(c_abi_args)
            arg_value_pointer = ctypes.pointer(arg_value)

        c_abi_args = [return_value_pointer] + [arg_value_pointer]
        return c_abi_args, numpy_arg_buffer

    def get_cmain(self, *args):
        """Get a string representing a C program that can be linked against the shared object."""
        _, buffer = self.args_to_memref_descs(self.restype, args)

        return get_template(self.func_name, self.restype, *buffer)

    def __call__(self, *args, **kwargs):
        if self.compile_options.abstracted_axes is not None:
            abstracted_axes = self.compile_options.abstracted_axes
            args = get_implicit_and_explicit_flat_args(abstracted_axes, *args, **kwargs)

        abi_args, _buffer = self.args_to_memref_descs(self.restype, args)

        numpy_dict = {nparr.ctypes.data: nparr for nparr in _buffer}

        result = CompiledFunction._exec(
            self.shared_object,
            self.restype,
            numpy_dict,
            *abi_args,
        )

        return result


# pylint: disable=too-many-instance-attributes
class QJIT:
    """Class representing a just-in-time compiled hybrid quantum-classical function.

    .. note::

        ``QJIT`` objects are created by the :func:`~.qjit` decorator. Please see
        the :func:`~.qjit` documentation for more details.

    Args:
        fn (Callable): the quantum or classical function
        compile_options (Optional[CompileOptions]): common compilation options
    """

    def __init__(self, fn, compile_options):
        self.compile_options = compile_options
        self.compiler = Compiler(compile_options)
        self.compiling_from_textual_ir = isinstance(fn, str)
        self.original_function = fn
        self.user_function = fn
        self.jaxed_function = None
        self.compiled_function = None
        self.mlir_module = None
        self.user_typed = False
        self.c_sig = None
        self.out_tree = None
        self._jaxpr = None
        self._mlir = None
        self._llvmir = None

        functools.update_wrapper(self, fn)

        if compile_options.autograph:
            self.user_function = run_autograph(fn)

        # QJIT is the owner of workspace.
        # do not move to compiler.
        preferred_workspace_dir = (
            pathlib.Path.cwd() if self.compile_options.keep_intermediate else None
        )
        # If we are compiling from textual ir, just use this as the name of the function.
        name = "compiled_function"
        if not self.compiling_from_textual_ir:
            # pylint: disable=no-member
            # Guaranteed to exist after functools.update_wrapper AND not compiling from textual IR
            name = self.__name__

        self.workspace = WorkspaceManager.get_or_create_workspace(name, preferred_workspace_dir)

        if self.compiling_from_textual_ir:
            EvaluationContext.check_is_not_tracing("Cannot compile from IR in tracing context.")
            return

        parameter_types = get_type_annotations(self.user_function)
        if parameter_types is not None:
            self.user_typed = True
            self.mlir_module = self.get_mlir(*parameter_types)
            if self.compile_options.target == "binary":
                self.compiled_function = self.compile()

    def print_stage(self, stage):
        """Print one of the recorded stages.

        Args:
            stage: string corresponding with the name of the stage to be printed
        """
        self.compiler.print(stage)  # pragma: nocover

    @property
    def mlir(self):
        """str: Returns the MLIR intermediate representation
        of the quantum program.
        """
        return self._mlir

    @property
    def jaxpr(self):
        """str: Returns the JAXPR intermediate representation
        of the quantum program.
        """
        return self._jaxpr

    @property
    def qir(self):
        """str: Returns the LLVM and QIR intermediate representation
        of the quantum program. Only available if the function was compiled to binary.
        """
        return self._llvmir

    def get_mlir(self, *args):
        """Trace :func:`~.user_function`

        Args:
            *args: either the concrete values to be passed as arguments to ``fn`` or abstract values

        Returns:
            an MLIR module
        """
        self.c_sig = CompiledFunction.get_runtime_signature(*args)

        with Patcher(
            (qml.QNode, "__call__", QFunc.__call__),
        ):
            func = self.user_function
            sig = self.c_sig
            abstracted_axes = self.compile_options.abstracted_axes
            mlir_module, ctx, jaxpr, _, self.out_tree = trace_to_mlir(func, abstracted_axes, *sig)

        inject_functions(mlir_module, ctx)
        self._jaxpr = jaxpr
        canonicalizer_options = deepcopy(self.compile_options)
        canonicalizer_options.pipelines = [("0_canonicalize", ["canonicalize"])]
        canonicalizer_options.lower_to_llvm = False
        canonicalizer = Compiler(canonicalizer_options)
        _, self._mlir, _ = canonicalizer.run(mlir_module, self.workspace)
        return mlir_module

    def compile(self):
        """Compile the current MLIR module."""

        if self.compiled_function and self.compiled_function.shared_object:
            self.compiled_function.shared_object.close()

        if self.compiling_from_textual_ir:
            # Module name can be anything.
            module_name = "catalyst_module"
            shared_object, llvm_ir, inferred_func_data = self.compiler.run_from_ir(
                self.user_function, module_name, self.workspace
            )
            qfunc_name = inferred_func_data[0]
            # Parse back the return types given as a semicolon-separated string
            with ir.Context():
                restype = [ir.RankedTensorType.parse(rt) for rt in inferred_func_data[1].split(",")]
        else:
            # This will make a check before sending it to the compiler that the return type
            # is actually available in most systems. f16 needs a special symbol and linking
            # will fail if it is not available.
            #
            # WARNING: assumption is that the first function
            # is the entry point to the compiled program.
            entry_point_func = self.mlir_module.body.operations[0]
            restype = entry_point_func.type.results

            for res in restype:
                baseType = ir.RankedTensorType(res).element_type
                mlir_type_to_numpy_type(baseType)

            # The function name out of MLIR has quotes around it, which we need to remove.
            # The MLIR function name is actually a derived type from string which has no
            # `replace` method, so we need to get a regular Python string out of it.
            qfunc_name = str(self.mlir_module.body.operations[0].name).replace('"', "")

            shared_object, llvm_ir, inferred_func_data = self.compiler.run(
                self.mlir_module, self.workspace
            )

        self._llvmir = llvm_ir
        options = self.compile_options
        compiled_function = CompiledFunction(shared_object, qfunc_name, restype, options)
        return compiled_function

    def _ensure_real_arguments_and_formal_parameters_are_compatible(self, function, *args):
        """Logic to decide whether the function needs to be recompiled
        given ``*args`` and whether ``*args`` need to be promoted.
        A function may need to be compiled if:
            1. It was not compiled before
            2. The real arguments sent to the function are not promotable to the type of the
                formal parameters.

        Args:
          function: an instance of ``CompiledFunction`` that may need recompilation
          *args: arguments that may be promoted.

        Returns:
          function: an instance of ``CompiledFunction`` that may have been recompiled
          *args: arguments that may have been promoted
        """
        r_sig = CompiledFunction.get_runtime_signature(*args)

        has_been_compiled = self.compiled_function is not None
        next_action = TypeCompatibility.UNKNOWN
        if not has_been_compiled:
            next_action = TypeCompatibility.NEEDS_COMPILATION
        else:
            abstracted_axes = self.compile_options.abstracted_axes
            next_action = CompiledFunction.typecheck(abstracted_axes, self.c_sig, r_sig)

        if next_action == TypeCompatibility.NEEDS_PROMOTION:
            args = CompiledFunction.promote_arguments(self.c_sig, *args)
        elif next_action == TypeCompatibility.NEEDS_COMPILATION:
            if self.user_typed:
                msg = "Provided arguments did not match declared signature, recompiling..."
                warnings.warn(msg, UserWarning)
            if not self.compiling_from_textual_ir:
                self.mlir_module = self.get_mlir(*r_sig)
            function = self.compile()
        else:
            assert next_action == TypeCompatibility.CAN_SKIP_PROMOTION

        return function, args

    def get_cmain(self, *args):
        """Return the C interface template for current arguments.

        Args:
          *args: Arguments to be used in the template.
        Returns:
          str: A C program that can be compiled with the current shared object.
        """
        msg = "C interface cannot be generated from tracing context."
        EvaluationContext.check_is_not_tracing(msg)
        function, args = self._ensure_real_arguments_and_formal_parameters_are_compatible(
            self.compiled_function, *args
        )
        return function.get_cmain(*args)

    def __call__(self, *args, **kwargs):
        if EvaluationContext.is_tracing():
            return self.user_function(*args, **kwargs)

        function, args = self._ensure_real_arguments_and_formal_parameters_are_compatible(
            self.compiled_function, *args
        )
        recompilation_needed = function != self.compiled_function
        self.compiled_function = function

        args_data, _args_shape = tree_flatten(args)
        if any(isinstance(arg, jax.core.Tracer) for arg in args_data):
            # Only compile a derivative version of the compiled function when needed.
            if self.jaxed_function is None or recompilation_needed:
                self.jaxed_function = JAX_QJIT(self)

            return self.jaxed_function(*args, **kwargs)

        data = self.compiled_function(*args, **kwargs)

        # Unflatten the return value w.r.t. the original PyTree definition if available
        if self.out_tree is not None:
            data = tree_unflatten(self.out_tree, data)

        # For the classical and pennylane_extensions compilation path,
        if isinstance(data, (list, tuple)) and len(data) == 1:
            data = data[0]

        return data


class JAX_QJIT:
    """Wrapper class around :class:`~.QJIT` that enables compatibility with JAX transformations.

    The primary mechanism through which this is effected is by wrapping the invocation of the QJIT
    object inside a JAX ``pure_callback``. Additionally, a custom JVP is defined in order to support
    JAX-based differentiation, which is itself a ``pure_callback`` around a second QJIT object which
    invokes :func:`~.grad` on the original function. Using this class thus incurs additional
    compilation time.

    Args:
        qjit_function (QJIT): the compiled quantum function object to wrap
    """

    def __init__(self, qjit_function):
        @jax.custom_jvp
        def jaxed_function(*args, **kwargs):
            return self.wrap_callback(qjit_function, *args, **kwargs)

        self.qjit_function = qjit_function
        self.derivative_functions = {}
        self.jaxed_function = jaxed_function
        jaxed_function.defjvp(self.compute_jvp, symbolic_zeros=True)

    @staticmethod
    def wrap_callback(qjit_function, *args, **kwargs):
        """Wrap a QJIT function inside a jax host callback."""
        data = jax.pure_callback(
            qjit_function, qjit_function.jaxpr.out_avals, *args, vectorized=False, **kwargs
        )

        # Unflatten the return value w.r.t. the original PyTree definition if available
        assert qjit_function.out_tree is not None, "PyTree shape must not be none."
        return tree_unflatten(qjit_function.out_tree, data)

    def get_derivative_qjit(self, argnums):
        """Compile a function computing the derivative of the wrapped QJIT for the given argnums."""

        argnum_key = "".join(str(idx) for idx in argnums)
        if argnum_key in self.derivative_functions:
            return self.derivative_functions[argnum_key]

        # Here we define the signature for the new QJIT object explicitly, rather than relying on
        # functools.wrap, in order to guarantee compilation is triggered on instantiation.
        # The signature of the original QJIT object is guaranteed to be defined by now, located
        # in QJIT.c_sig, however we don't update the original function with these annotations.
        annotations = {}
        updated_params = []
        signature = inspect.signature(self.qjit_function)
        for idx, (arg_name, param) in enumerate(signature.parameters.items()):
            annotations[arg_name] = self.qjit_function.c_sig[idx]
            updated_params.append(param.replace(annotation=annotations[arg_name]))

        def deriv_wrapper(*args, **kwargs):
            return catalyst.jacobian(self.qjit_function, argnum=argnums)(*args, **kwargs)

        deriv_wrapper.__name__ = "deriv_" + self.qjit_function.__name__
        deriv_wrapper.__annotations__ = annotations
        deriv_wrapper.__signature__ = signature.replace(parameters=updated_params)

        self.derivative_functions[argnum_key] = QJIT(
            deriv_wrapper, self.qjit_function.compile_options
        )
        return self.derivative_functions[argnum_key]

    def compute_jvp(self, primals, tangents):
        """Compute the set of results and JVPs for a QJIT function."""
        # Assume we have primals of shape `[a,b]` and results of shape `[c,d]`. Derivatives [2]
        # would get the shape `[c,d,a,b]` and tangents [1] would have the same shape as primals.
        # Now, In this function we apply tensordot using the pattern `[c,d,a,b]*[a,b] -> [c,d]`.

        # Optimization: Do not compute Jacobians for arguments which do not participate in
        #               differentiation.
        argnums = []
        for idx, tangent in enumerate(tangents):
            if not isinstance(tangent, jax.custom_derivatives.SymbolicZero):
                argnums.append(idx)

        results = self.wrap_callback(self.qjit_function, *primals)
        results_data, _results_shape = tree_flatten(results)
        derivatives = self.wrap_callback(self.get_derivative_qjit(argnums), *primals)
        derivatives_data, _derivatives_shape = tree_flatten(derivatives)

        jvps = [jnp.zeros_like(results_data[res_idx]) for res_idx in range(len(results_data))]
        for diff_arg_idx, arg_idx in enumerate(argnums):
            tangent = tangents[arg_idx]  # [1]
            taxis = list(range(tangent.ndim))
            for res_idx in range(len(results_data)):
                deriv_idx = diff_arg_idx + res_idx * len(argnums)
                deriv = derivatives_data[deriv_idx]  # [2]
                daxis = list(range(deriv.ndim - tangent.ndim, deriv.ndim))
                jvp = jnp.tensordot(deriv, tangent, axes=(daxis, taxis))
                jvps[res_idx] = jvps[res_idx] + jvp

        # jvps must match the type of primals
        # due to pytrees, primals are a tuple
        primal_type = type(primals)
        jvps = primal_type(jvps)
        if len(jvps) == 1:
            jvps = jvps[0]

        return results, jvps

    def __call__(self, *args, **kwargs):
        return self.jaxed_function(*args, **kwargs)


def qjit(
    fn=None,
    *,
    autograph=False,
    async_qnodes=False,
    target="binary",
    keep_intermediate=False,
    verbose=False,
    logfile=None,
    pipelines=None,
    abstracted_axes=None,
):  # pylint: disable=too-many-arguments
    """A just-in-time decorator for PennyLane and JAX programs using Catalyst.

    This decorator enables both just-in-time and ahead-of-time compilation,
    depending on whether function argument type hints are provided.

    .. note::

        Currently, ``lightning.qubit`` is the only supported backend device
        for Catalyst compilation. For a list of supported operations, observables,
        and measurements, please see the :doc:`/dev/quick_start`.

    Args:
        fn (Callable): the quantum or classical function
        autograph (bool): Experimental support for automatically converting Python control
            flow statements to Catalyst-compatible control flow. Currently supports Python ``if``,
            ``elif``, ``else``, and ``for`` statements. Note that this feature requires an
            available TensorFlow installation. For more details, see the
            :doc:`AutoGraph guide </dev/autograph>`.
        async_qnodes (bool): Experimental support for automatically executing
            QNodes asynchronously, if supported by the device runtime.
        target (str): the compilation target
        keep_intermediate (bool): Whether or not to store the intermediate files throughout the
            compilation. If ``True``, intermediate representations are available via the
            :attr:`~.QJIT.mlir`, :attr:`~.QJIT.jaxpr`, and :attr:`~.QJIT.qir`, representing
            different stages in the optimization process.
        verbosity (bool): If ``True``, the tools and flags used by Catalyst behind the scenes are
            printed out.
        logfile (Optional[TextIOWrapper]): File object to write verbose messages to (default -
            ``sys.stderr``).
        pipelines (Optional(List[Tuple[str,List[str]]])): A list of pipelines to be executed. The
            elements of this list are named sequences of MLIR passes to be executed. A ``None``
            value (the default) results in the execution of the default pipeline. This option is
            considered to be used by advanced users for low-level debugging purposes.
        abstracted_axes (Sequence[Sequence[str]] or Dict[int, str] or Sequence[Dict[int, str]]):
            An experimental option to specify dynamic tensor shapes.
            This option affects the compilation of the annotated function.
            Function arguments with ``abstracted_axes`` specified will be compiled to ranked tensors
            with dynamic shapes. For more details, please see the Dynamically-shaped Arrays section
            below.

    Returns:
        QJIT object.

    Raises:
        FileExistsError: Unable to create temporary directory
        PermissionError: Problems creating temporary directory
        OSError: Problems while creating folder for intermediate files
        AutoGraphError: Raised if there was an issue converting the given the function(s).
        ImportError: Raised if AutoGraph is turned on and TensorFlow could not be found.

    **Example**

    In just-in-time (JIT) mode, the compilation is triggered at the call site the
    first time the quantum function is executed. For example, ``circuit`` is
    compiled as early as the first call.

    .. code-block:: python

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(theta):
            qml.Hadamard(wires=0)
            qml.RX(theta, wires=1)
            qml.CNOT(wires=[0,1])
            return qml.expval(qml.PauliZ(wires=1))

    >>> circuit(0.5)  # the first call, compilation occurs here
    array(0.)
    >>> circuit(0.5)  # the precompiled quantum function is called
    array(0.)

    Alternatively, if argument type hints are provided, compilation
    can occur 'ahead of time' when the function is decorated.

    .. code-block:: python

        from jax.core import ShapedArray

        @qjit  # compilation happens at definition
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x: complex, z: ShapedArray(shape=(3,), dtype=jnp.float64)):
            theta = jnp.abs(x)
            qml.RY(theta, wires=0)
            qml.Rot(z[0], z[1], z[2], wires=0)
            return qml.state()

    >>> circuit(0.2j, jnp.array([0.3, 0.6, 0.9]))  # calls precompiled function
    array([0.75634905-0.52801002j, 0. +0.j,
           0.35962678+0.14074839j, 0. +0.j])

    For more details on compilation and debugging, please see :doc:`/dev/sharp_bits`.

    .. important::

        Most decomposition logic will be equivalent to PennyLane's decomposition.
        However, decomposition logic will differ in the following cases:

        1. All :class:`qml.Controlled <pennylane.ops.op_math.Controlled>` operations will decompose
            to :class:`qml.QubitUnitary <pennylane.QubitUnitary>` operations.

        2. :class:`qml.ControlledQubitUnitary <pennylane.ControlledQubitUnitary>` operations will
            decompose to :class:`qml.QubitUnitary <pennylane.QubitUnitary>` operations.

        3. The list of device-supported gates employed by Catalyst is currently different than that
            of the ``lightning.qubit`` device, as defined by the
            :class:`~.pennylane_extensions.QJITDevice`.

    .. details::
        :title: AutoGraph and Python control flow

        Catalyst also supports capturing imperative Python control flow in compiled programs. You
        can enable this feature via the ``autograph=True`` parameter. Note that it does come with
        some restrictions, in particular whenever global state is involved. Refer to the
        :doc:`AutoGraph guide </dev/autograph>` for a complete discussion of the
        supported and unsupported use-cases.

        .. code-block:: python

            @qjit(autograph=True)
            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circuit(x: int):

                if x < 5:
                    qml.Hadamard(wires=0)
                else:
                    qml.T(wires=0)

                return qml.expval(qml.PauliZ(0))

        >>> circuit(3)
        array(0.)

        >>> circuit(5)
        array(1.)

        Note that imperative control flow will still work in Catalyst even when the AutoGraph
        feature is turned off, it just won't be captured in the compiled program and cannot involve
        traced values. The example above would then raise a tracing error, as there is no value for
        ``x`` yet than can be compared in the if statement. A loop like ``for i in range(5)`` would
        be unrolled during tracing, "copy-pasting" the body 5 times into the program rather than
        appearing as is.

    .. details::
        :title: Dynamically-shaped arrays

        There are three ways to use ``abstracted_axes``; by passing a sequence of tuples, a
        dictionary, or a sequence of dictionaries. Passing a sequence of tuples:

        .. code-block:: python

            abstracted_axes=((), ('n',), ('m', 'n'))

        Each tuple in the sequence corresponds to one of the arguments in the annotated
        function. Empty tuples can
        be used and correspond to parameters with statically known shapes.
        Non-empty tuples correspond to parameters with dynamically known shapes.

        In this example above,

        - the first argument will have a statically known shape,

        - the second argument has its zeroth axis have dynamic
          shape ``n``, and

        - the third argument will have its zeroth axis with dynamic shape
          ``m`` and first axis with dynamic shape ``n``.

        Passing a dictionary:

        .. code-block:: python

            abstracted_axes={0: 'n'}

        This approach allows a concise expression of the relationships
        between axes for different function arguments. In this example,
        it specifies that for all function arguments, the zeroth axis will
        have dynamic shape ``n``.

        Passing a sequence of dictionaries:

        .. code-block:: python

            abstracted_axes=({}, {0: 'n'}, {1: 'm', 0: 'n'})

        The example here is a more verbose version of the tuple example. This convention
        allows axes to be omitted from the list of abstracted axes.

        Using ``abstracted_axes`` can help avoid the cost of recompilation.
        By using ``abstracted_axes``, a more general version of the compiled function will be
        generated. This more general version is parametrized over the abstracted axes and
        allows results to be computed over tensors independently of their axes lengths.

        For example:

        .. code-block:: python

            @qjit
            def sum(arr):
                return jnp.sum(arr)

            sum(jnp.array([1]))     # Compilation happens here.
            sum(jnp.array([1, 1]))  # And here!

        The ``sum`` function would recompile each time an array of different size is passed
        as an argument.

        .. code-block:: python

            @qjit(abstracted_axes={0: "n"})
            def sum_abstracted(arr):
                return jnp.sum(arr)

            sum(jnp.array([1]))     # Compilation happens here.
            sum(jnp.array([1, 1]))  # No need to recompile.

        the ``sum_abstracted`` function would only compile once and its definition would be
        reused for subsequent function calls.
    """

    axes = abstracted_axes
    if fn is not None:
        return QJIT(
            fn,
            CompileOptions(
                verbose,
                logfile,
                target,
                keep_intermediate,
                pipelines,
                autograph,
                async_qnodes,
                abstracted_axes=axes,
            ),
        )

    def wrap_fn(fn):
        return QJIT(
            fn,
            CompileOptions(
                verbose,
                logfile,
                target,
                keep_intermediate,
                pipelines,
                autograph,
                async_qnodes,
                abstracted_axes=axes,
            ),
        )

    return wrap_fn
