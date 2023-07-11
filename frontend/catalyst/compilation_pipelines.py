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

import ctypes
import functools
import inspect
import typing
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
from jax.interpreters.mlir import ir
from mlir_quantum.runtime import (
    as_ctype,
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
    make_zero_d_memref_descriptor,
)

import catalyst
import catalyst.jax_tracer as tracer
from catalyst.compiler import CompileOptions, Compiler
from catalyst.pennylane_extensions import QFunc
from catalyst.utils import wrapper  # pylint: disable=no-name-in-module
from catalyst.utils.c_template import get_template, mlir_type_to_numpy_type
from catalyst.utils.gen_mlir import inject_functions
from catalyst.utils.patching import Patcher
from catalyst.utils.tracing import TracingContext

# Required for JAX tracer objects as PennyLane wires.
# pylint: disable=unnecessary-lambda
setattr(jax.interpreters.partial_eval.DynamicJaxprTracer, "__hash__", lambda x: id(x))

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_array", True)


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
    ):
        self.shared_object_file = shared_object_file
        assert func_name  # make sure that func_name is not false-y
        self.func_name = func_name
        self.restype = restype

    @staticmethod
    def can_promote(compiled_signature, runtime_signature):
        """Whether arguments can be promoted.

        Args:
            compiled_signature: user supplied signature, obtain from either an annotation or a
                                previously compiled implementation of the compiled function
            runtime_signature: runtime signature

        Returns:
            bool.
        """
        len_compile = len(compiled_signature)
        len_runtime = len(runtime_signature)
        if len_compile != len_runtime:
            return False

        for c_param, r_param in zip(compiled_signature, runtime_signature):
            assert isinstance(c_param, jax.core.ShapedArray)
            assert isinstance(r_param, jax.core.ShapedArray)
            promote_to = jax.numpy.promote_types(r_param.dtype, c_param.dtype)
            if c_param.dtype != promote_to or c_param.shape != r_param.shape:
                return False
        return True

    @staticmethod
    def promote_arguments(compiled_signature, runtime_signature, *args):
        """Promote arguments from the type specified in args to the type specified by
           compiled_signature.

        Args:
            compiled_signature: user supplied signature, obtain from either an annotation or a
                                previously compiled implementation of the compiled function
            runtime_signature: runtime signature
            *args: actual arguments to the function

        Returns:
            promoted_args: Arguments after promotion.
        """
        len_compile = len(compiled_signature)
        len_runtime = len(runtime_signature)
        assert (
            len_compile == len_runtime
        ), "Compiled function incompatible with quantity of runtime arguments"

        promoted_args = []
        for c_param, r_param, arg in zip(compiled_signature, runtime_signature, args):
            assert isinstance(arg, jax.Array)
            assert isinstance(c_param, jax.core.ShapedArray)
            assert isinstance(r_param, jax.core.ShapedArray)
            arg_dtype = arg.dtype
            promote_to = jax.numpy.promote_types(arg_dtype, c_param.dtype)
            promoted_arg = jax.numpy.asarray(arg, dtype=promote_to)
            promoted_args.append(promoted_arg)
        return promoted_args

    @staticmethod
    def load_symbols(shared_object_file, func_name):
        """Load symbols necessary for for execution of the compiled function.

        Args:
            shared_object_file: path to shared object file
            func_name: name of compiled function to be executed

        Returns:
            shared_object: in memory shared object
            function: function handle
            setup: handle to the setup function, which initializes the device
            teardown: handle to the teardown function, which tears down the device
            mem_transfer: memory transfer shared object
        """
        shared_object = ctypes.CDLL(shared_object_file)

        setup = shared_object.setup
        setup.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        setup.restypes = ctypes.c_int

        teardown = shared_object.teardown
        teardown.argtypes = None
        teardown.restypes = None

        # We are calling the c-interface
        function = shared_object["_catalyst_pyface_" + func_name]
        # Guaranteed from _mlir_ciface specification
        function.restypes = None
        # Not needed, computed from the arguments.
        # function.argyptes

        mem_transfer = shared_object["_mlir_memory_transfer"]

        return shared_object, function, setup, teardown, mem_transfer

    @staticmethod
    def get_runtime_signature(*args):
        """Get signature from arguments.

        Args:
            *args: arguments to the compiled function

        Returns:
            a list of JAX shaped arrays
        """
        try:
            r_sig = []
            for arg in args:
                r_sig.append(jax.api_util.shaped_abstractify(arg))
            return r_sig
        except Exception as exc:
            arg_type = type(arg)
            raise TypeError(f"Unsupported argument type: {arg_type}") from exc

    @staticmethod
    def _exec(shared_object_file, func_name, has_return, numpy_dict, *args):
        """Execute the compiled function with arguments ``*args``.

        Args:
            shared_object_file: path to the shared object file containing the JIT compiled function
            func_name: name of compiled function to be executed
            has_return: whether the function returns a value or not
            numpy_dict: dictionary of numpy arrays of buffers from the runtime
            *args: arguments to the function

        Returns:
            retval: the value computed by the function or None if the function has no return value
        """

        shared_object, function, setup, teardown, mem_transfer = CompiledFunction.load_symbols(
            shared_object_file, func_name
        )

        params_to_setup = [b"jitted-function"]
        argc = len(params_to_setup)
        array_of_char_ptrs = (ctypes.c_char_p * len(params_to_setup))()
        array_of_char_ptrs[:] = params_to_setup

        setup(ctypes.c_int(argc), array_of_char_ptrs)
        result_desc = type(args[0].contents) if has_return else None

        retval = wrapper.wrap(function, args, result_desc, mem_transfer, numpy_dict)
        if len(retval) == 0:
            retval = None
        elif len(retval) == 1:
            retval = retval[0]

        # Teardown has to be made after the return valued has been copied.
        teardown()

        # Unmap the shared library. This is necessary in case the function is re-compiled.
        # Without unmapping the shared library, there would be a conflict in the name of
        # the function and the previous one would succeed.
        # Need to close after obtaining value, since the value can point to memory in the shared
        # object. This is in the case of returning a constant, for example.
        dlclose = ctypes.CDLL(None).dlclose
        dlclose.argtypes = [ctypes.c_void_p]
        # pylint: disable=protected-access
        dlclose(shared_object._handle)

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

    @staticmethod
    def restype_to_memref_descs(mlir_tensor_types):
        """Converts the return type to a compatible type for the expected ABI.

        Args:
            mlir_tensor_types: a list of MLIR tensor types which match the expected return type
        Returns:
            a pointer to a CompiledFunctionReturnValue, which corresponds to a structure in which
            fields match the expected return types
        """
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
        return return_value_pointer

    @staticmethod
    def args_to_memref_descs(restype, args):
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
            return_value_pointer = CompiledFunction.restype_to_memref_descs(restype)

        c_abi_args = []

        for arg in args:
            numpy_arg = np.asarray(arg)
            numpy_arg_buffer.append(numpy_arg)
            c_abi_ptr = ctypes.pointer(get_ranked_memref_descriptor(numpy_arg))
            c_abi_args.append(c_abi_ptr)

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
        _, buffer = CompiledFunction.args_to_memref_descs(self.restype, args)

        return get_template(self.func_name, self.restype, *buffer)

    def __call__(self, *args, **kwargs):
        abi_args, _buffer = CompiledFunction.args_to_memref_descs(self.restype, args)

        numpy_dict = {nparr.ctypes.data: nparr for nparr in _buffer}

        result = CompiledFunction._exec(
            self.shared_object_file,
            self.func_name,
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
        self.qfunc = fn
        self.jaxed_qfunc = None
        self.c_sig = None
        functools.update_wrapper(self, fn)
        self.compile_options = compile_options
        self._compiler = Compiler()
        self._jaxpr = None
        self._mlir = None
        self._llvmir = None
        self.mlir_module = None
        self.compiled_function = None
        parameter_types = get_type_annotations(self.qfunc)
        self.user_typed = False
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
        self._compiler.print(stage)  # pragma: nocover

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
        """Trace :func:`~.qfunc`

        Args:
            *args: either the concrete values to be passed as arguments to ``fn`` or abstract values

        Returns:
            an MLIR module
        """
        self.c_sig = CompiledFunction.get_runtime_signature(*args)

        with Patcher(
            (qml.QNode, "__call__", QFunc.__call__),
        ):
            mlir_module, ctx, jaxpr = tracer.get_mlir(self.qfunc, *self.c_sig)

        inject_functions(mlir_module, ctx)
        mod = mlir_module.operation
        self._jaxpr = jaxpr
        self._mlir = mod.get_asm(binary=False, print_generic_op_form=False, assume_verified=True)

        return mlir_module

    def compile(self):
        """Compile the current MLIR module."""

        # This will make a check before sending it to the compiler that the return type
        # is actually available in most systems. f16 needs a special symbol and linking
        # will fail if it is not available.
        restype = self.mlir_module.body.operations[0].type.results
        for res in restype:
            baseType = ir.RankedTensorType(res).element_type
            mlir_type_to_numpy_type(baseType)

        shared_object = self._compiler.run(
            self.mlir_module,
            options=self.compile_options,
        )

        self._llvmir = self._compiler.get_output_of("LLVMDialectToLLVMIR")

        # The function name out of MLIR has quotes around it, which we need to remove.
        # The MLIR function name is actually a derived type from string which has no
        # `replace` method, so we need to get a regular Python string out of it.
        qfunc_name = str(self.mlir_module.body.operations[0].name).replace('"', "")
        return CompiledFunction(shared_object, qfunc_name, restype)

    def _maybe_promote(self, function, *args):
        """Logic to decide whether the function needs to be recompiled
        given ``*args`` and whether ``*args`` need to be promoted.

        Args:
          function: an instance of ``CompiledFunction`` that may need recompilation
          *args: arguments that may be promoted.

        Returns:
          function: an instance of ``CompiledFunction`` that may have been recompiled
          *args: arguments that may have been promoted
        """
        args = [jax.numpy.array(arg) for arg in args]
        r_sig = CompiledFunction.get_runtime_signature(*args)
        is_prev_compile = self.compiled_function is not None
        can_promote = not is_prev_compile or CompiledFunction.can_promote(self.c_sig, r_sig)
        needs_compile = not is_prev_compile or not can_promote

        if needs_compile:
            if self.user_typed:
                msg = "Provided arguments did not match declared signature, recompiling..."
                warnings.warn(msg, UserWarning)
            self.mlir_module = self.get_mlir(*r_sig)
            function = self.compile()
        else:
            args = CompiledFunction.promote_arguments(self.c_sig, r_sig, *args)
        args = [jax.numpy.array(arg) for arg in args]

        return function, args

    def get_cmain(self, *args):
        """Return the C interface template for current arguments.

        Args:
          *args: Arguments to be used in the template.
        Returns:
          str: A C program that can be compiled with the current shared object.
        """
        msg = "C interface cannot be generated from tracing context."
        TracingContext.check_is_not_tracing(msg)
        function, args = self._maybe_promote(self.compiled_function, *args)
        return function.get_cmain(*args)

    def __call__(self, *args, **kwargs):
        if TracingContext.is_tracing():
            return self.qfunc(*args, **kwargs)

        function, args = self._maybe_promote(self.compiled_function, *args)
        recompilation_needed = function != self.compiled_function
        self.compiled_function = function

        if any(isinstance(arg, jax.core.Tracer) for arg in args):
            # Only compile a derivative version of the compiled function when needed.
            if self.jaxed_qfunc is None or recompilation_needed:
                self.jaxed_qfunc = JAX_QJIT(self)

            return self.jaxed_qfunc(*args, **kwargs)

        return self.compiled_function(*args, **kwargs)


class JAX_QJIT:
    """Wrapper class around :class:`~.QJIT` that enables compatibility with JAX transformations.

    The primary mechanism through which this is effected is by wrapping the invocation of the QJIT
    object inside a JAX ``pure_callback``. Additionally, a custom JVP is defined in order to support
    JAX-based differentiation, which is itself a ``pure_callback`` around a second QJIT object which
    invokes :func:`~.grad` on the original function. Using this class thus incurs additional
    compilation time.

    Args:
        qfunc (QJIT): the compiled quantum function object to wrap
    """

    def __init__(self, qfunc):
        @jax.custom_jvp
        def jaxed_qfunc(*args, **kwargs):
            results = self.wrap_callback(qfunc, *args, **kwargs)
            if len(results) == 1:
                results = results[0]
            return results

        self.qfunc = qfunc
        self.deriv_qfuncs = {}
        self.jaxed_qfunc = jaxed_qfunc
        jaxed_qfunc.defjvp(self.compute_jvp, symbolic_zeros=True)

    @staticmethod
    def wrap_callback(qfunc, *args, **kwargs):
        """Wrap a QJIT function inside a jax host callback."""
        return jax.pure_callback(qfunc, qfunc.jaxpr.out_avals, *args, vectorized=False, **kwargs)

    def get_derivative_qfunc(self, argnums):
        """Compile a function computing the derivative of the wrapped QJIT for the given argnums."""

        argnum_key = "".join(str(idx) for idx in argnums)
        if argnum_key in self.deriv_qfuncs:
            return self.deriv_qfuncs[argnum_key]

        # Here we define the signature for the new QJIT object explicitly, rather than relying on
        # functools.wrap, in order to guarantee compilation is triggered on instantiation.
        # The signature of the original QJIT object is guaranteed to be defined by now, located
        # in QJIT.c_sig, however we don't update the original function with these annotations.
        annotations = {}
        updated_params = []
        signature = inspect.signature(self.qfunc)
        for idx, (arg_name, param) in enumerate(signature.parameters.items()):
            annotations[arg_name] = self.qfunc.c_sig[idx]
            updated_params.append(param.replace(annotation=annotations[arg_name]))

        def deriv_wrapper(*args, **kwargs):
            return catalyst.grad(self.qfunc, argnum=argnums)(*args, **kwargs)

        deriv_wrapper.__name__ = "deriv_" + self.qfunc.__name__
        deriv_wrapper.__annotations__ = annotations
        deriv_wrapper.__signature__ = signature.replace(parameters=updated_params)

        self.deriv_qfuncs[argnum_key] = QJIT(deriv_wrapper, self.qfunc.compile_options)
        return self.deriv_qfuncs[argnum_key]

    def compute_jvp(self, primals, tangents):
        """Compute the set of results and JVPs for a QJIT function."""

        # Optimization: Do not compute Jacobians for arguments which do not participate in
        #               differentiation.
        argnums = []
        for idx, tangent in enumerate(tangents):
            if not isinstance(tangent, jax.custom_derivatives.SymbolicZero):
                argnums.append(idx)

        results = self.wrap_callback(self.qfunc, *primals)
        derivatives = self.wrap_callback(self.get_derivative_qfunc(argnums), *primals)

        jvps = [jnp.zeros_like(results[res_idx]) for res_idx in range(len(results))]
        for diff_arg_idx, arg_idx in enumerate(argnums):
            tangent = tangents[arg_idx]
            for res_idx in range(len(results)):
                deriv_idx = diff_arg_idx * len(results) + res_idx
                num_axes = 0 if tangent.ndim == 0 else 1
                jvp = jnp.tensordot(jnp.transpose(derivatives[deriv_idx]), tangent, axes=num_axes)
                jvps[res_idx] = jvps[res_idx] + jvp

        if len(results) == 1:
            results = results[0]
        if len(jvps) == 1:
            jvps = jvps[0]

        return results, jvps

    def __call__(self, *args, **kwargs):
        return self.jaxed_qfunc(*args, **kwargs)


def qjit(
    fn=None,
    *,
    target="binary",
    keep_intermediate=False,
    verbose=False,
    logfile=None,
    pipelines=None,
):
    """A just-in-time decorator for PennyLane and JAX programs using Catalyst.

    This decorator enables both just-in-time and ahead-of-time compilation,
    depending on whether function argument type hints are provided.

    .. note::

        Currently, ``lightning.qubit`` is the only supported backend device
        for Catalyst compilation. For a list of supported operations, observables,
        and measurements, please see the :doc:`/dev/quick_start`.

    Args:
        fn (Callable): the quantum or classical function
        target (str): the compilation target
        keep_intermediate (bool): Whether or not to store the intermediate files throughout the
            compilation. If ``True``, intermediate representations are available via the
            :attr:`~.QJIT.mlir`, :attr:`~.QJIT.jaxpr`, and :attr:`~.QJIT.qir`, representing
            different stages in the optimization process.
        verbosity (bool): If ``True``, the tools and flags used by Catalyst behind the scenes are
            printed out.
        logfile (Optional[TextIOWrapper]): File object to write verbose messages to (default -
            ``sys.stderr``).
        pipelines (Optional(List[AnyType]): A list of pipelines to be executed. The elements of
            the list are asked to implement a run method which takes the output of the previous run
            as an input to the next element, and so on.

    Returns:
        QJIT object.

    Raises:
        FileExistsError: Unable to create temporary directory
        PermissionError: Problems creating temporary directory
        OSError: Problems while creating folder for intermediate files

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
    """

    if fn is not None:
        return QJIT(fn, CompileOptions(verbose, logfile, target, keep_intermediate, pipelines))

    def wrap_fn(fn):
        return QJIT(fn, CompileOptions(verbose, logfile, target, keep_intermediate, pipelines))

    return wrap_fn
