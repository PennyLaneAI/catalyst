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

# pylint: disable=missing-module-docstring

import ctypes
import os
import inspect
import tempfile
import typing
import warnings
import functools

import numpy as np
import jax
from jax.interpreters.mlir import ir

import pennylane as qml

import catalyst.jax_tracer as tracer
from catalyst import compiler
from catalyst.utils.gen_mlir import append_modules
from catalyst.utils.patching import Patcher
from catalyst.pennylane_extensions import QFunc
from catalyst._configuration import INSTALLED
from catalyst.utils.tracing import TracingContext

default_bindings_path = os.path.join(
    os.path.dirname(__file__), "../../mlir/build/python_packages/quantum"
)
if not INSTALLED and os.path.exists(default_bindings_path):  # pragma: no cover
    import sys

    sys.path.insert(0, default_bindings_path)

# pylint: disable=wrong-import-position,wrong-import-order
from mlir_quantum.runtime import get_ranked_memref_descriptor, ranked_memref_to_numpy

# Required for JAX tracer objects as PennyLane wires.
# pylint: disable=unnecessary-lambda
setattr(jax.interpreters.partial_eval.DynamicJaxprTracer, "__hash__", lambda x: id(x))

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_array", True)


# pylint: disable=too-many-return-statements
def mlir_type_to_numpy_type(t):
    """Convert an MLIR type to a Numpy type.

    Args:
        t: an MLIR numeric type
    Returns:
        A numpy type
    Raises:
        TypeError
    """
    if ir.ComplexType.isinstance(t):
        base = ir.ComplexType(t).element_type
        if ir.F64Type.isinstance(base):
            return np.complex128
        if ir.F32Type.isinstance(base):
            return np.complex64
        raise TypeError("No known type")
    if ir.F64Type.isinstance(t):
        return np.float64
    if ir.F32Type.isinstance(t):
        return np.float32
    if ir.F16Type.isinstance(t):
        return np.float16
    if ir.IntegerType(t).width == 1:
        return np.bool_
    if ir.IntegerType(t).width == 8:
        return np.int8
    if ir.IntegerType(t).width == 16:
        return np.int16
    if ir.IntegerType(t).width == 32:
        return np.int32
    if ir.IntegerType(t).width == 64:
        return np.int64
    raise TypeError("No known type")


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
            compiled_signature: user supplied signature, obtain from either an annotation or a previously compiled
            implementation of the compiled function
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
            if c_param.dtype != promote_to:
                return False
        return True

    @staticmethod
    def promote_arguments(compiled_signature, runtime_signature, *args):
        """Promote arguments

        Promote arguments from the type specified in args to the type specified by compiled_signature.

        Args:
            compiled_signature: user supplied signature, obtain from either an annotation or a previously compiled
            implementation of the compiled function
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
        """
        shared_object = ctypes.CDLL(shared_object_file)

        setup = shared_object.setup
        setup.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        setup.restypes = ctypes.c_int

        teardown = shared_object.teardown
        teardown.argtypes = None
        teardown.restypes = None

        # We are calling the c-interface
        function = shared_object["_mlir_ciface_" + func_name]
        # Guaranteed from _mlir_ciface specification
        function.restypes = None
        # Not needed, computed from the arguments.
        # function.argyptes

        return shared_object, function, setup, teardown

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
    def are_all_signature_params_annotated(f: typing.Callable):
        """Determine if all parameters are typed.

        Args:
            f: callable, with possible annotation
        Returns:
            bool: whether all parameters are annotated
        """
        signature = inspect.signature(f)
        parameters = signature.parameters
        return all(p.annotation is not inspect.Parameter.empty for p in parameters.values())

    @staticmethod
    def get_compile_time_signature(f: typing.Callable) -> typing.List[typing.Any]:
        """Get signature from parameter annotations.

        Args:
            f: callable, with possible annotations
        Returns:
            annotations for all parameters if possible

        """
        can_validate = CompiledFunction.are_all_signature_params_annotated(f)

        if can_validate:
            # Needed instead of inspect.get_annotations for Python < 3.10.
            return getattr(f, "__annotations__", {}).values()

        return None

    @staticmethod
    def zero_ranked_memref_to_numpy(ranked_memref):
        """Cast a zero ranked memrefs to a numpy array.

        Args:
            ranked_memref: a zero ranked memref descriptor
        Returns:
            a numpy array with the contents of the ranked memref descriptor
        """
        assert not hasattr(ranked_memref, "shape")
        return np.array(ranked_memref.aligned.contents)

    @staticmethod
    def ranked_memref_to_numpy(memref_desc):
        """Cast a ranked memref to numpy array.

        Args:
            memref_desc: a descriptor of a ranked memref

        Returns:
            a numpy array
        """
        ranked_memref = memref_desc.contents
        if not hasattr(ranked_memref, "shape"):
            return CompiledFunction.zero_ranked_memref_to_numpy(ranked_memref)
        return np.copy(ranked_memref_to_numpy(memref_desc))

    @staticmethod
    def ranked_memrefs_to_numpy(ranked_memrefs):
        """Cast the ranked memrefs to numpy

        Args:
            ranked_memrefs: a list of memrefs

        Returns:
            a list of numpy arrays
        """
        # pylint: disable=redefined-outer-name
        ranked_memref_to_numpy = CompiledFunction.ranked_memref_to_numpy
        return [ranked_memref_to_numpy(ranked_memref) for ranked_memref in ranked_memrefs]

    @staticmethod
    def return_value_to_ranked_memrefs(return_value):
        """Cast the return value to a list of ranked memrefs.

        Args:
            return_value: to a return value descriptor

        Returns:
            list of ranked memrefs
        """
        return_value_type = type(return_value)
        # pylint: disable=protected-access
        memref_descs = [getattr(return_value, field) for field, _ in return_value_type._fields_]
        memrefs = [ctypes.pointer(memref_desc) for memref_desc in memref_descs]
        return memrefs

    @staticmethod
    def return_value_ptr_to_ranked_memrefs(return_value_ptr):
        """Cast the return value pointer to a list of ranked memrefs.

        Args:
            return_value_ptr: pointer to a return value descriptor

        Returns:
            list of ranked memrefs
        """
        return CompiledFunction.return_value_to_ranked_memrefs(return_value_ptr.contents)

    @staticmethod
    def return_value_ptr_to_numpy(return_value_ptr):
        """Cast the return value pointer to a list of numpy arrays.

        Args:
            return_value_ptr: pointer to a return value descriptor

        Returns:
            list or single numpy array
        """
        ranked_memrefs = CompiledFunction.return_value_ptr_to_ranked_memrefs(return_value_ptr)
        return_value = CompiledFunction.ranked_memrefs_to_numpy(ranked_memrefs)
        # TODO: Handle return types correctly. Tuple, lists of 1 item (?)
        return return_value[0] if len(return_value) == 1 else return_value

    @staticmethod
    def _exec(shared_object_file, func_name, has_return, *args):
        """Execute the compiled function with arguments `*args`.

        Args:
            shared_object_file: path to the shared object file containing the JIT compiled function
            func_name: name of the JIT compiled function
            has_return: whether the function returns a value or not
            *args: arguments to the function

        Returns:
            retval: the value computed by the function or None if the function has no return value
        """
        shared_object, function, setup, teardown = CompiledFunction.load_symbols(
            shared_object_file, func_name
        )

        params_to_setup = [b"jitted-function"]
        argc = len(params_to_setup)
        array_of_char_ptrs = (ctypes.c_char_p * len(params_to_setup))()
        array_of_char_ptrs[:] = params_to_setup

        setup(ctypes.c_int(argc), array_of_char_ptrs)
        function(*args)
        teardown()

        result = args[0] if has_return else None
        retval = CompiledFunction.return_value_ptr_to_numpy(result) if result else None

        # Unmap the shared library. This is necessary in case the function is re-compiled.
        # Without unmapping the shared library, there would be a conflict in the name of
        # the function and the previous one would succeed.
        # Need to close after obtaining value, since the value can point to memory in the shared object.
        # This is in the case of returning a constant, for example.
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
        array_numpy_type = np.empty(shape, dtype=numpy_element_type)
        memref_descriptor = get_ranked_memref_descriptor(array_numpy_type)
        return memref_descriptor

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

        # pylint: disable=too-few-public-methods
        class CompiledFunctionReturnValue(ctypes.Structure):
            """Programmatically create a structure which holds N tensors of possibly different T base types."""

            _fields_ = [("f" + str(i), type(t)) for i, t in enumerate(return_fields_types)]

        return_value = CompiledFunctionReturnValue()
        return_value_pointer = ctypes.pointer(return_value)
        return return_value_pointer

    @staticmethod
    def args_to_memref_descs(restype, args):
        """Convert args to memref descriptors.

        Besides converting the arguments to memrefs, it also prepares the return value. To respect
        the ABI, the return value is changed to a pointer and passed as the first parameter.

        Args:
            restype: the type of restype is a CompiledFunctionReturnValue
            args: the JAX arrays to be used as arguments to the function

        Returns:
            c_abi_args: a list of memref descriptor pointers to return values and parameters
            numpy_arg_buffer: A list to the return values. It must be kept around until the function
                finishes execution as the memref descriptors will point to memory locations inside
                numpy arrays.

        """
        c_abi_args = []
        numpy_arg_buffer = []

        if restype:
            return_value_pointer = CompiledFunction.restype_to_memref_descs(restype)
            c_abi_args.append(return_value_pointer)

        for arg in args:
            numpy_arg = np.asarray(arg)
            numpy_arg_buffer.append(numpy_arg)
            c_abi_arg = get_ranked_memref_descriptor(numpy_arg)
            c_abi_arg_ptr = ctypes.pointer(c_abi_arg)
            c_abi_args.append(c_abi_arg_ptr)
        return c_abi_args, numpy_arg_buffer

    def __call__(self, *args, **kwargs):
        abi_args, _buffer = CompiledFunction.args_to_memref_descs(self.restype, args)

        result = CompiledFunction._exec(
            self.shared_object_file,
            self.func_name,
            self.restype,
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
        target (str): the compilation target
        keep_intermediate (bool): Whether or not to store the intermediate files throughout the
            compilation. If ``True``, the current working directory keeps
            readable representations of the compiled module which remain available
            after the Python process ends. If ``False``, these representations
            will instead be stored in a temporary folder, which will be deleted
            as soon as the QJIT instance is deleted.
    """

    def __init__(self, fn, target, keep_intermediate):
        self.qfunc = fn
        self.c_sig = None
        functools.update_wrapper(self, fn)
        if keep_intermediate:
            dirname = fn.__name__
            parent_dir = os.getcwd()
            path = os.path.join(parent_dir, dirname)
            os.makedirs(path, exist_ok=True)
            self.workspace_name = path
        else:
            # The temporary directory must be referenced by the wrapper class
            # in order to avoid being garbage collected
            # pylint: disable=consider-using-with
            self.workspace = tempfile.TemporaryDirectory()
            self.workspace_name = self.workspace.name
        self.passes = {}
        self._jaxpr = None
        self._mlir = None
        self._llvmir = None
        self.mlir_module = None
        self.compiled_function = None

        parameter_types = CompiledFunction.get_compile_time_signature(self.qfunc)
        self.user_typed = False
        if parameter_types is not None:
            self.user_typed = True
            if target in ("mlir", "binary"):
                self.mlir_module = self.get_mlir(*parameter_types)
            if target == "binary":
                self.compiled_function = self.compile()

    def print_stage(self, stage):
        """
        Print one of the recorded stages.

        Args:
            stage: string corresponding with the name of the stage to be printed
        """
        if self.passes.get(stage):
            with open(self.passes[stage], "r", encoding="utf-8") as f:
                print(f.read())

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
        """Trace self.qfunc

        Args:
            *args: either the concrete values to be passed as arguments to `fn` or abstract values

        Returns:
            an MLIR module
        """
        assert args is not None
        self.c_sig = CompiledFunction.get_runtime_signature(*args)

        with Patcher(
            (qml.QNode, "__call__", QFunc.__call__),
        ):
            mlir_module, ctx, jaxpr = tracer.get_mlir(self.qfunc, *self.c_sig)

        mod = mlir_module.operation
        self._jaxpr = jaxpr
        self._mlir = mod.get_asm(binary=False, print_generic_op_form=False, assume_verified=True)

        # Inject setup and finalize functions.
        append_modules(mlir_module, ctx)

        return mlir_module

    def compile(self):
        """Compile the current MLIR module."""

        shared_object, self._llvmir = compiler.compile(
            self.mlir_module, self.workspace_name, self.passes
        )

        # The function name out of MLIR has quotes around it, which we need to remove.
        # The MLIR function name is actually a derived type from string which has no
        # `replace` method, so we need to get a regular Python string out of it.
        qfunc_name = str(self.mlir_module.body.operations[0].name).replace('"', "")
        restype = self.mlir_module.body.operations[0].type.results
        return CompiledFunction(shared_object, qfunc_name, restype)

    def __call__(self, *args, **kwargs):
        if TracingContext.is_tracing():
            return self.qfunc(*args, **kwargs)

        args = [jax.numpy.array(arg) for arg in args]
        r_sig = CompiledFunction.get_runtime_signature(*args)
        is_prev_compile = self.compiled_function is not None
        can_promote = not is_prev_compile or CompiledFunction.can_promote(self.c_sig, r_sig)
        needs_compile = not is_prev_compile or not can_promote

        if needs_compile:
            if self.user_typed:
                msg = "Provided arguments did not match declared signature, recompilation has been triggered"
                warnings.warn(msg, UserWarning)
            self.mlir_module = self.get_mlir(*r_sig)
            self.compiled_function = self.compile()
        else:
            args = CompiledFunction.promote_arguments(self.c_sig, r_sig, *args)

        return self.compiled_function(*args, **kwargs)


def qjit(fn=None, *, target="binary", keep_intermediate=False):
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
        return QJIT(fn, target, keep_intermediate)

    def wrap_fn(fn):
        return QJIT(fn, target, keep_intermediate)

    return wrap_fn
