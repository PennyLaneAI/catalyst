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

import ctypes
import os
import inspect
import tempfile
import typing
import warnings
import functools
from abc import ABC, abstractmethod

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

from mlir_quantum.runtime import get_ranked_memref_descriptor, ranked_memref_to_numpy

# Required for JAX tracer objects as PennyLane wires.
setattr(jax.interpreters.partial_eval.DynamicJaxprTracer, "__hash__", lambda x: id(x))

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_array", True)


def mlir_type_to_numpy_type(t):
    if ir.ComplexType.isinstance(t):
        base = ir.ComplexType(t).element_type
        if ir.F64Type.isinstance(base):
            return np.complex128
        elif ir.F32Type.isinstance(base):
            return np.complex64
        raise TypeError("No known type")
    elif ir.F64Type.isinstance(t):
        return np.float64
    elif ir.F32Type.isinstance(t):
        return np.float32
    elif ir.F16Type.isinstance(t):
        return np.float16
    elif ir.IntegerType(t).width == 1:
        return np.bool_
    elif ir.IntegerType(t).width == 8:
        return np.int8
    elif ir.IntegerType(t).width == 16:
        return np.int16
    elif ir.IntegerType(t).width == 32:
        return np.int32
    elif ir.IntegerType(t).width == 64:
        return np.int64
    raise TypeError("No known type")


class CompiledFunction:
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
    def can_promote(compiled_signature, runtime_signature, *args):
        len_compile = len(compiled_signature)
        len_runtime = len(runtime_signature)
        if len_compile != len_runtime:
            return False

        for c_param, r_param, arg in zip(compiled_signature, runtime_signature, args):
            assert isinstance(arg, jax.Array)
            assert isinstance(c_param, jax.core.ShapedArray)
            assert isinstance(r_param, jax.core.ShapedArray)
            arg_dtype = arg.dtype
            promote_to = jax.numpy.promote_types(arg_dtype, c_param.dtype)
            if c_param.dtype != promote_to:
                return False
        return True

    @staticmethod
    def promote_arguments(compiled_signature, runtime_signature, *args):
        len_compile = len(compiled_signature)
        len_runtime = len(runtime_signature)
        assert (
            len_compile == len_runtime
        ), "Compiled function incompatible with quantity of runtime arguments"

        promoted_args = list()
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
        # TODO: Compute earlier
        # function.argyptes = Computed later

        return shared_object, function, setup, teardown

    @staticmethod
    def get_runtime_signature(*args):
        try:
            r_sig = list()
            for arg in args:
                r_sig.append(jax.api_util.shaped_abstractify(arg))
            return r_sig
        except:
            arg_type = type(arg)
            raise TypeError(f"Unsupported argument type: {arg_type}")

    @staticmethod
    def get_numpy_array_from_abi_scalar_tensor(scalar_tensor):
        is_scalar = not hasattr(scalar_tensor.contents, "shape")
        assert is_scalar, "scalar_tensor must be scalar"
        numpy_array = np.array(scalar_tensor.contents.aligned.contents)
        return numpy_array

    @staticmethod
    def execute_abi(shared_object_file, func_name, restype, *py_args):
        shared_object, function, setup, teardown = CompiledFunction.load_symbols(
            shared_object_file, func_name
        )

        params_to_setup = [
            b"jitted-function",
            b"-qrt",
            b"ftqc",
        ]
        argc = len(params_to_setup)
        array_of_char_ptrs = (ctypes.c_char_p * len(params_to_setup))()
        array_of_char_ptrs[:] = params_to_setup

        setup(ctypes.c_int(argc), array_of_char_ptrs)
        function(*py_args)
        teardown()

        # Check if the return value is scalar array...
        retval = None
        if restype:
            # There ought to be a better way...
            is_tuple = isinstance(restype, list) and len(restype) > 1
            scalar = not hasattr(py_args[0].contents, "shape")
            if is_tuple:
                retval = []
                struct = py_args[0].contents
                a_struct_type = type(struct)
                for f, _ in a_struct_type._fields_:
                    memref = getattr(struct, f)
                    memref_ptr = ctypes.pointer(memref)
                    is_scalar = not hasattr(memref, "shape")
                    if not is_scalar:
                        retval.append(np.copy(ranked_memref_to_numpy(memref_ptr)))
                    else:
                        numpy_array = np.copy(
                            CompiledFunction.get_numpy_array_from_abi_scalar_tensor(memref_ptr)
                        )
                        retval.append(numpy_array)
            elif not scalar:
                retval = np.copy(ranked_memref_to_numpy(py_args[0]))
            else:
                retval = np.copy(
                    CompiledFunction.get_numpy_array_from_abi_scalar_tensor(py_args[0])
                )

        # Unmap the shared library. This is necessary in case the function is re-compiled.
        # Without unmapping the shared library, there would be a conflict in the name of
        # the function and the previous one would succeed.
        dlclose = ctypes.CDLL(None).dlclose
        dlclose.argtypes = [ctypes.c_void_p]
        dlclose(shared_object._handle)

        return retval

    def prepare_single_retval_for_tensor_abi(restype):
        assert restype is not tuple
        assert restype
        shape = ir.RankedTensorType(restype).shape
        element_type_mlir = ir.RankedTensorType(restype).element_type
        element_type = mlir_type_to_numpy_type(element_type_mlir)
        # Scalar tensor
        if shape == []:
            shape = element_type(0)
            qretval_numpy = np.array(shape, dtype=element_type)
        else:
            qretval_numpy = np.empty(shape, dtype=element_type)
        qretval_instance = get_ranked_memref_descriptor(qretval_numpy)
        return qretval_instance

    # TODO: Generalize and clean these methods
    # This method is currently used to check the return value
    # of the CountsOp
    def prepare_list_for_tensor_abi(c_abi_args, restype_list):
        fields = list()
        for restype in restype_list:
            mlir_data = CompiledFunction.prepare_single_retval_for_tensor_abi(restype)
            fields.append(mlir_data)

        class RetVal(ctypes.Structure):
            _fields_ = [("f" + str(i), type(t)) for i, t in enumerate(fields)]

        retval = RetVal()
        retval_ptr = ctypes.pointer(retval)
        c_abi_args.append(retval_ptr)

    def prepare_args_for_tensor_abi(compile_time_params, restype, py_args):
        c_abi_args = list()

        if isinstance(restype, list) and len(restype) > 1:
            CompiledFunction.prepare_list_for_tensor_abi(c_abi_args, restype)
        elif restype:
            qretval_instance = CompiledFunction.prepare_single_retval_for_tensor_abi(restype)
            qretval_pointer = ctypes.pointer(qretval_instance)
            c_abi_args.append(qretval_pointer)
        len_params = len(compile_time_params)
        len_args = len(py_args)
        assert len_params == len_args, "Different number of arguments"
        numpy_arg_buffer = list()
        for py_arg in py_args:
            numpy_arg = np.asarray(py_arg)
            numpy_arg_buffer.append(numpy_arg)
            c_abi_arg = get_ranked_memref_descriptor(numpy_arg)
            c_abi_arg_ptr = ctypes.pointer(c_abi_arg)
            c_abi_args.append(c_abi_arg_ptr)
        return c_abi_args, numpy_arg_buffer

    def __call__(self, *args, **kwargs):
        runtime_signature = CompiledFunction.get_runtime_signature(*args)
        abi_args, _buffer = CompiledFunction.prepare_args_for_tensor_abi(
            runtime_signature, self.restype, args
        )

        result = CompiledFunction.execute_abi(
            self.shared_object_file,
            self.func_name,
            self.restype,
            *abi_args,
        )

        return result


class QJIT:
    def __init__(self, fn, target, keep_intermediate):
        self.qfunc = fn
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
            self.workspace = tempfile.TemporaryDirectory()
            self.workspace_name = self.workspace.name
        self.passes = {}
        self._jaxpr = None
        self._mlir = None
        self._llvmir = None
        self.mlir_module = None
        self.compiled_function = None

        parameter_types = QJIT.validate_param_types(self.qfunc)
        self.user_typed = False
        if parameter_types is not None:
            self.user_typed = True
            if target in ("mlir", "binary"):
                self.mlir_module = self.get_mlir(parameter_types)
            if target == "binary":
                self.compiled_function = self.compile(parameter_types)

    def print_stage(self, stage):  # pragma: no cover
        if self.passes.get(stage):
            with open(self.passes[stage], "r") as f:
                print(f.read())

    @property
    def mlir(self):
        return self._mlir

    @property
    def jaxpr(self):
        return self._jaxpr

    @property
    def qir(self):
        return self._llvmir

    @staticmethod
    def are_all_signature_params_annotated(f: typing.Callable):
        signature = inspect.signature(f)
        parameters = signature.parameters
        return all(p.annotation is not inspect.Parameter.empty for p in parameters.values())

    @staticmethod
    def validate_param_types(f: typing.Callable) -> typing.List[typing.Any]:
        can_validate = QJIT.are_all_signature_params_annotated(f)

        if can_validate:
            # Needed instead of inspect.get_annotations for Python < 3.10.
            return getattr(f, "__annotations__", {}).values()

    def get_mlir(self, args_or_argtypes):
        with Patcher(
            (qml.QNode, "__call__", QFunc.__call__),
        ):
            mlir_module, ctx, jaxpr = tracer.get_mlir(self.qfunc, *args_or_argtypes)
        self._jaxpr = jaxpr
        self._mlir = mlir_module.operation.get_asm(
            binary=False, print_generic_op_form=False, assume_verified=True
        )

        # Inject setup and finalize functions.
        append_modules(mlir_module, ctx)

        return mlir_module

    def compile(self, args_or_argtypes):
        params = [jax.api_util.shaped_abstractify(p) for p in args_or_argtypes]
        self.c_sig = params

        shared_object, self._llvmir = compiler.compile(
            self.mlir_module, self.workspace_name, self.passes
        )

        # The function name out of MLIR has quotes around it, which we need to remove.
        # The MLIR function name is actually a derived type from string which has no
        # `replace` method, so we need to get a regular Python string out of it.
        qfunc_name = str(self.mlir_module.body.operations[0].name).replace('"', "")
        restype = self.mlir_module.body.operations[0].type.results
        if not restype:
            restype = None
        elif len(restype) == 1:
            restype = restype[0]
        else:
            restype = restype[:]

        return CompiledFunction(shared_object, qfunc_name, restype)

    def __call__(self, *args, **kwargs):
        if TracingContext.is_tracing():
            return self.qfunc(*args, **kwargs)

        args = [jax.numpy.array(arg) for arg in args]
        r_sig = CompiledFunction.get_runtime_signature(*args)
        is_prev_compile = getattr(self, "compiled_function", None) is not None
        self.c_sig = getattr(self, "c_sig", None) if is_prev_compile else None
        can_promote = not is_prev_compile or CompiledFunction.can_promote(self.c_sig, r_sig, *args)
        needs_compile = not is_prev_compile or not can_promote

        if needs_compile:
            if self.user_typed:
                warnings.warn(
                    "Provided arguments did not match declared signature, recompilation has been triggered",
                    UserWarning,
                )
            self.mlir_module = self.get_mlir(r_sig)
            self.compiled_function = self.compile(r_sig)
        else:
            args = CompiledFunction.promote_arguments(self.c_sig, r_sig, *args)

        return self.compiled_function(*args, **kwargs)


def qjit(fn=None, *, target="binary", keep_intermediate=False):
    """A just-in-time decorator for Pennylane programs.

    Args:
        fn (Callable): the quantum function. Defaults to ``None``
        target (str): the compilation target. Defaults to ``"binary"``
        keep_intermediate (bool): the flag to store the intermediate files throughout the
                                compilation. Defaults to ``False``

    Returns:
        QJIT object.

    Raises:
        FileExistsError: Unable to create temporary directory
        PermissionError: Problems creating temporary directory
        OSError: Problems while creating folder for intermediate files
    """

    if fn is not None:
        return QJIT(fn, target, keep_intermediate)

    def wrap_fn(fn):
        return QJIT(fn, target, keep_intermediate)

    return wrap_fn
