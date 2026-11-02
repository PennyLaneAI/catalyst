# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Execute pure classical slices needed to determine exact logical PBC depth."""

from functools import lru_cache
from math import prod
from sys import getsizeof
from threading import local

import numpy as np
from jax._src import xla_bridge
from jax.interpreters import mlir
from jaxlib import lapack, xla_client
from jaxlib.mlir import ir

from catalyst.utils.runtime_environment import get_lib_path, get_libpython_path

_contexts = local()


def _get_mlir_context():
    """Reuse a context per thread, periodically releasing its interned attributes."""
    if getattr(_contexts, "uses", 0) == 0:
        _contexts.context = mlir.make_ir_context()
    _contexts.uses = (getattr(_contexts, "uses", 0) + 1) % 256
    return _contexts.context


def resource_analysis_options() -> dict[str, str]:
    """Locate the native Python callback used by exact logical depth analysis."""
    return {
        "libQPD_path": get_lib_path("catalyst", "CATALYST_LIB_DIR"),
        "libpython_path": get_libpython_path(),
    }


def _validate_tensor_type(tensor_type):
    """Reject element types without a lossless NumPy/MLIR buffer conversion."""
    element_type = ir.RankedTensorType(tensor_type).element_type
    if ir.IntegerType.isinstance(element_type):
        supported = ir.IntegerType(element_type).width in (1, 8, 16, 32, 64)
    elif ir.ComplexType.isinstance(element_type):
        real_type = ir.ComplexType(element_type).element_type
        supported = ir.F32Type.isinstance(real_type) or ir.F64Type.isinstance(real_type)
    else:
        supported = any(
            scalar_type.isinstance(element_type)
            for scalar_type in (ir.F16Type, ir.F32Type, ir.F64Type)
        )
    if not supported:
        raise ValueError(f"Unsupported classical tensor element type: {element_type}")


def _attribute_to_array(text):
    """Decode a dense tensor attribute without losing its element representation."""
    payload, type_text = text.rsplit(" : ", 1)
    tensor_type = ir.RankedTensorType(ir.Type.parse(type_text))
    element_type = tensor_type.element_type
    if not ir.ComplexType.isinstance(element_type):
        return np.asarray(ir.DenseElementsAttr(ir.Attribute.parse(text))).copy()

    # jaxlib's attribute parser/buffer conversion does not support complex dense
    # attributes. Parse the same bits as pairs of real components instead.
    real_type = ir.ComplexType(element_type).element_type
    if ir.F32Type.isinstance(real_type):
        dtype = np.complex64
    elif ir.F64Type.isinstance(real_type):
        dtype = np.complex128
    else:
        raise ValueError(f"Unsupported classical tensor element type: {element_type}")
    is_splat = payload.startswith("dense<(")
    shape = (2,) if is_splat else (*tensor_type.shape, 2)
    component_type = ir.RankedTensorType.get(shape, real_type)
    components = payload.replace("(", "[").replace(")", "]")
    attribute = ir.DenseElementsAttr(ir.Attribute.parse(f"{components} : {component_type}"))
    values = np.asarray(attribute).copy().view(dtype)
    if is_splat:
        return np.broadcast_to(values.reshape(()), tensor_type.shape).copy()
    return values.reshape(tensor_type.shape)


def _array_to_attribute(array, type_text):
    """Encode an execution result using the corresponding function result type."""
    tensor_type = ir.RankedTensorType(ir.Type.parse(type_text))
    array = np.array(array, copy=True, order="C")
    if array.dtype == np.bool_:
        # Explicit i1 tensor types expect packed bits in DenseElementsAttr.get.
        array = np.packbits(array, axis=None, bitorder="little")
    return str(ir.DenseElementsAttr.get(array, type=tensor_type))


@lru_cache(maxsize=128)
def _compile_module(module_text):
    """Compile each pure slice once, independently of its concrete arguments."""
    with _get_mlir_context():
        module = ir.Module.parse(module_text)
        main = next(
            operation
            for operation in module.body.operations
            if operation.operation.name == "func.func"
            and ir.StringAttr(operation.attributes["sym_name"]).value == "main"
        )
        function_type = ir.FunctionType(ir.TypeAttr(main.attributes["function_type"]).value)
        for tensor_type in (*function_type.inputs, *function_type.results):
            _validate_tensor_type(tensor_type)
        input_types = tuple(str(value) for value in function_type.inputs)
        output_types = tuple(str(value) for value in function_type.results)
        # Include a conservative allowance for printed complex values, brackets,
        # types and container overhead before caching any concrete result.
        output_size_bound = sum(
            getsizeof(str(value))
            + 128
            + (128 + 2 * ir.RankedTensorType(value).rank) * prod(ir.RankedTensorType(value).shape)
            for value in function_type.results
        )

    backend = xla_bridge.get_backend("cpu")
    # Register the CPU LAPACK custom calls used by JAX's linear algebra lowering.
    lapack.prepare_lapack_call("getrf_ffi", np.float64)
    executable = backend.compile_and_load(
        module_text, backend.local_devices(), xla_client.CompileOptions()
    )
    return backend, executable, input_types, output_types, output_size_bound


def _execute_module(module_text, input_attributes):
    """Execute a pure slice and return immutable serialized results."""
    backend, executable, input_types, output_types, _ = _compile_module(module_text)
    if len(input_attributes) != len(input_types):
        raise ValueError("Classical slice argument count does not match its signature")
    with _get_mlir_context(), ir.Location.unknown():
        arrays = []
        for attribute, expected_type in zip(input_attributes, input_types):
            if ir.Type.parse(attribute.rsplit(" : ", 1)[1]) != ir.Type.parse(expected_type):
                raise ValueError("Classical slice argument type does not match its signature")
            arrays.append(_attribute_to_array(attribute))
        outputs = executable.execute([backend.buffer_from_pyval(array) for array in arrays])
        if len(outputs) != len(output_types):
            raise ValueError("Classical slice result count does not match its signature")
        return tuple(
            _array_to_attribute(value, type_text) for value, type_text in zip(outputs, output_types)
        )


@lru_cache(maxsize=64)
def _evaluate_cached(module_text, input_attributes):
    """Cache only small pure evaluations, including all argument bits in the key."""
    return _execute_module(module_text, input_attributes)


def evaluate_pbc_classical(module_text: str, input_attributes: list[str]) -> list[str]:
    """Evaluate a compiler-validated pure StableHLO module on concrete tensors.

    The public ``main`` function's arguments and results are passed as complete
    dense tensor attribute strings. This executes the classical decisions that
    select the logical PBC graph; it does not simulate quantum operations or
    estimate physical execution time. Unsupported modules raise an exception,
    which the native callback translates to an unavailable depth.
    """
    arguments = tuple(input_attributes)
    output_size_bound = _compile_module(module_text)[4]
    key_size = getsizeof(module_text) + getsizeof(arguments) + sum(map(getsizeof, arguments))
    # At most 64 entries, each with at most 1 MiB of serialized key/result data.
    # Large tensors still execute normally without retaining their concrete values.
    evaluate = _evaluate_cached if key_size + output_size_bound <= 1024**2 else _execute_module
    return list(evaluate(module_text, arguments))
