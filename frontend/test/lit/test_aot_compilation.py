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

# RUN: %PYTHON %s | FileCheck %s

import jax
import pennylane as qml
from jax.core import ShapedArray

from catalyst import measure, qjit

"""
Currently unsupported:
    * numpy types
"""


# CHECK-LABEL: public @jit_function_complex
# CHECK-SAME: tensor<complex<f64>>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_complex(x: complex, y: complex):
    x_r = x.real
    y_r = y.real
    val = jax.numpy.arctan2(x_r, y_r)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_complex.mlir)


# CHECK-LABEL: public @jit_function_jaxnumpy_csingle
# CHECK-SAME: tensor<complex<f32>>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_jaxnumpy_csingle(x: jax.numpy.csingle, y: jax.numpy.csingle):
    x_r = x.real
    y_r = y.real
    val = jax.numpy.arctan2(x_r, y_r)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_jaxnumpy_csingle.mlir)


# CHECK-LABEL: public @jit_function_jaxnumpy_cdouble
# CHECK-SAME: tensor<complex<f64>>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_jaxnumpy_cdouble(x: jax.numpy.cdouble, y: jax.numpy.cdouble):
    x_r = x.real
    y_r = y.real
    val = jax.numpy.arctan2(x_r, y_r)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_jaxnumpy_cdouble.mlir)


# CHECK-LABEL: public @jit_function_jaxnumpy_complex_
# CHECK-SAME: tensor<complex<f64>>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_jaxnumpy_complex_(x: jax.numpy.complex_, y: jax.numpy.complex_):
    x_r = x.real
    y_r = y.real
    val = jax.numpy.arctan2(x_r, y_r)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_jaxnumpy_complex_.mlir)


# CHECK-LABEL: public @jit_function_jaxnumpy_complex64
# CHECK-SAME: tensor<complex<f32>>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_jaxnumpy_complex64(x: jax.numpy.complex64, y: jax.numpy.complex64):
    x_r = x.real
    y_r = y.real
    val = jax.numpy.arctan2(x_r, y_r)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_jaxnumpy_complex64.mlir)


# CHECK-LABEL: public @jit_function_jaxnumpy_complex128
# CHECK-SAME: tensor<complex<f64>>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_jaxnumpy_complex128(x: jax.numpy.complex128, y: jax.numpy.complex128):
    x_r = x.real
    y_r = y.real
    val = jax.numpy.arctan2(x_r, y_r)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_jaxnumpy_complex128.mlir)


# CHECK-LABEL: public @jit_function_bool
# CHECK-SAME: tensor<i1>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_bool(x: bool, y: bool):
    val = jax.numpy.arctan2(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_bool.mlir)


# CHECK-LABEL: public @jit_function_int
# CHECK-SAME: tensor<i64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_int(x: int, y: int):
    val = jax.numpy.arctan2(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_int.mlir)


# CHECK-LABEL: public @jit_function_float
# CHECK-SAME: tensor<f64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_float(x: float, y: float):
    val = jax.numpy.arctan2(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_float.mlir)


# CHECK-LABEL: public @jit_function_jaxnumpy_float64
# CHECK-SAME: tensor<f64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_jaxnumpy_float64(x: jax.numpy.float64, y: jax.numpy.float64):
    val = jax.numpy.arctan2(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_jaxnumpy_float64.mlir)


# CHECK-LABEL: public @jit_function_jaxnumpy_double
# CHECK-SAME: tensor<f64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_jaxnumpy_double(x: jax.numpy.double, y: jax.numpy.double):
    val = jax.numpy.arctan2(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_jaxnumpy_double.mlir)


# CHECK-LABEL: public @jit_function_jaxnumpy_int_
# CHECK-SAME: tensor<i64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_jaxnumpy_int_(x: jax.numpy.int_, y: jax.numpy.int_):
    val = jax.numpy.arctan2(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_jaxnumpy_int_.mlir)


# CHECK-LABEL: public @jit_function_jaxnumpy_int64
# CHECK-SAME: tensor<i64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_jaxnumpy_int64(x: jax.numpy.int64, y: jax.numpy.int64):
    val = jax.numpy.arctan2(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_jaxnumpy_int64.mlir)


# CHECK-LABEL: public @jit_function_scalar_tensor_bool
# CHECK-SAME: tensor<i1>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_scalar_tensor_bool(x: ShapedArray([], bool), y: ShapedArray([], bool)):
    val = jax.numpy.arctan2(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_scalar_tensor_bool.mlir)


# CHECK-LABEL: public @jit_function_scalar_tensor_int
# CHECK-SAME: tensor<i64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_scalar_tensor_int(x: ShapedArray([], int), y: ShapedArray([], int)):
    val = jax.numpy.arctan2(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_scalar_tensor_int.mlir)


# CHECK-LABEL: public @jit_function_scalar_tensor_float
# CHECK-SAME: tensor<f64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_scalar_tensor_float(x: ShapedArray([], float), y: ShapedArray([], float)):
    val = jax.numpy.arctan2(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_scalar_tensor_float.mlir)


# CHECK-LABEL: public @jit_function_scalar_tensor_jaxnumpy_float64
# CHECK-SAME: tensor<f64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_scalar_tensor_jaxnumpy_float64(
    x: ShapedArray([], jax.numpy.float64), y: ShapedArray([], jax.numpy.float64)
):
    val = jax.numpy.arctan2(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_scalar_tensor_jaxnumpy_float64.mlir)


# CHECK-LABEL: public @jit_function_scalar_tensor_jaxnumpy_double
# CHECK-SAME: tensor<f64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_scalar_tensor_jaxnumpy_double(
    x: ShapedArray([], jax.numpy.double), y: ShapedArray([], jax.numpy.double)
):
    val = jax.numpy.arctan2(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_scalar_tensor_jaxnumpy_double.mlir)


# CHECK-LABEL: public @jit_function_scalar_tensor_jaxnumpy_int_
# CHECK-SAME: tensor<i64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_scalar_tensor_jaxnumpy_int_(
    x: ShapedArray([], jax.numpy.int_), y: ShapedArray([], jax.numpy.int_)
):
    val = jax.numpy.arctan2(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_scalar_tensor_jaxnumpy_int_.mlir)


# CHECK-LABEL: public @jit_function_scalar_tensor_jaxnumpy_int64
# CHECK-SAME: tensor<i64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_scalar_tensor_jaxnumpy_int64(
    x: ShapedArray([], jax.numpy.int64), y: ShapedArray([], jax.numpy.int64)
):
    val = jax.numpy.arctan2(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_scalar_tensor_jaxnumpy_int64.mlir)


# CHECK-LABEL: public @jit_function_tensor_bool
# CHECK-SAME: tensor<1xi1>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_tensor_bool(x: ShapedArray([1], bool), y: ShapedArray([1], bool)):
    val = jax.numpy.arctan2(x[0], y[0])
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_tensor_bool.mlir)


# CHECK-LABEL: public @jit_function_tensor_int
# CHECK-SAME: tensor<1xi64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_tensor_int(x: ShapedArray([1], int), y: ShapedArray([1], int)):
    val = jax.numpy.arctan2(x[0], y[0])
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_tensor_int.mlir)


# CHECK-LABEL: public @jit_function_tensor_float
# CHECK-SAME: tensor<1xf64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_tensor_float(x: ShapedArray([1], float), y: ShapedArray([1], float)):
    val = jax.numpy.arctan2(x[0], y[0])
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_tensor_float.mlir)


# CHECK-LABEL: public @jit_function_tensor_jaxnumpy_float64
# CHECK-SAME: tensor<1xf64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_tensor_jaxnumpy_float64(
    x: ShapedArray([1], jax.numpy.float64), y: ShapedArray([1], jax.numpy.float64)
):
    val = jax.numpy.arctan2(x[0], y[0])
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_tensor_jaxnumpy_float64.mlir)


# CHECK-LABEL: public @jit_function_tensor_jaxnumpy_double
# CHECK-SAME: tensor<1xf64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_tensor_jaxnumpy_double(
    x: ShapedArray([1], jax.numpy.double), y: ShapedArray([1], jax.numpy.double)
):
    val = jax.numpy.arctan2(x[0], y[0])
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_tensor_jaxnumpy_double.mlir)


# CHECK-LABEL: public @jit_function_tensor_jaxnumpy_int_
# CHECK-SAME: tensor<1xi64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_tensor_jaxnumpy_int_(
    x: ShapedArray([1], jax.numpy.int_), y: ShapedArray([1], jax.numpy.int_)
):
    val = jax.numpy.arctan2(x[0], y[0])
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_tensor_jaxnumpy_int_.mlir)


# CHECK-LABEL: public @jit_function_tensor_jaxnumpy_int64
# CHECK-SAME: tensor<1xi64>
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def function_tensor_jaxnumpy_int64(
    x: ShapedArray([1], jax.numpy.int64), y: ShapedArray([1], jax.numpy.int64)
):
    val = jax.numpy.arctan2(x[0], y[0])
    qml.RZ(val, wires=0)
    return measure(wires=0)


print(function_tensor_jaxnumpy_int64.mlir)


# CHECK-LABEL: module @lowering_to_stablehlo_custom_call
@qjit(target="mlir")
def lowering_to_stablehlo_custom_call(A: ShapedArray([2, 2], jax.numpy.float64)):
    """Test lowering to `stablehlo.custom_call @lapack_dsyevd`"""

    # CHECK: func.func private @eigh
    # CHECK: stablehlo.custom_call @lapack_dsyevd
    B = qml.math.sqrt_matrix(A)
    return B


print(lowering_to_stablehlo_custom_call.mlir)


# CHECK-LABEL: module @multiple_stablehlo_custom_call
@qjit(target="mlir")
def multiple_stablehlo_custom_call(A: ShapedArray([2, 2], jax.numpy.float64)):
    """Test lowering to `stablehlo.custom_call @lapack_dsyevd` of multiple lapack methods"""

    # CHECK: func.func private @eigh
    # CHECK: stablehlo.custom_call @lapack_dsyevd
    B = qml.math.sqrt_matrix(A) @ qml.math.sqrt_matrix(A)
    return B


print(multiple_stablehlo_custom_call.mlir)


@qjit(target="mlir")
# CHECK-LABEL: module @test_nested_module
def test_nested_module():
    # CHECK-LABEL: catalyst.call_function_in_module @module_function::@function

    # CHECK-LABEL: module @module_function
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK-LABEL: func.func private @function
    def function():
        return qml.state()

    return function()


print(test_nested_module.mlir)
