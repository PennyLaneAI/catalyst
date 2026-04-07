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

import numpy as np
import pennylane as qml
from jax import numpy as jnp
from jax.core import ShapedArray

from catalyst import measure, qjit


def tensor(val, size, dtype):
    return jnp.array([val] * size, dtype)


def test_tensor_accept(type):
    if type in {jnp.complex64, jnp.complex128}:
        in_zero = tensor(complex(0, 0), 1, type)
        in_one = tensor(complex(1, 0), 1, type)

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def jax_untyped(x):
            extract = x[0]
            real = jnp.real(extract)
            multiply = real * np.pi
            qml.RY(multiply, wires=0)
            return measure(wires=0)

        assert jax_untyped(in_zero) == False
        assert jax_untyped(in_one) == True

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def jax_typed(x: ShapedArray([1], type)):
            extract = x[0]
            real = jnp.real(extract)
            multiply = real * np.pi
            qml.RY(multiply, wires=0)
            return measure(wires=0)

        assert jax_typed(in_zero) == False
        assert jax_typed(in_one) == True

        print(jax_untyped.mlir)
        print(jax_typed.mlir)
        del jax_untyped
        del jax_typed

    else:
        in_zero = tensor(0, 1, type)
        in_one = tensor(1, 1, type)

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def jax_untyped(x):
            qml.RY(x[0] * np.pi, wires=0)
            return measure(wires=0)

        assert jax_untyped(in_zero) == False
        assert jax_untyped(in_one) == True

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def jax_typed(x: ShapedArray([1], type)):
            qml.RY(x[0] * np.pi, wires=0)
            return measure(wires=0)

        assert jax_typed(in_zero) == False
        assert jax_typed(in_one) == True

        print(jax_untyped.mlir)
        print(jax_typed.mlir)
        del jax_untyped
        del jax_typed


# The compiled quantum function should accept
# * JAX tensors of type int8
# CHECK-LABEL: jit_jax_untyped(%arg0: tensor<1xi8>)
# CHECK-LABEL: jit_jax_typed(%arg0: tensor<1xi8>)
test_tensor_accept(jnp.int8)
# * JAX tensors of type int16
# CHECK-LABEL: jit_jax_untyped(%arg0: tensor<1xi16>)
# CHECK-LABEL: jit_jax_typed(%arg0: tensor<1xi16>)
test_tensor_accept(jnp.int16)
# * JAX tensors of type int32
# CHECK-LABEL: jit_jax_untyped(%arg0: tensor<1xi32>)
# CHECK-LABEL: jit_jax_typed(%arg0: tensor<1xi32>)
test_tensor_accept(jnp.int32)
# * JAX tensors of type float32
# CHECK-LABEL: jit_jax_untyped(%arg0: tensor<1xf32>)
# CHECK-LABEL: jit_jax_typed(%arg0: tensor<1xf32>)
test_tensor_accept(jnp.float32)
# * JAX tensors of type float64
# CHECK-LABEL: jit_jax_untyped(%arg0: tensor<1xf64>)
# CHECK-LABEL: jit_jax_typed(%arg0: tensor<1xf64>)
test_tensor_accept(jnp.float64)
# * JAX tensors of type complex64
# CHECK-LABEL: jit_jax_untyped(%arg0: tensor<1xcomplex<f32>>)
# CHECK-LABEL: jit_jax_typed(%arg0: tensor<1xcomplex<f32>>)
test_tensor_accept(jnp.complex64)
# * JAX tensors of type complex128
# CHECK-LABEL: jit_jax_untyped(%arg0: tensor<1xcomplex<f64>>)
# CHECK-LABEL: jit_jax_typed(%arg0: tensor<1xcomplex<f64>>)
test_tensor_accept(jnp.complex128)


def test_python_accept(type):
    if type == complex:
        in_zero = type(0)
        in_one = type(1)

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def jax_untyped(x):
            qml.RY(x.real * np.pi, wires=0)
            return measure(wires=0)

        assert jax_untyped(in_zero) == False
        assert jax_untyped(in_one) == True

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def jax_typed(x: type):
            qml.RY(x.real * np.pi, wires=0)
            return measure(wires=0)

        assert jax_typed(in_zero) == False
        assert jax_typed(in_one) == True
        print(jax_untyped.mlir)
        print(jax_typed.mlir)
        del jax_untyped
        del jax_typed
    else:
        in_zero = type(0)
        in_one = type(1)

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def jax_untyped(x):
            qml.RY(x * np.pi, wires=0)
            return measure(wires=0)

        assert jax_untyped(in_zero) == False
        assert jax_untyped(in_one) == True

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def jax_typed(x: type):
            qml.RY(x * np.pi, wires=0)
            return measure(wires=0)

        assert jax_typed(in_zero) == False
        assert jax_typed(in_one) == True
        print(jax_untyped.mlir)
        print(jax_typed.mlir)
        del jax_untyped
        del jax_typed


# The compiled function should accept
# * python booleans
# CHECK-LABEL: jit_jax_untyped(%arg0: tensor<i1>)
# CHECK-LABEL: jit_jax_typed(%arg0: tensor<i1>)
test_python_accept(bool)
# * python integers
# CHECK-LABEL: jit_jax_untyped(%arg0: tensor<i64>)
# CHECK-LABEL: jit_jax_typed(%arg0: tensor<i64>)
test_python_accept(int)
# * python floats
# CHECK-LABEL: jit_jax_untyped(%arg0: tensor<f64>)
# CHECK-LABEL: jit_jax_typed(%arg0: tensor<f64>)
test_python_accept(float)
# * python complex
# CHECK-LABEL: jit_jax_untyped(%arg0: tensor<complex<f64>>)
# CHECK-LABEL: jit_jax_typed(%arg0: tensor<complex<f64>>)
test_python_accept(complex)
