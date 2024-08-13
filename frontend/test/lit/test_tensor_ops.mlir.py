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

import pennylane as qml
from jax import numpy as jnp

from catalyst import measure, qjit
from catalyst.debug import get_compilation_stage

# Test methodology:
# Each mathematical function found in numpy
#   https://numpy.org/doc/stable/reference/routines.math.html
# that is a binary elementwise operations is tested.
# If a tests succeeds, it is kept, otherwise a comment is added.

# Not sure why the following ops are not working
# perhaps they rely on another function?
# jnp.hypot


# CHECK-LABEL: test_ewise_arctan2
@qjit(keep_intermediate=True)
@qml.qnode(qml.device("lightning.qubit", wires=2))
def test_ewise_arctan2(x, y):
    # CHECK: math.atan2
    # CHECK-SAME: f64
    val = jnp.arctan2(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


test_ewise_arctan2(jnp.array(1.0), jnp.array(2.0))
print(get_compilation_stage(test_ewise_arctan2, "BufferizationPass"))

# Need more time to test
# jnp.ldexp

# Failed to legalize chlo dialect
# jnp.nextafter
# error: custom op 'chlo.next_after' is unknown
#    %1 = chlo.next_after %arg0, %arg1 : tensor<f64>, tensor<f32> -> tensor<f32>

# Interesting to test:
# jnp.lcm
# jnp.gcd
# def test_ewise_lcm (x, y):
#    val = jnp.lcm (x.astype(int), y.astype(int))
#    qml.RZ (val.astype(float), wires=0)
#    qml.sample (qml.PauliZ (wires=0))
# However, it likely falls in relying in another function
# and we currently support only leaf functions.


@qjit(keep_intermediate=True)
@qml.qnode(qml.device("lightning.qubit", wires=2))
# CHECK-LABEL: test_ewise_add
def test_ewise_add(x, y):
    # CHECK: arith.addf
    # CHECK-SAME: f64
    val = jnp.add(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


test_ewise_add(jnp.array(1.0), jnp.array(2.0))
print(get_compilation_stage(test_ewise_add, "BufferizationPass"))


# CHECK-LABEL: test_ewise_mult
@qjit(keep_intermediate=True)
@qml.qnode(qml.device("lightning.qubit", wires=2))
def test_ewise_mult(x, y):
    # CHECK: arith.mulf
    # CHECK-SAME: f64
    val = jnp.multiply(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


test_ewise_mult(jnp.array(1.0), jnp.array(2.0))
print(get_compilation_stage(test_ewise_mult, "BufferizationPass"))


# CHECK-LABEL: test_ewise_div
@qjit(keep_intermediate=True)
@qml.qnode(qml.device("lightning.qubit", wires=2))
def test_ewise_div(x, y):
    # CHECK: arith.divf
    # CHECK-SAME: f64
    val = jnp.divide(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


test_ewise_div(jnp.array(1.0), jnp.array(2.0))
print(get_compilation_stage(test_ewise_div, "BufferizationPass"))


# CHECK-LABEL: test_ewise_power
@qjit(keep_intermediate=True)
@qml.qnode(qml.device("lightning.qubit", wires=2))
def test_ewise_power(x, y):
    # CHECK: math.powf
    # CHECK-SAME: f64
    val = jnp.power(x, y.astype(int))
    qml.RZ(val, wires=0)
    return measure(wires=0)


test_ewise_power(jnp.array(1.0), jnp.array(2.0))
print(get_compilation_stage(test_ewise_power, "BufferizationPass"))


# CHECK-LABEL: test_ewise_sub
@qjit(keep_intermediate=True)
@qml.qnode(qml.device("lightning.qubit", wires=2))
def test_ewise_sub(x, y):
    # CHECK: arith.subf
    # CHECK-SAME: f64
    val = jnp.subtract(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


test_ewise_sub(jnp.array(1.0), jnp.array(2.0))
print(get_compilation_stage(test_ewise_sub, "BufferizationPass"))


@qjit(keep_intermediate=True)
@qml.qnode(qml.device("lightning.qubit", wires=2))
# CHECK-LABEL: test_ewise_true_div
def test_ewise_true_div(x, y):
    # CHECK: arith.divf
    # CHECK-SAME: f64
    val = jnp.true_divide(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


test_ewise_true_div(jnp.array(1.0), jnp.array(2.0))
print(get_compilation_stage(test_ewise_true_div, "BufferizationPass"))

# Not sure why the following ops are not working
# perhaps they rely on another function?
# jnp.floor_divide


# CHECK-LABEL: test_ewise_float_power
@qjit(keep_intermediate=True)
@qml.qnode(qml.device("lightning.qubit", wires=2))
def test_ewise_float_power(x, y):
    # CHECK: math.powf
    # CHECK-SAME: f64
    val = jnp.float_power(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


test_ewise_float_power(jnp.array(1.0), jnp.array(2.0))
print(get_compilation_stage(test_ewise_float_power, "BufferizationPass"))


# Not sure why the following ops are not working
# perhaps they rely on another function?
# jnp.fmod
# jnp.mod
# jnp.remainder

# divmod is interesting because it returns a tuple.
# Chose not to test.


# CHECK-LABEL: test_ewise_maximum
@qjit(keep_intermediate=True)
@qml.qnode(qml.device("lightning.qubit", wires=2))
def test_ewise_maximum(x, y):
    # CHECK: arith.maximumf
    # CHECK-SAME: f64
    val = jnp.maximum(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


test_ewise_maximum(jnp.array(1.0), jnp.array(2.0))
print(get_compilation_stage(test_ewise_maximum, "BufferizationPass"))

# Only single function support
# * jnp.fmax


# CHECK-LABEL: test_ewise_minimum
@qjit(keep_intermediate=True)
@qml.qnode(qml.device("lightning.qubit", wires=2))
def test_ewise_minimum(x, y):
    # CHECK: arith.minimumf
    # CHECK-SAME: f64
    val = jnp.minimum(x, y)
    qml.RZ(val, wires=0)
    return measure(wires=0)


test_ewise_minimum(jnp.array(1.0), jnp.array(2.0))
print(get_compilation_stage(test_ewise_minimum, "BufferizationPass"))

# Only single function support
# * jnp.fmin

# Not sure why the following ops are not working
# perhaps they rely on another function?
# jnp.heaviside
