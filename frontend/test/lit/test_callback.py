# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the callback feature."""

# RUN: %PYTHON %s | FileCheck %s

import pennylane as qml

from catalyst import pure_callback


def i(x):
    """Identity function"""
    return x


# CHECK-LABEL: module @one_callback_cached
@qml.qjit
# CHECK-NOT: catalyst.callback @callback
# CHECK-LABEL: func.func public @jit_one_callback_cached
def one_callback_cached(x: float):
    """Single callback is created, but called twice"""
    c = pure_callback(i, float)
    return c(x), c(x)


# CHECK-LABEL: catalyst.callback @callback
# CHECK-NOT: catalyst.callback @callback
print(one_callback_cached.mlir)


@pure_callback
def always_return_float(x) -> float:
    """Function that always returns float"""
    if x == 0.0:
        return x
    else:
        return x + 0.0


# CHECK-LABEL: module @test2
@qml.qjit
# CHECK-NOT: catalyst.callback @callback
# CHECK-LABEL: func.func public @jit_test2
def test2():
    return always_return_float(0.0), always_return_float(1)


# CHECK-LABEL: catalyst.callback @callback
# CHECK-LABEL: catalyst.callback @callback
# CHECK-NOT: catalyst.callback @callback

print(test2.mlir)


# CHECK-LABEL: module @test3
@pure_callback
# CHECK-LABEL func.func private @callback_custom_name
def custom_name(x) -> float:
    """A function with a custom name"""
    return x


@qml.qjit
def test3(x: float) -> float:
    """Tests that custom_name will be in the IR"""
    return custom_name(x)


print(test3.mlir)
