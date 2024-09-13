# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the AutoGraph source-to-source transformation feature."""

# RUN: %PYTHON %s | FileCheck %s

import jax.numpy as jnp
import pennylane as qml
from jax.core import ShapedArray
from lit_util_printers import print_jaxpr, print_mlir

from catalyst import qjit


# CHECK-LABEL: test_qjit_dynamic_argument
@qjit(abstracted_axes={0: "n"})
def test_qjit_dynamic_argument(a):
    """Test passing a dynamic argument"""
    # CHECK:        tensor<?xi64>
    return a


print_mlir(test_qjit_dynamic_argument, jnp.array([1, 2, 3]))


# CHECK-LABEL: test_qnode_dynamic_arg
@qjit(abstracted_axes={0: "n"})
def test_qnode_dynamic_arg(a):
    """Test passing a dynamic argument to qnode"""

    # CHECK:       { lambda ; [[a:.]]:i64[] [[b:.]]:i64[[[a]]]. let
    # CHECK:         [[c:.]]:i64[[[a]]] = func[
    # CHECK:                                  ] [[a]] [[b]]
    # CHECK:       in ([[c]],) }
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def _circuit(a):
        return a

    return _circuit(a)


print_jaxpr(test_qnode_dynamic_arg, jnp.array([1, 2, 3]))


# CHECK-LABEL: test_qjit_dynamic_result
@qjit
def test_qjit_dynamic_result(a):
    """Test getting a dynamic result from qjit"""
    # CHECK:       { lambda ; [[a:.]]:i64[]. let
    # CHECK:         [[b:.]]:i64[] = add [[a]] 1
    # CHECK:         [[c:.]]:f64[[[b]]] = {{[a-z_0-9.]+\[[^]]*\]}} 1.0 [[b]]
    # CHECK:       in ([[b]], [[c]]) }
    return jnp.ones((a + 1,), dtype=float)


print_jaxpr(test_qjit_dynamic_result, 3)


# CHECK-LABEL: test_qnode_dynamic_result
@qjit
def test_qnode_dynamic_result(a):
    """Test getting a dynamic result from qnode"""

    # CHECK:       { lambda ; [[a:.]]:i64[]. let
    # CHECK:         [[b:.]]:i64[] [[c:.]]:f64[[[b]]] = func[
    # CHECK:                                                ] [[a]]
    # CHECK:       in ([[b]], [[c]]) }
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def _circuit(a):
        return jnp.ones((a + 1,), dtype=float)

    return _circuit(a)


print_jaxpr(test_qnode_dynamic_result, 3)


# CHECK-LABEL: test_qjit_aot
@qjit(abstracted_axes={0: "n", 2: "m"})
def test_qjit_aot(a: ShapedArray([1, 3, 1], dtype=float)):
    """Test running aor compilation"""
    # CHECK:        tensor<?x3x?xf64>
    return a


print_mlir(test_qjit_aot, aot=True)


@qjit
def test_qjit_indexing(sz):
    """Check the usage of stablehlo.gather for indexing"""
    r = jnp.ones((sz + 1,), dtype=int)
    # CHECK:        gather
    return r[0]


print_mlir(test_qjit_indexing, 3)
