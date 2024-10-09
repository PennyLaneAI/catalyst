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

"""Integration tests for AutoGraph and Catalyst transformations."""

# RUN: %PYTHON %s | FileCheck %s

import inspect

import pennylane as qml
from jax.core import ShapedArray

from catalyst import jacobian, mitigate_with_zne, qjit, vmap

# pylint: disable=missing-function-docstring

# -----
# Test autograph on nested QJIT object.


@qjit(autograph=True, target="mlir")
@qjit(target="")
@qml.qnode(qml.device("lightning.qubit", wires=1))
def test_qjit(c: bool, data: float):
    if c:
        qml.RY(data, wires=0)
    return qml.probs()


# CHECK-LABEL: @test_qjit
# CHECK:         scf.if
print(test_qjit.mlir)


# -----
# Test autograph on nested VMAP object.


@vmap(in_axes=(None, 0))
@qml.qnode(qml.device("lightning.qubit", wires=1))
def test_vmap(c, data):
    if c:
        qml.RY(data, wires=0)
    return qml.probs()


# hack in the annotations for vmap
ptype = inspect.Parameter.POSITIONAL_OR_KEYWORD
annotated_params = [
    inspect.Parameter("c", ptype, annotation=bool),
    inspect.Parameter("data", ptype, annotation=ShapedArray((5,), dtype=float)),
]
test_vmap.__signature__ = inspect.Signature(annotated_params)

test_vmap = qjit(test_vmap, autograph=True, target="mlir")

# CHECK-LABEL: @test_vmap
# CHECK:         scf.if
print(test_vmap.mlir)


# -----
# Test autograph on nested Grad object.


@qjit(autograph=True, target="mlir")
@jacobian(argnums=1)
@qml.qnode(qml.device("lightning.qubit", wires=1))
def test_grad(c: bool, data: float):
    if c:
        qml.RY(data, wires=0)
    return qml.probs()


# CHECK-LABEL: @test_grad
# CHECK:         scf.if
print(test_grad.mlir)


# -----
# Test autograph on nested ZNE object.


@qjit(autograph=True, target="mlir")
@mitigate_with_zne(scale_factors=[1, 3, 5])
@qml.qnode(qml.device("lightning.qubit", wires=1))
def test_zne(c: bool, data: float):
    if c:
        qml.RY(data, wires=0)
    return qml.expval(qml.PauliZ(0))


# CHECK-LABEL: @test_zne
# CHECK:         scf.if
print(test_zne.mlir)
