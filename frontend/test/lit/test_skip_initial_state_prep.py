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

# RUN: %PYTHON %s | FileCheck %s

"""Tests code generation of state prep"""

import jax.numpy as jnp
import pennylane as qml


@qml.qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def state_prep_example():
    """Test example from
    https://docs.pennylane.ai/en/stable/code/api/pennylane.StatePrep.html
    as of July 31st 2024.

    Modified to use jax.numpy and a non trivial StatePrep
    """
    qml.StatePrep(jnp.array([0, 1, 0, 0]), wires=range(2))
    return qml.state()


# CHECK-LABEL: func.func private @state_prep_example
#       CHECK: quantum.set_state
print(state_prep_example.mlir)


@qml.qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def basis_state_example():
    """Test example from
    https://docs.pennylane.ai/en/stable/code/api/pennylane.BasisState.html
    as of July 31st 2024.

    Modified to use jax.numpy and a non trivial StatePrep
    """
    qml.BasisState(jnp.array([1, 1]), wires=range(2))
    return qml.state()


# CHECK-LABEL: func.func private @basis_state_example
#       CHECK: quantum.set_basis_state
print(basis_state_example.mlir)


@qml.qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def state_prep_example_double():
    """What happens if we have two? It shouldn't be repeated because
    we only skip the first one
    """
    qml.StatePrep(jnp.array([0, 1, 0, 0]), wires=range(2))
    qml.StatePrep(jnp.array([1, 0, 0, 0]), wires=range(2))
    return qml.state()


# CHECK-LABEL: func.func private @state_prep_example_double
#       CHECK:   quantum.set_state
#   CHECK-NOT:   quantum.set_state
print(state_prep_example_double.mlir)
