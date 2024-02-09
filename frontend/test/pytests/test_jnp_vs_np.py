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

import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from catalyst import qjit


def circuit_jnp():
    qml.QubitUnitary(1 / np.sqrt(2) * jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex), wires=0)
    return qml.expval(qml.PauliZ(0))


def circuit_np():
    qml.QubitUnitary(1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex), wires=0)
    return qml.expval(qml.PauliZ(0))


def test_variable_wires(backend):
    """Test variable wires."""

    jitted_fn_jnp = qjit()(qml.qnode(qml.device(backend, wires=1))(circuit_jnp))
    jitted_fn_np = qjit()(qml.qnode(qml.device(backend, wires=1))(circuit_np))
    assert np.isclose(jitted_fn_jnp(), jitted_fn_np())


if __name__ == "__main__":
    pytest.main(["-x", __file__])
