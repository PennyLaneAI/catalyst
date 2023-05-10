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
import pennylane as qml
import pytest

from catalyst import measure, qjit


@pytest.mark.parametrize(
    "_in,_out",
    [
        (0, False),
        (1, True),
    ],
)
def test_variable_capture(_in, _out, backend):
    """Test variable capture."""

    @qjit()
    def workflow1(n: int):
        @qml.qnode(qml.device(backend, wires=2))
        def f(x: float):
            qml.RX(n * x, wires=n)
            return measure(wires=n)

        @qml.qnode(qml.device(backend, wires=2))
        def g(x: float):
            qml.RX(x, wires=1)
            return measure(wires=1)

        return jnp.array_equal(f(jnp.pi), g(jnp.pi))

    assert workflow1(_in) == _out


if __name__ == "__main__":
    pytest.main(["-x", __file__])
