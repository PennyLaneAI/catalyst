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

import pytest
import jax.numpy as jnp
import pennylane as qml

from catalyst import qjit, measure, CompileError


@qjit()
def workflow1(n: int):
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def f(x: float):
        qml.RX(n * x, wires=n)
        return measure(wires=n)

    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def g(x: float):
        qml.RX(x, wires=1)
        return measure(wires=1)

    return jnp.array_equal(f(jnp.pi), g(jnp.pi))


@pytest.mark.parametrize(
    "workflow,_in,_out",
    [
        (workflow1, 0, False),
        (workflow1, 1, True),
    ],
)
def test_variable_capture(workflow, _in, _out):
    assert workflow(_in) == _out


def test_unsupported_device():
    @qml.qnode(qml.device("default.qubit", wires=2))
    def func():
        return qml.probs()

    with pytest.raises(CompileError, match="devices are supported for compilation at the moment."):
        qjit(func)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
