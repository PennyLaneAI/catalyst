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

import jax
import pennylane as qml
import pytest

from catalyst import measure, qjit


def test_strings_aot(backend):
    """Test strings AOT."""

    # Due to limitations in the frontend, we can only test qjit with scalar floats.
    @qjit()
    @qml.qnode(qml.device(backend, wires=2))
    def foo(x: float, y: float):
        val = jax.numpy.arctan2(x, y)
        qml.RZ(val, wires=0)
        return measure(wires=0)

    with pytest.raises(TypeError):
        foo("hello", "world")


def test_strings_jit(backend):
    """Test strings JIT."""

    # Due to limitations in the frontend, we can only test qjit with scalar floats.
    @qjit()
    @qml.qnode(qml.device(backend, wires=2))
    def bar(x, y):
        val = jax.numpy.arctan2(x, y)
        qml.RZ(val, wires=0)
        return measure(wires=0)

    with pytest.raises(TypeError):
        bar("hello", "world")


if __name__ == "__main__":
    pytest.main(["-x", __file__])
