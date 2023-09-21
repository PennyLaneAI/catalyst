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

from catalyst import CompileError, measure, qjit


class TestMidCircuitMeasurement:
    def test_pl_measure(self, backend):
        """Test PL measure."""

        def circuit():
            return qml.measure(0)

        with pytest.raises(CompileError, match="Must use 'measure' from Catalyst"):
            qjit(qml.qnode(qml.device(backend, wires=1))(circuit))()

    def test_measure_outside_qjit(self):
        """Test measure outside qjit."""

        def circuit():
            return measure(0)

        with pytest.raises(CompileError, match="can only be used from within @qjit"):
            circuit()

    def test_measure_outside_qnode(self):
        """Test measure outside qnode."""

        def circuit():
            return measure(0)

        with pytest.raises(CompileError, match="can only be used from within a qml.qnode"):
            qjit(circuit)()

    def test_invalid_arguments(self, backend):
        """Test invalid arguments exception."""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit():
            qml.RX(0.0, wires=0)
            m = measure(wires=[1, 2])
            return m

        with pytest.raises(TypeError, match=r"One classical argument \(a wire\) is expected"):
            qjit(circuit)()

    def test_basic(self, backend):
        """Test measure (basic)."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):
            qml.RX(x, wires=0)
            m = measure(wires=0)
            return m

        assert circuit(jnp.pi)  # m will be equal to True if wire 0 is measured in 1 state

    def test_more_complex(self, backend):
        """Test measure (more complex)."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x: float):
            qml.RX(x, wires=0)
            m1 = measure(wires=0)
            maybe_pi = m1 * jnp.pi
            qml.RX(maybe_pi, wires=1)
            m2 = measure(wires=1)
            return m2

        assert circuit(jnp.pi)  # m will be equal to True if wire 0 is measured in 1 state
        assert not circuit(0.0)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
