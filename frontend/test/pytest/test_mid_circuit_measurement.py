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

# TODO: add tests with other measurement processes (e.g. qml.sample, qml.probs, ...)


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
        """Test too many arguments to the wires parameter."""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit():
            qml.RX(0.0, wires=0)
            m = measure(wires=[1, 2])
            return m

        with pytest.raises(
            TypeError, match="Only one element is supported for the 'wires' parameter"
        ):
            qjit(circuit)()

    def test_invalid_arguments2(self, backend):
        """Test too large array for the wires parameter."""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit():
            qml.RX(0.0, wires=0)
            m = measure(wires=jnp.array([1, 2]))
            return m

        with pytest.raises(TypeError, match="Measure is only supported on 1 qubit"):
            qjit(circuit)()

    def test_basic(self, backend):
        """Test measure (basic)."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):
            qml.RX(x, wires=0)
            m = measure(wires=0)
            return m

        assert circuit(jnp.pi)  # m will be equal to True if wire 0 is measured in 1 state

    def test_scalar_array_wire(self, backend):
        """Test a scalar array wire."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(w):
            qml.PauliX(0)
            m = measure(wires=w)
            return m

        assert circuit(jnp.array(0)) == 1

    def test_1element_array_wire(self, backend):
        """Test a 1D single-element array wire."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(w):
            qml.PauliX(0)
            m = measure(wires=w)
            return m

        assert circuit(jnp.array([0])) == 1

    def test_more_complex(self, backend):
        """Test measure (more complex)."""

        @qjit
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

    def test_with_postselect_zero(self, backend):
        """Test measure (postselect = 0)."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):
            qml.RX(x, wires=0)
            m = measure(wires=0, postselect=0)
            return m

        assert not circuit(jnp.pi)  # m will be equal to False

    def test_with_postselect_one(self, backend):
        """Test measure (postselect = 1)."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):
            qml.RX(x, wires=0)
            m = measure(wires=0, postselect=1)
            return m

        assert circuit(jnp.pi)  # m will be equal to True

    def test_with_reset_false(self, backend):
        """Test measure (reset = False)."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit():
            qml.Hadamard(wires=0)
            m1 = measure(wires=0, reset=False, postselect=1)
            m2 = measure(wires=0)
            return m1 == m2

        assert circuit()  # both measures are the same

    def test_with_reset_true(self, backend):
        """Test measure (reset = True)."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit():
            qml.Hadamard(wires=0)
            m1 = measure(wires=0, reset=True, postselect=1)
            m2 = measure(wires=0)
            return m1 != m2

        assert circuit()  # measures are different


if __name__ == "__main__":
    pytest.main(["-x", __file__])
