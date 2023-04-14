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
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from catalyst import cond, measure, qjit, while_loop


class TestTracing:
    def test_fixed_tracing(self):
        @qjit()
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit():
            m = measure(wires=0)
            qml.RX(m + 0.0, wires=0)
            return m

        assert not circuit()

    def test_cond_inside_while_loop(self):
        def reset_measure(wires):
            """
            measure a wire and then reset it back to the |0> state
            """

            m = measure(wires)

            @cond(m)
            def cond_fn():
                qml.PauliX(wires=wires)

            cond_fn()

            return m

        @qjit()
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit(n):
            @while_loop(lambda i: i < n)
            def loop(i):
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=1)
                qml.Hadamard(wires=2)

                m0 = reset_measure(wires=0)
                m1 = reset_measure(wires=1)
                m2 = reset_measure(wires=2)

                return i + 1 + m0 + m1 + m2

            return loop(0)

        circuit(5)

    def test_discarded_measurements(self):
        @qjit()
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit():
            qml.state()
            return

        assert circuit() is None

    def test_mixed_result_types(self):
        @qjit()
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit():
            @while_loop(lambda _, repeat: repeat)
            def repeat_until_false(i, _):
                qml.Hadamard(0)
                return i + 1, measure(0)

            n_iter, _ = repeat_until_false(0, True)
            return n_iter, qml.state()

        n_iter, state = circuit()
        assert n_iter > 0
        assert np.allclose(np.abs(state), [1, 0])


def test_complex_dialect():
    """Test that we can use functions that turn into complex dialect operations in MLIR."""

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit():
        return qml.state()

    @qjit
    def workflow():
        x = circuit()[0]
        return jnp.sum(x).real

    assert workflow() == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
