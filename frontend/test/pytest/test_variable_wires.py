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
import pennylane as qp
import pytest

from catalyst import cond, measure, qjit, while_loop


class TestWireTypes:
    """Test different forms in which wires could be specified."""

    @pytest.mark.parametrize("dtype", (jnp.bool_, jnp.int8, jnp.int16, jnp.int32, jnp.int64))
    def test_32bit_integer(self, backend, dtype):
        """Test that wires can be a 32-bit integer."""

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(w):
            qp.PauliX(0)
            return measure(w)

        assert circuit(jnp.array(0, dtype=dtype)) == 1


class TestBasicCircuits:
    """Test gates and measurements with variable wires in basic circuits."""

    @pytest.mark.parametrize("args", [[np.pi, 0, 0], [np.pi, 0, 1], [np.pi, 1, 0], [np.pi, 1, 1]])
    def test_operations(self, args, backend):
        """Test operations."""

        def circuit(arg0: float, arg1: int, arg2: int):
            qp.RX(arg0, wires=[arg1])
            qp.RX(arg0, wires=[arg2])
            return qp.state()

        result = qjit(qp.qnode(qp.device(backend, wires=3))(circuit))(*args)
        expected = qp.qnode(qp.device("default.qubit", wires=3))(circuit)(*args)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("args", [[np.pi, 0, 0], [np.pi, 0, 1], [np.pi, 1, 0], [np.pi, 1, 1]])
    def test_operations_with_computation(self, args, backend):
        """Test operations with computation."""

        def circuit(arg0: float, arg1: int, arg2: int):
            qp.RX(arg0, wires=[arg1])
            qp.RX(arg0, wires=[arg2 + 1])
            return qp.state()

        result = qjit(qp.qnode(qp.device(backend, wires=3))(circuit))(*args)
        expected = qp.qnode(qp.device("default.qubit", wires=3))(circuit)(*args)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("args", [[np.pi, 0, 0], [np.pi, 0, 1], [np.pi, 1, 0], [np.pi, 1, 1]])
    def test_measurements(self, args, backend):
        """Test measurements."""

        def circuit(arg0: float, w1: int, w2: int):
            qp.RX(arg0, wires=w1)
            return qp.sample(wires=w2)

        result = qjit(qp.set_shots(qp.qnode(qp.device(backend, wires=3))(circuit), shots=10))(*args)
        expected = qp.set_shots(qp.qnode(qp.device("default.qubit", wires=3))(circuit), shots=10)(
            *args
        )
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("args", [[np.pi, 0, 0], [np.pi, 0, 1], [np.pi, 1, 0], [np.pi, 1, 1]])
    def test_measurements_with_computation(self, args, backend):
        """Test measurements with computation."""

        def circuit(arg0: float, w1: int, w2: int):
            qp.RX(arg0, wires=w1)
            return qp.sample(wires=[w2 + 1])

        result = qjit(qp.set_shots(qp.qnode(qp.device(backend, wires=3))(circuit), shots=10))(*args)
        expected = qp.set_shots(qp.qnode(qp.device("default.qubit", wires=3))(circuit), shots=10)(
            *args
        )
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("args", [[jnp.pi, 0, 1], [jnp.pi, 1, 0]])
    def test_observables(self, args, backend):
        """Test observables."""

        def circuit(arg0: float, w1: int, w2: int):
            qp.RX(arg0, wires=w1)
            A = np.array(
                [
                    [complex(2.0, 0.0), complex(1.0, 1.0), complex(9.0, 2.0), complex(0.0, 0.0)],
                    [complex(1.0, -1.0), complex(5.0, 0.0), complex(4.0, 6.0), complex(3.0, -2.0)],
                    [complex(9.0, -2.0), complex(4.0, -6.0), complex(10.0, 0.0), complex(1.0, 7.0)],
                    [complex(0.0, 0.0), complex(3.0, 2.0), complex(1.0, -7.0), complex(4.0, 0.0)],
                ]
            )
            return qp.expval(qp.Hermitian(A, wires=[w1, w2]))

        result = qjit(qp.qnode(qp.device(backend, wires=2))(circuit))(*args)
        expected = qp.qnode(qp.device("default.qubit", wires=2))(circuit)(*args)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("args", [[jnp.pi, 0, 0]])
    def test_observables_with_computation(self, args, backend):
        """Test observables with computation."""

        def circuit(arg0: float, w1: int, w2: int):
            qp.RX(arg0, wires=w1)
            A = np.array(
                [
                    [complex(2.0, 0.0), complex(1.0, 1.0), complex(9.0, 2.0), complex(0.0, 0.0)],
                    [complex(1.0, -1.0), complex(5.0, 0.0), complex(4.0, 6.0), complex(3.0, -2.0)],
                    [complex(9.0, -2.0), complex(4.0, -6.0), complex(10.0, 0.0), complex(1.0, 7.0)],
                    [complex(0.0, 0.0), complex(3.0, 2.0), complex(1.0, -7.0), complex(4.0, 0.0)],
                ]
            )
            return qp.expval(qp.Hermitian(A, wires=[w1 + 1, w2]))

        result = qjit(qp.qnode(qp.device(backend, wires=2))(circuit))(*args)
        expected = qp.qnode(qp.device("default.qubit", wires=2))(circuit)(*args)
        assert np.allclose(result, expected)

    def test_reinsertion_before_dynamic_wires(self, backend):
        """Test reinsertion before dynamic wires."""

        @qp.qnode(qp.device(backend, wires=7))
        def circuit(qb):
            # Without reinsertion the Rot gate would be folded away as it produces a dangling qubit.
            qp.Rot(0.3, 0.4, 0.5, wires=0)
            qp.CNOT(wires=[qb, 2])
            return qp.state()

        dyn_qubit = 1
        result = qjit(circuit)(jnp.array(dyn_qubit))
        expected = circuit(dyn_qubit)

        assert jnp.allclose(result, expected)


class TestControlFlow:
    """Test if-else, while, and for control flow operations with variable wires."""

    @pytest.mark.parametrize(
        "args,expected", [([0, 0], True), ([0, 1], False), ([1, 0], False), ([1, 1], True)]
    )
    def test_conditional(self, args, expected, backend):
        """Test conditional."""

        @qjit
        @qp.qnode(qp.device(backend, wires=2))
        def circuit(x: int, y: int):
            @cond(x > 4)
            def cond_fn():
                pass

            @cond_fn.otherwise
            def else_fn():
                qp.RX(jnp.pi, wires=y)

            cond_fn()
            return measure(wires=x)

        assert circuit(*args) == expected

    @pytest.mark.parametrize(
        "args,expected", [([0, 0], False), ([0, 1], False), ([1, 0], True), ([1, 1], True)]
    )
    def test_while_loop_with_func_arg_wires(self, args, expected, backend):
        """Test while loop with func arg wires."""

        @qjit
        @qp.qnode(qp.device(backend, wires=2))
        def circuit(n: int, m: int):
            @while_loop(lambda v: v[0] < v[1])
            def loop(v):
                qp.RX(jnp.pi, wires=m)
                return v[0] + 1, v[1]

            loop((0, n))
            return measure(wires=m)

        assert circuit(*args) == expected

    @pytest.mark.parametrize(
        "args,expected",
        [
            ([1, 0], True),
            ([1, 1], False),
            ([2, 1], True),
            ([2, 2], False),
            ([2, 3], False),
            ([3, 1], True),
        ],
    )
    def test_while_loop_with_loop_arg_wires(self, args, expected, backend):
        """Test while loop with loop arg wires."""

        @qjit
        @qp.qnode(qp.device(backend, wires=5))
        def circuit(n: int, m: int):
            qp.RX(jnp.pi, wires=0)

            @while_loop(lambda v: v[0] < v[1])
            def loop(v):
                qp.CNOT(wires=[0, v[0]])
                return v[0] + 1, v[1]

            loop((1, n))
            return measure(wires=m)

        assert circuit(*args) == expected


if __name__ == "__main__":
    pytest.main(["-x", __file__])
