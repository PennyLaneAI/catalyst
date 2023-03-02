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

from catalyst import qjit, measure, while_loop, for_loop
import numpy as np
import pennylane as qml


class TestWhileLoopToJaxpr:
    def test_simple_loop(self):
        expected = """{ lambda ; a:f64[]. let
    b:i64[] c:f64[] = qwhile[
      body_jaxpr={ lambda ; d:i64[] e:f64[]. let f:i64[] = add d 1 in (f, e) }
      body_nconsts=0
      cond_jaxpr={ lambda ; g:i64[] h:f64[]. let i:bool[] = lt g 10 in (i,) }
      cond_nconsts=0
    ] 0 a
  in (b, c) }"""

        @qjit
        def circuit(x: float):
            @while_loop(lambda v: v[0] < 10)
            def loop(v):
                return v[0] + 1, v[1]

            return loop((0, x))

        assert expected == str(circuit._jaxpr)


class TestWhileLoops:
    def test_alternating_loop(self):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(n):
            @while_loop(lambda v: v[0] < v[1])
            def loop(v):
                qml.PauliX(wires=0)
                return v[0] + 1, v[1]

            loop((0, n))
            return measure(wires=0)

        assert circuit(1)
        assert not circuit(2)
        assert circuit(3)
        assert not circuit(4)

    def test_closure_condition_fn(self):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(n):
            @while_loop(lambda i: i < n)
            def loop(i):
                qml.PauliX(wires=0)
                return i + 1

            loop(0)
            return measure(wires=0)

        assert circuit(1)
        assert not circuit(2)
        assert circuit(3)
        assert not circuit(4)

    def test_closure_body_fn(self):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(n):
            my_const = 1

            @while_loop(lambda v: v[0] < v[1])
            def loop(v):
                qml.PauliX(wires=0)
                return v[0] + my_const, v[1]

            loop((0, n))
            return measure(wires=0)

        assert circuit(1)
        assert not circuit(2)
        assert circuit(3)
        assert not circuit(4)

    def test_assert_joint_closure(self):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(n):
            my_const = 1

            @while_loop(lambda i: i < n)
            def loop(i):
                qml.PauliX(wires=0)
                return i + my_const

            loop(0)
            return measure(wires=0)

        assert circuit(1)
        assert not circuit(2)
        assert circuit(3)
        assert not circuit(4)

    def test_assert_reference_outside_measure(self):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(n):
            m = measure(wires=0)

            @while_loop(lambda i: i < n)
            def loop(i):
                qml.PauliX(wires=0)
                return i + 1 + m

            loop(0)
            return measure(wires=0)

        assert circuit(1)
        assert not circuit(2)
        assert circuit(3)
        assert not circuit(4)

    def test_multiple_loop_arguments(self):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(n: int):
            @while_loop(lambda v, _: v[0] < v[1])
            def loop(v, inc):
                qml.PauliX(wires=0)
                return (v[0] + inc, v[1]), inc

            loop((0, n), 1)
            return measure(wires=0)

        assert circuit(1)
        assert not circuit(2)
        assert circuit(3)
        assert not circuit(4)

    def test_nested_loops(self):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(n, m):
            @while_loop(lambda i, _: i < n)
            def outer(i, sum):
                @while_loop(lambda j: j < m)
                def inner(j):
                    return j + 1

                return i + 1, sum + inner(0)

            return outer(0, 0)[1]

        assert circuit(5, 6) == 30  # 5 * 6
        assert circuit(4, 7) == 28  # 4 * 7


class TestForLoops:
    def test_required_index(self):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(n):
            @for_loop(0, n, 1)
            def loop_fn():
                pass

            loop_fn()
            return

        # TODO: raise better error for user
        with pytest.raises(TypeError, match="takes 0 positional arguments but 1 was given"):
            circuit(5)

    def test_basic_loop(self):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(n):
            @for_loop(0, n, 1)
            def loop_fn(i):
                qml.PauliX(0)

            loop_fn()
            return measure(0)

        assert circuit(1)
        assert not circuit(2)

    def test_loop_caried_values(self):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(n):
            @for_loop(0, n, 1)
            def loop_fn(i, x):
                qml.RY(x, wires=0)
                return x + np.pi / 4

            loop_fn(0.0)
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(1), 1.0)
        assert np.allclose(circuit(2), np.sqrt(0.5))
        assert np.allclose(circuit(3), -np.sqrt(0.5))
        assert np.allclose(circuit(4), 0.0)

    def test_dynamic_wires(self):
        @qjit()
        @qml.qnode(qml.device("lightning.qubit", wires=6))
        def circuit(n: int):
            qml.Hadamard(wires=0)

            @for_loop(0, n - 1, 1)
            def loop_fn(i):
                qml.CNOT(wires=[i, i + 1])

            loop_fn()
            return qml.state()

        expected = np.zeros(2**6)
        expected[[0, 2**6 - 1]] = 1 / np.sqrt(2)

        assert np.allclose(circuit(6), expected)

    def test_closure(self):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(x):
            y = 2 * x

            @for_loop(0, 1, 1)
            def loop_fn(i):
                qml.RY(y, wires=0)

            loop_fn()
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(np.pi / 4), 0.0)

    def test_nested_loops(self):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=4))
        def circuit(n):
            # Input state: equal superposition
            @for_loop(0, n, 1)
            def init(i):
                qml.Hadamard(wires=i)

            # QFT
            @for_loop(0, n, 1)
            def qft(i):
                qml.Hadamard(wires=i)

                @for_loop(i + 1, n, 1)
                def inner(j):
                    qml.ControlledPhaseShift(np.pi / 2 ** (n - j + 1), [i, j])

                inner()

            init()
            qft()

            # Expected output: |100...>
            return qml.state()

        assert np.allclose(circuit(4), np.eye(2**4)[0])


class TestInterpretationControlFlow:
    def test_while_loop(self):
        def muli(x: int, n: int):
            @while_loop(lambda v, _: v < n)
            def loop(v, i):
                return v + 1, i + x

            counter, x_times_n = loop(0, 0)
            return x_times_n

        mulc = qjit(muli)
        assert mulc(1, 2) == muli(1, 2)

    def test_for_loop(self):
        def muli(x: int, n: int):
            @for_loop(0, n, 1)
            def loop(i, agg):
                return (agg + x,)

            x_times_n = loop(0)
            return x_times_n

        mulc = qjit(muli)
        assert mulc(1, 2) == muli(1, 2)

    def test_qnode_with_while_loop(self):
        num_wires = 2
        device = qml.device("lightning.qubit", wires=num_wires)

        @qml.qnode(device)
        def interpreted_circuit(n):
            @while_loop(lambda i: i < n)
            def loop(i):
                qml.RX(np.pi, wires=i)
                return (i + 1,)

            loop(0)
            return qml.state()

        compiled_circuit = qjit(interpreted_circuit)
        assert np.allclose(compiled_circuit(num_wires), interpreted_circuit(num_wires))

    def test_qnode_with_for_loop(self):
        num_wires = 2
        device = qml.device("lightning.qubit", wires=num_wires)

        @qml.qnode(device)
        def interpreted_circuit(n):
            @for_loop(0, n, 1)
            def loop(i):
                qml.RX(np.pi, wires=i)
                return ()

            loop()
            return qml.state()

        compiled_circuit = qjit(interpreted_circuit)
        assert np.allclose(compiled_circuit(num_wires), interpreted_circuit(num_wires))


class TestClassicalCompilation:
    @pytest.mark.parametrize("x,n", [(2, 3), (4, 5)])
    def test_while_loop(self, x, n):
        @qjit
        def mulc(x: int, n: int):
            @while_loop(lambda v, _: v < n)
            def loop(v, i):
                return v + 1, i + x

            counter, x_times_n = loop(0, 0)
            return x_times_n

        assert mulc.mlir
        assert mulc(x, n) == x * n

    @pytest.mark.parametrize("x,n", [(2, 3), (4, 5)])
    def test_while_nested_loop(self, x, n):
        @qjit
        def mulc(x: int, n: int):
            @while_loop(lambda i, _: i < x)
            def loop(i, sum):
                @while_loop(lambda j: j < n)
                def loop2(j):
                    return j + 1

                return i + 1, sum + loop2(0)

            counter, x_times_n = loop(0, 0)
            return x_times_n

        assert mulc(x, n) == x * n

    @pytest.mark.parametrize("x,n", [(2, 3), (4, 5)])
    def test_for_loop(self, x, n):
        @qjit
        def mulc(x: int, n: int):
            @for_loop(0, n, 1)
            def loop(i, agg):
                return agg + x

            x_times_n = loop(0)
            return x_times_n

        assert mulc.mlir
        assert mulc(x, n) == x * n

    @pytest.mark.parametrize("x,n", [(2, 3), (4, 5)])
    def test_nested_for_loop(self, x, n):
        @qjit
        def mulc(x: int, n: int):
            @for_loop(0, x, 1)
            def loop(i, counter):
                @for_loop(0, n, 1)
                def loop2(j, carry):
                    return j + 1

                return counter + loop2(0)

            x_times_n = loop(0)
            return x_times_n

        assert mulc.mlir
        assert mulc(x, n) == x * n

    @pytest.mark.parametrize("x,n", [(2, 4), (4, 6)])
    def test_for_loop(self, x, n):
        @qjit
        def mulc(x: int, n: int):
            @for_loop(0, n, 2)
            def loop(i, agg):
                return agg + x

            x_times_n = loop(0)
            return x_times_n

        def muli(x, n):
            agg = 0
            for y in range(0, n, 2):
                agg += x
            return agg

        assert mulc.mlir
        assert mulc(x, n) == muli(x, n)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
