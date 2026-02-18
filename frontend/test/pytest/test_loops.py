# Copyright 2022-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test suite for loop operations in Catalyst."""

from textwrap import dedent

import numpy as np
import pennylane as qml
import pytest

from catalyst import api_extensions
from catalyst import for_loop as catalyst_for_loop
from catalyst import measure as catalyst_measure
from catalyst import qjit
from catalyst import while_loop as catalyst_while_loop
from catalyst.utils.exceptions import PlxprCaptureCFCompatibilityError

# pylint: disable=no-value-for-parameter,unused-argument


# Helper functions that select the appropriate primitive based on capture state
def for_loop(*args, **kwargs):
    """Select qml.for_loop when capture is enabled, else catalyst.for_loop."""
    if qml.capture.enabled():
        return qml.for_loop(*args, **kwargs)
    return catalyst_for_loop(*args, **kwargs)


def while_loop(*args, **kwargs):
    """Select qml.while_loop when capture is enabled, else catalyst.while_loop."""
    if qml.capture.enabled():
        return qml.while_loop(*args, **kwargs)
    return catalyst_while_loop(*args, **kwargs)


def measure(*args, **kwargs):
    """Use catalyst.measure always - qml.measure not yet compatible with qjit(capture=True)."""
    if qml.capture.enabled():
        return qml.measure(*args, **kwargs)
    return catalyst_measure(*args, **kwargs)


class TestLoopToJaxpr:
    """Collection of tests that examine the generated JAXPR of loops."""

    def test_while_loop(self):
        """Check the while loop JAXPR."""

        expected = dedent(
            """
            { lambda ; a:f64[]. let
                b:i64[] c:f64[] = while_loop[
                  body_jaxpr={ lambda ; d:i64[] e:f64[]. let
                      f:i64[] = add d 1:i64[]
                    in (f, e) }
                  body_nconsts=0
                  cond_jaxpr={ lambda ; g:i64[] h:f64[]. let
                      i:bool[] = lt g 10:i64[]
                    in (i,) }
                  cond_nconsts=0
                  num_implicit_inputs=0
                  preserve_dimensions=True
                ] 0:i64[] a
              in (b, c) }
            """
        )

        @qjit
        def circuit(x: float):
            @while_loop(lambda v: v[0] < 10)
            def loop(v):
                return v[0] + 1, v[1]

            return loop((0, x))

        assert expected.strip() == str(circuit.jaxpr).strip()

    def test_for_loop(self):
        """Check the for loop JAXPR."""

        expected = dedent(
            """
            { lambda ; a:f64[] b:i64[]. let
                c:i64[] d:f64[] = for_loop[
                  apply_reverse_transform=False
                  body_jaxpr={ lambda ; e:i64[] f:i64[] g:f64[]. let
                      h:i64[] = add f 1:i64[]
                    in (h, g) }
                  body_nconsts=0
                  num_implicit_inputs=0
                  preserve_dimensions=True
                ] 0:i64[] b 1:i64[] 0:i64[] 0:i64[] a
              in (c, d) }
        """
        )

        @qjit
        def circuit(x: float, n: int):
            @for_loop(0, n, 1)
            def loop(_, v):
                return v[0] + 1, v[1]

            return loop((0, x))

        assert expected.strip() == str(circuit.jaxpr).strip()


class TestWhileLoops:
    """Test the Catalyst while_loop operation."""

    def test_alternating_loop(self, backend, capture_mode):
        """Test simple while loop."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(n):
            @while_loop(lambda v: v[0] < v[1])
            def loop(v):
                qml.PauliX(wires=0)
                return v[0] + 1, v[1]

            loop((0, n))
            return qml.expval(qml.PauliZ(0))

        # PauliX applied n times: odd -> |1⟩ (Z=-1), even -> |0⟩ (Z=+1)
        assert circuit(1) == -1.0  # 1 PauliX -> |1⟩
        assert circuit(2) == 1.0  # 2 PauliX -> |0⟩
        assert circuit(3) == -1.0  # 3 PauliX -> |1⟩
        assert circuit(4) == 1.0  # 4 PauliX -> |0⟩

    @pytest.mark.old_frontend  # Error message specific to catalyst.while_loop
    def test_non_bool_condition_error(self, backend, capture_mode):
        """Test error messages issues when the non-bool conditions are provided."""

        def workflow(R):
            @qjit(capture=capture_mode)
            @qml.qnode(qml.device(backend, wires=1))
            def circuit():
                @while_loop(lambda i: R)
                def loop(i):
                    qml.PauliX(wires=0)
                    return i + 1

                loop(0)
                return qml.expval(qml.PauliZ(0))

            return circuit()

        with pytest.raises(TypeError, match="boolean scalar was expected, got the value"):
            workflow((44, 33))
        with pytest.raises(TypeError, match="boolean scalar was expected, got the value"):
            workflow(33)

    def test_closure_condition_fn(self, backend, capture_mode):
        """Test while loop with captured values (closures) in the condition function."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(n):
            @while_loop(lambda i: i < n)
            def loop(i):
                qml.PauliX(wires=0)
                return i + 1

            loop(0)
            return qml.expval(qml.PauliZ(0))

        assert circuit(1) == -1.0
        assert circuit(2) == 1.0
        assert circuit(3) == -1.0
        assert circuit(4) == 1.0

    def test_closure_body_fn(self, backend, capture_mode):
        """Test while loop with captured values (closures) in the body function."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(n):
            my_const = 1

            @while_loop(lambda v: v[0] < v[1])
            def loop(v):
                qml.PauliX(wires=0)
                return v[0] + my_const, v[1]

            loop((0, n))
            return qml.expval(qml.PauliZ(0))

        assert circuit(1) == -1.0
        assert circuit(2) == 1.0
        assert circuit(3) == -1.0
        assert circuit(4) == 1.0

    def test_assert_joint_closure(self, backend, capture_mode):
        """Test while loop with captured values (closures) in both body and condition functions."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(n):
            my_const = 1

            @while_loop(lambda i: i < n)
            def loop(i):
                qml.PauliX(wires=0)
                return i + my_const

            loop(0)
            return qml.expval(qml.PauliZ(0))

        assert circuit(1) == -1.0
        assert circuit(2) == 1.0
        assert circuit(3) == -1.0
        assert circuit(4) == 1.0

    @pytest.mark.old_frontend  # MCM used in arithmetic inside loop - bool type issues with capture
    def test_assert_reference_outside_measure(self, backend, capture_mode):
        """Test while loop in conjunction with the measure op."""

        @qjit(capture=capture_mode)
        @qml.set_shots(1)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(n):
            m = measure(wires=0)

            @while_loop(lambda i: i < n)
            def loop(i):
                qml.PauliX(wires=0)
                return i + 1 + m

            loop(0)
            return qml.sample(measure(wires=0))

        # Initial state |0⟩: m=0, loop runs n times applying PauliX
        # n=1: 1 PauliX -> |1⟩ -> sample=1
        # n=2: 2 PauliX -> |0⟩ -> sample=0
        assert circuit(1)[0] == 1
        assert circuit(2)[0] == 0
        assert circuit(3)[0] == 1
        assert circuit(4)[0] == 0

    def test_multiple_loop_arguments(self, backend, capture_mode):
        """Test while loop with multiple (loop-carried) arguments."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(n: int):
            @while_loop(lambda v, _: v[0] < v[1])
            def loop(v, inc):
                qml.PauliX(wires=0)
                return (v[0] + inc, v[1]), inc

            loop((0, n), 1)
            return qml.expval(qml.PauliZ(0))

        assert circuit(1) == -1.0
        assert circuit(2) == 1.0
        assert circuit(3) == -1.0
        assert circuit(4) == 1.0

    def test_nested_loops_classical(self, capture_mode):
        """Test nested while loops without qnode."""

        @qjit(capture=capture_mode)
        def circuit(n, m):
            @while_loop(lambda i, _: i < n)
            def outer(i, accum):
                @while_loop(lambda j: j < m)
                def inner(j):
                    return j + 1

                return i + 1, accum + inner(0)

            return outer(0, 0)[1]

        assert circuit(5, 6) == 30  # 5 * 6
        assert circuit(4, 7) == 28  # 4 * 7

    def test_nested_loops_quantum(self, backend, capture_mode):
        """Test nested while loops with quantum operations."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(n: int, m: int):
            @while_loop(lambda i: i < n)
            def outer(i):
                @while_loop(lambda j: j < m)
                def inner(j):
                    qml.RX(0.1, wires=0)
                    return j + 1

                inner(0)
                return i + 1

            outer(0)
            return qml.expval(qml.PauliZ(0))

        # Total rotation angle = n * m * 0.1
        # Expectation value of Z for state RX(theta)|0> is cos(theta)
        # Note: RX(theta) = exp(-i*theta/2*X).
        # State = (cos(theta/2)I - i*sin(theta/2)X)|0> = cos(theta/2)|0> - i*sin(theta/2)|1>
        # <Z> = |cos(theta/2)|^2 - |-i*sin(theta/2)|^2 = cos^2(theta/2) - sin^2(theta/2) = cos(theta)

        n, m = 4, 3
        expected = np.cos(n * m * 0.1)
        assert np.isclose(circuit(n, m), expected)

    @pytest.mark.usefixtures("disable_capture")
    def test_while_loop_raises_compatibility_error_with_capture(self):
        """Test that while_loop raises PlxprCaptureCFCompatibilityError when
        capture mode is enabled."""
        qml.capture.enable()

        def condition(i):
            return i < 5

        with pytest.raises(PlxprCaptureCFCompatibilityError) as exc_info:

            @catalyst_while_loop(condition)
            def loop_fn(x):
                return x + 1

        # Verify the error message is specific and helpful
        error_msg = str(exc_info.value)
        assert "catalyst.while_loop is not supported with PennyLane's capture enabled" in error_msg

    @pytest.mark.usefixtures("disable_capture")
    def test_while_loop_raises_compatibility_error_with_capture_integration(self):
        """Test that while_loop raises PlxprCaptureCFCompatibilityError when
        capture mode is enabled."""
        qml.capture.enable()

        with pytest.raises(PlxprCaptureCFCompatibilityError) as exc_info:

            @qml.qjit
            @qml.qnode(qml.device("lightning.qubit", wires=3))
            def test(n):
                def condition(x):
                    return x < n

                @catalyst_while_loop(condition)
                def loop(i):
                    qml.X(i)

                loop()

            test(4)

        # Verify the error message is specific and helpful
        error_msg = str(exc_info.value)
        assert "catalyst.while_loop is not supported with PennyLane's capture enabled" in error_msg


class TestForLoops:
    """Test the Catalyst for_loop operation."""

    @pytest.mark.old_frontend  # Error message specific to catalyst.for_loop
    def test_required_index(self, backend, capture_mode):
        """Check for loop error message when the iteration index is missing."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(n):
            @for_loop(0, n, 1)
            def loop_fn():
                pass

            loop_fn()

        # TODO: raise better error for user
        with pytest.raises(TypeError, match="takes 0 positional arguments but 1 was given"):
            circuit(5)

    def test_basic_loop(self, backend, capture_mode):
        """Test simple for loop."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(n):
            @for_loop(0, n, 1)
            def loop_fn(_):
                qml.PauliX(0)

            loop_fn()
            return qml.expval(qml.PauliZ(0))

        assert circuit(1) == -1.0  # 1 PauliX -> |1⟩
        assert circuit(2) == 1.0  # 2 PauliX -> |0⟩

    def test_loop_carried_values(self, backend, capture_mode):
        """Test for loop with updating loop carried values."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(n):
            @for_loop(0, n, 1)
            def loop_fn(_, x):
                qml.RY(x, wires=0)
                return x + np.pi / 4

            loop_fn(0.0)
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(1), 1.0)
        assert np.allclose(circuit(2), np.sqrt(0.5))
        assert np.allclose(circuit(3), -np.sqrt(0.5))
        assert np.allclose(circuit(4), 0.0)

    def test_dynamic_wires(self, backend, capture_mode):
        """Test for loops with iteration index-dependant wires."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=6))
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

    def test_closure(self, backend, capture_mode):
        """Test for loop with captured values (closures)."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x):
            y = 2 * x

            @for_loop(0, 1, 1)
            def loop_fn(_):
                qml.RY(y, wires=0)

            loop_fn()
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(np.pi / 4), 0.0)

    def test_nested_loops(self, backend, capture_mode):
        """Test nested for loops."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=4))
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

    def test_negative_step(self, backend, capture_mode):
        """Test loops with a negative step size."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(n):
            @for_loop(n, 0, -1)
            def loop_fn(_):
                qml.PauliX(0)

            loop_fn()
            return qml.expval(qml.PauliZ(0))

        assert circuit(1) == -1.0  # 1 PauliX
        assert circuit(0) == 1.0  # 0 PauliX

    @pytest.mark.usefixtures("disable_capture")
    def test_for_loop_raises_compatibility_error_with_capture(self):
        """Test that for_loop raises PlxprCaptureCFCompatibilityError when
        capture mode is enabled."""
        # Enable capture mode
        qml.capture.enable()

        with pytest.raises(PlxprCaptureCFCompatibilityError) as exc_info:

            @catalyst_for_loop(0, 3, 1)
            def loop_fn(i, acc):
                return acc + i

        # Verify the error message is specific and helpful
        error_msg = str(exc_info.value)
        assert "catalyst.for_loop is not supported with PennyLane's capture enabled" in error_msg

    @pytest.mark.usefixtures("disable_capture")
    def test_for_loop_raises_compatibility_error_with_capture_integration(self):
        """Test that for_loop raises PlxprCaptureCFCompatibilityError when
        capture mode is enabled."""
        # Enable capture mode
        qml.capture.enable()

        with pytest.raises(PlxprCaptureCFCompatibilityError) as exc_info:

            @qml.qjit
            @qml.qnode(qml.device("lightning.qubit", wires=3))
            def test(n):
                @catalyst_for_loop(0, n, 1)
                def loop(i):
                    qml.X(i)

                loop()

            test(4)

        # Verify the error message is specific and helpful
        error_msg = str(exc_info.value)
        assert "catalyst.for_loop is not supported with PennyLane's capture enabled" in error_msg


class TestClassicalCompilation:
    """Test that Catalyst loops can be used outside of quantum functions."""

    @pytest.mark.parametrize("x,n", [(2, 3), (4, 5)])
    def test_while_loop(self, x, n, capture_mode):
        """Test while loop in classical function."""

        @qjit(capture=capture_mode)
        def mulc(x: int, n: int):
            @while_loop(lambda v, _: v < n)
            def loop(v, i):
                return v + 1, i + x

            _, x_times_n = loop(0, 0)
            return x_times_n

        assert mulc.mlir
        assert mulc(x, n) == x * n

    @pytest.mark.parametrize("x,n", [(2, 3), (4, 5)])
    def test_while_nested_loop(self, x, n, capture_mode):
        """Test nested while loops in classical function."""

        @qjit(capture=capture_mode)
        def mulc(x: int, n: int):
            @while_loop(lambda i, _: i < x)
            def loop(i, accum):
                @while_loop(lambda j: j < n)
                def loop2(j):
                    return j + 1

                return i + 1, accum + loop2(0)

            _, x_times_n = loop(0, 0)
            return x_times_n

        assert mulc(x, n) == x * n

    @pytest.mark.parametrize("x,n", [(2, 3), (4, 5)])
    def test_for_loop(self, x, n, capture_mode):
        """Test for loop in classical function."""

        @qjit(capture=capture_mode)
        def mulc(x: int, n: int):
            @for_loop(0, n, 1)
            def loop(_, agg):
                return agg + x

            x_times_n = loop(0)
            return x_times_n

        assert mulc.mlir
        assert mulc(x, n) == x * n

    @pytest.mark.parametrize("x,n", [(2, 3), (4, 5)])
    def test_nested_for_loop(self, x, n, capture_mode):
        """Test nested for loops in classical function."""

        @qjit(capture=capture_mode)
        def mulc(x: int, n: int):
            @for_loop(0, x, 1)
            def loop(_, counter):
                @for_loop(0, n, 1)
                def loop2(j, _):
                    return j + 1

                return counter + loop2(0)

            x_times_n = loop(0)
            return x_times_n

        assert mulc.mlir
        assert mulc(x, n) == x * n

    @pytest.mark.parametrize("x,n", [(2, 4), (4, 6)])
    def test_for_loop_2(self, x, n, capture_mode):
        """Test for loop in classical function with different step size."""

        @qjit(capture=capture_mode)
        def mulc(x: int, n: int):
            @for_loop(0, n, 2)
            def loop(_, agg):
                return agg + x

            x_times_n = loop(0)
            return x_times_n

        def muli(x, n):
            agg = 0
            for _ in range(0, n, 2):
                agg += x
            return agg

        assert mulc.mlir
        assert mulc(x, n) == muli(x, n)

    def test_for_loop_inf(self, capture_mode):
        """
        Test for loop with a negative step size (that would produce an infinite range) iterates 0
        times.
        """

        @qjit(capture=capture_mode)
        def revc():
            @for_loop(5, 10, -1)
            def loop(i, agg):
                return agg + i

            return loop(27)

        assert revc.mlir
        assert revc() == 27

    def test_for_loop_neg_step_expression(self, capture_mode):
        """
        Test for loop in classical function with a nontrivial expression that evaluates to a
        negative step, but is constant w.r.t. function args.
        """

        @qjit(capture=capture_mode)
        def revc(m: int):
            y = 7
            x = y * 7

            @for_loop(m, -3, x - 51 + (y - y))
            def loop(i, agg):
                return agg + i

            return loop(0)

        assert revc.mlir
        assert revc(7) == 15


class TestInterpretationControlFlow:
    """Test that the loops' executions are semantically equivalent when compiled and interpreted."""

    def test_while_loop(self):
        """Test simple while loop."""

        def muli(x: int, n: int):
            @while_loop(lambda v, _: v < n)
            def loop(v, i):
                return v + 1, i + x

            _, x_times_n = loop(0, 0)
            return x_times_n

        mulc = qjit(muli)
        assert mulc(1, 2) == muli(1, 2)

    def test_for_loop(self):
        """Test simple for loop."""

        def muli(x: int, n: int):
            @for_loop(0, n, 1)
            def loop(_, agg):
                return agg + x

            x_times_n = loop(0)
            return x_times_n

        mulc = qjit(muli)
        assert np.allclose(mulc(1, 2), muli(1, 2))

    def test_qnode_with_while_loop(self, backend):
        """Test while loop inside QNode."""
        num_wires = 2
        device = qml.device(backend, wires=num_wires)

        @qml.qnode(device)
        def interpreted_circuit(n):
            @while_loop(lambda i: i < n)
            def loop(i):
                qml.RX(np.pi, wires=i)
                return i + 1

            loop(0)
            return qml.state()

        compiled_circuit = qjit(interpreted_circuit)
        assert np.allclose(compiled_circuit(num_wires), interpreted_circuit(num_wires))

    def test_qnode_with_for_loop(self, backend):
        """Test for loop inside QNode."""
        num_wires = 2
        device = qml.device(backend, wires=num_wires)

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


class TestResultStructureInterpreted:
    """Test that interpreted loops preserve the (tuple) structure of arguments/results:
    - no arguments: return None
    - scalar argument: return scalar
    - single tuple argument: return tuple of same size (tested for 0-, 1-, 2-length tuples)
    - multiple arguments: return tuple preserving individual argument structure
    """

    def test_for_loop_no_args(self):
        """Test for loop result structure with no arguments."""

        def loop(_):
            pass

        assert for_loop(0, 0, 1)(loop)() is None
        assert for_loop(0, 1, 1)(loop)() is None
        assert for_loop(0, 2, 1)(loop)() is None

    @pytest.mark.parametrize("x", [1, (), (1,), (1, 1)])
    def test_for_loop_one_arg(self, x):
        """Test for loop result structure with one argument."""

        def loop(_, x):
            return x

        assert for_loop(0, 0, 1)(loop)(x) == x
        assert for_loop(0, 1, 1)(loop)(x) == x
        assert for_loop(0, 2, 1)(loop)(x) == x

    @pytest.mark.parametrize("x", [1, (), (1,), (1, 1)])
    @pytest.mark.parametrize("y", [1, (), (1,), (1, 1)])
    def test_for_loop_two_arg(self, x, y):
        """Test for loop result structure with two arguments."""

        def loop(_, x, y):
            return x, y

        assert for_loop(0, 0, 1)(loop)(x, y) == (x, y)
        assert for_loop(0, 1, 1)(loop)(x, y) == (x, y)
        assert for_loop(0, 2, 1)(loop)(x, y) == (x, y)

    def test_while_loop_no_args(self):
        """Test while loop result structure with no arguments."""

        def loop():
            pass

        assert while_loop(lambda: False)(loop)() is None

    def test_while_loop_one_scalar(self):
        """Test while loop result structure with one scalar."""

        def loop(x):
            return x + 1

        assert while_loop(lambda x: x < 1)(loop)(1) == 1
        assert while_loop(lambda x: x < 2)(loop)(1) == 2
        assert while_loop(lambda x: x < 3)(loop)(1) == 3

    def test_while_loop_two_scalars(self):
        """Test while loop result structure with two scalars."""

        def loop(x, y):
            return x, y + 1

        assert while_loop(lambda _, y: y < 1)(loop)(1, 1) == (1, 1)
        assert while_loop(lambda _, y: y < 2)(loop)(1, 1) == (1, 2)
        assert while_loop(lambda _, y: y < 3)(loop)(1, 1) == (1, 3)

    def test_while_loop_one_empty_tuple(self):
        """Test while loop result structure with one empty tuple."""

        def loop(x):
            return x

        assert while_loop(lambda _: False)(loop)(()) == ()

    def test_while_loop_two_empty_tuples(self):
        """Test while loop result structure with two empty tuples."""

        def loop(x, y):
            return x, y

        assert while_loop(lambda *_: False)(loop)((), ()) == ((), ())

    @pytest.mark.parametrize("x", [(1,), (1, 1)])
    def test_while_loop_one_tuple(self, x):
        """Test while loop result structure with one tuple."""

        def loop(x):
            return (x[0] + 1,) + x[1:]

        assert while_loop(lambda x: x[0] < 1)(loop)(x) == (x[0] + 0,) + x[1:]
        assert while_loop(lambda x: x[0] < 2)(loop)(x) == (x[0] + 1,) + x[1:]
        assert while_loop(lambda x: x[0] < 3)(loop)(x) == (x[0] + 2,) + x[1:]

    @pytest.mark.parametrize("x", [(1,), (1, 1)])
    @pytest.mark.parametrize("y", [(1,), (1, 1)])
    def test_while_loop_two_tuples(self, x, y):
        """Test while loop result structure with two tuples."""

        def loop(x, y):
            return x, (y[0] + 1,) + y[1:]

        assert while_loop(lambda _, y: y[0] < 1)(loop)(x, y) == (x, (y[0] + 0,) + y[1:])
        assert while_loop(lambda _, y: y[0] < 2)(loop)(x, y) == (x, (y[0] + 1,) + y[1:])
        assert while_loop(lambda _, y: y[0] < 3)(loop)(x, y) == (x, (y[0] + 2,) + y[1:])


@pytest.mark.old_frontend
class TestForLoopOperatorAccess:
    """Test suite for accessing the ForLoop operation in quantum contexts in Catalyst."""

    def test_for_loop_access_quantum(self, backend, capture_mode):
        """Test ForLoop operation access in quantum context."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit():
            @for_loop(0, 4, 1)
            def body(i, accum):
                qml.PauliZ(0)
                return accum + 1

            body(0)
            assert isinstance(body.operation, api_extensions.control_flow.ForLoop)

            return qml.probs()

        assert np.allclose(circuit(), [1, 0])

    def test_for_loop_access_classical(self, capture_mode):
        """Test ForLoop operation access in classical context."""

        @qjit(capture=capture_mode)
        def circuit(x):
            @for_loop(0, 10, 1)
            def body(i, accum):
                return accum + x

            x_times_10 = body(0)
            with pytest.raises(
                AttributeError,
                match=r"""
                The for_loop\(\) was not called \(or has not been called\) in a quantum context,
                and thus has no associated quantum operation.
                """,
            ):
                isinstance(body.operation, api_extensions.control_flow.ForLoop)

            return x_times_10

        assert circuit(5) == 50
        assert circuit(3) == 30

    def test_for_loop_access_interpreted(self):
        """Test ForLoop operation access in interpreted context."""

        def func(n):
            @for_loop(0, n, 1)
            def body(i, prod):
                return prod * 2

            two_to_the_n = body(1)
            with pytest.raises(
                AttributeError,
                match=r"""
                The for_loop\(\) was not called \(or has not been called\) in a quantum context,
                and thus has no associated quantum operation.
                """,
            ):
                isinstance(body.operation, api_extensions.control_flow.ForLoop)

            return two_to_the_n

        assert func(10) == 1024
        assert func(5) == 32
        assert func(0) == 1


@pytest.mark.old_frontend
class TestWhileLoopOperatorAccess:
    """Test suite for accessing the WhileLoop operation in quantum contexts in Catalyst."""

    def test_while_loop_access_quantum(self, backend, capture_mode):
        """Test WhileLoop operation access in quantum context."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit():
            @while_loop(lambda i: i < 5)
            def body(i):
                qml.PauliX(0)
                return i + 1

            body(0)
            assert isinstance(body.operation, api_extensions.control_flow.WhileLoop)

            return qml.probs()

        assert np.allclose(circuit(), [0, 1])

    def test_while_loop_access_classical(self, capture_mode):
        """Test WhileLoop operation access in classical context."""

        @qjit(capture=capture_mode)
        def circuit(x):
            @while_loop(lambda i, _: i < 10)
            def body(i, accum):
                return i + 1, accum + x

            _, x_times_10 = body(0, 0)
            with pytest.raises(
                AttributeError,
                match=r"""
                The while_loop\(\) was not called \(or has not been called\) in a quantum context,
                and thus has no associated quantum operation.
                """,
            ):
                isinstance(body.operation, api_extensions.control_flow.WhileLoop)

            return x_times_10

        assert circuit(5) == 50
        assert circuit(3) == 30

    def test_while_loop_access_interpreted(self):
        """Test WhileLoop operation access in interpreted context."""

        def func(n):
            @while_loop(lambda i, _: i < n)
            def body(i, prod):
                return i + 1, prod * 2

            _, two_to_the_n = body(0, 1)
            with pytest.raises(
                AttributeError,
                match=r"""
                The while_loop\(\) was not called \(or has not been called\) in a quantum context,
                and thus has no associated quantum operation.
                """,
            ):
                isinstance(body.operation, api_extensions.control_flow.WhileLoop)

            return two_to_the_n

        assert func(10) == 1024
        assert func(5) == 32
        assert func(0) == 1


if __name__ == "__main__":
    pytest.main(["-x", __file__])
