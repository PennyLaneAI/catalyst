# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the Catalyst adjoint function.
"""

from functools import partial

import jax.numpy as jnp
import pennylane as qml
import pennylane.numpy as pnp
import pytest
from numpy.testing import assert_allclose
from pennylane import adjoint as PL_adjoint

from catalyst import adjoint as C_adjoint
from catalyst import cond, for_loop, qjit, while_loop


def verify_catalyst_adjoint_against_pennylane(quantum_func, device, *args):
    """
    A helper function for verifying Catalyst's native adjoint against the behaviour of PennyLane's
    adjoint function. This is specialized to verifying the behaviour of a single function that has
    its adjoint computed.
    """

    @qjit
    @qml.qnode(device)
    def catalyst_workflow(*args):
        C_adjoint(quantum_func)(*args)
        return qml.state()

    @qml.qnode(device)
    def pennylane_workflow(*args):
        PL_adjoint(quantum_func)(*args)
        return qml.state()

    assert_allclose(catalyst_workflow(*args), pennylane_workflow(*args))


def test_adjoint_func(backend):
    """Ensures that catalyst.adjoint accepts simple Python functions as argument. Makes sure that
    simple quantum gates are adjointed correctly."""

    def func():
        qml.PauliX(wires=0)
        qml.PauliY(wires=0)
        qml.PauliZ(wires=1)

    device = qml.device(backend, wires=2)

    @qjit
    @qml.qnode(device)
    def C_workflow():
        qml.PauliX(wires=0)
        C_adjoint(func)()
        qml.PauliY(wires=0)
        return qml.state()

    @qml.qnode(device)
    def PL_workflow():
        qml.PauliX(wires=0)
        PL_adjoint(func)()
        qml.PauliY(wires=0)
        return qml.state()

    actual = C_workflow()
    desired = PL_workflow()
    assert_allclose(actual, desired)


@pytest.mark.parametrize("theta, val", [(jnp.pi, 0), (-100.0, 1)])
def test_adjoint_op(theta, val, backend):
    """Ensures that catalyst.adjoint accepts single PennyLane operators classes as argument."""
    device = qml.device(backend, wires=2)

    @qjit
    @qml.qnode(device)
    def C_workflow(theta, val):
        C_adjoint(qml.RY)(jnp.pi, val)
        C_adjoint(qml.RZ)(theta, wires=val)
        return qml.state()

    @qml.qnode(device)
    def PL_workflow(theta, val):
        PL_adjoint(qml.RY)(jnp.pi, val)
        PL_adjoint(qml.RZ)(theta, wires=val)
        return qml.state()

    actual = C_workflow(theta, val)
    desired = PL_workflow(theta, val)
    assert_allclose(actual, desired)


@pytest.mark.parametrize("theta, val", [(pnp.pi, 0), (-100.0, 2)])
def test_adjoint_bound_op(theta, val, backend):
    """Ensures that catalyst.adjoint accepts single PennyLane operators objects as argument."""

    device = qml.device(backend, wires=3)

    @qjit
    @qml.qnode(device)
    def C_workflow(theta, val):
        C_adjoint(qml.RX(jnp.pi, val))
        C_adjoint(qml.PauliY(val))
        C_adjoint(qml.RZ(theta, wires=val))
        return qml.state()

    @qml.qnode(device)
    def PL_workflow(theta, val):
        PL_adjoint(qml.RX(jnp.pi, val))
        PL_adjoint(qml.PauliY(val))
        PL_adjoint(qml.RZ(theta, wires=val))
        return qml.state()

    actual = C_workflow(theta, val)
    desired = PL_workflow(theta, val)
    assert_allclose(actual, desired, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("w, p", [(0, 0.5), (0, -100.0), (1, 123.22)])
def test_adjoint_param_fun(w, p, backend):
    """Ensures that catalyst.adjoint accepts parameterized Python functions as arguments."""

    def func(w, theta1, theta2, theta3=1):
        qml.RX(theta1 * pnp.pi / 2, wires=w)
        qml.RY(theta2 / 2, wires=w)
        qml.RZ(theta3, wires=1)

    device = qml.device(backend, wires=2)

    @qjit
    @qml.qnode(device)
    def C_workflow(w, theta):
        qml.PauliX(wires=0)
        C_adjoint(func)(w, theta, theta2=theta)
        qml.PauliY(wires=0)
        return qml.state()

    @qml.qnode(device)
    def PL_workflow(w, theta):
        qml.PauliX(wires=0)
        PL_adjoint(func)(w, theta, theta2=theta)
        qml.PauliY(wires=0)
        return qml.state()

    actual = C_workflow(w, p)
    desired = PL_workflow(w, p)
    assert_allclose(actual, desired)


def test_adjoint_nested_fun(backend):
    """Ensures that catalyst.adjoint allows arbitrary nesting."""

    def func(A, I):
        qml.RX(I, wires=1)
        qml.RY(I, wires=1)
        if I < 5:
            I = I + 1
            A(partial(func, A=A, I=I))()

    @qjit
    @qml.qnode(qml.device(backend, wires=2))
    def C_workflow():
        qml.RX(pnp.pi / 2, wires=0)
        C_adjoint(partial(func, A=C_adjoint, I=0))()
        qml.RZ(pnp.pi / 2, wires=0)
        return qml.state()

    @qml.qnode(qml.device("default.qubit", wires=2))
    def PL_workflow():
        qml.RX(pnp.pi / 2, wires=0)
        PL_adjoint(partial(func, A=PL_adjoint, I=0))()
        qml.RZ(pnp.pi / 2, wires=0)
        return qml.state()

    assert_allclose(C_workflow(), PL_workflow())


def test_adjoint_qubitunitary(backend):
    """Ensures that catalyst.adjoint supports QubitUnitary oprtations."""

    def func():
        qml.QubitUnitary(
            jnp.array(
                [
                    [0.99500417 - 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 - 0.09983342j],
                ]
            ),
            wires=[0, 1],
        )

    verify_catalyst_adjoint_against_pennylane(func, qml.device(backend, wires=2))


def test_adjoint_qubitunitary_dynamic_variable_loop(backend):
    """Ensures that catalyst.adjoint supports QubitUnitary oprtations."""

    def func(gate):
        @for_loop(0, 4, 1)
        def loop_body(_i, s):
            # Nonsensical, but good enough
            gate_modified = gate + s
            qml.QubitUnitary(gate_modified, wires=[0, 1])
            return s + 1

        loop_body(1)

    _input = jnp.array(
        [
            [0.99500417 - 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 - 0.09983342j],
        ]
    )

    verify_catalyst_adjoint_against_pennylane(func, qml.device(backend, wires=2), _input)


def test_adjoint_multirz(backend):
    """Ensures that catalyst.adjoint supports MultiRZ operations."""

    def func():
        qml.PauliX(0)
        qml.MultiRZ(theta=pnp.pi / 2, wires=[0, 1])

    verify_catalyst_adjoint_against_pennylane(func, qml.device(backend, wires=2))


def test_adjoint_no_measurements():
    """Checks that catalyst.adjoint rejects functions containing quantum measurements."""

    def func():
        qml.RX(pnp.pi / 2, wires=0)
        qml.sample()

    with pytest.raises(ValueError, match="Quantum measurements are not allowed"):

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def C_workflow():
            C_adjoint(func)()
            return qml.state()

        C_workflow()


def test_adjoint_invalid_argument():
    """Checks that catalyst.adjoint rejects non-quantum program arguments."""
    with pytest.raises(ValueError, match="Expected a callable"):

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def C_workflow():
            C_adjoint(33)()
            return qml.state()

        C_workflow()


def test_adjoint_classical_loop(backend):
    """Checks that catalyst.adjoint supports purely-classical Control-flows."""

    def func(w=0):
        @for_loop(0, 2, 1)
        def loop(_i, s):
            return s + 1

        qml.PauliX(wires=loop(w))
        qml.RX(pnp.pi / 2, wires=w)

    verify_catalyst_adjoint_against_pennylane(func, qml.device(backend, wires=3), 0)


@pytest.mark.parametrize("pred", [True, False])
def test_adjoint_cond(backend, pred):
    """Tests that the correct gates are applied in reverse in a conditional branch"""

    def func(pred, theta):
        @cond(pred)
        def cond_fn():
            qml.RX(theta, wires=0)

        cond_fn()

    dev = qml.device(backend, wires=1)
    verify_catalyst_adjoint_against_pennylane(func, dev, pred, jnp.pi)


def test_adjoint_while_loop(backend):
    """
    Tests that the correct gates are applied in reverse in a while loop with a statically unknown
    number of iterations.
    """

    def func(limit):
        qml.PauliY(wires=0)

        @while_loop(lambda carried: carried < limit)
        def loop_body(carried):
            qml.RX(carried, wires=0)
            return carried * 2

        final = loop_body(1)
        qml.RZ(final, wires=0)

    dev = qml.device(backend, wires=1)
    verify_catalyst_adjoint_against_pennylane(func, dev, 10)


def test_adjoint_for_loop(backend):
    """Tests the correct application of gates (with dynamic wires)"""

    def func(ub):
        @for_loop(0, ub, 1)
        def loop_body(i):
            qml.CNOT(wires=(i, i + 1))

        loop_body()

    dev = qml.device(backend, wires=5)
    verify_catalyst_adjoint_against_pennylane(func, dev, 4)


def test_adjoint_while_nested(backend):
    """Tests the correct handling of nested while loops."""

    def func(limit, inner_iters):
        @while_loop(lambda carried: carried < limit)
        def loop_outer(carried):
            qml.RX(carried, wires=0)

            @while_loop(lambda counter: counter < inner_iters[carried])
            def loop_inner(counter):
                @cond(counter > 3)
                def cond_fn():
                    qml.RY(counter, wires=0)

                @cond_fn.otherwise
                def cond_otherwise():
                    qml.RZ(counter / jnp.pi, wires=0)

                cond_fn()
                return counter + 1

            loop_inner(0)
            return carried + 2

        final = loop_outer(1)
        qml.MultiRZ(final / 30, wires=(0, 1))

    dev = qml.device(backend, wires=2)
    verify_catalyst_adjoint_against_pennylane(
        func, dev, 10, jnp.array([2, 4, 3, 5, 1, 7, 4, 6, 9, 10])
    )


def test_adjoint_nested_with_control_flow(backend):
    """
    Tests that nested adjoint ops produce correct results in the presence of nested control flow.
    """

    def c_quantum_func(theta):
        @for_loop(0, 4, 1)
        def loop_outer(_):
            qml.PauliX(wires=0)

            def inner_func():
                @for_loop(0, 4, 1)
                def loop_inner(_):
                    qml.RX(theta, wires=0)

                loop_inner()

            C_adjoint(inner_func)()

        loop_outer()

    def pl_quantum_func(theta):
        @for_loop(0, 4, 1)
        def loop_outer(_):
            qml.PauliX(wires=0)

            def inner_func():
                @for_loop(0, 4, 1)
                def loop_inner(_):
                    qml.RX(theta, wires=0)

                loop_inner()

            PL_adjoint(inner_func)()

        loop_outer()

    dev = qml.device(backend, wires=1)

    @qjit
    @qml.qnode(dev)
    def catalyst_workflow(*args):
        C_adjoint(c_quantum_func)(*args)
        return qml.state()

    @qml.qnode(dev)
    def pennylane_workflow(*args):
        PL_adjoint(pl_quantum_func)(*args)
        return qml.state()

    assert_allclose(catalyst_workflow(jnp.pi), pennylane_workflow(jnp.pi))


def test_adjoint_for_nested(backend):
    """
    Tests the adjoint op with nested and interspersed for/while loops that produce classical
    values in addition to quantum ones
    """

    def func(theta):
        @for_loop(0, 6, 1)
        def loop_outer(iv):
            qml.RX(theta / 2, wires=0)

            @for_loop(0, iv, 2)
            def loop_inner(jv, ub):
                qml.RY(theta, wires=0)
                return ub + jv

            ub = loop_inner(1)

            qml.RX(theta / ub, wires=0)

            @while_loop(lambda counter: counter < ub)
            def while_loop_inner(counter):
                qml.RZ(ub / 5, wires=0)
                return counter + 1

            final = while_loop_inner(0)

            qml.RX(theta / final, wires=0)

        loop_outer()

    dev = qml.device(backend, wires=1)
    verify_catalyst_adjoint_against_pennylane(func, dev, jnp.pi)


def test_adjoint_outside_qjit():
    """Test that the Catalyst adjoint function can be used without jitting."""

    assert C_adjoint(qml.T(wires=0)) == PL_adjoint(qml.T(wires=0))


def test_adjoint_wires(backend):
    """Test the wires property of Adjoint"""

    @qml.qjit
    @qml.qnode(qml.device(backend, wires=3))
    def circuit(theta):
        def func(theta):
            qml.RX(theta, wires=[0])
            qml.Hadamard(2)
            qml.CNOT([0, 2])

        qctrl = C_adjoint(func)(theta)
        return qctrl.wires

    # Without the `wires` property, returns `[-1]`
    assert circuit(0.3) == qml.wires.Wires([0, 2])


def test_adjoint_wires_qubitunitary(backend):
    """Test the wires property of nested Adjoint with QubitUnitary"""

    @qml.qjit
    @qml.qnode(qml.device(backend, wires=3))
    def circuit():
        def func():
            qml.QubitUnitary(
                jnp.array(
                    [
                        [0.99500417 - 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 - 0.09983342j],
                    ]
                ),
                wires=[0, 1],
            )

        qadj = C_adjoint(C_adjoint(C_adjoint(func)))()
        return qadj.wires

    # Without the `wires` property, returns `[-1]`
    assert circuit() == qml.wires.Wires([0, 1])


@pytest.mark.xfail(reason="adjoint.wires is not supported with variable wires")
def test_adjoint_var_wires(backend):
    """Test catalyst.adjoint.wires with variable wires."""

    from catalyst import debug

    device = qml.device(backend, wires=3)

    def func(w0, w1, theta):
        qml.RX(theta * pnp.pi / 2, wires=w0)
        qml.RY(theta / 2, wires=w1)
        qml.RZ(theta, wires=2)

    @qml.qjit
    @qml.qnode(device)
    def C_workflow(w0, w1, theta):
        qml.PauliX(wires=0)
        cadj = C_adjoint(func)(w0, w1, theta)
        debug.print(cadj.wires)
        # <Wires =
        #    [Traced<ShapedArray(int64[], weak_type=True)>with<DynamicJaxprTrace(level=3/1)>,
        #     Traced<ShapedArray(int64[], weak_type=True)>with<DynamicJaxprTrace(level=3/1)>,
        #     2]>

        return qml.state()

    C_workflow(0, 1, 0.23)


@pytest.mark.xfail(reason="adjoint.wires is not supported with control-flow branches")
def test_adjoint_wires_controlflow(backend):
    """Test the wires property of Adjoint  in a conditional branch"""

    @qml.qjit
    @qml.qnode(qml.device(backend, wires=3))
    def circuit():
        def func(pred, theta):
            @cond(pred)
            def cond_fn():
                qml.RX(theta, wires=0)

            cond_fn()

        qadj = C_adjoint(func)(True, 3.14)
        return qadj.wires

    # It returns `-1` instead of `0`
    assert circuit() == qml.wires.Wires([0])


def test_adjoint_ctrl_ctrl_subroutine(backend):
    """https://github.com/PennyLaneAI/catalyst/issues/589"""

    def subsubroutine():
        qml.ctrl(qml.PhaseShift, control=2)(0.1, wires=3)

    def subroutine():
        subsubroutine()
        qml.adjoint(qml.ctrl(subsubroutine, control=0))()

    dev = qml.device(backend, wires=4, shots=500)

    @qml.qnode(dev)
    def circuit():
        qml.adjoint(subroutine)()
        return qml.probs(wires=dev.wires)

    expected = circuit()
    observed = qjit(circuit)()
    assert_allclose(expected, observed)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
