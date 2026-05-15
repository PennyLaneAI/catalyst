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
"""Unit tests for the Catalyst adjoint function."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pennylane.numpy as pnp
import pytest
from numpy.testing import assert_allclose
from pennylane import adjoint, cond, for_loop, qjit, while_loop
from pennylane.ops.op_math.adjoint import Adjoint, AdjointOperation

import catalyst
import catalyst as cat
from catalyst import debug, measure, qjit

# pylint: disable=too-many-lines,missing-class-docstring,missing-function-docstring,too-many-public-methods


class TestCatalyst:
    """Integration tests for Catalyst adjoint functionality."""

    def verify_catalyst_adjoint_against_pennylane(
        self, quantum_func, device, *args, capture_mode="global"
    ):
        """
        A helper function for verifying Catalyst's native adjoint against the behaviour of
        PennyLane's adjoint function. This is specialized to verifying the behaviour of a single
        function that has its adjoint computed.
        """

        @qjit(capture=capture_mode)
        @qp.qnode(device)
        def catalyst_workflow(*args):
            adjoint(quantum_func)(*args)
            return qp.state()

        @qp.qnode(device)
        def pennylane_workflow(*args):
            qp.adjoint(quantum_func)(*args)
            return qp.state()

        capture_enabled = qp.capture.enabled()
        pass
        try:
            pl_res = pennylane_workflow(*args)
        finally:
            if capture_enabled:
                qp.capture.enable()

        assert_allclose(catalyst_workflow(*args), pl_res)

    def test_adjoint_func(self, backend, capture_mode):
        """Ensures that catalyst.adjoint accepts simple Python functions as argument. Makes sure
        that simple quantum gates are adjointed correctly."""

        def func():
            qp.PauliX(wires=0)
            qp.PauliY(wires=0)
            qp.PauliZ(wires=1)

        device = qp.device(backend, wires=2)

        @qjit(capture=capture_mode)
        @qp.qnode(device)
        def C_workflow():
            qp.PauliX(wires=0)
            adjoint(func)()
            qp.PauliY(wires=0)
            return qp.state()

        @qp.qnode(device)
        def PL_workflow():
            qp.PauliX(wires=0)
            qp.adjoint(func)()
            qp.PauliY(wires=0)
            return qp.state()

        actual = C_workflow()
        desired = PL_workflow()
        assert_allclose(actual, desired)

    @pytest.mark.parametrize("theta, val", [(jnp.pi, 0), (-100.0, 1)])
    def test_adjoint_op(self, theta, val, backend, capture_mode):
        """Ensures that catalyst.adjoint accepts single PennyLane operators classes as argument."""
        device = qp.device(backend, wires=2)

        @qjit(capture=capture_mode)
        @qp.qnode(device)
        def C_workflow(theta, val):
            adjoint(qp.RY)(jnp.pi, val)
            adjoint(qp.RZ)(theta, val)
            return qp.state()

        @qp.qnode(device)
        def PL_workflow(theta, val):
            qp.adjoint(qp.RY)(jnp.pi, val)
            qp.adjoint(qp.RZ)(theta, val)
            return qp.state()

        actual = C_workflow(theta, val)
        desired = PL_workflow(theta, val)
        assert_allclose(actual, desired)

    @pytest.mark.parametrize("theta, val", [(np.pi, 0), (-100.0, 2)])
    def test_adjoint_bound_op(self, theta, val, backend, capture_mode):
        """Ensures that catalyst.adjoint accepts single PennyLane operators objects as argument."""

        device = qp.device(backend, wires=3)

        @qjit(capture=capture_mode)
        @qp.qnode(device)
        def C_workflow(theta, val):
            adjoint(qp.RX(jnp.pi, val))
            adjoint(qp.PauliY(val))
            adjoint(qp.RZ(theta, wires=val))
            return qp.state()

        @qp.qnode(device)
        def PL_workflow(theta, val):
            qp.adjoint(qp.RX(jnp.pi, val))
            qp.adjoint(qp.PauliY(val))
            qp.adjoint(qp.RZ(theta, wires=val))
            return qp.state()

        actual = C_workflow(theta, val)
        desired = PL_workflow(theta, val)
        assert_allclose(actual, desired, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("w, p", [(0, 0.5), (0, -100.0), (1, 123.22)])
    def test_adjoint_param_fun(self, w, p, backend, capture_mode):
        """Ensures that catalyst.adjoint accepts parameterized Python functions as arguments."""

        def func(w, theta1, theta2, theta3=1):
            qp.RX(theta1 * np.pi / 2, wires=w)
            qp.RY(theta2 / 2, wires=w)
            qp.RZ(theta3, wires=1)

        device = qp.device(backend, wires=2)

        @qjit(capture=capture_mode)
        @qp.qnode(device)
        def C_workflow(w, theta):
            qp.PauliX(wires=0)
            adjoint(func)(w, theta, theta)
            qp.PauliY(wires=0)
            return qp.state()

        @qp.qnode(device)
        def PL_workflow(w, theta):
            qp.PauliX(wires=0)
            qp.adjoint(func)(w, theta, theta)
            qp.PauliY(wires=0)
            return qp.state()

        actual = C_workflow(w, p)
        desired = PL_workflow(w, p)
        assert_allclose(actual, desired)

    def test_adjoint_nested_fun(self, backend, capture_mode):
        """Ensures that catalyst.adjoint allows arbitrary nesting."""

        def func(A, I):
            qp.RX(I, wires=1)
            qp.RY(I, wires=1)
            if I < 5:
                I = I + 1
                A(partial(func, A=A, I=I))()

        @qjit(capture=capture_mode)
        @qp.qnode(qp.device(backend, wires=2))
        def C_workflow():
            qp.RX(np.pi / 2, wires=0)
            adjoint(partial(func, A=adjoint, I=0))()
            qp.RZ(np.pi / 2, wires=0)
            return qp.state()

        @qp.qnode(qp.device("default.qubit", wires=2))
        def PL_workflow():
            qp.RX(np.pi / 2, wires=0)
            qp.adjoint(partial(func, A=qp.adjoint, I=0))()
            qp.RZ(np.pi / 2, wires=0)
            return qp.state()

        assert_allclose(C_workflow(), PL_workflow())

    def test_adjoint_qubitunitary(self, backend, capture_mode):
        """Ensures that catalyst.adjoint supports QubitUnitary oprtations."""

        def func():
            qp.QubitUnitary(
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

        self.verify_catalyst_adjoint_against_pennylane(
            func, qp.device(backend, wires=2), capture_mode=capture_mode
        )

    def test_adjoint_qubitunitary_dynamic_variable_loop(self, backend, capture_mode):
        """Ensures that catalyst.adjoint supports QubitUnitary oprtations."""

        def func(gate):
            @for_loop(0, 4, 1)
            def loop_body(_i, s):
                # Nonsensical, but good enough
                gate_modified = gate + s
                qp.QubitUnitary(gate_modified, wires=[0, 1])
                return s + 1

            loop_body(1)  # pylint: disable=no-value-for-parameter

        _input = jnp.array(
            [
                [0.99500417 - 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 - 0.09983342j],
            ]
        )

        self.verify_catalyst_adjoint_against_pennylane(
            func, qp.device(backend, wires=2), _input, capture_mode=capture_mode
        )

    def test_adjoint_multirz(self, backend, capture_mode):
        """Ensures that catalyst.adjoint supports MultiRZ operations."""

        def func():
            qp.PauliX(0)
            qp.MultiRZ(theta=np.pi / 2, wires=[0, 1])

        self.verify_catalyst_adjoint_against_pennylane(
            func, qp.device(backend, wires=2), capture_mode=capture_mode
        )

    def test_adjoint_pcphase(self, backend):
        """Ensures that catalyst.adjoint supports PCPhase operations."""

        def func():
            qp.PauliX(0)
            qp.PCPhase(np.pi / 2, dim=0, wires=[0, 1])

        self.verify_catalyst_adjoint_against_pennylane(func, qp.device(backend, wires=2))

    def test_adjoint_no_measurements(self):
        """Checks that catalyst.adjoint rejects functions containing quantum measurements."""

        def func():
            qp.RX(np.pi / 2, wires=0)
            qp.sample()

        with pytest.raises(ValueError, match="Measurement process cannot be used"):

            @qjit
            @qp.qnode(qp.device("lightning.qubit", wires=2))
            def C_workflow():
                adjoint(func)()
                return qp.state()

            C_workflow()

    def test_adjoint_invalid_argument(self):
        """Checks that catalyst.adjoint rejects non-quantum program arguments."""
        with pytest.raises(ValueError, match="Expected a callable"):

            @qjit
            @qp.qnode(qp.device("lightning.qubit", wires=2))
            def C_workflow():
                adjoint(33)()
                return qp.state()

            C_workflow()

    def test_adjoint_classical_loop(self, backend, capture_mode):
        """Checks that catalyst.adjoint supports purely-classical Control-flows."""

        def func(w=0):
            @for_loop(0, 2, 1)
            def loop(_i, s):
                return s + 1

            qp.PauliX(wires=loop(w))  # pylint: disable=no-value-for-parameter
            qp.RX(np.pi / 2, wires=w)

        self.verify_catalyst_adjoint_against_pennylane(
            func, qp.device(backend, wires=3), 0, capture_mode=capture_mode
        )

    @pytest.mark.parametrize("pred", [True, False])
    def test_adjoint_cond(self, backend, pred, capture_mode):
        """Tests that the correct gates are applied in reverse in a conditional branch"""

        def func(pred, theta):
            @cond(pred)
            def cond_fn():
                qp.RX(theta, wires=0)

            cond_fn()

        dev = qp.device(backend, wires=1)
        self.verify_catalyst_adjoint_against_pennylane(
            func, dev, pred, jnp.pi, capture_mode=capture_mode
        )

    def test_adjoint_while_loop(self, backend, capture_mode):
        """
        Tests that the correct gates are applied in reverse in a while loop with a statically
        unknown number of iterations.
        """

        def func(limit):
            qp.PauliY(wires=0)

            @while_loop(lambda carried: carried < limit)
            def loop_body(carried):
                qp.RX(carried, wires=0)
                return carried * 2

            final = loop_body(1)  # pylint: disable=no-value-for-parameter
            qp.RZ(final, wires=0)

        dev = qp.device(backend, wires=1)
        self.verify_catalyst_adjoint_against_pennylane(func, dev, 10, capture_mode=capture_mode)

    def test_adjoint_for_loop(self, backend, capture_mode):
        """Tests the correct application of gates (with dynamic wires)"""

        def func(ub):
            @for_loop(0, ub, 1)
            def loop_body(i):
                qp.CNOT(wires=(i, i + 1))

            loop_body()  # pylint: disable=no-value-for-parameter

        dev = qp.device(backend, wires=5)
        self.verify_catalyst_adjoint_against_pennylane(func, dev, 4, capture_mode=capture_mode)

    def test_adjoint_while_nested(self, backend, capture_mode):
        """Tests the correct handling of nested while loops."""

        def func(limit, inner_iters):
            @while_loop(lambda carried: carried < limit)
            def loop_outer(carried):
                qp.RX(carried, wires=0)

                @while_loop(lambda counter: counter < inner_iters[carried])
                def loop_inner(counter):
                    @cond(counter > 3)
                    def cond_fn():
                        qp.RY(counter, wires=0)

                    @cond_fn.otherwise
                    def cond_otherwise():
                        qp.RZ(counter / jnp.pi, wires=0)

                    cond_fn()
                    return counter + 1

                loop_inner(0)  # pylint: disable=no-value-for-parameter
                return carried + 2

            final = loop_outer(1)  # pylint: disable=no-value-for-parameter
            qp.MultiRZ(final / 30, wires=(0, 1))

        dev = qp.device(backend, wires=2)
        self.verify_catalyst_adjoint_against_pennylane(
            func,
            dev,
            10,
            jnp.array([2, 4, 3, 5, 1, 7, 4, 6, 9, 10]),
            capture_mode=capture_mode,
        )

    def test_adjoint_nested_with_control_flow(self, backend, capture_mode):
        """
        Tests that nested adjoint ops produce correct results in the presence of nested control
        flow.
        """

        def c_quantum_func(theta):
            @for_loop(0, 4, 1)
            def loop_outer(_):
                qp.PauliX(wires=0)

                def inner_func():
                    @for_loop(0, 4, 1)
                    def loop_inner(_):
                        qp.RX(theta, wires=0)

                    loop_inner()  # pylint: disable=no-value-for-parameter

                adjoint(inner_func)()

            loop_outer()  # pylint: disable=no-value-for-parameter

        def pl_quantum_func(theta):
            @for_loop(0, 4, 1)
            def loop_outer(_):
                qp.PauliX(wires=0)

                def inner_func():
                    @for_loop(0, 4, 1)
                    def loop_inner(_):
                        qp.RX(theta, wires=0)

                    loop_inner()  # pylint: disable=no-value-for-parameter

                qp.adjoint(inner_func)()

            loop_outer()  # pylint: disable=no-value-for-parameter

        dev = qp.device(backend, wires=1)

        @qjit(capture=capture_mode)
        @qp.qnode(dev)
        def catalyst_workflow(*args):
            adjoint(c_quantum_func)(*args)
            return qp.state()

        @qp.qnode(dev)
        def pennylane_workflow(*args):
            qp.adjoint(pl_quantum_func)(*args)
            return qp.state()

        assert_allclose(catalyst_workflow(jnp.pi), pennylane_workflow(jnp.pi))

    def test_adjoint_for_nested(self, backend, capture_mode):
        """
        Tests the adjoint op with nested and interspersed for/while loops that produce classical
        values in addition to quantum ones
        """

        def func(theta):
            @for_loop(0, 6, 1)
            def loop_outer(iv):
                qp.RX(theta / 2, wires=0)

                @for_loop(0, iv, 2)
                def loop_inner(jv, ub):
                    qp.RY(theta, wires=0)
                    return ub + jv

                ub = loop_inner(1)  # pylint: disable=no-value-for-parameter

                qp.RX(theta / ub, wires=0)

                @while_loop(lambda counter: counter < ub)
                def while_loop_inner(counter):
                    qp.RZ(ub / 5, wires=0)
                    return counter + 1

                final = while_loop_inner(0)  # pylint: disable=no-value-for-parameter

                qp.RX(theta / final, wires=0)

            loop_outer()  # pylint: disable=no-value-for-parameter

        dev = qp.device(backend, wires=1)
        self.verify_catalyst_adjoint_against_pennylane(func, dev, jnp.pi, capture_mode=capture_mode)

    def test_adjoint_wires(self, backend):
        """Test the wires property of Adjoint"""

        @qjit
        @qp.qnode(qp.device(backend, wires=3))
        def circuit(theta):
            def func(theta):
                qp.RX(theta, wires=[0])
                qp.Hadamard(2)
                qp.CNOT([0, 2])

            qctrl = adjoint(func)(theta)
            return qctrl.wires

        # Without the `wires` property, returns `[-1]`
        assert circuit(0.3) == qp.wires.Wires([0, 2])

    def test_adjoint_wires_qubitunitary(self, backend):
        """Test the wires property of nested Adjoint with QubitUnitary"""

        @qjit
        @qp.qnode(qp.device(backend, wires=3))
        def circuit():
            def func():
                qp.QubitUnitary(
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

            qadj = adjoint(adjoint(adjoint(func)))()
            return qadj.wires

        # Without the `wires` property, returns `[-1]`
        assert circuit() == qp.wires.Wires([0, 1])

    @pytest.mark.xfail(reason="adjoint.wires is not supported with variable wires")
    def test_adjoint_var_wires(self, backend):
        """Test catalyst.adjoint.wires with variable wires."""

        device = qp.device(backend, wires=3)

        def func(w0, w1, theta):
            qp.RX(theta * np.pi / 2, wires=w0)
            qp.RY(theta / 2, wires=w1)
            qp.RZ(theta, wires=2)

        @qjit
        @qp.qnode(device)
        def C_workflow(w0, w1, theta):
            qp.PauliX(wires=0)
            cadj = adjoint(func)(w0, w1, theta)
            debug.print(cadj.wires)
            # <Wires =
            #    [Traced<ShapedArray(int64[], weak_type=True)>with<DynamicJaxprTrace(level=3/1)>,
            #     Traced<ShapedArray(int64[], weak_type=True)>with<DynamicJaxprTrace(level=3/1)>,
            #     2]>

            return qp.state()

        C_workflow(0, 1, 0.23)

    @pytest.mark.xfail(reason="adjoint.wires is not supported with control-flow branches")
    def test_adjoint_wires_controlflow(self, backend):
        """Test the wires property of Adjoint  in a conditional branch"""

        @qjit
        @qp.qnode(qp.device(backend, wires=3))
        def circuit():
            def func(pred, theta):
                @cond(pred)
                def cond_fn():
                    qp.RX(theta, wires=0)

                cond_fn()

            qadj = adjoint(func)(True, 3.14)
            return qadj.wires

        # It returns `-1` instead of `0`
        assert circuit() == qp.wires.Wires([0])

    def test_adjoint_ctrl_ctrl_subroutine(self, backend, capture_mode):
        """https://github.com/PennyLaneAI/catalyst/issues/589"""

        def subsubroutine():
            qp.ctrl(qp.PhaseShift, control=2)(0.1, wires=3)

        def subroutine():
            subsubroutine()
            qp.adjoint(qp.ctrl(subsubroutine, control=0))()

        dev = qp.device(backend, wires=4)

        @qp.set_shots(shots=500)
        @qp.qnode(dev)
        def circuit():
            qp.adjoint(subroutine)()
            return qp.probs(wires=dev.wires)

        expected = circuit()
        observed = qjit(circuit, capture=capture_mode)()
        assert_allclose(expected, observed)

    def test_adjoint_subroutine_with_classical_args(self, backend, capture_mode):
        """Test an adjoint on a subroutine, with classical arguments"""

        @qp.templates.Subroutine
        def f(x, wires):
            qp.IsingXX(x, wires)

        dev = qp.device(backend, wires=4)

        @qp.qnode(dev)
        def circuit():
            qp.adjoint(f)(0.5, (0, 1))
            return qp.probs(wires=0)

        expected = circuit()
        observed = qjit(circuit, capture=capture_mode)()
        assert_allclose(expected, observed)

    def test_adjoint_outside_qjit(self, backend):
        """Test that the hybrid adjoint can be used from outside qjit & qnode."""

        adj_op = adjoint(qp.RY(np.pi / 2, wires=0))

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit():
            qp.Hadamard(0)
            qp.apply(adj_op)
            return qp.probs()

        assert_allclose(circuit(), [1.0, 0.0], atol=1e-7)

    def test_adjoint_decomposition(self):
        """Test that the hybrid adjoint can be decomposed."""

        def qfunc(x):
            qp.RY(x, wires=0)
            qp.Hadamard(0)

        adj_op = catalyst.adjoint(qfunc)(0.7)
        decomp = adj_op.decomposition()

        assert len(decomp) == 2
        assert all(isinstance(op, qp.ops.op_math.Adjoint) for op in decomp)
        assert isinstance(decomp[0].base, qp.Hadamard)
        assert isinstance(decomp[1].base, qp.RY)
        assert decomp[1].base.data == (0.7,)


#####################################################################################
#### ADJOINT TEST SUITE COPIED OVER FROM PENNYLANE FOR UNIFIED BEHAVIOUR TESTING ####
#####################################################################################

# Notes:
# - instead of qp.Adjoint instantiation use catalyst.adjoint
# - remove Adjoint.id attribute checking from tests
# - update metadata size (1 -> 2)
# - do not pass base op as `base` keyword argument (related to point 1)
# - remove torch, tf, autograd tests
# - remove non-callable error message test (duplicates catalyst test)
# - change string wires to integers
# - remove pickle tetst


# pylint: disable=too-few-public-methods
class PlainOperator(qp.operation.Operator):
    """just an operator."""


class TestInheritanceMixins:
    """Test inheritance structure and mixin addition through dynamic __new__ method."""

    def test_plain_operator(self):
        """Test when base directly inherits from Operator, Adjoint only inherits
        from Adjoint and Operator."""

        base = PlainOperator(1.234, wires=0)
        op = adjoint(base)

        assert isinstance(op, Adjoint)
        assert isinstance(op, qp.operation.Operator)
        assert not isinstance(op, qp.operation.Operation)
        assert not isinstance(op, AdjointOperation)

        # checking we can call `dir` without problems
        assert "num_params" in dir(op)

    def test_operation(self):
        """When the operation inherits from `Operation`, the `AdjointOperation` mixin is
        added and the Adjoint has Operation functionality."""

        # pylint: disable=too-few-public-methods
        class CustomOp(qp.operation.Operation):
            num_wires = 1
            num_params = 1

        base = CustomOp(1.234, wires=0)
        op = adjoint(base)

        assert isinstance(op, Adjoint)
        assert isinstance(op, qp.operation.Operator)
        assert isinstance(op, qp.operation.Operation)
        assert isinstance(op, AdjointOperation)

        # check operation-specific properties made it into the mapping
        assert "grad_recipe" in dir(op)
        assert "control_wires" in dir(op)


class TestInitialization:
    """Test the initialization process and standard properties."""

    # pylint: disable=use-implicit-booleaness-not-comparison
    def test_nonparametric_ops(self):
        """Test adjoint initialization for a non parameteric operation."""
        base = qp.PauliX("a")

        op = adjoint(base)

        assert op.base is base
        assert op.hyperparameters["base"] is base
        assert op.name == "Adjoint(PauliX)"

        assert op.num_params == 0
        assert op.parameters == []
        assert op.data == ()

        assert op.wires == qp.wires.Wires("a")

    def test_parametric_ops(self):
        """Test adjoint initialization for a standard parametric operation."""
        params = [1.2345, 2.3456, 3.4567]
        base = qp.Rot(*params, wires="b")

        op = adjoint(base)

        assert op.base is base
        assert op.hyperparameters["base"] is base
        assert op.name == "Adjoint(Rot)"

        assert op.num_params == 3
        assert qp.math.allclose(params, op.parameters)
        assert qp.math.allclose(params, op.data)

        assert op.wires == qp.wires.Wires("b")

    def test_template_base(self):
        """Test adjoint initialization for a template."""
        rng = np.random.default_rng(seed=42)
        shape = qp.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
        params = rng.random(shape)

        base = qp.StronglyEntanglingLayers(params, wires=[0, 1])
        op = adjoint(base)

        assert op.base is base
        assert op.hyperparameters["base"] is base
        assert op.name == "Adjoint(StronglyEntanglingLayers)"

        assert op.num_params == 1
        assert qp.math.allclose(params, op.parameters[0])
        assert qp.math.allclose(params, op.data[0])

        assert op.wires == qp.wires.Wires((0, 1))

    def test_hamiltonian_base(self):
        """Test adjoint initialization for a hamiltonian."""
        base = 2.0 * qp.PauliX(0) @ qp.PauliY(1) + qp.PauliZ("b")

        op = adjoint(base)

        assert op.base is base
        assert op.hyperparameters["base"] is base
        assert op.name == "Adjoint(Sum)"

        assert op.num_params == 1
        assert qp.math.allclose(op.parameters, [2.0])
        assert qp.math.allclose(op.data, [2.0])

        assert op.wires == qp.wires.Wires([0, 1, "b"])


class TestProperties:
    """Test Adjoint properties."""

    def test_data(self):
        """Test base data can be get and set through Adjoint class."""
        x = np.array(1.234)

        base = qp.RX(x, wires="a")
        adj = adjoint(base)

        assert adj.data == (x,)

        # update parameters through adjoint
        x_new = np.array(2.3456)
        adj.data = (x_new,)
        assert base.data == (x_new,)
        assert adj.data == (x_new,)

        # update base data updates Adjoint data
        x_new2 = np.array(3.456)
        base.data = (x_new2,)
        assert adj.data == (x_new2,)

    def test_has_matrix_true(self):
        """Test `has_matrix` property carries over when base op defines matrix."""
        base = qp.PauliX(0)
        op = adjoint(base)

        assert op.has_matrix is True

    def test_has_matrix_false(self):
        """Test has_matrix property carries over when base op does not define a matrix."""
        base = qp.StatePrep([1, 0], wires=0)
        op = adjoint(base)

        assert op.has_matrix is False

    def test_has_decomposition_true_via_base_adjoint(self):
        """Test `has_decomposition` property is activated because the base operation defines an
        `adjoint` method."""
        base = qp.PauliX(0)
        op = adjoint(base)

        assert op.has_decomposition is True

    def test_has_decomposition_true_via_base_decomposition(self):
        """Test `has_decomposition` property is activated because the base operation defines a
        `decomposition` method."""

        # pylint: disable=too-few-public-methods
        class MyOp(qp.operation.Operation):
            num_wires = 1

            def decomposition(self):
                return [qp.RX(0.2, self.wires)]

        base = MyOp(0)
        op = adjoint(base)

        assert op.has_decomposition is True

    def test_has_decomposition_false(self):
        """Test `has_decomposition` property is not activated if the base neither
        `has_adjoint` nor `has_decomposition`."""

        # pylint: disable=too-few-public-methods
        class MyOp(qp.operation.Operation):
            num_wires = 1

        base = MyOp(0)
        op = adjoint(base)

        assert op.has_decomposition is False

    def test_has_adjoint_true_always(self):
        """Test `has_adjoint` property to always be true, irrespective of the base."""

        # pylint: disable=too-few-public-methods
        class MyOp(qp.operation.Operation):
            """Operation that does not define `adjoint` and hence has `has_adjoint=False`."""

            num_wires = 1

        base = MyOp(0)
        op = adjoint(base)

        assert op.has_adjoint is True
        assert op.base.has_adjoint is False

        base = qp.PauliX(0)
        op = adjoint(base)

        assert op.has_adjoint is True
        assert op.base.has_adjoint is True

    def test_has_diagonalizing_gates_true_via_base_diagonalizing_gates(self):
        """Test `has_diagonalizing_gates` property is activated because the
        base operation defines a `diagonalizing_gates` method."""

        op = adjoint(qp.PauliX(0))

        assert op.has_diagonalizing_gates is True

    def test_has_diagonalizing_gates_false(self):
        """Test `has_diagonalizing_gates` property is not activated if the base neither
        `has_adjoint` nor `has_diagonalizing_gates`."""

        # pylint: disable=too-few-public-methods
        class MyOp(qp.operation.Operation):
            num_wires = 1
            has_diagonalizing_gates = False

        op = adjoint(MyOp(0))

        assert op.has_diagonalizing_gates is False

    def test_queue_category(self):
        """Test that the queue category `"_ops"` carries over."""
        op = adjoint(qp.PauliX(0))
        assert op._queue_category == "_ops"  # pylint: disable=protected-access

    @pytest.mark.parametrize("value", (True, False))
    def test_is_verified_hermitian(self, value):
        """Test `is_verified_hermitian` property mirrors that of the base."""

        # pylint: disable=too-few-public-methods
        class DummyOp(qp.operation.Operator):
            num_wires = 1
            is_verified_hermitian = value

        op = adjoint(DummyOp(0))
        assert op.is_verified_hermitian == value

    def test_batching_properties(self):
        """Test the batching properties and methods."""

        base = qp.RX(np.array([1.2, 2.3, 3.4]), 0)
        op = adjoint(base)
        assert op.batch_size == 3
        assert op.ndim_params == (0,)


class TestSimplify:
    """Test Adjoint simplify method and depth property."""

    def test_depth_property(self):
        """Test depth property."""
        adj_op = adjoint(adjoint(qp.RZ(1.32, wires=0)))
        assert adj_op.arithmetic_depth == 2

    def test_simplify_method(self):
        """Test that the simplify method reduces complexity to the minimum."""
        adj_op = adjoint(adjoint(adjoint(qp.RZ(1.32, wires=0))))
        final_op = qp.RZ(4 * np.pi - 1.32, wires=0)
        simplified_op = adj_op.simplify()

        # TODO: Use qp.equal when supported for nested operators

        assert isinstance(simplified_op, qp.RZ)
        assert final_op.data == simplified_op.data
        assert final_op.wires == simplified_op.wires
        assert final_op.arithmetic_depth == simplified_op.arithmetic_depth

    def test_simplify_adj_of_sums(self):
        """Test that the simplify methods converts an adjoint of sums to a sum of adjoints."""
        adj_op = adjoint(qp.sum(qp.RX(1, 0), qp.RY(1, 0), qp.RZ(1, 0)))
        sum_op = qp.sum(qp.RX(4 * np.pi - 1, 0), qp.RY(4 * np.pi - 1, 0), qp.RZ(4 * np.pi - 1, 0))
        simplified_op = adj_op.simplify()

        # TODO: Use qp.equal when supported for nested operators

        assert isinstance(simplified_op, qp.ops.Sum)
        assert sum_op.data == simplified_op.data
        assert sum_op.wires == simplified_op.wires
        assert sum_op.arithmetic_depth == simplified_op.arithmetic_depth

        for s1, s2 in zip(sum_op.operands, simplified_op.operands):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_simplify_adj_of_prod(self):
        """Test that the simplify method converts an adjoint of products to a (reverse) product
        of adjoints."""
        adj_op = adjoint(qp.prod(qp.RX(1, 0), qp.RY(1, 0), qp.RZ(1, 0)))
        final_op = qp.prod(
            qp.RZ(4 * np.pi - 1, 0), qp.RY(4 * np.pi - 1, 0), qp.RX(4 * np.pi - 1, 0)
        )
        simplified_op = adj_op.simplify()

        assert isinstance(simplified_op, qp.ops.Prod)
        assert final_op.data == simplified_op.data
        assert final_op.wires == simplified_op.wires
        assert final_op.arithmetic_depth == simplified_op.arithmetic_depth

        for s1, s2 in zip(final_op.operands, simplified_op.operands):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_simplify_with_adjoint_not_defined(self):
        """Test the simplify method with an operator that has not defined the op.adjoint method."""
        op = adjoint(qp.T(0))
        simplified_op = op.simplify()
        assert isinstance(simplified_op, Adjoint)
        assert op.data == simplified_op.data
        assert op.wires == simplified_op.wires
        assert op.arithmetic_depth == simplified_op.arithmetic_depth


class TestMiscMethods:
    """Test miscellaneous small methods on the Adjoint class."""

    def test_repr(self):
        """Test __repr__ method."""
        assert repr(adjoint(qp.S(0))) == "Adjoint(S(0))"

        base = qp.S(0) + qp.T(0)
        op = adjoint(base)
        assert repr(op) == "Adjoint(S(0) + T(0))"

    def test_label(self):
        """Test that the label method for the adjoint class adds a † to the end."""
        base = qp.Rot(1.2345, 2.3456, 3.4567, wires="b")
        op = adjoint(base)
        assert op.label(decimals=2) == "Rot\n(1.23,\n2.35,\n3.46)†"

        base = qp.S(0) + qp.T(0)
        op = adjoint(base)
        assert op.label() == "𝓗†"

    def test_adjoint_of_adjoint(self):
        """Test that the adjoint of an adjoint is the original operation."""
        base = qp.PauliX(0)
        op = adjoint(base)

        assert op.adjoint() is base

    def test_diagonalizing_gates(self):
        """Assert that the diagonalizing gates method gives the base's diagonalizing gates."""
        base = qp.Hadamard(0)
        diag_gate = adjoint(base).diagonalizing_gates()[0]

        assert isinstance(diag_gate, qp.RY)
        assert qp.math.allclose(diag_gate.data[0], -np.pi / 4)

    # pylint: disable=protected-access
    def test_flatten_unflatten(self):
        """Test the flatten and unflatten methods."""

        # pylint: disable=too-few-public-methods
        class CustomOp(qp.operation.Operator):
            pass

        op = CustomOp(1.2, 2.3, wires=0)
        adj_op = adjoint(op)
        data, metadata = adj_op._flatten()
        assert len(data) == 1
        assert data[0] is op

        assert metadata == tuple()

        new_op = type(adj_op)._unflatten(*adj_op._flatten())
        assert qp.equal(adj_op, new_op)


class TestAdjointOperation:
    """Test methods in the AdjointOperation mixin."""

    def test_has_generator_true(self):
        """Test `has_generator` property carries over when base op defines generator."""
        base = qp.RX(0.5, 0)
        op = adjoint(base)

        assert op.has_generator is True

    def test_has_generator_false(self):
        """Test `has_generator` property carries over when base op does not define a generator."""
        base = qp.PauliX(0)
        op = adjoint(base)

        assert op.has_generator is False

    def test_generator(self):
        """Assert that the generator of an Adjoint is -1.0 times the base generator."""
        base = qp.RX(1.23, wires=0)
        op = adjoint(base)

        assert qp.equal(base.generator(), -1.0 * op.generator())

    def test_no_generator(self):
        """Test that an adjointed non-Operation raises a GeneratorUndefinedError."""

        with pytest.raises(qp.operation.GeneratorUndefinedError):
            adjoint(1.0 * qp.PauliX(0)).generator()

    def test_single_qubit_rot_angles(self):
        param = 1.234
        base = qp.RX(param, wires=0)
        op = adjoint(base)

        base_angles = base.single_qubit_rot_angles()
        angles = op.single_qubit_rot_angles()

        for angle1, angle2 in zip(angles, reversed(base_angles)):
            assert angle1 == -angle2

    @pytest.mark.parametrize(
        "base, basis",
        (
            (qp.RX(1.234, wires=0), "X"),
            (qp.PauliY("a"), "Y"),
            (qp.PhaseShift(4.56, wires="b"), "Z"),
            (qp.SX(-1), "X"),
        ),
    )
    def test_basis_property(self, base, basis):
        op = adjoint(base)
        assert op.basis == basis

    def test_control_wires(self):
        """Test the control_wires of an adjoint are the same as the base op."""
        op = adjoint(qp.CNOT(wires=("a", "b")))
        assert op.control_wires == qp.wires.Wires("a")


class TestAdjointOperationDiffInfo:
    """Test differention related properties and methods of AdjointOperation."""

    def test_grad_method_None(self):
        """Test grad_method copies base grad_method when it is None."""
        base = qp.PauliX(0)
        op = adjoint(base)

        assert op.grad_method is None

    @pytest.mark.parametrize("op", (qp.RX(1.2, wires=0),))
    def test_grad_method_not_None(self, op):
        """Make sure the grad_method property of a Adjoint op is the same as the base op."""
        assert adjoint(op).grad_method == op.grad_method

    @pytest.mark.parametrize(
        "base", (qp.PauliX(0), qp.RX(1.234, wires=0), qp.Rotation(1.234, wires=0))
    )
    def test_grad_recipe(self, base):
        """Test that the grad_recipe of the Adjoint is the same as the grad_recipe of the base."""
        assert adjoint(base).grad_recipe == base.grad_recipe

    @pytest.mark.parametrize(
        "base",
        (qp.RX(1.23, wires=0), qp.Rot(1.23, 2.345, 3.456, wires=0), qp.CRX(1.234, wires=(0, 1))),
    )
    def test_parameter_frequencies(self, base):
        """Test that the parameter frequencies of an Adjoint are the same as those of the base."""
        assert adjoint(base).parameter_frequencies == base.parameter_frequencies


class TestQueueing:
    """Test that Adjoint operators queue and update base metadata"""

    def test_queueing(self):
        """Test queuing and metadata when both Adjoint and base defined inside a recording
        context."""

        with qp.queuing.AnnotatedQueue() as q:
            base = qp.Rot(1.2345, 2.3456, 3.4567, wires="b")
            _ = adjoint(base)

        assert base not in q
        assert len(q) == 1

    def test_queuing_base_defined_outside(self):
        """Test that base isn't added to queue if it's defined outside the recording context."""

        base = qp.Rot(1.2345, 2.3456, 3.4567, wires="b")
        with qp.queuing.AnnotatedQueue() as q:
            op = adjoint(base)

        assert len(q) == 1
        assert q.queue[0] is op


class TestMatrix:
    """Test the matrix method for a variety of interfaces."""

    def test_batching_support(self):
        """Test that adjoint matrix has batching support."""
        x = qp.numpy.array([0.1, 0.2, 0.3])
        base = qp.RX(x, wires=0)
        op = adjoint(base)
        mat = op.matrix()
        compare = qp.RX(-x, wires=0)

        assert qp.math.allclose(mat, compare.matrix())
        assert mat.shape == (3, 2, 2)

    def check_matrix(self, x, interface):
        """Compares matrices in a interface independent manner."""
        base = qp.RX(x, wires=0)
        base_matrix = base.matrix()
        expected = qp.math.conj(qp.math.transpose(base_matrix))

        mat = adjoint(base).matrix()

        assert qp.math.allclose(expected, mat)
        assert qp.math.get_interface(mat) == interface

    def test_matrix_jax(self):
        """Test the matrix of an adjoint operator with a jax parameter."""

        self.check_matrix(jnp.array(1.2345), "jax")

    def test_no_matrix_defined(self):
        """Test that if the base has no matrix defined, then Adjoint.matrix also raises a
        MatrixUndefinedError."""
        rng = np.random.default_rng(seed=42)
        shape = qp.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
        params = rng.random(shape)

        base = qp.StronglyEntanglingLayers(params, wires=[0, 1])

        with pytest.raises(qp.operation.MatrixUndefinedError):
            adjoint(base).matrix()

    def test_adj_hamiltonian(self):
        """Test that a we can take the adjoint of a hamiltonian."""
        U = qp.Hamiltonian([1.0], [qp.PauliX(wires=0) @ qp.PauliZ(wires=1)])
        adj_op = adjoint(U)  # hamiltonian = hermitian = self-adjoint
        mat = adj_op.matrix()

        true_mat = qp.matrix(U)
        assert np.allclose(mat, true_mat)


def test_sparse_matrix():
    """Test that the spare_matrix method returns the adjoint of the base sparse matrix."""
    # pylint: disable=import-outside-toplevel
    from scipy.sparse import coo_matrix, csr_matrix

    H = np.array([[6 + 0j, 1 - 2j], [1 + 2j, -1]])
    H = csr_matrix(H)
    base = qp.SparseHamiltonian(H, wires=0)

    op = adjoint(base)

    base_sparse_mat = base.sparse_matrix()
    base_conj_T = qp.numpy.conj(qp.numpy.transpose(base_sparse_mat))
    op_sparse_mat = op.sparse_matrix()

    assert isinstance(op_sparse_mat, csr_matrix)
    assert isinstance(op.sparse_matrix(format="coo"), coo_matrix)

    assert qp.math.allclose(base_conj_T.toarray(), op_sparse_mat.toarray())


class TestEigvals:
    """Test the Adjoint class adjoint methods."""

    @pytest.mark.parametrize(
        "base", (qp.PauliX(0), qp.Hermitian(np.array([[6 + 0j, 1 - 2j], [1 + 2j, -1]]), wires=0))
    )
    def test_hermitian_eigvals(self, base):
        """Test adjoint's eigvals are the same as base eigvals when op is Hermitian."""
        base_eigvals = base.eigvals()
        adj_eigvals = adjoint(base).eigvals()
        assert qp.math.allclose(base_eigvals, adj_eigvals)

    def test_non_hermitian_eigvals(self):
        """Test that the Adjoint eigvals are the conjugate of the base's eigvals."""

        base = qp.SX(0)
        base_eigvals = base.eigvals()
        adj_eigvals = adjoint(base).eigvals()

        assert qp.math.allclose(qp.math.conj(base_eigvals), adj_eigvals)

    def test_batching_eigvals(self):
        """Test that eigenvalues work with batched parameters."""
        x = np.array([1.2, 2.3, 3.4])
        base = qp.RX(x, 0)
        adj = adjoint(base)
        compare = qp.RX(-x, 0)

        # eigvals might have different orders
        assert qp.math.allclose(adj.eigvals()[:, 0], compare.eigvals()[:, 1])
        assert qp.math.allclose(adj.eigvals()[:, 1], compare.eigvals()[:, 0])

    def test_no_matrix_defined_eigvals(self):
        """Test that if the base does not define eigvals, The Adjoint raises the same error."""
        base = qp.StatePrep([1, 0], wires=0)

        with pytest.raises(qp.operation.EigvalsUndefinedError):
            adjoint(base).eigvals()


class TestDecomposition:
    """Test the decomposition methods for the Adjoint class."""

    def test_decomp_custom_adjoint_defined(self):
        """Test decomposition method when a custom adjoint is defined."""
        decomp = adjoint(qp.Hadamard(0)).decomposition()
        assert len(decomp) == 1
        assert isinstance(decomp[0], qp.Hadamard)

    def test_decomp(self):
        """Test decomposition when base has decomposition but no custom adjoint."""
        base = qp.SX(0)
        base_decomp = base.decomposition()
        decomp = adjoint(base).decomposition()

        for adj_op, base_op in zip(decomp, reversed(base_decomp)):
            assert isinstance(adj_op, Adjoint)
            assert adj_op.base.__class__ == base_op.__class__
            assert qp.math.allclose(adj_op.data, base_op.data)

    def test_no_base_gate_decomposition(self):
        """Test that when the base gate doesn't have a decomposition, the Adjoint decomposition
        method raises the proper error."""
        nr_wires = 2
        rho = np.zeros((2**nr_wires, 2**nr_wires), dtype=np.complex128)
        rho[0, 0] = 1  # initialize the pure state density matrix for the |0><0| state
        base = qp.QubitDensityMatrix(rho, wires=(0, 1))

        with pytest.raises(qp.operation.DecompositionUndefinedError):
            adjoint(base).decomposition()

    def test_adjoint_of_adjoint(self):
        """Test that the adjoint an adjoint returns the base operator through both decomposition."""

        base = qp.PauliX(0)
        adj1 = adjoint(base)
        adj2 = adjoint(adj1)

        assert adj2.decomposition()[0] is base


class TestIntegration:
    """Test the integration of the Adjoint class with qnodes and gradients."""

    @pytest.mark.parametrize(
        "diff_method", ("parameter-shift", "finite-diff", "adjoint", "backprop")
    )
    def test_gradient_adj_rx(self, diff_method):
        @qp.qnode(qp.device("default.qubit", wires=1), diff_method=diff_method)
        def circuit(x):
            adjoint(qp.RX(x, wires=0))
            return qp.expval(qp.PauliY(0))

        x = pnp.array(1.2345, requires_grad=True)

        res = circuit(x)
        expected = np.sin(x)
        assert qp.math.allclose(res, expected)

        grad = qp.grad(circuit)(x)
        expected_grad = np.cos(x)

        assert qp.math.allclose(grad, expected_grad)

    def test_adj_batching(self):
        """Test execution of the adjoint of an operation with batched parameters."""
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circuit(x):
            adjoint(qp.RX(x, wires=0))
            return qp.expval(qp.PauliY(0))

        x = qp.numpy.array([1.234, 2.34, 3.456])
        res = circuit(x)

        expected = np.sin(x)
        assert qp.math.allclose(res, expected)


##### TESTS FOR THE ADJOINT CONSTRUCTOR ######

noncallable_objects = [
    [qp.Hadamard(1), qp.RX(-0.2, wires=1)],
    qp.tape.QuantumScript(),
]


class TestAdjointConstructorPreconstructedOp:
    """Test providing an already initalized operator to the transform."""

    @pytest.mark.parametrize("base", (qp.IsingXX(1.23, wires=("c", "d")), qp.QFT(wires=(0, 1, 2))))
    def test_single_op(self, base):
        """Test passing a single preconstructed op in a queuing context."""
        with qp.queuing.AnnotatedQueue() as q:
            base.queue()
            out = adjoint(base)

        assert len(q) == 1
        assert q.queue[0] is out

    def test_single_op_defined_outside_queue_eager(self):
        """Test if base is defined outside context and the function eagerly simplifies
        the adjoint, the base is not added to queue."""
        base = qp.RX(1.2, wires=0)
        with qp.queuing.AnnotatedQueue() as q:
            out = adjoint(base, lazy=False)

        assert len(q) == 1
        assert q.queue[0] is out

    def test_single_observable(self):
        """Test passing a single preconstructed observable in a queuing context."""

        with qp.queuing.AnnotatedQueue() as q:
            base = qp.Hermitian([[1, 0], [0, 1]], wires=0)
            out = adjoint(base)

        assert len(q) == 1
        assert q.queue[0] is out
        assert out.base is base
        assert isinstance(out, Adjoint)

        qs = qp.tape.QuantumScript.from_queue(q)
        assert len(qs) == 1


class TestAdjointConstructorDifferentCallableTypes:
    """Test the adjoint transform on a variety of possible inputs."""

    def test_adjoint_single_op_function(self):
        """Test the adjoint transform on a single operation."""

        with qp.queuing.AnnotatedQueue() as q:
            out = adjoint(qp.RX)(1.234, wires=0)

        tape = qp.tape.QuantumScript.from_queue(q)
        assert out is tape[0]
        assert isinstance(out, Adjoint)
        assert qp.equal(out.base, qp.RX(1.234, 0))

    def test_adjoint_template(self):
        """Test the adjoint transform on a template."""

        with qp.queuing.AnnotatedQueue() as q:
            out = adjoint(qp.QFT)(wires=(0, 1, 2))

        tape = qp.tape.QuantumScript.from_queue(q)
        assert len(tape) == 1
        assert out is tape[0]
        assert isinstance(out, Adjoint)
        assert out.base.__class__ is qp.QFT
        assert out.wires == qp.wires.Wires((0, 1, 2))

    @pytest.mark.skip(reason="Catalyst and PL are not unified in the qfunc case.")
    def test_adjoint_on_function(self):
        """Test adjoint transform on a function"""

        def func(x, y, z):
            qp.RX(x, wires=0)
            qp.RY(y, wires=0)
            qp.RZ(z, wires=0)

        x = 1.23
        y = 2.34
        z = 3.45
        with qp.queuing.AnnotatedQueue() as q:
            out = adjoint(func)(x, y, z)

        tape = qp.tape.QuantumScript.from_queue(q)
        assert out == tape.circuit

        for op in tape:
            assert isinstance(op, Adjoint)

        # check order reversed
        assert tape[0].base.__class__ is qp.RZ
        assert tape[1].base.__class__ is qp.RY
        assert tape[2].base.__class__ is qp.RX

        # check parameters assigned correctly
        assert tape[0].data == (z,)
        assert tape[1].data == (y,)
        assert tape[2].data == (x,)

    def test_nested_adjoint(self):
        """Test the adjoint transform on an adjoint transform."""
        x = 4.321
        with qp.queuing.AnnotatedQueue() as q:
            out = adjoint(adjoint(qp.RX))(x, wires=1)

        tape = qp.tape.QuantumScript.from_queue(q)
        assert out is tape[0]
        assert isinstance(out, Adjoint)
        assert isinstance(out.base, Adjoint)
        assert out.base.base.__class__ is qp.RX
        assert out.data == (x,)
        assert out.wires == qp.wires.Wires(1)


class TestAdjointConstructorNonLazyExecution:
    """Test the lazy=False keyword."""

    def test_single_decomposeable_op(self):
        """Test lazy=False for a single op that gets decomposed."""

        x = 1.23
        with qp.queuing.AnnotatedQueue() as q:
            base = qp.RX(x, wires=1)
            out = adjoint(base, lazy=False)

        assert len(q) == 1
        assert q.queue[0] is out

        assert isinstance(out, qp.RX)
        assert out.data == (-1.23,)

    def test_single_nondecomposable_op(self):
        """Test lazy=false for a single op that can't be decomposed."""
        with qp.queuing.AnnotatedQueue() as q:
            base = qp.S(0)
            out = adjoint(base, lazy=False)

        assert len(q) == 1
        assert q.queue[0] is out

        assert isinstance(out, Adjoint)
        assert isinstance(out.base, qp.S)

    def test_single_decomposable_op_function(self):
        """Test lazy=False for a single op callable that gets decomposed."""
        x = 1.23
        with qp.queuing.AnnotatedQueue() as q:
            out = adjoint(qp.RX, lazy=False)(x, wires=1)

        tape = qp.tape.QuantumScript.from_queue(q)
        assert out is tape[0]
        assert not isinstance(out, Adjoint)
        assert isinstance(out, qp.RX)
        assert out.data == (-x,)

    def test_single_nondecomposable_op_function(self):
        """Test lazy=False for a single op function that can't be decomposed."""
        with qp.queuing.AnnotatedQueue() as q:
            out = adjoint(qp.S, lazy=False)(0)

        tape = qp.tape.QuantumScript.from_queue(q)
        assert out is tape[0]
        assert isinstance(out, Adjoint)
        assert isinstance(out.base, qp.S)

    @pytest.mark.skip(reason="Catalyst and PL are not unified in the qfunc case.")
    def test_mixed_function(self):
        """Test lazy=False with a function that applies operations of both types."""
        x = 1.23

        def qfunc(x):
            qp.RZ(x, wires="b")
            qp.T("b")

        with qp.queuing.AnnotatedQueue() as q:
            out = adjoint(qfunc, lazy=False)(x)

        tape = qp.tape.QuantumScript.from_queue(q)
        assert len(tape) == len(out) == 2
        assert isinstance(tape[0], Adjoint)
        assert isinstance(tape[0].base, qp.T)

        assert isinstance(tape[1], qp.RZ)
        assert tape[1].data[0] == -x


class TestAdjointConstructorOutsideofQueuing:
    """Test the behaviour of the adjoint transform when not called in a queueing context."""

    def test_single_op(self):
        """Test providing a single op outside of a queuing context."""

        x = 1.234
        out = adjoint(qp.RZ(x, wires=0))

        assert isinstance(out, Adjoint)
        assert out.base.__class__ is qp.RZ
        assert out.data == (1.234,)
        assert out.wires == qp.wires.Wires(0)

    def test_single_op_eager(self):
        """Test a single op that can be decomposed in eager mode outside of a queuing context."""

        x = 1.234
        base = qp.RX(x, wires=0)
        out = adjoint(base, lazy=False)

        assert isinstance(out, qp.RX)
        assert out.data == (-x,)

    def test_single_op_function(self):
        """Test the transform on a single op as a callable outside of a queuing context."""
        x = 1.234
        out = adjoint(qp.IsingXX)(x, wires=(0, 1))

        assert isinstance(out, Adjoint)
        assert out.base.__class__ is qp.IsingXX
        assert out.data == (1.234,)
        assert out.wires == qp.wires.Wires((0, 1))

    @pytest.mark.skip(reason="Catalyst and PL are not unified in the qfunc case.")
    def test_function(self):
        """Test the transform on a function outside of a queuing context."""

        def func(wire):
            qp.S(wire)
            qp.SX(wire)

        wire = 1.234
        out = adjoint(func)(wire)

        assert len(out) == 2
        assert all(isinstance(op, Adjoint) for op in out)
        assert all(op.wires == qp.wires.Wires(wire) for op in out)

    def test_nonlazy_op_function(self):
        """Test non-lazy mode on a simplifiable op outside of a queuing context."""

        out = adjoint(qp.PauliX, lazy=False)(0)

        assert not isinstance(out, Adjoint)
        assert isinstance(out, qp.PauliX)


class TestAdjointConstructorIntegration:
    """Test circuit execution and gradients with the adjoint transform."""

    def test_single_op(self):
        """Test the adjoint of a single op against analytically expected results."""

        @qp.qnode(qp.device("default.qubit", wires=1))
        def circ():
            qp.PauliX(0)
            adjoint(qp.S)(0)
            return qp.state()

        res = circ()
        expected = np.array([0, -1j])

        assert np.allclose(res, expected)

    @pytest.mark.parametrize("diff_method", ("backprop", "adjoint", "parameter-shift"))
    def test_gradient_jax(self, diff_method):
        """Test gradients through the adjoint transform with jax."""

        @qp.qnode(qp.device("default.qubit", wires=1), diff_method=diff_method)
        def circ(x):
            adjoint(qp.RX)(x, wires=0)
            return qp.expval(qp.PauliY(0))

        x = jnp.array(0.234)
        expected_res = jnp.sin(x)
        expected_grad = jnp.cos(x)
        assert qp.math.allclose(circ(x), expected_res)
        assert qp.math.allclose(jax.grad(circ)(x), expected_grad)


class TestMidCircuitMeasurementAfterAdjoint:

    def test_issue_1055(self, backend):
        """See https://github.com/PennyLaneAI/catalyst/issues/1055"""

        def subroutine():
            qp.Hadamard(wires=1)

        @qjit
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            # Comment/uncomment to toggle bug
            adjoint(subroutine)()

            res = measure(0)

            # This call is just to show that it works after the measurement
            adjoint(subroutine)()

            return res

        assert not circuit()


class TestAdjointOfTemplates:
    """Test behaviour of adjoint around complex templates."""

    def test_adjoint_for_loop(self, backend):
        """Test operator adjoint works around templates that decompose into for loops."""

        @qp.qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(n: int):
            qp.H(0)

            @cat.for_loop(0, n, 1)
            def f(_):
                qp.T(0)

            f()
            qp.adjoint(f.operation)  # orig f is dequeued

            qp.H(0)
            return qp.expval(qp.Z(0))

        assert np.allclose(circuit(0), 1.0)
        assert np.allclose(circuit(1), 1.0 / np.sqrt(2))
        assert np.allclose(circuit(2), 0.0)

    def test_adjoint_while_loop(self, backend):
        """Test operator adjoint works around templates that decompose into for loops."""

        @qp.qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(n: int):
            qp.H(0)

            @cat.while_loop(lambda i: i < n)
            def f(i):
                qp.T(0)
                return i + 1

            f(0)
            qp.adjoint(f.operation)  # orig f is dequeued

            qp.H(0)
            return qp.expval(qp.Z(0))

        assert np.allclose(circuit(0), 1.0)
        assert np.allclose(circuit(1), 1.0 / np.sqrt(2))
        assert np.allclose(circuit(2), 0.0)

    def test_adjoint_cond(self, backend):
        """Test operator adjoint works around templates that decompose into if conditionals."""

        @qp.qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(b: bool):
            qp.H(0)

            @cat.cond(b)
            def f():
                qp.T(0)

            f()
            qp.adjoint(f.operation)  # orig f is dequeued

            qp.H(0)
            return qp.expval(qp.Z(0))

        assert np.allclose(circuit(False), 1.0)
        assert np.allclose(circuit(True), 1.0 / np.sqrt(2))

    def test_adjoint_switch(self, backend):
        """Test operator adjoint works on the switch operation."""

        @qp.qnode(qp.device(backend, wires=1))
        def circuit(s: int):

            @cat.switch(s)
            def f():
                qp.T(0)

            f()
            qp.adjoint(f.operation)  # orig f is dequeued

            return qp.expval(qp.Z(0))

        result = qjit(circuit)(0)
        assert np.isclose(result, 1.0)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
