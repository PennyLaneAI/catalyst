# Copyright 2024-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module tests the from_plxpr conversion function.
"""

import jax
import numpy as np
import pennylane as qml
import pytest
from pennylane.capture.primitives import adjoint_transform_prim, for_loop_prim, while_loop_prim

import catalyst
from catalyst.from_plxpr import from_plxpr
from catalyst.from_plxpr.qref_jax_primitives import (
    qref_alloc_p,
    qref_get_p,
    qref_qinst_p,
)
from catalyst.jax_primitives import (
    adjoint_p,
    qinsert_p,
)

pytestmark = pytest.mark.usefixtures("disable_capture")


def catalyst_execute_jaxpr(jaxpr):
    """Create a function capable of executing the provided catalyst-variant jaxpr."""

    # pylint: disable=arguments-differ, too-few-public-methods
    class JAXPRRunner(catalyst.QJIT):
        """A variant of catalyst.QJIT with a pre-constructed jaxpr."""

        # pylint: disable=missing-function-docstring
        def capture(self, args):

            result_treedef = jax.tree_util.tree_structure((0,) * len(jaxpr.out_avals))
            arg_signature = catalyst.tracing.type_signatures.get_abstract_signature(args)

            return jaxpr, None, result_treedef, arg_signature

    return JAXPRRunner(fn=lambda: None, compile_options=catalyst.CompileOptions())


@pytest.mark.usefixtures("use_capture")
class TestErrors:
    """Test that errors are raised in unsupported situations."""

    def test_measuring_eigvals_not_supported(self):
        """Test that a NotImplementedError is raised for converting a measurement
        specified via eigvals and wires."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.set_shots(50)
        @qml.qnode(dev)
        def circuit():
            return qml.measurements.SampleMP(
                wires=qml.wires.Wires((0, 1)), eigvals=np.array([-1.0, -1.0, 1.0, 1.0])
            )

        jaxpr = jax.make_jaxpr(circuit)()
        with pytest.raises(NotImplementedError, match="does not yet support measurements with"):
            from_plxpr(jaxpr)()

    def test_unsupported_measurement(self):
        """Test that a NotImplementedError is raised if a measurement
        is not yet supported for conversion."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return qml.vn_entropy(wires=0)

        jaxpr = jax.make_jaxpr(circuit)()

        with pytest.raises(NotImplementedError, match="not yet supported"):
            from_plxpr(jaxpr)()

    def test_no_shot_vectors(self):
        """Test that a NotImplementedError is raised with shot vectors."""

        dev = qml.device("lightning.qubit", wires=1)

        @qml.set_shots((10, 10, 20))
        @qml.qnode(dev)
        def c():
            return qml.sample(wires=0)

        jaxpr = jax.make_jaxpr(c)()

        with pytest.raises(NotImplementedError, match="not yet supported"):
            from_plxpr(jaxpr)()

    def test_errors_transform_inside_qnode(self):
        """Test that an error is raised if a transform is applied inside a transform."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        @qml.transforms.cancel_inverses
        def c():
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(c)()

        with pytest.raises(
            NotImplementedError, match="transforms cannot currently be applied inside a QNode."
        ):
            from_plxpr(jaxpr)()


class TestCatalystCompareJaxpr:
    """Test comparing catalyst and pennylane jaxpr for a variety of situations."""

    def test_qubit_unitary(self):
        """Test that qubit unitary can be converted."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(U):
            qml.QubitUnitary(U, wires=0)
            return qml.expval(qml.Z(0))

        x = qml.X.compute_matrix()
        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(x)
        converted = from_plxpr(plxpr)(x)
        qml.capture.disable()

        catalyst_res = catalyst_execute_jaxpr(converted)(x)
        assert len(catalyst_res) == 1
        assert qml.math.allclose(catalyst_res[0], -1)

    def test_globalphase(self):
        """Test conversion of a global phase."""

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(phi):
            qml.GlobalPhase(phi)
            return qml.state()

        phi = 0.5
        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(phi)
        converted = from_plxpr(plxpr)(phi)
        qml.capture.disable()
        catalyst_res = catalyst_execute_jaxpr(converted)(phi)
        assert qml.math.allclose(catalyst_res, np.exp(-0.5j) * np.array([1.0, 0.0]))

    def test_expval(self):
        """Test comparison and execution of the jaxpr for a simple qnode."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.Z(0))

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(0.5)
        converted = from_plxpr(plxpr)(0.5)
        qml.capture.disable()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.quantum_kernel_p
        assert converted.eqns[0].params["qnode"] is circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(0.5)
        assert len(catalyst_res) == 1
        assert qml.math.allclose(catalyst_res[0], jax.numpy.cos(0.5))

    def test_probs(self):
        """Test comparison and execution of a jaxpr containing a probability measurement."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.probs(wires=0)

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(0.5)

        converted = from_plxpr(plxpr)(0.5)
        qml.capture.disable()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.quantum_kernel_p
        assert converted.eqns[0].params["qnode"] is circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(0.5)
        assert len(catalyst_res) == 1
        expected = np.array([np.cos(0.5 / 2) ** 2, np.sin(0.5 / 2) ** 2])
        assert qml.math.allclose(catalyst_res[0], expected)

    def test_state(self):
        """Test that the state can be converted to catalxpr."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(phi):
            qml.Hadamard(0)
            qml.IsingXX(phi, wires=(0, 1))
            return qml.state()

        phi = np.array(-0.6234)

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(phi)

        converted = from_plxpr(plxpr)(phi)
        qml.capture.disable()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.quantum_kernel_p
        assert converted.eqns[0].params["qnode"] is circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(phi)
        assert len(catalyst_res) == 1

        x1 = np.cos(phi / 2) / np.sqrt(2)
        x2 = -1j * np.sin(phi / 2) / np.sqrt(2)
        expected = np.array([x1, x2, x1, x2])

        assert qml.math.allclose(catalyst_res[0], expected)

    def test_variance(self):
        """Test comparison and execution of a jaxpr containing a variance measurement."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.var(qml.Y(0))

        x = np.array(0.724)

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(x)

        converted = from_plxpr(plxpr)(np.array(0.724))
        qml.capture.disable()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.quantum_kernel_p
        assert converted.eqns[0].params["qnode"] is circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(x)
        assert len(catalyst_res) == 1
        expected = 1 - np.sin(x) ** 2
        assert qml.math.allclose(catalyst_res[0], expected)

    def test_sample(self):
        """Test comparison and execution of a jaxpr returning samples."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.set_shots(50)
        @qml.qnode(dev, mcm_method="single-branch-statistics")
        def circuit():
            qml.X(0)
            return qml.sample()

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)()

        converted = from_plxpr(plxpr)()
        qml.capture.disable()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.quantum_kernel_p
        assert converted.eqns[0].params["qnode"] is circuit

        catalyst_res = catalyst_execute_jaxpr(converted)()
        assert len(catalyst_res) == 1
        expected = np.transpose(np.vstack([np.ones(50), np.zeros(50)]))
        assert qml.math.allclose(catalyst_res[0], expected)

    def test_counts(self):
        """Test comparison and execution of a jaxpr returning counts."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.set_shots(50)
        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return qml.counts(all_outcomes=True)

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)()
        converted = from_plxpr(plxpr)()
        qml.capture.disable()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.quantum_kernel_p
        assert converted.eqns[0].params["qnode"] is circuit

        catalyst_res = catalyst_execute_jaxpr(converted)()
        assert len(catalyst_res) == 2
        expected_keys = np.array([0, 1, 2, 3])
        expected_values = np.array([0, 0, 50, 0])
        assert qml.math.allclose(catalyst_res[0], expected_keys)
        assert qml.math.allclose(catalyst_res[1], expected_values)

    def test_basis_state(self):
        """Test comparison and execution of a jaxpr containing BasisState."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(_basis_state):
            qml.BasisState(_basis_state, wires=[0, 1])
            return qml.state()

        basis_state = np.array([1, 1])
        expected_state_vector = np.array([0, 0, 0, 1], dtype=np.complex128)

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(basis_state)
        converted = from_plxpr(plxpr)(basis_state)
        qml.capture.disable()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.quantum_kernel_p
        assert converted.eqns[0].params["qnode"] is circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(basis_state)
        assert len(catalyst_res) == 1
        assert qml.math.allclose(catalyst_res[0], expected_state_vector)

    def test_state_prep(self):
        """Test comparison and execution of a jaxpr containing StatePrep."""
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(_init_state):
            # NOTE: Require validate_norm=False here otherwise Catalyst jaxpr contains
            # unused function that computes norm
            qml.StatePrep(_init_state, wires=0, validate_norm=False)
            return qml.state()

        init_state = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(init_state)
        converted = from_plxpr(plxpr)(init_state)
        qml.capture.disable()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.quantum_kernel_p
        assert converted.eqns[0].params["qnode"] is circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(init_state)
        assert len(catalyst_res) == 1
        assert qml.math.allclose(catalyst_res[0], init_state)

    def test_multiple_measurements(self):
        """Test that we can convert a circuit with multiple measurement returns."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y, z):
            qml.Rot(x, y, z, 0)
            return qml.expval(qml.X(0)), qml.expval(qml.Y(0)), qml.probs(wires=0)

        x, y, z = 0.9, 0.2, 0.5

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(x, y, z)

        converted = from_plxpr(plxpr)(x, y, z)
        qml.capture.disable()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.quantum_kernel_p
        assert converted.eqns[0].params["qnode"] is circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(x, y, z)
        assert len(catalyst_res) == 3

        a = np.cos(y / 2) * np.exp(-0.5j * (x + z))
        b = np.sin(y / 2) * np.exp(-0.5j * (x - z))
        state = np.array([a, b])
        expected_probs = np.abs(state) ** 2
        expected_expval_x = np.conj(state) @ qml.X.compute_matrix() @ state
        expected_expval_y = np.conj(state) @ qml.Y.compute_matrix() @ state
        assert qml.math.allclose(catalyst_res[0], expected_expval_x)
        assert qml.math.allclose(catalyst_res[1], expected_expval_y)
        assert qml.math.allclose(catalyst_res[2], expected_probs)

    def test_dynamic_shots(self):
        """Test that shots can be specified on qnode call."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.set_shots(50)
        @qml.qnode(dev)
        def circuit():
            return qml.sample(wires=0)

        def f():
            return qml.set_shots(circuit, shots=100)()

        qml.capture.enable()
        jaxpr = jax.make_jaxpr(f)()

        converted = from_plxpr(jaxpr)()
        qml.capture.disable()

        assert converted.out_avals[0].shape == (100, 1)
        [samples] = catalyst_execute_jaxpr(converted)()
        assert qml.math.allclose(samples, np.zeros((100, 1)))


class TestAdjointCtrl:
    """Test the conversion of adjoint and control operations."""

    @pytest.mark.parametrize("num_adjoints", (1, 2, 3))
    def test_adjoint_op(self, num_adjoints):
        """Test the conversion of a simple adjoint op."""
        qml.capture.enable()

        @qml.qnode(qml.device("lightning.qubit", wires=4))
        def c():
            op = qml.S(0)
            for _ in range(num_adjoints):
                op = qml.adjoint(op)

            return qml.state()

        plxpr = jax.make_jaxpr(c)()
        catalyst_xpr = from_plxpr(plxpr)()
        qfunc_xpr = catalyst_xpr.eqns[0].params["call_jaxpr"]

        assert qfunc_xpr.eqns[-5].primitive == qref_qinst_p
        assert qfunc_xpr.eqns[-5].params == {
            "adjoint": num_adjoints % 2 == 1,
            "ctrl_len": 0,
            "op": "S",
            "qubits_len": 1,
            "params_len": 0,
        }

    @pytest.mark.parametrize("inner_adjoint", (True, False))
    @pytest.mark.parametrize("outer_adjoint", (True, False))
    def test_ctrl_op(self, inner_adjoint, outer_adjoint):
        """Test the conversion of a simple adjoint op."""

        qml.capture.enable()

        @qml.qnode(qml.device("lightning.qubit", wires=4))
        def c(x, wire3):
            op = qml.RX(x, 0)
            if inner_adjoint:
                op = qml.adjoint(op)
            op = qml.ctrl(op, (1, 2, wire3), [0, 1, 0])
            if outer_adjoint:
                op = qml.adjoint(op)
            return qml.state()

        plxpr = jax.make_jaxpr(c)(0.5, 3)
        catalyst_xpr = from_plxpr(plxpr)(0.5, 3)

        qfunc_xpr = catalyst_xpr.eqns[0].params["call_jaxpr"]
        eqn = qfunc_xpr.eqns[6]  # dev, qreg, four allocations
        assert eqn.primitive == qref_qinst_p
        assert eqn.params == {
            "adjoint": (inner_adjoint + outer_adjoint) % 2 == 1,
            "ctrl_len": 3,
            "op": "RX",
            "qubits_len": 1,
            "params_len": 1,
        }
        assert eqn.invars[0] is qfunc_xpr.eqns[2].outvars[0]
        assert eqn.invars[1] is qfunc_xpr.invars[0]
        for i in range(3):
            assert eqn.invars[2 + i] is qfunc_xpr.eqns[3 + i].outvars[0]
        assert eqn.invars[5].val == False
        assert eqn.invars[6].val == True
        assert eqn.invars[7].val == False

    @pytest.mark.parametrize("as_qfunc", (True, False))
    def test_doubly_ctrl(self, as_qfunc):
        """Test doubly controlled op."""

        qml.capture.enable()

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def c():
            if as_qfunc:
                qml.ctrl(qml.ctrl(qml.S, 1), 2, control_values=[False])(0)
            else:
                qml.ctrl(qml.ctrl(qml.S(0), 1), 2, control_values=[False])
            return qml.state()

        plxpr = jax.make_jaxpr(c)()
        catalyst_xpr = from_plxpr(plxpr)()

        qfunc_xpr = catalyst_xpr.eqns[0].params["call_jaxpr"]
        eqn = qfunc_xpr.eqns[5]
        assert eqn.primitive == qref_qinst_p
        assert eqn.params == {
            "adjoint": False,
            "ctrl_len": 2,
            "op": "S",
            "qubits_len": 1,
            "params_len": 0,
        }

        for i in range(3):
            assert eqn.invars[i] == qfunc_xpr.eqns[2 + i].outvars[0]
        assert eqn.invars[3].val == False
        assert eqn.invars[4].val == True

    @pytest.mark.parametrize("with_return", (True, False))
    def test_adjoint_transform(self, with_return):
        """Test the adjoint transform."""

        qml.capture.enable()

        # pylint: disable=inconsistent-return-statements
        def f(x):
            op = qml.IsingXX(2 * x, wires=(0, 1))
            if with_return:
                return op

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def c(x):
            qml.X(0)
            qml.adjoint(f)(x)
            return qml.state()

        plxpr = jax.make_jaxpr(c)(0.5)
        catalyst_xpr = from_plxpr(plxpr)(0.5)
        qfunc_xpr = catalyst_xpr.eqns[0].params["call_jaxpr"]

        assert qfunc_xpr.eqns[1].primitive == qref_alloc_p
        assert qfunc_xpr.eqns[2].primitive == qref_get_p
        assert qfunc_xpr.eqns[3].primitive == qref_qinst_p

        eqn = qfunc_xpr.eqns[4]
        assert eqn.primitive == adjoint_transform_prim
        assert eqn.invars[0] == qfunc_xpr.eqns[1].outvars[0]  # the qreg, as a closure variable
        assert eqn.invars[1] == qfunc_xpr.invars[0]  # x
        assert len(eqn.outvars) == 0

        target_xpr = eqn.params["jaxpr"]
        assert target_xpr.eqns[1].primitive == qref_get_p
        assert target_xpr.eqns[2].primitive == qref_get_p
        assert target_xpr.eqns[3].primitive == qref_qinst_p
        assert target_xpr.eqns[3].params == {
            "adjoint": False,
            "ctrl_len": 0,
            "op": "IsingXX",
            "params_len": 1,
            "qubits_len": 2,
        }

    @pytest.mark.parametrize("as_qfunc", (True, False))
    def test_dynamic_control_wires(self, as_qfunc):
        """Test that dynamic wires are re-inserted if a dynamic wire is present."""

        qml.capture.enable()

        @qml.qnode(qml.device("lightning.qubit", wires=4))
        def c(wire):
            qml.CNOT((0, wire))
            if as_qfunc:
                qml.ctrl(qml.T, wire)(0)
            else:
                qml.ctrl(qml.T(0), wire)
            return qml.state()

        plxpr = jax.make_jaxpr(c)(3)
        catalyst_xpr = from_plxpr(plxpr)(3)

        qfunc_xpr = catalyst_xpr.eqns[0].params["call_jaxpr"]

        assert qfunc_xpr.eqns[2].primitive == qref_get_p
        assert qfunc_xpr.eqns[3].primitive == qref_get_p
        assert qfunc_xpr.eqns[4].primitive == qref_qinst_p
        assert qfunc_xpr.eqns[5].primitive == qref_get_p
        assert qfunc_xpr.eqns[6].primitive == qref_get_p

        assert qfunc_xpr.eqns[7].primitive == qref_qinst_p
        assert qfunc_xpr.eqns[7].params == {
            "adjoint": False,
            "ctrl_len": 1,
            "op": "T",
            "params_len": 0,
            "qubits_len": 1,
        }

    def test_ctrl_around_for_loop(self):
        """Test that ctrl applied to a for loop."""

        qml.capture.enable()

        @qml.for_loop(3)
        def g(i):
            qml.X(i)

        @qml.qnode(qml.device("lightning.qubit", wires=4))
        def c():
            qml.ctrl(g, [4, 5])()
            return qml.state()

        plxpr = jax.make_jaxpr(c)()
        catalyst_xpr = from_plxpr(plxpr)()

        qfunc_xpr = catalyst_xpr.eqns[0].params["call_jaxpr"]
        for_loop_xpr = qfunc_xpr.eqns[2].params["jaxpr_body_fn"]

        for i in [0, 1, 2]:
            assert for_loop_xpr.eqns[i].primitive == qref_get_p
        assert for_loop_xpr.eqns[3].primitive == qref_qinst_p
        assert for_loop_xpr.eqns[3].params == {
            "adjoint": False,
            "ctrl_len": 2,
            "op": "PauliX",
            "params_len": 0,
            "qubits_len": 1,
        }


class TestControlFlow:
    """Tests for for and while loops."""

    @pytest.mark.parametrize("reverse", (True, False))
    def test_for_loop_outside_qnode(self, reverse):
        """Test the conversion of a for loop outside the qnode."""

        qml.capture.enable()
        if reverse:
            start, stop, step = 6, 0, -2  # 6, 4, 2
        else:
            start, stop, step = 2, 7, 2  # 2, 4, 6

        def f(i0):
            @qml.for_loop(start, stop, step)
            def g(i, x):
                return i + x

            return g(i0)  # pylint: disable=no-value-for-parameter

        jaxpr = jax.make_jaxpr(f)(2)
        catalyst_jaxpr = from_plxpr(jaxpr)(2)

        eqn = catalyst_jaxpr.eqns[0]

        assert eqn.primitive == for_loop_prim
        assert eqn.params["abstract_shapes_slice"] == (0, 0, None)
        assert eqn.params["args_slice"] == (0, None, None)
        assert eqn.params["consts_slice"] == (0, 0, None)

        assert len(eqn.params["jaxpr_body_fn"].eqns) == 3 if reverse else 1

        if reverse:
            assert eqn.invars[0].val == 0
            assert eqn.invars[1].val == 3
            assert eqn.invars[2].val == 1
        else:
            assert eqn.invars[0].val == start
            assert eqn.invars[1].val == stop
            assert eqn.invars[2].val == step

    def test_while_loop_outside_qnode(self):
        """Test that a while loop outside a qnode can be translated."""
        qml.capture.enable()

        def f(x):

            y = jax.numpy.array([0, 1, 2])

            @qml.while_loop(lambda i: jax.numpy.sum(i) < 5 * jax.numpy.sum(y))
            def g(i):
                return i + y

            return g(x)

        x = jax.numpy.array([0, 0, 0])

        plxpr = jax.make_jaxpr(f)(x)
        catalyst_xpr = from_plxpr(plxpr)(x)

        eqn = catalyst_xpr.eqns[0]

        assert eqn.primitive == while_loop_prim
        assert eqn.params["args_slice"] == (2, None, None)
        assert eqn.params["body_slice"] == (0, 1, None)
        assert eqn.params["cond_slice"] == (1, 2, None)

        assert eqn.params["jaxpr_body_fn"].eqns[0].primitive.name == "add"
        assert eqn.params["jaxpr_cond_fn"].eqns[-1].primitive.name == "lt"


class TestHybridPrograms:
    """from_plxpr conversion tests for hybrid programs."""

    def test_pre_post_processing(self):
        """Test converting a workflow with pre and post processing."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, 0)
            qml.RY(3 * y + 1, 1)
            qml.CNOT((0, 1))
            return qml.expval(qml.X(1)), qml.expval(qml.Y(0))

        def workflow(z):
            a, b = circuit(z, 2 * z)
            return a + b

        qml.capture.enable()
        plxpr = jax.make_jaxpr(workflow)(0.5)

        converted = from_plxpr(plxpr)(0.5)
        qml.capture.disable()

        res = catalyst_execute_jaxpr(converted)(0.5)

        x = 0.5
        y = 3 * 2 * 0.5 + 1

        expval_x1 = np.sin(y)
        expval_y0 = -np.sin(x) * np.sin(y)
        expected = expval_x1 + expval_y0

        assert qml.math.allclose(expected, res[0])

    def test_multiple_qnodes(self):
        """Test that a workflow with multiple qnodes can be converted."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.Y(0))

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def g(y):
            qml.Hadamard(0)
            qml.IsingXX(y, wires=(0, 1))
            return qml.expval(qml.PauliZ(1))

        def workflow(x, y):
            return f(x) + g(y)

        qml.capture.enable()
        jaxpr = jax.make_jaxpr(workflow)(0.5, 1.2)
        catalxpr = from_plxpr(jaxpr)(0.5, 1.2)
        qml.capture.disable()
        results = catalyst_execute_jaxpr(catalxpr)(0.5, 1.2)

        expected = -np.sin(0.5) + np.cos(1.2)

        assert qml.math.allclose(results, expected)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
