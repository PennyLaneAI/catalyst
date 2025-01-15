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

"""Test quantum control decomposition in Catalyst."""

# pylint: disable=too-many-public-methods
# pylint: disable=protected-access
# pylint: disable=pointless-statement
# pylint: disable=expression-not-assigned
# pylint: disable=too-many-arguments
# pylint: disable=too-many-lines

import copy
from typing import Callable

import jax.numpy as jnp
import pennylane as qml
import pennylane.numpy as pnp
import pytest
from numpy.testing import assert_allclose
from pennylane import adjoint as PL_adjoint
from pennylane import ctrl as PL_ctrl
from pennylane.operation import DecompositionUndefinedError, Operation, Operator, Wires
from pennylane.ops.op_math.controlled import Controlled
from pennylane.tape import QuantumTape
from scipy import sparse

from catalyst import adjoint as C_adjoint
from catalyst import cond
from catalyst import ctrl as C_ctrl
from catalyst import for_loop, measure, qjit, while_loop
from catalyst.api_extensions.quantum_operators import HybridCtrl
from catalyst.jax_tracer import HybridOpRegion


def verify_catalyst_ctrl_against_pennylane(
    quantum_func: Callable, device, *args, with_adjoint_arg=False
):
    """
    A helper function for verifying Catalyst's native quantum control against the behaviour of
    PennyLane's quantum control function.
    """

    @qjit
    @qml.qnode(device)
    def catalyst_workflow(*args):
        if with_adjoint_arg:
            return quantum_func(*args, ctrl_fn=C_ctrl, adjoint_fn=C_adjoint)
        else:
            return quantum_func(*args, ctrl_fn=C_ctrl)

    @qml.qnode(device)
    def pennylane_workflow(*args):
        if with_adjoint_arg:
            return quantum_func(*args, ctrl_fn=PL_ctrl, adjoint_fn=PL_adjoint)
        else:
            return quantum_func(*args, ctrl_fn=PL_ctrl)

    assert_allclose(catalyst_workflow(*args), pennylane_workflow(*args), atol=1e-7)


class TestCatalystControlled:
    """Integration tests for Catalyst ctrl functionality."""

    def test_qctrl_op_object(self, backend):
        """Test the quantum control application to an operation object"""

        def circuit(theta, w, cw, ctrl_fn):
            ctrl_fn(qml.RX(theta, wires=[w]), control=[cw], control_values=[False])
            ctrl_fn(qml.RX, control=[cw], control_values=[False])(theta, wires=[w])
            return qml.state()

        verify_catalyst_ctrl_against_pennylane(circuit, qml.device(backend, wires=3), 0.1, 0, 1)

    def test_ctrl_invalid_argument(self):
        """Checks that ctrl rejects non-quantum program arguments."""

        with pytest.raises(ValueError, match="Expected a callable"):

            @qjit
            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def workflow():
                C_ctrl(0, control=1)(2)
                return qml.state()

            workflow()

    def test_qctrl_op_class(self, backend):
        """Test the quantum control application to a single operation class"""

        def circuit(theta, w, cw, ctrl_fn):
            ctrl_fn(qml.RX, control=[w], control_values=[True])(theta, wires=[cw])
            return qml.state()

        verify_catalyst_ctrl_against_pennylane(circuit, qml.device(backend, wires=3), 0.1, 0, 1)

    def test_qctrl_adjoint_func_simple(self, backend):
        """Test the quantum control distribution over the group of operations"""

        def circuit(arg, ctrl_fn, adjoint_fn):
            def _func(theta):
                qml.RX(theta, wires=[0])
                qml.RZ(theta, wires=2)

            ctrl_fn(adjoint_fn(_func), control=[1], control_values=[True])(arg)
            return qml.state()

        verify_catalyst_ctrl_against_pennylane(
            circuit, qml.device(backend, wires=3), 0.1, with_adjoint_arg=True
        )

    def test_adjoint_qctrl_func_simple(self, backend):
        """Test the quantum control distribution over the group of operations"""

        def circuit(arg, ctrl_fn, adjoint_fn):
            def _func(theta):
                qml.RX(theta, wires=[0])
                qml.RZ(theta, wires=2)

            adjoint_fn(ctrl_fn(_func, control=[1], control_values=[True]))(arg)
            return qml.state()

        verify_catalyst_ctrl_against_pennylane(
            circuit, qml.device(backend, wires=3), 0.1, with_adjoint_arg=True
        )

    @pytest.mark.xfail(
        reason="adjoint fails on quantum.unitary with 'operand #0 does not dominate this use'"
    )
    def test_qctrl_adjoint_hybrid(self, backend):
        """Test the quantum control distribution over the group of operations"""

        def circuit(theta, w2, cw, ctrl_fn, adjoint_fn):
            def _func():
                @while_loop(lambda s: s < w2)
                def _while_loop(s):
                    qml.RY(theta, wires=s)
                    return s + 1

                _while_loop(0)

            ctrl_fn(adjoint_fn(_func), control=[cw], control_values=[True])()
            return qml.state()

        verify_catalyst_ctrl_against_pennylane(
            circuit, qml.device(backend, wires=3), 0.1, 2, 2, with_adjoint_arg=True
        )

    def test_qctrl_func_simple(self, backend):
        """Test the quantum control distribution over the group of operations"""

        def circuit(arg, ctrl_fn):
            def _func(theta):
                qml.RX(theta, wires=[0])
                qml.RZ(theta, wires=2)

            ctrl_fn(_func, control=[1], control_values=[True])(arg)
            return qml.state()

        verify_catalyst_ctrl_against_pennylane(circuit, qml.device(backend, wires=3), 0.1)

    def test_qctrl_func_hybrid(self, backend):
        """Test the quantum control distribution over the Catalyst hybrid operation"""

        def circuit(theta, w1, w2, cw, ctrl_fn):
            def _func():
                qml.RX(theta, wires=[w1])

                s = 0

                @while_loop(lambda s: s < w2)
                def _while_loop(s):
                    qml.RY(theta, wires=s)
                    return s + 1

                s = _while_loop(s)

                @for_loop(0, w2, 1)
                def _for_loop(i, s):
                    qml.RY(theta, wires=i)
                    return s + 1

                s = _for_loop(s)

                @cond(True)
                def _branch():
                    qml.RZ(theta, wires=w2 - 1)
                    return 1

                @_branch.otherwise
                def _branch():
                    qml.RZ(theta, wires=w2 - 1)
                    return 0

                x = _branch()

                qml.RZ((s + x) * theta, wires=w1)

            ctrl_fn(_func, control=[cw], control_values=[True])()
            return qml.state()

        verify_catalyst_ctrl_against_pennylane(circuit, qml.device(backend, wires=3), 0.1, 0, 2, 2)

    def test_qctrl_func_nested(self, backend):
        """Test the quantum control distribution over the nested control operations"""

        def circuit(theta, w1, w2, cw1, cw2, ctrl_fn):
            def _func1():
                qml.RX(theta, wires=[w1])

                def _func2():
                    qml.RY(theta, wires=[w2])

                ctrl_fn(_func2, control=[cw2], control_values=[True])()

                qml.RZ(theta, wires=w1)

            ctrl_fn(_func1, control=[cw1], control_values=[True])()
            return qml.state()

        verify_catalyst_ctrl_against_pennylane(
            circuit, qml.device(backend, wires=4), 0.1, 0, 1, 2, 3
        )

    def test_qctrl_func_work_wires(self, backend):
        """Test the quantum control distribution over the nested control operations"""

        def circuit(theta, ctrl_fn):
            def _func1():
                qml.RX(theta, wires=[0])

                def _func2():
                    qml.RY(theta, wires=[0])

                ctrl_fn(_func2, control=[3], work_wires=[4])()

                qml.RZ(theta, wires=[0])

            ctrl_fn(_func1, control=[1], work_wires=[2])()
            return qml.state()

        verify_catalyst_ctrl_against_pennylane(circuit, qml.device(backend, wires=5), 0.1)

    def test_qctrl_valid_input_types(self, backend):
        """Test the quantum control input types"""

        def circuit(theta, w, cw, ctrl_fn):
            ctrl_fn(qml.RX(theta, wires=[w]), control=[cw])
            ctrl_fn(qml.RX(theta, wires=[w]), control=cw)
            ctrl_fn(qml.RX(theta, wires=[w]), control=[cw], control_values=[True])
            ctrl_fn(qml.RX(theta, wires=[w]), control=[cw], control_values=True)
            ctrl_fn(qml.RX(theta, wires=[w]), control=[cw], control_values=0)
            # FIXME: fails if work_wires is not None and other values are tracers
            # ctrl_fn(qml.RX(theta, wires=[0]), control=[1], work_wires=[2])
            return qml.state()

        verify_catalyst_ctrl_against_pennylane(circuit, qml.device(backend, wires=3), 0.1, 0, 1)

    def test_qctrl_raises_on_invalid_input(self, backend):
        """Test the no-measurements exception"""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit(theta):
            C_ctrl(qml.RX(theta, wires=[0]), control=[1], control_values=[])()
            return qml.state()

        with pytest.raises(ValueError, match="Length of the control_values"):
            qjit(circuit)(0.1)

    def test_qctrl_no_mid_circuit_measurements(self, backend):
        """Test the no-measurements exception"""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit(theta):
            def _func1():
                m = measure(0)
                qml.RX(m * theta, wires=[0])

            C_ctrl(_func1, control=[1], control_values=[True])()
            return qml.state()

        with pytest.raises(ValueError, match="Mid-circuit measurements cannot be used"):
            qjit(circuit)(0.1)

    def test_qctrl_no_end_circuit_measurements(self, backend):
        """Test the no-measurements exception"""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit(theta):
            def _func1():
                qml.RX(theta, wires=[0])
                return qml.state()

            C_ctrl(_func1, control=[1], control_values=[True])()
            return qml.state()

        with pytest.raises(ValueError, match="Measurement process cannot be used"):
            qjit(circuit)(0.1)

    def test_qctrl_wires(self, backend):
        """Test the wires property of HybridCtrl"""

        @qml.qjit
        @qml.qnode(qml.device(backend, wires=3))
        def circuit(theta):
            def func(theta):
                qml.RX(theta, wires=[0])
                qml.Hadamard(2)
                qml.CNOT([0, 2])

            qctrl = C_ctrl(func, control=[1])(theta)
            return qctrl.wires

        # Without the `wires` property, returns `[-1]`
        assert circuit(0.3) == qml.wires.Wires([1, 0, 2])

    def test_qctrl_wires_arg_fun(self, backend):
        """Test the wires property of HybridCtrl with argument wires"""

        @qml.qjit
        @qml.qnode(qml.device(backend, wires=4))
        def circuit():
            def func(anc, wires):
                qml.Hadamard(anc)
                h = pnp.array([[1, 1], [1, -1]]) / pnp.sqrt(2)
                qml.ctrl(qml.BlockEncode, control=anc)(h, wires=wires)
                qml.Hadamard(anc)

            qctrl = C_ctrl(func, control=[1])(0, [2, 3])
            return qctrl.wires

        assert circuit() == qml.wires.Wires([1, 0, 2, 3])

    def test_qctrl_var_wires(self, backend):
        """Test the wires property of HybridCtrl with variable wires"""

        @qml.qjit
        @qml.qnode(qml.device(backend, wires=4))
        def circuit(anc, wires):
            def func(anc, wires):
                qml.Hadamard(anc)
                h = pnp.array([[1, 1], [1, -1]]) / pnp.sqrt(2)
                qml.ctrl(qml.BlockEncode, control=anc)(h, wires=wires)
                qml.Hadamard(anc)

            qctrl = C_ctrl(func, control=[1])(anc, wires)
            return qctrl.wires

        assert circuit(0, [2, 3]) == qml.wires.Wires([1, 0, 2, 3])

    def test_qctrl_wires_nested(self, backend):
        """Test the wires property of HybridCtrl with nested branches"""

        @qml.qjit
        @qml.qnode(qml.device(backend, wires=4))
        def circuit(theta, w1, w2, cw1, cw2):
            def _func1():
                qml.RX(theta, wires=[w1])

                def _func2():
                    qml.RY(theta, wires=[w2])

                C_ctrl(_func2, control=[cw2], control_values=[True])()

                qml.RZ(theta, wires=w1)

            qctrl = C_ctrl(_func1, control=[cw1], control_values=[True])()
            return qctrl.wires

        assert circuit(0.1, 0, 1, 2, 3) == qml.wires.Wires([2, 0, 3, 1])

    def test_qctrl_work_wires(self, backend):
        """Test the wires property of HybridCtrl with work-wires"""

        @qml.qjit
        @qml.qnode(qml.device(backend, wires=5))
        def circuit(theta):
            def _func1():
                qml.RX(theta, wires=[0])

                def _func2():
                    qml.RY(theta, wires=[0])

                C_ctrl(_func2, control=[3], work_wires=[4])()

                qml.RZ(theta, wires=[0])

            qctrl = C_ctrl(_func1, control=[1], work_wires=[2])()
            return qctrl.wires

        assert circuit(0.1) == qml.wires.Wires([1, 0, 3])

    @pytest.mark.xfail(reason="ctrl.wires fails in control-flow branches is not supported")
    def test_qctrl_wires_controlflow(self, backend):
        """Test the wires property of HybridCtrl with control flow branches"""

        @qml.qjit
        @qml.qnode(qml.device(backend, wires=3))
        def circuit(theta, w1, w2, cw):
            def _func():
                qml.RX(theta, wires=[w1])
                s = 0

                @for_loop(0, w2, 1)
                def _for_loop(i, s):
                    qml.RY(theta, wires=i)
                    return s + 1

                s = _for_loop(s)
                qml.RZ(s * theta, wires=w1)

            qctrl = C_ctrl(_func, control=[cw], control_values=[True])()
            return qctrl.wires

        # It returns `[2, 0, -1]`
        assert circuit(0.1, 0, 2, 2) == qml.wires.Wires([2, 0, 1])

    def test_native_controlled_custom(self):
        """Test native control of a custom operation."""
        dev = qml.device("lightning.qubit", wires=4)

        @qml.qnode(dev)
        def native_controlled():
            qml.ctrl(qml.PauliZ(wires=[0]), control=[1, 2, 3])
            return qml.state()

        compiled = qjit()(native_controlled)
        assert all(sign in compiled.mlir for sign in ["ctrls", "ctrlvals"])
        result = compiled()
        expected = native_controlled()
        assert_allclose(result, expected, atol=1e-5, rtol=1e-5)

    def test_native_controlled_unitary(self):
        """Test native control of a custom operation."""
        dev = qml.device("lightning.qubit", wires=4)

        @qml.qnode(dev)
        def native_controlled():
            qml.ctrl(
                qml.QubitUnitary(
                    jnp.array(
                        [
                            [0.70710678 + 0.0j, 0.70710678 + 0.0j],
                            [0.70710678 + 0.0j, -0.70710678 + 0.0j],
                        ],
                        dtype=jnp.complex128,
                    ),
                    wires=[0],
                ),
                control=[1, 2, 3],
            )
            return qml.state()

        compiled = qjit()(native_controlled)
        result = compiled()
        expected = native_controlled()
        assert_allclose(result, expected, atol=1e-5, rtol=1e-5)

    def test_map_wires(self):
        """Test map wires."""

        X = HybridOpRegion(
            quantum_tape=QuantumTape([qml.X(wires=[1])], []),
            arg_classical_tracers=[],
            res_classical_tracers=[],
            trace=None,
        )
        qctrl = HybridCtrl([], [], [X], control_wires=[0])
        new_qctrl = qctrl.map_wires({1: 0, 0: 1})
        assert new_qctrl._control_wires == [1]  # pylint: disable=protected-access
        assert new_qctrl.regions[0].quantum_tape.operations[0].wires == Wires([0])

    def test_control_outside_qjit(self):
        """Test that the Catalyst control function can be used without jitting."""

        result = C_ctrl(qml.T(wires=0), control=[1, 2], control_values=[False, True], work_wires=3)
        expected = PL_ctrl(
            qml.T(wires=0), control=[1, 2], control_values=[False, True], work_wires=3
        )

        assert isinstance(result, type(expected))
        assert result.name == expected.name
        assert result.base == expected.base
        assert result.control_wires == expected.control_wires
        assert result.control_values == expected.control_values
        assert result.work_wires == expected.work_wires

    def test_control_decomp_trotter(self):
        """Test that the Catalyst control can safelt decompose TrotterProduct."""

        coeffs = [0.25, 0.75]
        ops = [qml.X(0), qml.Z(0)]
        H = qml.dot(coeffs, ops)

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.ControlledSequence(qml.TrotterProduct(H, time=2.4, order=2), control=[1])
            return qml.expval(qml.PauliZ(0))

        assert qml.math.allclose(qml.qjit(circuit)(), circuit())

    def test_distribute_controlled_with_adj(self):
        """Test that the distribute_controlled function with a PennyLane Adjoint,
        creates the equivalent Adjoint(Ctrl(base)) instead of Ctrl(Adj(base))"""

        # pylint: disable=import-outside-toplevel
        from catalyst.api_extensions.quantum_operators import ctrl_distribute

        tape = qml.tape.QuantumScript([qml.ops.Adjoint(qml.RX(1.2, 0)), qml.Hadamard(1)])

        new_ops = ctrl_distribute(tape, control_wires=[2, 3], control_values=[True, True])

        assert new_ops[0] == qml.ops.Adjoint(Controlled(qml.RX(1.2, 0), control_wires=[2, 3]))
        assert new_ops[1] == Controlled(qml.Hadamard(1), control_wires=[2, 3])


########################################################################################
#### Controlled TEST SUITE COPIED OVER FROM PENNYLANE FOR UNIFIED BEHAVIOUR TESTING ####
########################################################################################

# Notes:
# - instead of qml.Controlled and qml.ControlledOp instantiation use catalyst.ctrl
# - remove Controlled.id attribute checking from tests
# - update metadata size (1 -> 2)
# - remove hash(metadata) as `HybridOp` is not hashable
# - remove torch, tf, autograd, and custom decompostion tests
# - remove non-callable error message test (duplicates catalyst test)


class TempOperator(Operator):
    """A custom operator."""

    num_wires = 1


class TempOperation(Operation):
    """A custom operation."""

    num_wires = 1


class OpWithDecomposition(Operation):
    """A custom operation with a decomposition method."""

    @staticmethod
    def compute_decomposition(*params, wires=None, **_):
        return [
            qml.Hadamard(wires=wires[0]),
            qml.S(wires=wires[1]),
            qml.RX(params[0], wires=wires[0]),
        ]


class TestControlledInit:
    """Test the initialization process and standard properties."""

    temp_op = TempOperator("a")

    def test_nonparametric_ops(self):
        """Test pow initialization for a non parameteric operation."""

        op = C_ctrl(
            self.temp_op,
            (0, 1),
            control_values=[True, False],
            work_wires="aux",
        )

        assert op.base is self.temp_op
        assert op.hyperparameters["base"] is self.temp_op

        # In C_ctrl, wires include the list of all wires
        assert op.wires == Wires((0, 1, "a"))

        assert op.control_wires == Wires((0, 1))
        assert op.hyperparameters["control_wires"] == Wires((0, 1))

        assert op.target_wires == Wires("a")

        assert op.control_values == [True, False]
        assert op.hyperparameters["control_values"] == [True, False]

        assert op.work_wires == Wires(("aux"))

        assert op.name == "C(TempOperator)"

        assert op.num_params == 0
        assert not op.parameters
        assert not op.data

        assert op.num_wires == 3

    def test_default_control_values(self):
        """Test assignment of default control_values."""
        op = C_ctrl(self.temp_op, (0, 1))
        assert op.control_values == [True, True]

    def test_zero_one_control_values(self):
        """Test assignment of provided control_values."""
        op = C_ctrl(self.temp_op, (0, 1), control_values=[0, 1])
        assert op.control_values == [False, True]

    @pytest.mark.parametrize("control_values", [True, False, 0, 1])
    def test_scalar_control_values(self, control_values):
        """Test assignment of provided control_values."""
        op = Controlled(self.temp_op, 0, control_values=control_values)
        assert op.control_values == [control_values]

    def test_tuple_control_values(self):
        """Test assignment of provided control_values."""
        op = C_ctrl(self.temp_op, (0, 1), control_values=(0, 1))
        assert op.control_values == [False, True]

    def test_non_boolean_control_values(self):
        """Test control values are converted to booleans."""
        op = C_ctrl(self.temp_op, (0, 1, 2), control_values=["", None, 5])
        assert op.control_values == [False, False, True]

    def test_control_values_wrong_length(self):
        """Test checking control_values length error."""
        with pytest.raises(ValueError, match="Length of the control_values"):
            C_ctrl(self.temp_op, (0, 1), [True])

    def test_target_control_wires_overlap(self):
        """Test checking overlap of target wires and control_wires"""
        with pytest.raises(ValueError, match="The control wires must be different"):
            C_ctrl(self.temp_op, "a")

    def test_work_wires_overlap_target(self):
        """Test checking work wires are not in target wires."""
        with pytest.raises(ValueError, match="Work wires must be different"):
            C_ctrl(self.temp_op, "b", work_wires="a")

    def test_work_wires_overlap_control(self):
        """Test checking work wires are not in contorl wires."""
        with pytest.raises(ValueError, match="Work wires must be different."):
            C_ctrl(self.temp_op, control="b", work_wires="b")


class TestControlledProperties:
    """Test the properties of the `catalyst.ctrl` symbolic operator."""

    def test_data(self):
        """Test that the base data can be get and set through HybridCtrl class."""

        x = pnp.array(1.234)

        base = qml.RX(x, wires="a")
        op = C_ctrl(base, (0, 1))

        assert op.data == (x,)

        x_new = (pnp.array(2.3454),)
        op.data = x_new
        assert op.data == (x_new,)
        assert base.data == (x_new,)

        x_new2 = (pnp.array(3.456),)
        base.data = x_new2
        assert op.data == (x_new2,)
        assert op.parameters == [x_new2]

    @pytest.mark.parametrize(
        "val, arr", ((4, [1, 0, 0]), (6, [1, 1, 0]), (1, [0, 0, 1]), (5, [1, 0, 1]))
    )
    def test_control_int(self, val, arr):
        """Test private `_control_int` property converts control_values to integer
        representation."""

        op = C_ctrl(TempOperator(5), (0, 1, 2), control_values=arr)
        assert op._control_int == val

    @pytest.mark.parametrize("value", (True, False))
    def test_has_matrix(self, value):
        """Test that `catalyst.ctrl` defers has_matrix to base operator."""

        class DummyOp(Operator):
            """DummyOp"""

            num_wires = 1
            has_matrix = value

        op = C_ctrl(DummyOp(1), 0)
        assert op.has_matrix is value

    @pytest.mark.parametrize(
        "base", (qml.RX(1.23, 0), qml.Rot(1.2, 2.3, 3.4, 0), qml.QubitUnitary([[0, 1], [1, 0]], 0))
    )
    def test_ndim_params(self, base):
        """Test that `catalyst.ctrl` defers to base ndim_params"""

        op = C_ctrl(base, 1)
        assert op.ndim_params == base.ndim_params

    @pytest.mark.parametrize("cwires, cvalues", [(0, [0]), ([3, 0, 2], [1, 1, 0])])
    def test_has_decomposition_true_via_control_values(self, cwires, cvalues):
        """Test that `catalyst.ctrl` claims `has_decomposition` to be true if there are
        any negated control values."""

        op = C_ctrl(TempOperation(0.2, wires=1), cwires, cvalues)
        assert op.has_decomposition is True

    def test_has_decomposition_true_via_base_has_ctrl_single_cwire(self):
        """Test that `catalyst.ctrl` claims `has_decomposition` to be true if
        only one control wire is used and the base has a `_controlled` method."""

        op = C_ctrl(qml.RX(0.2, wires=1), 4)
        assert op.has_decomposition is True

    def test_has_decomposition_true_via_pauli_x(self):
        """Test that `catalyst.ctrl` claims `has_decomposition` to be true if
        the base is a `PauliX` operator"""

        op = C_ctrl(qml.PauliX(3), [0, 4])
        assert op.has_decomposition is True

    def test_has_decomposition_multicontrolled_special_unitary(self):
        """Test that a one qubit special unitary with any number of control
        wires has a decomposition."""
        op = C_ctrl(qml.RX(1.234, wires=0), (1, 2, 3, 4, 5))
        assert op.has_decomposition

    def test_has_decomposition_true_via_base_has_decomp(self):
        """Test that `catalyst.ctrl` claims `has_decomposition` to be true if
        the base has a decomposition and indicates this via `has_decomposition`."""

        op = C_ctrl(qml.IsingXX(0.6, [1, 3]), [0, 4])
        assert op.has_decomposition is True

    def test_has_decomposition_false_single_cwire(self):
        """Test that `catalyst.ctrl` claims `has_decomposition` to be false if
        no path of decomposition would work, here we use a single control wire."""

        # all control values are 1, there is only one control wire but TempOperator does
        # not have `_controlled`, is not `PauliX`, doesn't have a ZYZ decomposition,
        # and reports `has_decomposition=False`
        op = C_ctrl(TempOperator(0.5, 1), 0)
        assert op.has_decomposition is False

    def test_has_decomposition_false_multi_cwire(self):
        """Test that `catalyst.ctrl` claims `has_decomposition` to be false if
        no path of decomposition would work, here we use multiple control wires."""

        # all control values are 1, there are multiple control wires,
        # `TempOperator` is not `PauliX`, and reports `has_decomposition=False`
        op = C_ctrl(TempOperator(0.5, 1), [0, 5])
        assert op.has_decomposition is False

    @pytest.mark.parametrize("value", (True, False))
    def test_has_adjoint(self, value):
        """Test that `catalyst.ctrl` defers has_adjoint to base operator."""

        class DummyOp(Operator):
            """DummyOp"""

            num_wires = 1
            has_adjoint = value

        op = C_ctrl(DummyOp(1), 0)
        assert op.has_adjoint is value

    @pytest.mark.parametrize("value", (True, False))
    def test_has_diagonalizing_gates(self, value):
        """Test that `catalyst.ctrl` defers has_diagonalizing_gates to base operator."""

        class DummyOp(Operator):
            """DummyOp"""

            num_wires = 1
            has_diagonalizing_gates = value

        op = C_ctrl(DummyOp(1), 0)
        assert op.has_diagonalizing_gates is value

    @pytest.mark.parametrize("value", ("_ops", None))
    def test_queue_cateogry(self, value):
        """Test that `catalyst.ctrl` defers `_queue_category` to base operator."""

        class DummyOp(Operator):
            """DummyOp"""

            num_wires = 1
            _queue_category = value

        op = C_ctrl(DummyOp(1), 0)
        assert op._queue_category == value

    @pytest.mark.parametrize("value", (True, False))
    def test_is_hermitian(self, value):
        """Test that `catalyst.ctrl` defers `is_hermitian` to base operator."""

        class DummyOp(Operator):
            """DummyOp"""

            num_wires = 1
            is_hermitian = value

        op = C_ctrl(DummyOp(1), 0)
        assert op.is_hermitian is value

    def test_map_wires(self):
        """Test that we can get and set private wires."""

        base = qml.IsingXX(1.234, wires=(0, 1))
        op = C_ctrl(base, (3, 4), work_wires="aux")

        assert op.wires == Wires((3, 4, 0, 1))

        op = op.map_wires(wire_map={3: "a", 4: "b", 0: "c", 1: "d", "aux": "extra"})

        assert op.base.wires == Wires(("c", "d"))
        assert op.control_wires == Wires(("a", "b"))
        assert op.work_wires == Wires(("extra"))


class TestControlledMiscMethods:
    """Test miscellaneous minor catalyst.ctrl methods."""

    def test_repr(self):
        """Test __repr__ method."""
        assert repr(C_ctrl(qml.S(0), [1])) == "Controlled(S(0), control_wires=[1])"

        base = qml.S(0) + qml.T(1)
        op = C_ctrl(base, [2])
        assert repr(op) == "Controlled(S(0) + T(1), control_wires=[2])"

        op = C_ctrl(base, [2, 3], control_values=[True, False], work_wires=[4])
        assert (
            repr(op) == "Controlled(S(0) + T(1), control_wires=[2, 3], work_wires=[4],"
            " control_values=[True, False])"
        )

    def test_flatten_unflatten(self):
        """Tests the _flatten and _unflatten methods."""
        target = qml.S(0)
        control_wires = qml.wires.Wires((1, 2))
        control_values = (False, False)  # (0, 0)
        work_wires = qml.wires.Wires(3)

        op = C_ctrl(target, control_wires, control_values=control_values, work_wires=work_wires)

        data, metadata = op._flatten()
        assert data[0] is target
        assert len(data) == 1

        assert len(metadata) == 3
        assert metadata[0] == control_wires
        assert metadata[1] == control_values
        assert metadata[2] == work_wires

        assert hash(metadata)

        new_op = type(op)._unflatten(*op._flatten())
        assert qml.equal(op, new_op)
        assert new_op._name == "C(S)"  # make sure initialization was called

    def test_copy(self):
        """Test that a copy of a controlled oeprator can have its parameters updated
        independently of the original operator."""

        param1 = 1.234
        base_wire = "a"
        control_wires = [0, 1]
        base = qml.RX(param1, base_wire)
        op = C_ctrl(base, control_wires, control_values=[0, 1])

        copied_op = copy.copy(op)

        assert copied_op.__class__ is op.__class__
        assert copied_op.control_wires == op.control_wires
        assert copied_op.control_values == op.control_values
        assert copied_op.data == (param1,)

        copied_op.data = (6.54,)
        assert op.data == (param1,)

    def test_label(self):
        """Test that the label method defers to the label of the base."""
        base = qml.U1(1.23, wires=0)
        op = C_ctrl(base, "a")

        assert op.label() == base.label()
        assert op.label(decimals=2) == base.label(decimals=2)
        assert op.label(base_label="hi") == base.label(base_label="hi")

    def test_label_matrix_param(self):
        """Test that the label method simply returns the label of the base and updates the cache."""
        U = pnp.eye(2)
        base = qml.QubitUnitary(U, wires=0)
        op = C_ctrl(base, ["a", "b"])

        cache = {"matrices": []}
        assert op.label(cache=cache) == base.label(cache=cache)
        assert cache["matrices"] == [U]

    def test_eigvals(self):
        """Test the eigenvalues against the matrix eigenvalues."""
        base = qml.IsingXX(1.234, wires=(0, 1))
        op = C_ctrl(base, (2, 3))

        mat = op.matrix()
        mat_eigvals = pnp.sort(qml.math.linalg.eigvals(mat))

        eigs = op.eigvals()
        sort_eigs = pnp.sort(eigs)

        assert qml.math.allclose(mat_eigvals, sort_eigs)

    def test_has_generator_true(self):
        """Test `has_generator` property carries over when base op defines generator."""
        base = qml.RX(0.5, 0)
        op = C_ctrl(base, ("b", "c"))

        assert op.has_generator is True

    def test_has_generator_false(self):
        """Test `has_generator` property carries over when base op does not define a generator."""
        base = qml.PauliX(0)
        op = C_ctrl(base, ("b", "c"))

        assert op.has_generator is False

    def test_generator(self):
        """Test that the generator is a tensor product of projectors and the base's generator."""

        base = qml.RZ(-0.123, wires="a")
        control_values = [0, 1]
        op = C_ctrl(base, ("b", "c"), control_values=control_values)

        base_gen, base_gen_coeff = qml.generator(base, format="prefactor")
        gen_tensor, gen_coeff = qml.generator(op, format="prefactor")

        assert base_gen_coeff == gen_coeff

        for wire, val in zip(op.control_wires, control_values):
            ob = list(op for op in gen_tensor.operands if op.wires == qml.wires.Wires(wire))
            assert len(ob) == 1
            assert ob[0].data == ([val],)

        ob = list(op for op in gen_tensor.operands if op.wires == base.wires)
        assert len(ob) == 1
        assert ob[0].__class__ is base_gen.__class__

        expected = qml.exp(op.generator(), 1j * op.data[0])
        assert qml.math.allclose(
            expected.matrix(wire_order=["a", "b", "c"]), op.matrix(wire_order=["a", "b", "c"])
        )

    def test_diagonalizing_gates(self):
        """Test that the Controlled diagonalizing gates is the same as the base
        diagonalizing gates."""

        base = qml.PauliX(0)
        op = C_ctrl(base, (1, 2))

        op_gates = op.diagonalizing_gates()
        base_gates = base.diagonalizing_gates()

        assert len(op_gates) == len(base_gates)

        for op1, op2 in zip(op_gates, base_gates):
            assert op1.__class__ is op2.__class__
            assert op1.wires == op2.wires

    def test_hash(self):
        """Test that op.hash uniquely describes an op up to work wires."""

        base = qml.RY(1.2, wires=0)
        # different control wires
        op1 = C_ctrl(base, (1, 2), [0, 1])
        op2 = C_ctrl(base, (2, 1), [0, 1])
        assert op1.hash != op2.hash

        # different control values
        op3 = C_ctrl(base, (1, 2), [1, 0])
        assert op1.hash != op3.hash
        assert op2.hash != op3.hash

        # all variations on default control_values
        op4 = C_ctrl(base, (1, 2))
        op5 = C_ctrl(base, (1, 2), [True, True])
        op6 = C_ctrl(base, (1, 2), [1, 1])
        assert op4.hash == op5.hash
        assert op4.hash == op6.hash

        # work wires
        op7 = C_ctrl(base, (1, 2), [0, 1], work_wires="aux")
        assert op7.hash != op1.hash


class TestControlledOperationProperties:
    """Test Controlled specific properties."""

    # pylint:disable=no-member

    @pytest.mark.parametrize("gm", (None, "A", "F"))
    def test_grad_method(self, gm):
        """Check grad_method defers to that of the base operation."""

        class DummyOp(Operation):
            """DummyOp"""

            num_wires = 1
            grad_method = gm

        base = DummyOp(1)
        op = C_ctrl(base, 2)
        assert op.grad_method == gm

    def test_basis(self):
        """Test that controlled mimics the basis attribute of the base op."""

        class DummyOp(Operation):
            """DummyOp"""

            num_wires = 1
            basis = "Z"

        base = DummyOp(1)
        op = C_ctrl(base, 2)
        assert op.basis == "Z"

    @pytest.mark.parametrize(
        "base, expected",
        [
            (qml.RX(1.23, wires=0), [(0.5, 1.0)]),
            (qml.PhaseShift(-2.4, wires=0), [(1,)]),
            (qml.IsingZZ(-9.87, (0, 1)), [(0.5, 1.0)]),
            (qml.DoubleExcitationMinus(0.7, [0, 1, 2, 3]), [(0.5, 1.0)]),
        ],
    )
    def test_parameter_frequencies(self, base, expected):
        """Test parameter-frequencies against expected values."""

        op = C_ctrl(base, (4, 5))
        assert op.parameter_frequencies == expected

    def test_parameter_frequencies_no_generator_error(self):
        """An error should be raised if the base doesn't have a generator."""
        base = TempOperation(1.234, 1)
        op = C_ctrl(base, 2)

        with pytest.raises(
            qml.operation.ParameterFrequenciesUndefinedError,
            match=r"does not have parameter frequencies",
        ):
            op.parameter_frequencies

    def test_parameter_frequencies_multiple_params_error(self):
        """An error should be raised if the base has more than one parameter."""
        base = TempOperation(1.23, 2.234, 1)
        op = C_ctrl(base, (2, 3))

        with pytest.raises(
            qml.operation.ParameterFrequenciesUndefinedError,
            match=r"does not have parameter frequencies",
        ):
            op.parameter_frequencies


class TestControlledSimplify:
    """Test qml.sum simplify method and depth property."""

    def test_depth_property(self):
        """Test depth property."""
        controlled_op = C_ctrl(qml.RZ(1.32, wires=0) + qml.Identity(wires=0), control=1)
        assert controlled_op.arithmetic_depth == 2

    def test_simplify_method(self):
        """Test that the simplify method reduces complexity to the minimum."""
        controlled_op = C_ctrl(
            qml.RZ(1.32, wires=0) + qml.Identity(wires=0) + qml.RX(1.9, wires=1), control=2
        )
        final_op = C_ctrl(
            qml.sum(qml.RZ(1.32, wires=0), qml.Identity(wires=0), qml.RX(1.9, wires=1)),
            control=2,
        )
        simplified_op = controlled_op.simplify()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(simplified_op, Controlled)
        for s1, s2 in zip(final_op.base.operands, simplified_op.base.operands):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_simplify_nested_controlled_ops(self):
        """Test the simplify method with nested control operations on different wires."""
        controlled_op = C_ctrl(C_ctrl(qml.Hadamard(0), 1), 2)
        final_op = C_ctrl(qml.Hadamard(0), [2, 1])
        simplified_op = controlled_op.simplify()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(simplified_op, Controlled)
        assert isinstance(simplified_op.base, qml.Hadamard)
        assert simplified_op.name == final_op.name
        assert simplified_op.wires == final_op.wires
        assert simplified_op.data == final_op.data
        assert simplified_op.arithmetic_depth == final_op.arithmetic_depth


class TestControlledQueuing:
    """Test that `catalyst.ctrl` operators queue and update base metadata."""

    def test_queuing(self):
        """Test that `catalyst.ctrl` is queued upon initialization and updates base metadata."""
        with qml.queuing.AnnotatedQueue() as q:
            base = qml.Rot(1.234, 2.345, 3.456, wires=2)
            op = C_ctrl(base, (0, 1))

        assert base not in q
        assert qml.equal(q.queue[0], op)

    def test_queuing_base_defined_outside(self):
        """Test that base isn't added to queue if its defined outside the recording context."""

        base = qml.IsingXX(1.234, wires=(0, 1))
        with qml.queuing.AnnotatedQueue() as q:
            op = C_ctrl(base, ("a", "b"))

        assert len(q) == 1
        assert q.queue[0] is op


CSWAP = qml.math.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)  #: CSWAP gate

CH = qml.math.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1 / qml.math.sqrt(2), 1 / qml.math.sqrt(2)],
        [0, 0, 1 / qml.math.sqrt(2), -1 / qml.math.sqrt(2)],
    ]
)  # CH gate


def CRotx(theta):
    r"""Two-qubit controlled rotation about the x axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_x(\theta)`
    """
    return qml.math.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, qml.math.cos(theta / 2), -1j * qml.math.sin(theta / 2)],
            [0, 0, -1j * qml.math.sin(theta / 2), qml.math.cos(theta / 2)],
        ]
    )


def CRoty(theta):
    r"""Two-qubit controlled rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_y(\theta)`
    """
    return qml.math.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, qml.math.cos(theta / 2), -qml.math.sin(theta / 2)],
            [0, 0, qml.math.sin(theta / 2), qml.math.cos(theta / 2)],
        ]
    )


def CRotz(theta):
    r"""Two-qubit controlled rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_z(\theta)`
    """
    return qml.math.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, qml.math.exp(-1j * theta / 2), 0],
            [0, 0, 0, qml.math.exp(1j * theta / 2)],
        ],
        like=theta,
    )


def CRot3(a, b, c):
    r"""Arbitrary two-qubit controlled rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        array: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R(a,b,c)`
    """
    return qml.math.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [
                0,
                0,
                qml.math.exp(-1j * (a + c) / 2) * qml.math.cos(b / 2),
                -qml.math.exp(1j * (a - c) / 2) * qml.math.sin(b / 2),
            ],
            [
                0,
                0,
                qml.math.exp(-1j * (a - c) / 2) * qml.math.sin(b / 2),
                qml.math.exp(1j * (a + c) / 2) * qml.math.cos(b / 2),
            ],
        ],
        like=a,
    )


def ControlledPhaseShift(phi):
    r"""Controlled phase shift.

    Args:
        phi (float): rotation angle

    Returns:
        array: the two-wire controlled-phase matrix
    """
    return qml.math.diag([1, 1, 1, qml.math.exp(1j * phi)])


# Failed with Catalyst because of different decomposition:
# (qml.PauliX("a"), 2, qml.math.diag([1 for i in range(8)])),
# (qml.CNOT(["a", "b"]), 1, qml.math.diag([1 for i in range(8)])),
base_num_control_mats = [
    (qml.PauliX("a"), 1, qml.math.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])),
    (
        qml.PauliY("a"),
        1,
        qml.math.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]),
    ),
    (qml.PauliZ("a"), 1, qml.math.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])),
    (qml.PauliZ("a"), 2, qml.math.diag([1] * 7 + [-1])),
    (qml.SWAP(("a", "b")), 1, CSWAP),
    (qml.Hadamard("a"), 1, CH),
    (qml.RX(1.234, "b"), 1, CRotx(1.234)),
    (qml.RY(-0.432, "a"), 1, CRoty(-0.432)),
    (qml.RZ(6.78, "a"), 1, CRotz(6.78)),
    (qml.Rot(1.234, -0.432, 9.0, "a"), 1, CRot3(1.234, -0.432, 9.0)),
    (qml.PhaseShift(1.234, wires="a"), 1, ControlledPhaseShift(1.234)),
]


class TestMatrix:
    """Tests of Controlled.matrix and Controlled.sparse_matrix"""

    def test_correct_matrix_dimensions_with_batching(self):
        """Test batching returns a matrix of the correct dimensions"""

        x = pnp.array([1.0, 2.0, 3.0])
        base = qml.RX(x, 0)
        op = Controlled(base, 1)
        matrix = op.matrix()
        assert matrix.shape == (3, 4, 4)

    @pytest.mark.parametrize("base, num_control, mat", base_num_control_mats)
    def test_matrix_compare_with_gate_data(self, base, num_control, mat):
        """Test the matrix against matrices provided by `gate_data` file."""
        op = Controlled(base, list(range(num_control)))
        assert qml.math.allclose(op.matrix(), mat)

    def test_aux_wires_included(self):
        """Test that matrix expands to have identity on work wires."""

        base = qml.PauliX(1)
        op = Controlled(
            base,
            0,
            work_wires="aux",
        )
        mat = op.matrix()
        assert mat.shape == (4, 4)

    def test_wire_order(self):
        """Test that the ``wire_order`` keyword argument alters the matrix as expected."""
        base = qml.RX(-4.432, wires=1)
        op = Controlled(base, 0)

        method_order = op.matrix(wire_order=(1, 0))
        function_order = qml.math.expand_matrix(op.matrix(), op.wires, (1, 0))

        assert qml.math.allclose(method_order, function_order)

    @pytest.mark.parametrize("control_values", ([0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]))
    def test_control_values(self, control_values):
        """Test that the matrix with specified control_values is the same as using PauliX flips
        to reverse the control values."""
        control_wires = (0, 1, 2)

        base = qml.RX(3.456, wires=3)
        op = Controlled(base, control_wires, control_values=control_values)

        mat = op.matrix()
        with qml.queuing.AnnotatedQueue() as q:
            [qml.PauliX(w) for w, val in zip(control_wires, control_values) if not val]
            Controlled(base, control_wires, control_values=[1, 1, 1])
            [qml.PauliX(w) for w, val in zip(control_wires, control_values) if not val]
        tape = qml.tape.QuantumScript.from_queue(q)
        decomp_mat = qml.matrix(tape, wire_order=op.wires)

        assert qml.math.allclose(mat, decomp_mat)

    def test_sparse_matrix_base_defines(self):
        """Check that an op that defines a sparse matrix has it used in the controlled
        sparse matrix."""

        Hmat = (1.0 * qml.PauliX(0)).sparse_matrix()
        H_sparse = qml.SparseHamiltonian(Hmat, wires="0")
        op = Controlled(H_sparse, "a")

        sparse_mat = op.sparse_matrix()
        assert isinstance(sparse_mat, sparse.csr_matrix)
        assert qml.math.allclose(sparse_mat.toarray(), op.matrix())

    @pytest.mark.parametrize("control_values", ([0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]))
    def test_sparse_matrix_only_matrix_defined(self, control_values):
        """Check that an base doesn't define a sparse matrix but defines a dense matrix
        still provides a controlled sparse matrix."""
        control_wires = (0, 1, 2)
        base = qml.U2(1.234, -3.2, wires=3)
        op = Controlled(base, control_wires, control_values=control_values)

        sparse_mat = op.sparse_matrix()
        assert isinstance(sparse_mat, sparse.csr_matrix)
        assert qml.math.allclose(op.sparse_matrix().toarray(), op.matrix())

    def test_sparse_matrix_wire_order_error(self):
        """Check a NonImplementedError is raised if the user requests specific wire order."""
        control_wires = (0, 1, 2)
        base = qml.U2(1.234, -3.2, wires=3)
        op = Controlled(base, control_wires)

        with pytest.raises(NotImplementedError):
            op.sparse_matrix(wire_order=[3, 2, 1, 0])

    def test_no_matrix_defined_sparse_matrix_error(self):
        """Check that if the base gate defines neither a sparse matrix nor a dense matrix, a
        SparseMatrixUndefined error is raised."""

        base = TempOperator(1)
        op = Controlled(base, 2)

        with pytest.raises(qml.operation.SparseMatrixUndefinedError):
            op.sparse_matrix()

    def test_sparse_matrix_format(self):
        """Test format keyword determines output type of sparse matrix."""
        base = qml.PauliX(0)
        op = Controlled(base, 1)

        lil_mat = op.sparse_matrix(format="lil")
        assert isinstance(lil_mat, sparse.lil_matrix)


special_non_par_op_decomps = [
    (qml.PauliY, [], [0], [1], qml.CY, [qml.CRY(pnp.pi, wires=[1, 0]), qml.S(1)]),
    (qml.PauliZ, [], [1], [0], qml.CZ, [qml.ControlledPhaseShift(pnp.pi, wires=[0, 1])]),
    (
        qml.Hadamard,
        [],
        [1],
        [0],
        qml.CH,
        [qml.RY(-pnp.pi / 4, wires=1), qml.CZ(wires=[0, 1]), qml.RY(pnp.pi / 4, wires=1)],
    ),
    (
        qml.PauliZ,
        [],
        [0],
        [2, 1],
        qml.CCZ,
        [
            qml.CNOT(wires=[1, 0]),
            qml.adjoint(qml.T(wires=0)),
            qml.CNOT(wires=[2, 0]),
            qml.T(wires=0),
            qml.CNOT(wires=[1, 0]),
            qml.adjoint(qml.T(wires=0)),
            qml.CNOT(wires=[2, 0]),
            qml.T(wires=0),
            qml.T(wires=1),
            qml.CNOT(wires=[2, 1]),
            qml.Hadamard(wires=0),
            qml.T(wires=2),
            qml.adjoint(qml.T(wires=1)),
            qml.CNOT(wires=[2, 1]),
            qml.Hadamard(wires=0),
        ],
    ),
    (
        qml.CZ,
        [],
        [1, 2],
        [0],
        qml.CCZ,
        [
            qml.CNOT(wires=[1, 2]),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[0, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[1, 2]),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[0, 2]),
            qml.T(wires=2),
            qml.T(wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=2),
            qml.T(wires=0),
            qml.adjoint(qml.T(wires=1)),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[2]),
        ],
    ),
    (
        qml.SWAP,
        [],
        [1, 2],
        [0],
        qml.CSWAP,
        [qml.Toffoli(wires=[0, 2, 1]), qml.Toffoli(wires=[0, 1, 2]), qml.Toffoli(wires=[0, 2, 1])],
    ),
]

special_par_op_decomps = [
    (
        qml.RX,
        [0.123],
        [1],
        [0],
        qml.CRX,
        [
            qml.RZ(pnp.pi / 2, wires=1),
            qml.RY(0.123 / 2, wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.RY(-0.123 / 2, wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(-pnp.pi / 2, wires=1),
        ],
    ),
    (
        qml.RY,
        [0.123],
        [1],
        [0],
        qml.CRY,
        [
            qml.RY(0.123 / 2, 1),
            qml.CNOT(wires=(0, 1)),
            qml.RY(-0.123 / 2, 1),
            qml.CNOT(wires=(0, 1)),
        ],
    ),
    (
        qml.RZ,
        [0.123],
        [0],
        [1],
        qml.CRZ,
        [
            qml.PhaseShift(0.123 / 2, wires=0),
            qml.CNOT(wires=[1, 0]),
            qml.PhaseShift(-0.123 / 2, wires=0),
            qml.CNOT(wires=[1, 0]),
        ],
    ),
    (
        qml.Rot,
        [0.1, 0.2, 0.3],
        [1],
        [0],
        qml.CRot,
        [
            qml.RZ((0.1 - 0.3) / 2, wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(-(0.1 + 0.3) / 2, wires=1),
            qml.RY(-0.2 / 2, wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.RY(0.2 / 2, wires=1),
            qml.RZ(0.3, wires=1),
        ],
    ),
    (
        qml.PhaseShift,
        [0.123],
        [1],
        [0],
        qml.ControlledPhaseShift,
        [
            qml.PhaseShift(0.123 / 2, wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.PhaseShift(-0.123 / 2, wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.PhaseShift(0.123 / 2, wires=1),
        ],
    ),
]

custom_ctrl_op_decomps = special_non_par_op_decomps + special_par_op_decomps

pauli_x_based_op_decomps = [
    (
        qml.PauliX,
        [2],
        [0, 1],
        qml.Toffoli.compute_decomposition(wires=[0, 1, 2]),
    ),
    (
        qml.CNOT,
        [1, 2],
        [0],
        qml.Toffoli.compute_decomposition(wires=[0, 1, 2]),
    ),
    (
        qml.PauliX,
        [3],
        [0, 1, 2],
        qml.MultiControlledX.compute_decomposition(wires=[0, 1, 2, 3], work_wires=Wires("aux")),
    ),
    (
        qml.CNOT,
        [2, 3],
        [0, 1],
        qml.MultiControlledX.compute_decomposition(wires=[0, 1, 2, 3], work_wires=Wires("aux")),
    ),
    (
        qml.Toffoli,
        [1, 2, 3],
        [0],
        qml.MultiControlledX.compute_decomposition(wires=[0, 1, 2, 3], work_wires=Wires("aux")),
    ),
]


class TestDecomposition:
    """Test decomposition of Controlled."""

    @pytest.mark.parametrize(
        "target, decomp",
        [
            (
                OpWithDecomposition(0.123, wires=[0, 1]),
                [
                    qml.CH(wires=[2, 0]),
                    Controlled(qml.S(wires=1), control_wires=2),
                    qml.CRX(0.123, wires=[2, 0]),
                ],
            ),
            (
                qml.IsingXX(0.123, wires=[0, 1]),
                [
                    qml.Toffoli(wires=[2, 0, 1]),
                    qml.CRX(0.123, wires=[2, 0]),
                    qml.Toffoli(wires=[2, 0, 1]),
                ],
            ),
        ],
    )
    def test_decomposition(self, target, decomp):
        """Test that we decompose a normal controlled operation"""
        op = C_ctrl(target, 2)
        assert op.decomposition() == decomp

    def test_non_differentiable_one_qubit_special_unitary(self):
        """Assert that a non-differentiable on qubit special unitary uses the bisect
        decomposition."""

        op = C_ctrl(qml.RZ(1.2, wires=0), (1, 2, 3, 4))
        decomp = op.decomposition()

        assert qml.equal(decomp[0], qml.Toffoli(wires=(1, 2, 0)))
        assert isinstance(decomp[1], qml.QubitUnitary)
        assert qml.equal(decomp[2], qml.Toffoli(wires=(3, 4, 0)))
        assert isinstance(decomp[3].base, qml.QubitUnitary)
        assert qml.equal(decomp[4], qml.Toffoli(wires=(1, 2, 0)))
        assert isinstance(decomp[5], qml.QubitUnitary)
        assert qml.equal(decomp[6], qml.Toffoli(wires=(3, 4, 0)))
        assert isinstance(decomp[7].base, qml.QubitUnitary)

        decomp_mat = qml.matrix(op.decomposition, wire_order=op.wires)()
        assert qml.math.allclose(op.matrix(), decomp_mat)

    def test_differentiable_one_qubit_special_unitary(self):
        """Assert that a differentiable qubit special unitary uses the zyz decomposition."""

        pytest.xfail("ValueError: The control_wires should be a single wire, instead got: 4-wires")

        op = C_ctrl(qml.RZ(qml.numpy.array(1.2), 0), (1, 2, 3, 4))
        decomp = op.decomposition()

        assert qml.equal(decomp[0], qml.RZ(qml.numpy.array(1.2), 0))
        assert qml.equal(decomp[1], qml.MultiControlledX(wires=(1, 2, 3, 4, 0)))
        assert qml.equal(decomp[2], qml.RZ(qml.numpy.array(-0.6), wires=0))
        assert qml.equal(decomp[3], qml.MultiControlledX(wires=(1, 2, 3, 4, 0)))
        assert qml.equal(decomp[4], qml.RZ(qml.numpy.array(-0.6), wires=0))

        decomp_mat = qml.matrix(op.decomposition, wire_order=op.wires)()
        assert qml.math.allclose(op.matrix(), decomp_mat)

    @pytest.mark.parametrize(
        "base_cls, base_wires, ctrl_wires, expected",
        pauli_x_based_op_decomps,
    )
    def test_decomposition_pauli_x(self, base_cls, base_wires, ctrl_wires, expected):
        """Tests decompositions where the base is PauliX"""

        base_op = base_cls(wires=base_wires)
        ctrl_op = C_ctrl(base_op, control=ctrl_wires, work_wires=Wires("aux"))

        assert ctrl_op.decomposition() == expected

    def test_decomposition_nested(self):
        """Tests decompositions of nested controlled operations"""

        ctrl_op = C_ctrl(C_ctrl(lambda: qml.RZ(0.123, wires=0), control=1), control=2)()
        expected = [
            qml.ops.Controlled(qml.RZ(0.123, wires=0), control_wires=[1, 2]),
        ]
        assert ctrl_op.decomposition() == expected

    def test_decomposition_undefined(self):
        """Tests error raised when decomposition is undefined"""
        op = C_ctrl(TempOperator(0), (1, 2))
        with pytest.raises(DecompositionUndefinedError):
            op.decomposition()

    def test_global_phase_decomp_raises_warning(self):
        """Test that ctrl(GlobalPhase).decomposition() raises a warning."""
        op = qml.ctrl(qml.GlobalPhase(1.23), control=[0, 1])
        with pytest.warns(
            UserWarning, match="Multi-Controlled-GlobalPhase currently decomposes to nothing"
        ):
            assert op.decomposition() == []

    def test_control_on_zero(self):
        """Test decomposition applies PauliX gates to flip any control-on-zero wires."""

        control = (0, 1, 2)
        control_values = [True, False, False]

        base = TempOperator("a")
        op = C_ctrl(base, control, control_values)

        decomp = op.decomposition()

        assert qml.equal(decomp[0], qml.PauliX(1))
        assert qml.equal(decomp[1], qml.PauliX(2))

        assert isinstance(decomp[2], Controlled)
        assert decomp[2].control_values == [True, True, True]

        assert qml.equal(decomp[3], qml.PauliX(1))
        assert qml.equal(decomp[4], qml.PauliX(2))

    @pytest.mark.parametrize(
        "base_cls, params, base_wires, ctrl_wires, _, expected",
        custom_ctrl_op_decomps,
    )
    def test_control_on_zero_custom_ops(
        self, base_cls, params, base_wires, ctrl_wires, _, expected
    ):
        """Tests that custom ops are not converted when wires are control-on-zero."""

        base_op = base_cls(*params, wires=base_wires)
        op = C_ctrl(base_op, control=ctrl_wires, control_values=[False] * len(ctrl_wires))

        decomp = op.decomposition()

        i = 0
        for ctrl_wire in ctrl_wires:
            assert decomp[i] == qml.PauliX(wires=ctrl_wire)
            i += 1

        for exp in expected:
            assert decomp[i] == exp
            i += 1

        for ctrl_wire in ctrl_wires:
            assert decomp[i] == qml.PauliX(wires=ctrl_wire)
            i += 1


if __name__ == "__main__":
    pytest.main(["-x", __file__])
