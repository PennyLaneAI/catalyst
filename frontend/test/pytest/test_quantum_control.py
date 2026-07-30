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

from typing import Callable

import jax.numpy as jnp
import pennylane as qp
import pennylane.numpy as pnp
import pytest
from numpy.testing import assert_allclose
from pennylane import adjoint as PL_adjoint
from pennylane import cond
from pennylane import ctrl as PL_ctrl
from pennylane import for_loop, qjit, while_loop
from pennylane.operation import Wires
from pennylane.ops.op_math.controlled import Controlled
from pennylane.tape import QuantumTape

from catalyst import adjoint as C_adjoint
from catalyst import ctrl as C_ctrl
from catalyst import measure
from catalyst.api_extensions.quantum_operators import HybridCtrl
from catalyst.jax_tracer import HybridOpRegion


def verify_catalyst_ctrl_against_pennylane(
    quantum_func: Callable, device, *args, with_adjoint_arg=False, capture_mode="global"
):
    """
    A helper function for verifying Catalyst's native quantum control against the behaviour of
    PennyLane's quantum control function.
    """

    @qjit(capture=capture_mode)
    @qp.qnode(device)
    def catalyst_workflow(*args):
        if capture_mode is True:
            if with_adjoint_arg:
                return quantum_func(*args, ctrl_fn=PL_ctrl, adjoint_fn=PL_adjoint)
            else:
                return quantum_func(*args, ctrl_fn=PL_ctrl)

        if with_adjoint_arg:
            return quantum_func(*args, ctrl_fn=C_ctrl, adjoint_fn=C_adjoint)
        else:
            return quantum_func(*args, ctrl_fn=C_ctrl)

    @qp.qnode(device)
    def pennylane_workflow(*args):
        if with_adjoint_arg:
            return quantum_func(*args, ctrl_fn=PL_ctrl, adjoint_fn=PL_adjoint)
        else:
            return quantum_func(*args, ctrl_fn=PL_ctrl)

    compare = pennylane_workflow(*args)
    assert_allclose(catalyst_workflow(*args), compare, atol=1e-7)


class TestControlled:
    """Integration tests for Catalyst ctrl functionality."""

    def test_qctrl_op_object(self, backend, capture_mode):
        """Test the quantum control application to an operation object"""

        def circuit(theta, w, cw, ctrl_fn):
            ctrl_fn(qp.RX(theta, wires=[w]), control=[cw], control_values=[False])
            ctrl_fn(qp.RX, control=[cw], control_values=[False])(theta, wires=[w])
            return qp.state()

        verify_catalyst_ctrl_against_pennylane(
            circuit, qp.device(backend, wires=3), 0.1, 0, 1, capture_mode=capture_mode
        )

    def test_qctrl_op_class(self, backend, capture_mode):
        """Test the quantum control application to a single operation class"""

        def circuit(theta, w, cw, ctrl_fn):
            ctrl_fn(qp.RX, control=[w], control_values=[True])(theta, wires=[cw])
            return qp.state()

        verify_catalyst_ctrl_against_pennylane(
            circuit, qp.device(backend, wires=3), 0.1, 0, 1, capture_mode=capture_mode
        )

    def test_qctrl_adjoint_func_simple(self, backend, capture_mode):
        """Test the quantum control distribution over the group of operations"""

        def circuit(arg, ctrl_fn, adjoint_fn):
            def _func(theta):
                qp.RX(theta, wires=[0])
                qp.RZ(theta, wires=2)

            ctrl_fn(adjoint_fn(_func), control=[1], control_values=[True])(arg)
            return qp.state()

        verify_catalyst_ctrl_against_pennylane(
            circuit,
            qp.device(backend, wires=3),
            0.1,
            with_adjoint_arg=True,
            capture_mode=capture_mode,
        )

    def test_adjoint_qctrl_func_simple(self, backend, capture_mode):
        """Test the quantum control distribution over the group of operations"""

        def circuit(arg, ctrl_fn, adjoint_fn):
            def _func(theta):
                qp.RX(theta, wires=[0])
                qp.RZ(theta, wires=2)

            adjoint_fn(ctrl_fn(_func, control=[1], control_values=[True]))(arg)
            return qp.state()

        verify_catalyst_ctrl_against_pennylane(
            circuit,
            qp.device(backend, wires=3),
            0.1,
            with_adjoint_arg=True,
            capture_mode=capture_mode,
        )

    def test_qctrl_adjoint_hybrid(self, backend, capture_mode):
        """Test the quantum control distribution over the group of operations"""

        def circuit(theta, w2, cw, ctrl_fn, adjoint_fn):
            def _func():
                @while_loop(lambda s: s < w2)
                def _while_loop(s):
                    qp.RY(theta, wires=s)
                    return s + 1

                _while_loop(0)  # pylint: disable=no-value-for-parameter

            ctrl_fn(adjoint_fn(_func), control=[cw], control_values=[True])()
            return qp.state()

        verify_catalyst_ctrl_against_pennylane(
            circuit,
            qp.device(backend, wires=3),
            0.1,
            2,
            2,
            with_adjoint_arg=True,
            capture_mode=capture_mode,
        )

    def test_qctrl_func_simple(self, backend, capture_mode):
        """Test the quantum control distribution over the group of operations"""

        def circuit(arg, ctrl_fn):
            def _func(theta):
                qp.RX(theta, wires=[0])
                qp.RZ(theta, wires=2)

            ctrl_fn(_func, control=[1], control_values=[True])(arg)
            return qp.state()

        verify_catalyst_ctrl_against_pennylane(
            circuit, qp.device(backend, wires=3), 0.1, capture_mode=capture_mode
        )

    def test_qctrl_func_hybrid(self, backend, capture_mode):
        """Test the quantum control distribution over the Catalyst hybrid operation"""

        def circuit(theta, w1, w2, cw, ctrl_fn):
            def _func():
                qp.RX(theta, wires=[w1])

                s = 0

                @while_loop(lambda s: s < w2)
                def _while_loop(s):
                    qp.RY(theta, wires=s)
                    return s + 1

                s = _while_loop(s)  # pylint: disable=no-value-for-parameter

                @for_loop(0, w2, 1)
                def _for_loop(i, s):
                    qp.RY(theta, wires=i)
                    return s + 1

                s = _for_loop(s)  # pylint: disable=no-value-for-parameter

                @cond(True)
                def _branch():
                    qp.RZ(theta, wires=w2 - 1)
                    return 1

                @_branch.otherwise
                def _branch():
                    qp.RZ(theta, wires=w2 - 1)
                    return 0

                x = _branch()

                qp.RZ((s + x) * theta, wires=w1)

            ctrl_fn(_func, control=[cw], control_values=[True])()
            return qp.state()

        verify_catalyst_ctrl_against_pennylane(
            circuit, qp.device(backend, wires=3), 0.1, 0, 2, 2, capture_mode=capture_mode
        )

    def test_qctrl_func_nested(self, backend, capture_mode):
        """Test the quantum control distribution over the nested control operations"""

        def circuit(theta, w1, w2, cw1, cw2, ctrl_fn):
            def _func1():
                qp.RX(theta, wires=[w1])

                def _func2():
                    qp.RY(theta, wires=[w2])

                ctrl_fn(_func2, control=[cw2], control_values=[True])()

                qp.RZ(theta, wires=w1)

            ctrl_fn(_func1, control=[cw1], control_values=[True])()
            return qp.state()

        verify_catalyst_ctrl_against_pennylane(
            circuit,
            qp.device(backend, wires=4),
            0.1,
            0,
            1,
            2,
            3,
            capture_mode=capture_mode,
        )

    def test_qctrl_func_work_wires(self, backend, capture_mode):
        """Test the quantum control distribution over the nested control operations"""

        def circuit(theta, ctrl_fn):
            def _func1():
                qp.RX(theta, wires=[0])

                def _func2():
                    qp.RY(theta, wires=[0])

                ctrl_fn(_func2, control=[3], work_wires=[4])()

                qp.RZ(theta, wires=[0])

            ctrl_fn(_func1, control=[1], work_wires=[2])()
            return qp.state()

        verify_catalyst_ctrl_against_pennylane(
            circuit, qp.device(backend, wires=5), 0.1, capture_mode=capture_mode
        )

    def test_qctrl_valid_input_types(self, backend, capture_mode):
        """Test the quantum control input types"""

        def circuit(theta, w, cw, ctrl_fn):
            ctrl_fn(qp.RX(theta, wires=[w]), control=[cw])
            ctrl_fn(qp.RX(theta, wires=[w]), control=cw)
            ctrl_fn(qp.RX(theta, wires=[w]), control=[cw], control_values=[True])
            ctrl_fn(qp.RX(theta, wires=[w]), control=[cw], control_values=True)
            ctrl_fn(qp.RX(theta, wires=[w]), control=[cw], control_values=0)
            # FIXME: fails if work_wires is not None and other values are tracers
            # ctrl_fn(qp.RX(theta, wires=[0]), control=[1], work_wires=[2])
            return qp.state()

        verify_catalyst_ctrl_against_pennylane(
            circuit, qp.device(backend, wires=3), 0.1, 0, 1, capture_mode=capture_mode
        )

    def test_native_controlled_custom(self, capture_mode):
        """Test native control of a custom operation."""
        dev = qp.device("lightning.qubit", wires=4)

        @qp.qnode(dev)
        def native_controlled():
            qp.ctrl(qp.PauliZ(wires=[0]), control=[1, 2, 3])
            return qp.state()

        compiled = qjit(native_controlled, capture=capture_mode)
        assert all(sign in compiled.mlir for sign in ["ctrls", "ctrlvals"])
        result = compiled()
        expected = native_controlled()
        assert_allclose(result, expected, atol=1e-5, rtol=1e-5)

    def test_native_controlled_unitary(self, capture_mode):
        """Test native control of a custom operation."""
        dev = qp.device("lightning.qubit", wires=4)

        @qp.qnode(dev)
        def native_controlled():
            qp.ctrl(
                qp.QubitUnitary(
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
            return qp.state()

        compiled = qjit(native_controlled, capture=capture_mode)
        result = compiled()
        expected = native_controlled()
        assert_allclose(result, expected, atol=1e-5, rtol=1e-5)


class TestCatalystOnlyControlled:
    """Tests for just catalyst's version of control."""

    def test_ctrl_invalid_argument(self):
        """Checks that ctrl rejects non-quantum program arguments."""

        with pytest.raises(ValueError, match="Expected a callable"):

            @qjit
            @qp.qnode(qp.device("lightning.qubit", wires=2))
            def workflow():
                C_ctrl(0, control=1)(2)
                return qp.state()

            workflow()

    def test_qctrl_raises_on_invalid_input(self, backend):
        """Test the no-measurements exception"""

        @qp.qnode(qp.device(backend, wires=2))
        def circuit(theta):
            C_ctrl(qp.RX(theta, wires=[0]), control=[1], control_values=[])()
            return qp.state()

        with pytest.raises(ValueError, match="Length of the control_values"):
            qjit(circuit)(0.1)

    def test_qctrl_no_mid_circuit_measurements(self, backend):
        """Test the no-measurements exception"""

        @qp.qnode(qp.device(backend, wires=2))
        def circuit(theta):
            def _func1():
                m = measure(0)
                qp.RX(m * theta, wires=[0])

            C_ctrl(_func1, control=[1], control_values=[True])()
            return qp.state()

        with pytest.raises(ValueError, match="Mid-circuit measurements cannot be used"):
            qjit(circuit)(0.1)

    def test_qctrl_no_end_circuit_measurements(self, backend):
        """Test the no-measurements exception"""

        @qp.qnode(qp.device(backend, wires=2))
        def circuit(theta):
            def _func1():
                qp.RX(theta, wires=[0])
                return qp.state()

            C_ctrl(_func1, control=[1], control_values=[True])()
            return qp.state()

        with pytest.raises(ValueError, match="Measurement process cannot be used"):
            qjit(circuit)(0.1)

    def test_qctrl_wires(self, backend):
        """Test the wires property of HybridCtrl"""

        @qjit
        @qp.qnode(qp.device(backend, wires=3))
        def circuit(theta):
            def func(theta):
                qp.RX(theta, wires=[0])
                qp.Hadamard(2)
                qp.CNOT([0, 2])

            qctrl = C_ctrl(func, control=[1])(theta)
            return qctrl.wires

        # Without the `wires` property, returns `[-1]`
        assert circuit(0.3) == qp.wires.Wires([1, 0, 2])

    def test_qctrl_wires_arg_fun(self, backend):
        """Test the wires property of HybridCtrl with argument wires"""

        @qjit
        @qp.qnode(qp.device(backend, wires=4))
        def circuit():
            def func(anc, wires):
                qp.Hadamard(anc)
                h = pnp.array([[1, 1], [1, -1]]) / pnp.sqrt(2)
                qp.ctrl(qp.BlockEncode, control=anc)(h, wires=wires)
                qp.Hadamard(anc)

            qctrl = C_ctrl(func, control=[1])(0, [2, 3])
            return qctrl.wires

        assert circuit() == qp.wires.Wires([1, 0, 2, 3])

    def test_qctrl_var_wires(self, backend):
        """Test the wires property of HybridCtrl with variable wires"""

        @qjit
        @qp.qnode(qp.device(backend, wires=4))
        def circuit(anc, wires):
            def func(anc, wires):
                qp.Hadamard(anc)
                h = pnp.array([[1, 1], [1, -1]]) / pnp.sqrt(2)
                qp.ctrl(qp.BlockEncode, control=anc)(h, wires=wires)
                qp.Hadamard(anc)

            qctrl = C_ctrl(func, control=[1])(anc, wires)
            return qctrl.wires

        assert circuit(0, [2, 3]) == qp.wires.Wires([1, 0, 2, 3])

    def test_qctrl_wires_nested(self, backend):
        """Test the wires property of HybridCtrl with nested branches"""

        @qjit
        @qp.qnode(qp.device(backend, wires=4))
        def circuit(theta, w1, w2, cw1, cw2):
            def _func1():
                qp.RX(theta, wires=[w1])

                def _func2():
                    qp.RY(theta, wires=[w2])

                C_ctrl(_func2, control=[cw2], control_values=[True])()

                qp.RZ(theta, wires=w1)

            qctrl = C_ctrl(_func1, control=[cw1], control_values=[True])()
            return qctrl.wires

        assert circuit(0.1, 0, 1, 2, 3) == qp.wires.Wires([2, 0, 3, 1])

    def test_qctrl_work_wires(self, backend):
        """Test the wires property of HybridCtrl with work-wires"""

        @qjit
        @qp.qnode(qp.device(backend, wires=5))
        def circuit(theta):
            def _func1():
                qp.RX(theta, wires=[0])

                def _func2():
                    qp.RY(theta, wires=[0])

                C_ctrl(_func2, control=[3], work_wires=[4])()

                qp.RZ(theta, wires=[0])

            qctrl = C_ctrl(_func1, control=[1], work_wires=[2])()
            return qctrl.wires

        assert circuit(0.1) == qp.wires.Wires([1, 0, 3])

    @pytest.mark.xfail(reason="ctrl.wires fails in control-flow branches is not supported")
    def test_qctrl_wires_controlflow(self, backend):
        """Test the wires property of HybridCtrl with control flow branches"""

        @qjit
        @qp.qnode(qp.device(backend, wires=3))
        def circuit(theta, w1, w2, cw):
            def _func():
                qp.RX(theta, wires=[w1])
                s = 0

                @for_loop(0, w2, 1)
                def _for_loop(i, s):
                    qp.RY(theta, wires=i)
                    return s + 1

                s = _for_loop(s)  # pylint: disable=no-value-for-parameter
                qp.RZ(s * theta, wires=w1)

            qctrl = C_ctrl(_func, control=[cw], control_values=[True])()
            return qctrl.wires

        # It returns `[2, 0, -1]`
        assert circuit(0.1, 0, 2, 2) == qp.wires.Wires([2, 0, 1])

    def test_map_wires(self):
        """Test map wires."""

        X = HybridOpRegion(
            quantum_tape=QuantumTape([qp.X(wires=[1])], []),
            arg_classical_tracers=[],
            res_classical_tracers=[],
            trace=None,
        )
        qctrl = HybridCtrl([], [], [X], control_wires=[0])
        new_qctrl = qctrl.map_wires({1: 0, 0: 1})
        assert new_qctrl._control_wires == [1]  # pylint: disable=protected-access
        assert new_qctrl.regions[0].quantum_tape.operations[0].wires == Wires([0])

    @pytest.mark.parametrize("work_wire_type", ["zeroed", "borrowed"])
    def test_qctrl_work_wire_type_operator(self, work_wire_type):
        """Test that work_wire_type is preserved on a Controlled op inside qjit"""
        c_wire = 0
        x_wires = [1, 2, 3]
        output = [4, 5, 6]
        work_wires_add = [7, 8]
        work_wires_ctrl = [9]

        @qjit
        def func():
            return PL_ctrl(
                qp.SemiAdder(
                    x_wires=x_wires,
                    y_wires=output,
                    work_wires=work_wires_add,
                ),
                control=c_wire,
                work_wires=work_wires_ctrl,
                work_wire_type=work_wire_type,
            )

        op = func()
        assert op.hyperparameters["work_wire_type"] == work_wire_type
        assert op.work_wire_type == work_wire_type
        assert op.control_wires == Wires([0])
        assert op.work_wires == Wires([9])

        @qjit
        def func_native():
            return C_ctrl(
                qp.SemiAdder(
                    x_wires=x_wires,
                    y_wires=output,
                    work_wires=work_wires_add,
                ),
                control=c_wire,
                work_wires=work_wires_ctrl,
                work_wire_type=work_wire_type,
            )

        op = func_native()
        assert op.hyperparameters["work_wire_type"] == work_wire_type
        assert op.work_wire_type == work_wire_type

    @pytest.mark.parametrize("work_wire_type", ["zeroed", "borrowed"])
    def test_qctrl_work_wire_type_callable(self, work_wire_type):
        """Test that work_wire_type is preserved on a Controlled op when wrapping a callable"""
        c_wire = 0
        x_wires = [1, 2, 3]
        output = [4, 5, 6]
        work_wires_add = [7, 8]
        work_wires_ctrl = [9]

        def _func():
            qp.SemiAdder(x_wires=x_wires, y_wires=output, work_wires=work_wires_add)

        hybrid_ctrl = C_ctrl(
            _func, control=c_wire, work_wires=work_wires_ctrl, work_wire_type=work_wire_type
        )()
        assert hybrid_ctrl.work_wire_type == work_wire_type

        decomposed = hybrid_ctrl.decomposition()
        assert len(decomposed) == 1
        assert decomposed[0].hyperparameters["work_wire_type"] == work_wire_type

    def test_control_outside_qjit(self):
        """Test that the Catalyst control function can be used without jitting."""

        result = C_ctrl(qp.T(wires=0), control=[1, 2], control_values=[False, True], work_wires=3)
        expected = PL_ctrl(
            qp.T(wires=0), control=[1, 2], control_values=[False, True], work_wires=3
        )

        assert result.name == expected.name
        assert qp.equal(result.base, expected.base)
        assert result.control_wires == expected.control_wires
        assert result.control_values == expected.control_values
        assert result.work_wires == expected.work_wires

    def test_control_decomp_trotter(self):
        """Test that the Catalyst control can safelt decompose TrotterProduct."""

        coeffs = [0.25, 0.75]
        ops = [qp.X(0), qp.Z(0)]
        H = qp.dot(coeffs, ops)

        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(0)
            qp.ControlledSequence(qp.TrotterProduct(H, time=2.4, order=2), control=[1])
            return qp.expval(qp.PauliZ(0))

        assert qp.math.allclose(qjit(circuit)(), circuit())

    def test_distribute_controlled_with_adj(self):
        """Test that the distribute_controlled function with a PennyLane Adjoint,
        creates the equivalent Adjoint(Ctrl(base)) instead of Ctrl(Adj(base))"""

        # pylint: disable=import-outside-toplevel
        from catalyst.api_extensions.quantum_operators import ctrl_distribute

        tape = qp.tape.QuantumScript([qp.ops.Adjoint(qp.RX(1.2, 0)), qp.Hadamard(1)])

        new_ops = ctrl_distribute(tape, control_wires=[2, 3], control_values=[True, True])

        assert new_ops[0] == qp.ops.Adjoint(Controlled(qp.RX(1.2, 0), control_wires=[2, 3]))
        assert new_ops[1] == Controlled(qp.Hadamard(1), control_wires=[2, 3])


if __name__ == "__main__":
    pytest.main(["-x", __file__])
