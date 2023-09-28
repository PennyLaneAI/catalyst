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

# pylint: disable=too-many-arguments

from typing import Callable

import pennylane as qml
import pytest
from numpy.testing import assert_allclose
from pennylane import adjoint as PL_adjoint
from pennylane import ctrl as PL_ctrl

from catalyst import adjoint as C_adjoint
from catalyst import cond
from catalyst import ctrl as C_ctrl
from catalyst import for_loop, measure, qjit, while_loop


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

    assert_allclose(catalyst_workflow(*args), pennylane_workflow(*args))


def test_qctrl_op_object(backend):
    """Test the quantum control application to an operation object"""

    def circuit(theta, w, cw, ctrl_fn):
        ctrl_fn(qml.RX(theta, wires=[w]), control=[cw], control_values=[False])
        ctrl_fn(qml.RX, control=[cw], control_values=[False])(theta, wires=[w])
        return qml.state()

    verify_catalyst_ctrl_against_pennylane(circuit, qml.device(backend, wires=3), 0.1, 0, 1)


def test_qctrl_op_class(backend):
    """Test the quantum control application to a single operation class"""

    def circuit(theta, w, cw, ctrl_fn):
        ctrl_fn(qml.RX, control=[w], control_values=[True])(theta, wires=[cw])
        return qml.state()

    verify_catalyst_ctrl_against_pennylane(circuit, qml.device(backend, wires=3), 0.1, 0, 1)


def test_qctrl_adjoint_func_simple(backend):
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


def test_adjoint_qctrl_func_simple(backend):
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
def test_qctrl_adjoint_hybrid(backend):
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


def test_qctrl_func_simple(backend):
    """Test the quantum control distribution over the group of operations"""

    def circuit(arg, ctrl_fn):
        def _func(theta):
            qml.RX(theta, wires=[0])
            qml.RZ(theta, wires=2)

        ctrl_fn(_func, control=[1], control_values=[True])(arg)
        return qml.state()

    verify_catalyst_ctrl_against_pennylane(circuit, qml.device(backend, wires=3), 0.1)


def test_qctrl_func_hybrid(backend):
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


def test_qctrl_func_nested(backend):
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

    verify_catalyst_ctrl_against_pennylane(circuit, qml.device(backend, wires=4), 0.1, 0, 1, 2, 3)


def test_qctrl_func_work_wires(backend):
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


def test_qctrl_valid_input_types(backend):
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


def test_qctrl_raises_on_invalid_input(backend):
    """Test the no-measurements exception"""

    @qml.qnode(qml.device(backend, wires=2))
    def circuit(theta):
        C_ctrl(qml.RX(theta, wires=[0]), control=[1], control_values=[])()
        return qml.state()

    with pytest.raises(ValueError, match="Length of the control_values"):
        qjit(circuit)(0.1)


def test_qctrl_no_mid_circuit_measurements(backend):
    """Test the no-measurements exception"""

    @qml.qnode(qml.device(backend, wires=2))
    def circuit(theta):
        def _func1():
            m = measure(0)
            qml.RX(m * theta, wires=[0])

        C_ctrl(_func1, control=[1], control_values=[True])()
        return qml.state()

    with pytest.raises(ValueError, match="measurements are not allowed"):
        qjit(circuit)(0.1)


def test_qctrl_no_end_circuit_measurements(backend):
    """Test the no-measurements exception"""

    @qml.qnode(qml.device(backend, wires=2))
    def circuit(theta):
        def _func1():
            qml.RX(theta, wires=[0])
            return qml.state()

        C_ctrl(_func1, control=[1], control_values=[True])()
        return qml.state()

    with pytest.raises(ValueError, match="measurements are not allowed"):
        qjit(circuit)(0.1)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
