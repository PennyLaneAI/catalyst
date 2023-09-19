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

import pennylane as qml
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

from catalyst import for_loop, measure, qjit

# This is used just for internal testing
from catalyst.pennylane_extensions import qctrl, qfunc

lightning = qml.device("lightning.qubit", wires=3)
copy = lightning.operations.copy()
copy.discard("MultiControlledX")
copy.discard("Rot")
copy.discard("S")


class CustomDevice(qml.QubitDevice):
    name = "Device without MultiControlledX, Rot, and S gates"
    short_name = "dummy.device"
    pennylane_requires = "0.1.0"
    version = "0.0.1"
    author = "CV quantum"

    operations = copy
    observables = lightning.observables.copy()

    def __init__(self, shots=None, wires=None, backend_name=None, backend_kwargs=None):
        self.backend_name = backend_name if backend_name else "default"
        self.backend_kwargs = backend_kwargs if backend_kwargs else ""
        super().__init__(wires=wires, shots=shots)

    def apply(self, operations, **kwargs):
        pass


dev = CustomDevice(wires=2)


@pytest.mark.parametrize("param,expected", [(0.0, True), (jnp.pi, False)])
def test_decomposition(param, expected):
    @qjit()
    @qfunc(2, device=dev)
    def mid_circuit(x: float):
        qml.Hadamard(wires=0)
        qml.Rot(0, 0, x, wires=0)
        qml.Hadamard(wires=0)
        m = measure(wires=0)
        b = m ^ 0x1
        qml.Hadamard(wires=1)
        qml.Rot(0, 0, b * jnp.pi, wires=1)
        qml.Hadamard(wires=1)
        return measure(wires=1)

    assert mid_circuit(param) == expected


def verify_catalyst_ctrl_against_pennylane(quantum_func, device, *args):
    """
    A helper function for verifying Catalyst's native adjoint against the behaviour of PennyLane's
    adjoint function. This is specialized to verifying the behaviour of a single function that has
    its adjoint computed.
    """

    @qjit
    @qml.qnode(device)
    def catalyst_workflow(*args):
        return quantum_func(*args, ctrl=qctrl)

    @qml.qnode(device)
    def pennylane_workflow(*args):
        return quantum_func(*args, ctrl=qml.ctrl)

    assert_allclose(catalyst_workflow(*args), pennylane_workflow(*args))


def test_qctrl_op_simple(backend):
    def circuit(theta, w, cw, ctrl):
        ctrl(qml.RX(theta, wires=[w]), control=[cw], control_values=[True])
        ctrl(qml.RX, control=[cw], control_values=[True])(theta, wires=[w])
        return qml.state()

    verify_catalyst_ctrl_against_pennylane(circuit, qml.device(backend, wires=3), 0.1, 0, 1)


def test_qctrl_op_indirect(backend):
    def circuit(theta, w, cw, ctrl):
        ctrl(qml.RX, control=[w], control_values=[True])(theta, wires=[cw])
        return qml.state()

    verify_catalyst_ctrl_against_pennylane(circuit, qml.device(backend, wires=3), 0.1, 0, 1)


def test_qctrl_func_simple(backend):
    def circuit(arg, ctrl):
        def _func(theta):
            qml.RX(theta, wires=[0])
            qml.RZ(theta, wires=2)

        ctrl(_func, control=[1], control_values=[True])(arg)
        return qml.state()

    verify_catalyst_ctrl_against_pennylane(circuit, qml.device(backend, wires=3), 0.1)


def test_qctrl_func_hybrid(backend):
    def circuit(theta, w1, w2, cw, ctrl):
        def _func():
            qml.RX(theta, wires=[w1])

            @for_loop(0, w2, 1)
            def _loop(i):
                qml.RY(theta, wires=i)

            _loop()

            qml.RZ(theta, wires=w1)

        ctrl(_func, control=[cw], control_values=[True])()
        return qml.state()

    verify_catalyst_ctrl_against_pennylane(circuit, qml.device(backend, wires=3), 0.1, 0, 2, 2)


def test_qctrl_func_nested(backend):
    def circuit(theta, w1, w2, cw1, cw2, ctrl):
        def _func():
            qml.RX(theta, wires=[w1])

            def _func2():
                qml.RY(theta, wires=[w2])

            ctrl(_func2, control=[cw2], control_values=[True])()

            qml.RZ(theta, wires=w1)

        ctrl(_func, control=[cw1], control_values=[True])()
        return qml.state()

    verify_catalyst_ctrl_against_pennylane(circuit, qml.device(backend, wires=4), 0.1, 0, 1, 2, 3)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
