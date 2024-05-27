# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Test program verification routines """

from copy import deepcopy

import numpy as np
import pennylane as qml
import pytest
from unittest.mock import patch

import catalyst.utils.calculate_grad_shape as infer
from catalyst import (
    CompileError,
    DifferentiableCompileError,
    adjoint,
    cond,
    ctrl,
    for_loop,
    grad,
    qjit,
    while_loop,
)
from catalyst.device import get_device_capabilities
from catalyst.utils.toml import (
    OperationProperties,
    ProgramFeatures,
    pennylane_operation_set,
)

def get_custom_device(
    non_differentiable_gates=frozenset(),
    non_differentiable_obs=frozenset(),
    non_invertible_gates=frozenset(),
    non_controllable_gates=frozenset(),
    native_gates=frozenset(),
    **kwargs
):
    """Generate a custom device where certain gates are marked as non-differensiable."""

    lightning_device = qml.device("lightning.qubit", wires=0)

    class CustomDevice(qml.devices.Device):
        """Custom Gate Set Device"""

        name = lightning_device.name
        author = "Tester"

        config = None
        backend_name = "default"
        backend_lib = "default"
        backend_kwargs = {}

        def __init__(self, shots=None, wires=None):
            super().__init__(wires=wires, shots=shots)
            program_features = ProgramFeatures(shots_present=kwargs.get("shots") is not None)
            lightning_capabilities = get_device_capabilities(lightning_device, program_features)
            custom_capabilities = deepcopy(lightning_capabilities)
            for gate in native_gates:
                custom_capabilities.native_ops[gate] = OperationProperties(True, True, True)
            for gate in non_differentiable_gates:
                custom_capabilities.native_ops[gate].differentiable = False
            for gate in non_invertible_gates:
                custom_capabilities.native_ops[gate].invertible = False
            for gate in non_controllable_gates:
                custom_capabilities.native_ops[gate].controllable = False
            for obs in non_differentiable_obs:
                custom_capabilities.native_obs[obs].differentiable = False
            self.qjit_capabilities = custom_capabilities

        def execute(self, _circuits, _execution_config):
            """
            Raises: RuntimeError
            """
            raise RuntimeError("QJIT devices cannot execute tapes.")

        @property
        def operations(self):
            """Return operations using PennyLane's C(.) syntax"""
            return (
                pennylane_operation_set(self.qjit_capabilities.native_ops)
                | pennylane_operation_set(self.qjit_capabilities.to_decomp_ops)
                | pennylane_operation_set(self.qjit_capabilities.to_matrix_ops)
            )

        @property
        def observables(self):
            """Return PennyLane observables"""
            return pennylane_operation_set(self.qjit_capabilities.native_obs)

        def supports_derivatives(self, config, circuit=None):
            """Pretend we support any derivatives"""
            return True

    return CustomDevice(**kwargs)

@qml.transform
def null_transform(tape, *args, **kwargs):
    """A null transform that passes on the tape and the null post processing function.
    Used to overwrite transforms in the device preprocess with mocker when we want to 
    skip them for testing purproses"""

    return (tape,), lambda x: x[0]


@patch('catalyst.device.qjit_device.catalyst_decompose', null_transform)
class TestAdjointMethodVerification:

    def test_non_differentiable_gate_simple(self):
        """Emulate a device with a non-differentiable gate."""

        @qml.qnode(
            get_custom_device(non_differentiable_gates={"RX"}, wires=[0]), diff_method="adjoint"
        )
        def f(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliX(0))

        with pytest.raises(DifferentiableCompileError, match="RX.*non-differentiable"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    def test_non_differentiable_observable(self):
        """Emulate a device with a non-differentiable gate."""

        @qml.qnode(
            get_custom_device(non_differentiable_obs={"PauliX"}, wires=[0]), diff_method="adjoint"
        )
        def f(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliX(0))

        with pytest.raises(DifferentiableCompileError, match="PauliX.*non-differentiable"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    def test_non_differentiable_gate_nested_cond(self):
        """Emulate a device with a non-differentiable gate."""

        @qml.qnode(
            get_custom_device(non_differentiable_gates={"RX"}, wires=1), diff_method="adjoint"
        )
        def f(x):
            @cond(True)
            def true_path():
                qml.RX(x, wires=0)

            @true_path.otherwise
            def false_path():
                qml.RX(x, wires=0)

            true_path()

            return qml.expval(qml.PauliX(0))

        with pytest.raises(DifferentiableCompileError, match="RX.*non-differentiable"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    def test_non_differentiable_gate_nested_adjoint(self):
        """Emulate a device with a non-differentiable gate."""

        @qml.qnode(
            get_custom_device(non_differentiable_gates={"RX"}, wires=1), diff_method="adjoint"
        )
        def f(x):
            adjoint(qml.RX(x, wires=[0]))
            return qml.expval(qml.PauliX(0))

        with pytest.raises(DifferentiableCompileError, match="RX.*non-differentiable"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)


@patch('catalyst.device.qjit_device.catalyst_decompose', null_transform)
class TestHybridOpVerification:
    """Test that the verification catches situations where a HybridOp subtape contains
    an operation the given device can't support inside that HybridOp"""

    def test_non_invertible_gate_simple(self):
        """Emulate a device with a non-invertible gate applied inside an Adjoint HybridOp."""

        dev = get_custom_device(non_invertible_gates={"RX"}, wires=1)

        @qml.qnode(dev)
        def f(x):
            adjoint(qml.RX(x, wires=0))
            return qml.expval(qml.PauliX(0))

        with pytest.raises(CompileError, match="RX.*not invertible"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    def test_non_invertible_gate_nested_cond(self):
        """Emulate a device with a non-invertible gate inside an Adjoint that
        is further nested in a Cond operation."""

        @qml.qnode(get_custom_device(non_invertible_gates={"RX"}, wires=1))
        def f(x):
            @cond(True)
            def true_path():
                adjoint(qml.RX(x, wires=0))

            @true_path.otherwise
            def false_path():
                adjoint(qml.RX(x, wires=0))

            true_path()

            return qml.expval(qml.PauliX(0))

        with pytest.raises(CompileError, match="RX.*not invertible"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    def test_non_invertible_gate_nested_for(self):
        """Emulate a device with a non-invertible gate inside an Adjoint that
        is further nested in a For operation."""

        @qml.qnode(get_custom_device(non_invertible_gates={"RX"}, wires=1))
        def f(x):
            @for_loop(0, 10, 1)
            def loop(_i):
                adjoint(qml.RX(x, wires=0))

            loop()
            return qml.expval(qml.PauliX(0))

        with pytest.raises(CompileError, match="RX.*not invertible"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    def test_non_controllable_gate_simple(self):
        """Emulate a device with a non-controllable gate applied inside a QCtrl."""

        with pytest.raises(CompileError, match="PauliZ.*not controllable"):

            @qjit
            @qml.qnode(get_custom_device(non_controllable_gates={"PauliZ"}, wires=3))
            def f(x: float):
                ctrl(qml.PauliZ(wires=0), control=[1, 2])
                return qml.expval(qml.PauliX(0))


class PauliX2(qml.PauliX):
    """Test operation without the analytic gradient"""

    name = "PauliX2"
    grad_method = "F"


@patch('catalyst.device.qjit_device.catalyst_decompose', null_transform)
class TestParameterShiftMethodVerification:
    """Test the verification of operators and observables when the parameter shift method
    is used for differentiation"""

    def test_paramshift_obs_simple(self):
        """Emulate a device with a non-invertible observable."""

        assert qml.Hermitian.grad_method != "A"

        @qml.qnode(get_custom_device(wires=2), diff_method="parameter-shift")
        def f(x):
            qml.PauliX(wires=1)
            qml.RX(x, wires=0)
            A = np.array(
                [[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]]
            )
            return qml.expval(qml.Hermitian(A, wires=0))

        with pytest.raises(
            DifferentiableCompileError, match="Hermitian does not support analytic differentiation"
        ):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    def test_paramshift_gate_simple(self):
        """Emulate a device with a non-invertible gate."""

        @qml.qnode(
            get_custom_device(native_gates={"PauliX2"}, wires=1), diff_method="parameter-shift"
        )
        def f(_):
            PauliX2(wires=0)
            return qml.expval(qml.PauliX(0))

        with pytest.raises(
            DifferentiableCompileError, match="PauliX2 does not support analytic differentiation"
        ):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    def test_paramshift_gate_while(self):
        """Emulate a device with a non-invertible gate."""

        @qml.qnode(
            get_custom_device(native_gates={"PauliX2"}, wires=1), diff_method="parameter-shift"
        )
        def f(_):
            @while_loop(lambda s: s > 0)
            def loop(s):
                PauliX2(wires=0)
                return s + 1

            loop(0)
            return qml.expval(qml.PauliX(0))

        with pytest.raises(
            DifferentiableCompileError, match="PauliX2 does not support analytic differentiation"
        ):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)


def test_no_state_returns():
    """Test state returns are rejected in gradients."""

    @qml.qnode(get_custom_device(wires=1))
    def f(_):
        qml.PauliX(wires=0)
        return qml.state()

    with pytest.raises(DifferentiableCompileError, match="State returns.*forbidden"):

        @qml.qjit
        def cir(x: float):
            return grad(f)(x)


def test_no_variance_returns():
    """Test variance returns are rejected in gradients."""

    @qml.qnode(get_custom_device(wires=1))
    def f(_):
        qml.PauliX(wires=0)
        return qml.var(qml.PauliX(0))

    with pytest.raises(DifferentiableCompileError, match="Variance returns.*forbidden"):

        @qml.qjit
        def cir(x: float):
            return grad(f)(x)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
