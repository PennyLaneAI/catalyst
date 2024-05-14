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

import jax
import numpy as np
import pennylane as qml
import pytest
from jax import numpy as jnp
from jax.tree_util import tree_flatten

import catalyst.utils.calculate_grad_shape as infer
from catalyst import (
    CompileError,
    DifferentiableCompileError,
    adjoint,
    cond,
    ctrl,
    for_loop,
    grad,
    jacobian,
    measure,
    mitigate_with_zne,
    pure_callback,
    qjit,
    value_and_grad,
    while_loop,
)
from catalyst.utils.runtime import pennylane_operation_set
from catalyst.utils.toml import (
    OperationProperties,
    ProgramFeatures,
    get_device_capabilities,
)


def get_custom_device(
    non_differentiable_gates=set(),
    non_invertible_gates=set(),
    non_controllable_gates=set(),
    native_gates=set(),
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

        # By this we disable tape expansion in the deduced QJITDevice
        max_expansion = 0

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
            self.qjit_capabilities = custom_capabilities

        def execute(self, circuits, execution_config):
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


def test_non_differentiable_gate_simple():
    """Emulate a device with a non-differentiable gate."""

    @qml.qnode(get_custom_device(non_differentiable_gates={"RX"}, wires=[0]), diff_method="adjoint")
    def f(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliX(0))

    with pytest.raises(DifferentiableCompileError, match="RX.*non-differentiable"):

        @qml.qjit
        def cir(x: float):
            return grad(f)(x)


def test_non_differentiable_gate_nested_cond():
    """Emulate a device with a non-differentiable gate."""

    @qml.qnode(get_custom_device(non_differentiable_gates={"RX"}, wires=1), diff_method="adjoint")
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


def test_non_differentiable_gate_nested_adjoint():
    """Emulate a device with a non-differentiable gate."""

    @qml.qnode(get_custom_device(non_differentiable_gates={"RX"}, wires=1), diff_method="adjoint")
    def f(x):
        adjoint(qml.RX(x, wires=[0]))
        return qml.expval(qml.PauliX(0))

    with pytest.raises(DifferentiableCompileError, match="RX.*non-differentiable"):

        @qml.qjit
        def cir(x: float):
            return grad(f)(x)


def test_non_invertible_gate_simple():
    """Emulate a device with a non-invertible gate."""

    dev = get_custom_device(non_invertible_gates={"RX"}, wires=1)

    @qml.qnode(dev)
    def f(x):
        adjoint(qml.RX(x, wires=0))
        return qml.expval(qml.PauliX(0))

    with pytest.raises(CompileError, match="RX.*not invertible"):

        @qml.qjit
        def cir(x: float):
            return grad(f)(x)


def test_non_invertible_gate_nested_cond():
    """Emulate a device with a non-invertible gate."""

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


def test_non_controllable_gate_simple():
    """Emulate a device with a non-controllable gate."""

    with pytest.raises(CompileError, match="PauliZ.*not controllable"):

        @qjit
        @qml.qnode(get_custom_device(non_controllable_gates={"PauliZ"}, wires=3))
        def f(x: float):
            ctrl(qml.PauliZ(wires=0), control=[1, 2])
            return qml.expval(qml.PauliX(0))


def test_non_invertible_gate_nested_for():
    """Emulate a device with a non-invertible gate."""

    @qml.qnode(get_custom_device(non_invertible_gates={"RX"}, wires=1))
    def f(x):
        @for_loop(0, 10, 1)
        def loop(i):
            adjoint(qml.RX(x, wires=0))

        loop()
        return qml.expval(qml.PauliX(0))

    with pytest.raises(CompileError, match="RX.*not invertible"):

        @qml.qjit
        def cir(x: float):
            return grad(f)(x)


class PauliX2(qml.PauliX):
    """Test operation without the analytic gradient"""

    name = "PauliX2"
    grad_method = "F"


def test_paramshift_obs_simple():
    """Emulate a device with a non-invertible observable."""

    assert qml.Hermitian.grad_method != "A"

    @qml.qnode(get_custom_device(wires=2), diff_method="parameter-shift")
    def f(x):
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


def test_paramshift_gate_simple():
    """Emulate a device with a non-invertible gate."""

    @qml.qnode(get_custom_device(native_gates={"PauliX2"}, wires=1), diff_method="parameter-shift")
    def f(x):
        PauliX2(wires=0)
        return qml.expval(qml.PauliX(0))

    with pytest.raises(
        DifferentiableCompileError, match="PauliX2 does not support analytic differentiation"
    ):

        @qml.qjit
        def cir(x: float):
            return grad(f)(x)


def test_paramshift_gate_while():
    """Emulate a device with a non-invertible gate."""

    @qml.qnode(get_custom_device(native_gates={"PauliX2"}, wires=1), diff_method="parameter-shift")
    def f(x):
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


if __name__ == "__main__":
    pytest.main(["-x", __file__])
