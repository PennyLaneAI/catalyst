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

"""Test integration for catalyst.mitigate_with_zne."""

import jax
import numpy as np
import pennylane as qml
import pytest

import catalyst


@pytest.mark.parametrize("params", [0.1, 0.2, 0.3, 0.4, 0.5])
def test_single_measurement(params):
    """Test that without noise the same results are returned for single measurements."""
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(device=dev)
    def circuit(x):
        qml.Hadamard(wires=0)
        qml.RZ(x, wires=0)
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.Hadamard(wires=1)
        return qml.expval(qml.PauliY(wires=0))

    @catalyst.qjit
    def mitigated_qnode(args):
        return catalyst.mitigate_with_zne(circuit, scale_factors=jax.numpy.array([1, 2, 3]), deg=2)(
            args
        )

    assert np.allclose(mitigated_qnode(params), circuit(params))


@pytest.mark.parametrize("params", [0.1, 0.2, 0.3, 0.4, 0.5])
def test_multiple_measurements(params):
    """Test that without noise the same results are returned for multiple measurements"""
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(device=dev)
    def circuit(x):
        qml.Hadamard(wires=0)
        qml.RZ(x, wires=0)
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.Hadamard(wires=1)
        return qml.expval(qml.PauliY(wires=0)), qml.expval(qml.PauliY(wires=1))

    @catalyst.qjit
    def mitigated_qnode(args):
        return catalyst.mitigate_with_zne(circuit, scale_factors=jax.numpy.array([1, 2, 3]), deg=2)(
            args
        )

    assert np.allclose(mitigated_qnode(params), circuit(params))


@pytest.mark.parametrize("params", [0.1, 0.2, 0.3, 0.4, 0.5])
def test_single_measurement_control_flow(params):
    """Test that without noise the same results are returned for single measurement and with control flow."""
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(device=dev)
    def circuit(x, n):
        @catalyst.for_loop(0, n, 1)
        def loop_0(i):
            qml.RX(x, wires=0)

        loop_0()

        qml.Hadamard(wires=0)
        qml.RZ(x, wires=0)
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.Hadamard(wires=1)

        @catalyst.for_loop(0, n, 1)
        def loop_1(i):
            qml.RX(x, wires=0)

        loop_1()
        return qml.expval(qml.PauliY(wires=0))

    @catalyst.qjit(keep_intermediate=True)
    def mitigated_qnode(args, n):
        return catalyst.mitigate_with_zne(circuit, scale_factors=jax.numpy.array([1, 2, 3]))(
            args, n
        )

    assert np.allclose(mitigated_qnode(params, 3), catalyst.qjit(circuit)(params, 3))


def test_not_qnode_error():
    """Test that when applied not on a QNode the transform raises an error."""
    dev = qml.device("lightning.qubit", wires=2)

    def circuit(x):
        return jax.numpy.sin(x)

    @catalyst.qjit(keep_intermediate=True)
    def mitigated_function(args):
        return catalyst.mitigate_with_zne(circuit, scale_factors=jax.numpy.array([1, 2, 3]))(args)

    with pytest.raises(TypeError, match="A QNode is expected, got the classical function"):
        mitigated_function(0.1)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
