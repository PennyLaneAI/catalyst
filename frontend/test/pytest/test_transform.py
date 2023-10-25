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
"""End to end tests for QNode transforms.

As a note, these tests do not attempt to be as exhaustive as the
tests in PL. Maybe they should be, but we need to find a way to
avoid duplication of code.

The transforms tested here are the transforms listed in
https://github.com/PennyLaneAI/pennylane/pull/4440:
    * batch_input
    * batch_params

which correspond to QNode transforms.
"""

from functools import partial

import jax
import numpy as np
import pennylane as qml
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose
from pennylane import numpy as pnp
from pennylane.transforms import (
    hamiltonian_expand,
    qcut,
    sum_expand,
)
from pennylane_lightning.lightning_qubit import LightningQubit

from catalyst import qjit
from catalyst.pennylane_extensions import QJITDevice


@pytest.mark.skip(reason="Uses part of old API")
def test_batch_input(backend):
    """Test that batching works for a simple circuit"""

    def qnode_builder(device_name):
        @partial(qml.batch_input, argnum=1)
        @qml.qnode(
            qml.device(backend_name, wires=2), interface="jax", diff_method="parameter-shift"
        )
        def qfunc(inputs, weights):
            qml.RY(weights[0], wires=0)
            qml.AngleEmbedding(inputs, wires=range(2), rotation="Y")
            qml.RY(weights[1], wires=1)
            return qml.expval(qml.PauliZ(1))

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    batch_size = 5
    inputs = pnp.random.uniform(0, pnp.pi, (batch_size, 2), requires_grad=False)
    weights = pnp.random.uniform(-pnp.pi, pnp.pi, (2,))

    expected_output = jax.jit(qnode_control)(inputs, weights)
    observed_output = qjit(qnode_backend)(inputs, weights)

    assert_allclose(expected_output, observed_output)


@pytest.mark.skip(reason="Temporary, please investigate")
def test_batch_params(backend):
    def qnode_builder(device_name):
        @qml.batch_params
        @qml.qnode(qml.device(device_name, wires=3), interface="jax")
        def qfunc(data, x, weights):
            qml.templates.AmplitudeEmbedding(data, wires=[0, 1, 2], normalize=True)
            qml.RX(x, wires=0)
            qml.RY(0.2, wires=1)
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
            return qml.probs(wires=[0, 2])

        return qfunc

    batch_size = 5
    data = pnp.random.random((batch_size, 8))
    data = data[0]
    data = pnp.array([data, data, data, data, data])
    x = pnp.linspace(0.1, 0.5, batch_size, requires_grad=True)
    weights = pnp.ones((batch_size, 10, 3, 3), requires_grad=True)

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)
    expected = jax_jit(data, x, weights)
    observed = compiled(data, x, weights)
    assert np.allclose(expected, observed)


@pytest.mark.skipif(
    backend="lightning.kokkos", reason="https://github.com/PennyLaneAI/pennylane/issues/4731"
)
def test_split_non_commuting(backend):
    def qnode_builder(device_name):
        @qml.transforms.split_non_commuting
        @qml.qnode(qml.device(backend, wires=6), interface="jax")
        def qfunc():
            qml.Hadamard(1)
            qml.Hadamard(0)
            qml.PauliZ(0)
            qml.Hadamard(3)
            qml.Hadamard(5)
            qml.T(5)
            return (
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(qml.PauliX(1) @ qml.PauliX(4)),
                qml.expval(qml.PauliX(3)),
                qml.expval(qml.PauliY(5)),
            )

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)
    expected = jax.jit(qnode_control)()
    observed = qjit(qnode_backend)()
    assert np.allclose(expected, observed)


class RX_broadcasted(qml.RX):
    """A version of qml.RX that detects batching."""

    ndim_params = (0,)
    compute_decomposition = staticmethod(lambda theta, wires=None: [qml.RX(theta, wires=wires)])


class RZ_broadcasted(qml.RZ):
    """A version of qml.RZ that detects batching."""

    ndim_params = (0,)
    compute_decomposition = staticmethod(lambda theta, wires=None: [qml.RZ(theta, wires=wires)])


parameters_and_size = [
    [(0.2, np.array([0.1, 0.8, 2.1]), -1.5), 3],
    [(0.2, np.array([0.1]), np.array([-0.3])), 1],
    [
        (
            0.2,
            pnp.array([0.1, 0.3], requires_grad=True),
            pnp.array([-0.3, 2.1], requires_grad=False),
        ),
        2,
    ],
]

coeffs0 = [0.3, -5.1]
H0 = qml.Hamiltonian(qml.math.array(coeffs0), [qml.PauliZ(0), qml.PauliY(1)])

# Here we exploit the product structure of our circuit
exp_fn_Z0 = lambda x, y, z: -qml.math.cos(x) * qml.math.ones_like(y) * qml.math.ones_like(z)
exp_fn_Y1 = lambda x, y, z: qml.math.sin(y) * qml.math.cos(z) * qml.math.ones_like(x)
exp_fn_Z0Y1 = lambda x, y, z: exp_fn_Z0(x, y, z) * exp_fn_Y1(x, y, z)
exp_fn_Z0_and_Y1 = lambda x, y, z: qml.math.array(
    [exp_fn_Z0(x, y, z), exp_fn_Y1(x, y, z)],
    like=exp_fn_Z0(x, y, z) + exp_fn_Y1(x, y, z),
)
exp_fn_H0 = lambda x, y, z: exp_fn_Z0(x, y, z) * coeffs0[0] + exp_fn_Y1(x, y, z) * coeffs0[1]

observables_and_exp_fns = [
    ([qml.PauliZ(0)], exp_fn_Z0),
    ([qml.PauliZ(0) @ qml.PauliY(1)], exp_fn_Z0Y1),
    ([qml.PauliZ(0), qml.PauliY(1)], exp_fn_Z0_and_Y1),
    #  TODO: Uncomment when fixed: https://github.com/PennyLaneAI/pennylane/issues/4601
    # ([H0], exp_fn_H0),
]


class TestBroadcastExpand:
    @pytest.mark.skip(reason="https://github.com/PennyLaneAI/pennylane/issues/4734")
    @pytest.mark.parametrize("params, size", parameters_and_size)
    @pytest.mark.parametrize("obs, exp_fn", observables_and_exp_fns)
    def test_expansion_qnode(self, backend, params, size, obs, exp_fn):
        def qnode_builder(device_name):
            @qml.transforms.broadcast_expand
            @qml.qnode(qml.device(device_name, wires=2), interface="jax")
            def circuit(x, y, z, obs):
                qml.StatePrep(
                    np.array([complex(1, 0), complex(0, 0), complex(0, 0), complex(0, 0)]),
                    wires=[0, 1],
                )
                RX_broadcasted(x, wires=0)
                qml.PauliY(0)
                RX_broadcasted(y, wires=1)
                RZ_broadcasted(z, wires=1)
                qml.Hadamard(1)
                return [qml.expval(ob) for ob in obs]

            return circuit

        qnode_control = qnode_builder("lightning.qubit")
        qnode_backend = qnode_builder(backend)

        expected = jax.jit(qnode_control)(*params, obs)
        observed = qjit(qnode_backend)(*params, obs)

        assert np.allclose(e, observed)


class TestCutCircuitMCTransform:
    @pytest.mark.skipif(
        backend="lightning.kokkos", reason="https://github.com/PennyLaneAI/pennylane/issues/4731"
    )
    def test_cut_circuit_mc_sample(self, backend):
        """
        Tests that a circuit containing sampling measurements can be cut and
        postprocessed to return bitstrings of the original circuit size.
        """

        def qnode_builder(device_name):
            @qml.qnode(qml.device(device_name, wires=2, shots=None))
            def qfunc(x):
                """Example"""
                qml.RX(x, wires=0)
                qml.RY(0.543, wires=1)
                qml.WireCut(wires=0)
                qml.CNOT(wires=[0, 1])
                qml.RZ(0.240, wires=0)
                qml.RZ(0.133, wires=1)
                return qml.expval(qml.PauliZ(wires=[0]))

            return qfunc

        qnode_default = qnode_builder("default.qubit")
        qnode_backend = qnode_builder(backend)

        x = jnp.array(0.531)
        cut_circuit_jit = jax.jit(qcut.cut_circuit(qnode_default, use_opt_einsum=False))
        cut_circuit_qjit = qjit(qcut.cut_circuit(qnode_backend, use_opt_einsum=False))

        expected = cut_circuit_jit(x)
        observed = cut_circuit_qjit(x)

        assert np.allclose(expected, observed)


class TestHamiltonianExpand:
    @pytest.mark.skipif(
        backend="lightning.kokkos", reason="https://github.com/PennyLaneAI/pennylane/issues/4731"
    )
    def test_hamiltonian_expand(self, backend):
        """Test hamiltonian expand."""

        H4 = (
            qml.PauliX(0) @ qml.PauliZ(2)
            + 3 * qml.PauliZ(2)
            - 2 * qml.PauliX(0)
            + qml.PauliZ(2)
            + qml.PauliZ(2)
        )
        H4 += qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliY(2)

        def qnode_builder(device_name):
            @hamiltonian_expand
            @qml.qnode(qml.device(device_name, wires=3))
            def qfunc():
                qml.Hadamard(0)
                qml.Hadamard(1)
                qml.PauliZ(1)
                qml.PauliX(2)
                return qml.expval(H4)

            return qfunc

        qnode_backend = qnode_builder(backend)
        qnode_control = qnode_builder("default.qubit")
        expected = jax.jit(qnode_control)()
        observed = jax.jit(qnode_backend)()

        assert np.allclose(expected, observed)


class TestSumExpand:
    def test_sum_expand(self):
        dev = qml.device("lightning.qubit", wires=2, shots=None)

        @sum_expand
        @qml.qnode(dev)
        def circuit():
            obs1 = qml.prod(qml.PauliX(0), qml.PauliX(1))
            obs2 = qml.prod(qml.PauliX(0), qml.PauliY(1))
            return [qml.expval(obs1), qml.expval(obs2)]

        assert np.allclose(circuit(), qjit(circuit)())
