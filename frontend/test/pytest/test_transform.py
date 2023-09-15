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

from numpy.testing import assert_allclose
from functools import partial
import pennylane as qml
from pennylane import numpy as pnp
from pennylane_lightning.lightning_qubit import LightningQubit
import numpy as np
import pytest
from catalyst.jax_tracer import QJITDevice


from catalyst import qjit
import jax

@pytest.mark.skip(reason="Uses part of old API")
def test_batch_input(backend):
    """Test that batching works for a simple circuit"""

    @partial(qml.batch_input, argnum=1)
    @qml.qnode(qml.device(backend, wires=2), interface="jax", diff_method="parameter-shift")
    def interpreted(inputs, weights):
        qml.RY(weights[0], wires=0)
        qml.AngleEmbedding(inputs, wires=range(2), rotation="Y")
        qml.RY(weights[1], wires=1)
        return qml.expval(qml.PauliZ(1))

    batch_size = 5
    inputs = pnp.random.uniform(0, pnp.pi, (batch_size, 2), requires_grad=False)
    weights = pnp.random.uniform(-pnp.pi, pnp.pi, (2,))

    jax_jit = jax.jit(interpreted)(inputs, weights)
    compiled = qjit(interpreted, keep_intermediate=True)

    assert_allclose(jax_jit, compiled(inputs, weights))

def test_batch_params(backend):

    @qml.batch_params
    @qml.qnode(qml.device(backend, wires=3), interface="jax")
    def interpreted(data, x, weights):
        qml.templates.AmplitudeEmbedding(data, wires=[0, 1, 2], normalize=True)
        qml.RX(x, wires=0)
        qml.RY(0.2, wires=1)
        qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
        return qml.probs(wires=[0, 2])

    batch_size = 5
    data = pnp.random.random((batch_size, 8))
    data = data[0]
    data = pnp.array([data, data, data, data, data])
    x = pnp.linspace(0.1, 0.5, batch_size, requires_grad=True)
    weights = pnp.ones((batch_size, 10, 3, 3), requires_grad=True)
    jax_jit = jax.jit(interpreted)
    compiled = qjit(interpreted)
    expected = jax_jit(data, x, weights)
    observed = compiled(data, x, weights)
    assert np.allclose(expected, observed)

def test_split_non_commuting(backend):

        @qml.transforms.split_non_commuting
        @qml.qnode(qml.device(backend, wires=6), interface="jax")
        def interpreted():
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

        jax_jit = jax.jit(interpreted)
        jax_jit()
        compiled = qjit(interpreted)
        assert np.allclose(jax_jit(), compiled())

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
# TODO: Uncomment when fixed: https://github.com/PennyLaneAI/pennylane/issues/4601
#    ([H0], exp_fn_H0),
]

"""Broadcast expand appears to only work on default.qubit."""
class TestBroadcastExpand:
    @pytest.mark.parametrize("params, size", parameters_and_size)
    @pytest.mark.parametrize("obs, exp_fn", observables_and_exp_fns)
    def test_expansion_qnode(self, params, size, obs, exp_fn):

        @qml.transforms.broadcast_expand
        @qml.qnode(qml.device("lightning.qubit", wires=2), interface="jax")
        def circuit(x, y, z, obs):
            qml.StatePrep(np.array([complex(1, 0), complex(0, 0), complex(0, 0), complex(0, 0)]), wires=[0, 1])
            RX_broadcasted(x, wires=0)
            qml.PauliY(0)
            RX_broadcasted(y, wires=1)
            RZ_broadcasted(z, wires=1)
            qml.Hadamard(1)
            return [qml.expval(ob) for ob in obs]

        expected = jax.jit(circuit)(*params, obs)
        result = qjit(circuit)(*params, obs)

        assert np.allclose(result, expected)


"""
class TestSignExpand:

    def test_sign_expand(self):

        @qjit
        @qml.transforms.sign_expand
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            qml.PauliX(0)
            return qml.var(qml.Hamiltonian([1.5], [qml.PauliZ(0) @ qml.PauliZ(1)]))
        print(circuit())

"""
