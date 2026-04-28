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
import pennylane as qp
import pytest
import scipy
from jax import numpy as jnp
from numpy.testing import assert_allclose
from pennylane import numpy as pnp

try:
    from pennylane import qcut
except:  # pylint: disable=bare-except
    from pennylane.transforms import qcut

from pennylane.transforms import merge_rotations

from catalyst import measure, qjit
from catalyst.device import QJITDevice
from catalyst.device.decomposition import measurements_from_counts, measurements_from_samples
from catalyst.utils.exceptions import CompileError

# pylint: disable=too-many-lines,line-too-long


def test_add_noise(backend):
    """Test the add_noise transform on a simple circuit"""

    def qnode_builder(device_name):
        """Builder"""

        fcond1 = qp.noise.op_eq(qp.RX) & qp.noise.wires_in([0, 1])
        noise1 = qp.noise.partial_wires(qp.RX, 0.4)

        fcond2 = qp.noise.op_in([qp.RX, qp.RZ])

        def noise2(op, **_):
            qp.CRX(op.data[0], wires=[op.wires[0], (op.wires[0] + 1) % 2])

        noise_model = qp.NoiseModel({fcond1: noise1, fcond2: noise2}, t1=2.0, t2=0.2)

        @partial(qp.noise.add_noise, noise_model=noise_model)
        @qp.qnode(qp.device(device_name, wires=2), interface="jax")
        def qfunc(w, x, y, z):
            qp.RX(w, wires=0)
            qp.RY(x, wires=1)
            qp.CNOT(wires=[0, 1])
            qp.RY(y, wires=0)
            qp.RX(z, wires=1)
            return qp.expval(qp.Z(0) @ qp.Z(1))

        return qfunc

    qnode_control = qnode_builder("default.mixed")
    qnode_backend = qnode_builder(backend)

    expected = jax.jit(qnode_control)(0.9, 0.4, 0.5, 0.6)
    observed = qjit(qnode_backend)(0.9, 0.4, 0.5, 0.6)
    assert np.allclose(expected, observed)

    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)
    assert expected_shape == observed_shape


@pytest.mark.skip(reason="Uses part of old API")
def test_batch_input(backend):
    """Test that batching works for a simple circuit"""

    def qnode_builder(device_name):
        """Builder"""

        @partial(qp.batch_input, argnums=1)
        @qp.qnode(qp.device(device_name, wires=2), interface="jax", diff_method="parameter-shift")
        def qfunc(inputs, weights):
            """Example taken from tests"""
            qp.RY(weights[0], wires=0)
            qp.AngleEmbedding(inputs, wires=range(2), rotation="Y")
            qp.RY(weights[1], wires=1)
            return qp.expval(qp.PauliZ(1))

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    batch_size = 5
    inputs = pnp.random.uniform(0, pnp.pi, (batch_size, 2), requires_grad=False)
    weights = pnp.random.uniform(-pnp.pi, pnp.pi, (2,))

    expected = jax.jit(qnode_control)(inputs, weights)
    observed = qjit(qnode_backend)(inputs, weights)
    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)
    assert np.allclose(expected, observed)
    assert expected_shape == observed_shape


def test_batch_params(backend):
    """Test batch param"""

    def qnode_builder(device_name):
        """Builder"""

        @qp.batch_params
        @qp.qnode(qp.device(device_name, wires=3), interface="jax")
        def qfunc(data, x, weights):
            """Example taken from PL tests"""
            qp.templates.AmplitudeEmbedding(data, wires=[0, 1, 2], normalize=True)
            qp.RX(x, wires=0)
            qp.RY(0.2, wires=1)
            qp.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
            return qp.probs(wires=[0, 2])

        return qfunc

    batch_size = 5
    data = pnp.random.random((batch_size, 8))
    data = data[0]
    data = pnp.array([data, data, data, data, data])
    x = pnp.array([0.1, 0.1, 0.1, 0.1, 0.1], requires_grad=True)
    weights = pnp.ones((batch_size, 10, 3, 3), requires_grad=True)

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)
    expected = jax_jit(data, x, weights)
    observed = compiled(data, x, weights)
    assert np.allclose(expected, observed)

    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)
    assert expected_shape == observed_shape


def test_batch_partial(backend):
    """Test batch_partial"""

    def qnode_builder(device_name):
        """Builder"""

        @qp.qnode(qp.device(device_name, wires=2), interface="jax")
        def qfunc(x, y):
            qp.RX(x, wires=0)
            qp.RY(y, wires=1)
            return qp.expval(qp.Z(0) @ qp.Z(1))

        return qfunc

    batch_size = 4
    x = np.linspace(0.1, 0.5, batch_size)
    y = np.array(0.2)

    qnode_control = qp.batch_partial(qnode_builder("default.qubit"), all_operations=True, y=y)
    qnode_backend = qp.batch_partial(qnode_builder(backend), all_operations=True, y=y)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)
    expected = jax_jit(x)
    observed = compiled(x)
    assert np.allclose(expected, observed)

    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)
    assert expected_shape == observed_shape


def test_cancel_inverses(backend):
    """Test cancel_inverses"""

    def qnode_builder(device_name):
        """Builder"""

        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device(device_name, wires=3), interface="jax")
        def qfunc(x, y, z):
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=1)
            qp.Hadamard(wires=0)
            qp.RX(x, wires=2)
            qp.RY(y, wires=1)
            qp.X(1)
            qp.RZ(z, wires=0)
            qp.RX(y, wires=2)
            qp.CNOT(wires=[0, 2])
            qp.X(1)
            return qp.expval(qp.Z(0))

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    x, y, z = 0.1, 0.2, 0.3
    expected = jax_jit(x, y, z)
    observed = compiled(x, y, z)
    assert np.allclose(expected, observed)

    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)
    assert expected_shape == observed_shape


def test_commute_controlled(backend):
    """Test commute_controlled"""

    def qnode_builder(device_name):
        """Builder"""

        @partial(qp.transforms.commute_controlled, direction="right")
        @qp.qnode(qp.device(device_name, wires=3), interface="jax")
        def qfunc(theta):
            qp.CZ(wires=[0, 2])
            qp.X(2)
            qp.S(wires=0)
            qp.CNOT(wires=[0, 1])
            qp.Y(1)
            qp.CRY(theta, wires=[0, 1])
            qp.PhaseShift(theta / 2, wires=0)
            qp.Toffoli(wires=[0, 1, 2])
            qp.T(wires=0)
            qp.RZ(theta / 2, wires=1)
            return qp.expval(qp.Z(0))

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    expected = jax_jit(0.5)
    observed = compiled(0.5)
    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)

    assert np.allclose(expected, observed)
    assert expected_shape == observed_shape


def test_convert_to_numpy_parameters(backend):
    """Test convert_to_numpy_parameters"""

    def qnode_builder(device_name):
        """Builder"""

        @qp.transforms.convert_to_numpy_parameters
        @qp.qnode(qp.device(device_name, wires=1), interface="jax")
        def qfunc():
            qp.S(0)
            qp.RX(0.1234, 0)
            return qp.expval(qp.X(0))

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    expected = jax_jit()
    observed = compiled()
    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)

    assert np.allclose(expected, observed)
    assert expected_shape == observed_shape


def test_decompose(backend):
    """Test decompose"""

    def qnode_builder(device_name):
        """Builder"""

        @qp.transforms.decompose
        @qp.qnode(qp.device(device_name, wires=3), interface="jax")
        def qfunc():
            qp.Hadamard(wires=[0])
            qp.Toffoli(wires=[0, 1, 2])
            return qp.expval(qp.Z(0))

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    expected = jax_jit()
    observed = compiled()
    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)

    assert np.allclose(expected, observed)
    assert expected_shape == observed_shape


def test_diagonalize_measurements(backend):
    """Test diagonalize_measurements"""

    def qnode_builder(device_name):
        """Builder"""

        @qp.transforms.diagonalize_measurements
        @qp.qnode(qp.device(device_name, wires=3), interface="jax")
        def qfunc(x):
            qp.RY(x[0], wires=0)
            qp.RX(x[1], wires=1)
            return qp.expval(qp.X(0) @ qp.Z(1)), qp.var(0.5 * qp.Y(2) + qp.X(0))

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    x = [np.pi / 4, np.pi / 4]
    expected = jax_jit(x)
    observed = compiled(x)
    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)

    assert np.allclose(expected, observed)
    assert expected_shape == observed_shape


def test_insert(backend):
    """Test insert"""

    def qnode_builder(device_name):
        """Builder"""

        @partial(qp.noise.insert, op=qp.X, op_args=(), position="end")
        @qp.qnode(qp.device(device_name, wires=2), interface="jax")
        def qfunc(w, x, y, z):
            qp.RX(w, wires=0)
            qp.RY(x, wires=1)
            qp.CNOT(wires=[0, 1])
            qp.RY(y, wires=0)
            qp.RX(z, wires=1)
            return qp.expval(qp.Z(0) @ qp.Z(1))

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    expected = jax_jit(0.9, 0.4, 0.5, 0.6)
    observed = compiled(0.9, 0.4, 0.5, 0.6)
    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)

    assert np.allclose(expected, observed)
    assert expected_shape == observed_shape


def test_merge_amplitude_embedding(backend):
    """Test merge_amplitude_embedding"""

    def qnode_builder(device_name):
        """Builder"""

        @qp.transforms.merge_amplitude_embedding
        @qp.qnode(qp.device(device_name, wires=4), interface="jax")
        def qfunc():
            qp.CNOT(wires=[0, 1])
            qp.AmplitudeEmbedding([0, 1], wires=2)
            qp.AmplitudeEmbedding([0, 1], wires=3)
            return qp.state()

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    expected = jax_jit()
    observed = compiled()
    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)

    assert np.allclose(expected, observed)
    assert expected_shape == observed_shape


def test_remove_barrier(backend):
    """Test remove_barrier"""

    def qnode_builder(device_name):
        """Builder"""

        @qp.transforms.remove_barrier
        @qp.qnode(qp.device(device_name, wires=2))
        def qfunc():
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=1)
            qp.Barrier(wires=[0, 1])
            qp.X(0)
            return qp.expval(qp.Z(0))

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    expected = jax_jit()
    observed = compiled()
    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)

    assert np.allclose(expected, observed)
    assert expected_shape == observed_shape


def test_single_qubit_fusion(backend):
    """Test single_qubit_fusion"""

    def qnode_builder(device_name):
        """Builder"""

        @qp.transforms.single_qubit_fusion
        @qp.qnode(qp.device(device_name, wires=1), interface="jax")
        def qfunc(r1, r2):
            qp.Hadamard(wires=0)
            qp.Rot(*r1, wires=0)
            qp.Rot(*r2, wires=0)
            qp.RZ(r1[0], wires=0)
            qp.RZ(r2[0], wires=0)
            return qp.expval(qp.X(0))

        return qfunc

    r1 = [0.1, 0.2, 0.3]
    r2 = [0.4, 0.5, 0.6]

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    expected = jax_jit(r1, r2)
    observed = compiled(r1, r2)
    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)

    assert np.allclose(expected, observed)
    assert expected_shape == observed_shape


def test_split_non_commuting(backend):
    """Test split non commuting"""

    def qnode_builder(device_name):
        """Builder"""

        @qp.transforms.split_non_commuting
        @qp.qnode(qp.device(device_name, wires=6), interface="jax")
        def qfunc():
            """Example taken from PL tests"""
            qp.Hadamard(1)
            qp.Hadamard(0)
            qp.PauliZ(0)
            qp.Hadamard(3)
            qp.Hadamard(5)
            qp.T(5)
            return (
                qp.expval(qp.PauliZ(0) @ qp.PauliZ(1)),
                qp.expval(qp.PauliX(0)),
                qp.expval(qp.PauliZ(1)),
                qp.expval(qp.PauliX(1) @ qp.PauliX(4)),
                qp.expval(qp.PauliX(3)),
                qp.expval(qp.PauliY(5)),
            )

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)
    expected = jax.jit(qnode_control)()
    observed = qjit(qnode_backend)()
    assert np.allclose(expected, observed)

    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)
    assert expected_shape == observed_shape


def test_transpile(backend):
    """Test transpile"""

    def qnode_builder(device_name):
        """Builder"""

        @partial(qp.transforms.transpile, coupling_map=[(0, 1), (1, 3), (3, 2), (2, 0)])
        @qp.qnode(qp.device(device_name, wires=4), interface="jax")
        def qfunc():
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[2, 3])
            qp.CNOT(wires=[1, 3])
            qp.CNOT(wires=[1, 2])
            qp.CNOT(wires=[2, 3])
            qp.CNOT(wires=[0, 3])
            return qp.probs(wires=[0, 1, 2, 3])

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    expected = jax_jit()
    observed = compiled()
    assert np.allclose(expected, observed)

    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)
    assert expected_shape == observed_shape


def test_undo_swaps(backend):
    """Test undo_swaps"""

    def qnode_builder(device_name):
        """Builder"""

        @qp.transforms.undo_swaps
        @qp.qnode(qp.device(device_name, wires=3), interface="jax")
        def qfunc():
            qp.Hadamard(wires=0)
            qp.X(1)
            qp.SWAP(wires=[0, 1])
            qp.SWAP(wires=[0, 2])
            qp.Y(0)
            return qp.expval(qp.Z(0))

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    expected = jax_jit()
    observed = compiled()
    assert np.allclose(expected, observed)

    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)
    assert expected_shape == observed_shape


class TestMitigate:
    """Test error mitigation transforms"""

    def test_fold_global(self, backend):
        """Test fold_global"""

        def qnode_builder(device_name):
            """Builder"""

            @partial(qp.noise.fold_global, scale_factor=2)
            @qp.qnode(qp.device(device_name, wires=3), interface="jax")
            def qfunc(x):
                qp.RX(x[0], wires=0)
                qp.RY(x[1], wires=1)
                qp.RZ(x[2], wires=2)
                qp.CNOT(wires=(0, 1))
                qp.CNOT(wires=(1, 2))
                qp.RX(x[3], wires=0)
                qp.RY(x[4], wires=1)
                qp.RZ(x[5], wires=2)
                return qp.expval(qp.Z(0) @ qp.Z(1) @ qp.Z(2))

            return qfunc

        qnode_control = qnode_builder("default.qubit")
        qnode_backend = qnode_builder(backend)
        x = np.arange(6)

        compiled = qjit(qnode_backend)
        observed = compiled(x)
        expected = qnode_control(x)
        assert np.allclose(expected, observed)

        jax_jit = jax.jit(qnode_control)
        expected = jax_jit(x)
        assert np.allclose(expected, observed)

        _, expected_shape = jax.tree_util.tree_flatten(expected)
        _, observed_shape = jax.tree_util.tree_flatten(observed)
        assert expected_shape == observed_shape

    def test_mitigate_with_zne(self, backend):
        """Test mitigate_with_zne"""

        def qnode_builder(device_name):
            """Builder"""

            @partial(
                qp.noise.mitigate_with_zne,
                scale_factors=[1.0, 2.0, 3.0],
                folding=qp.noise.fold_global,
                extrapolate=qp.noise.poly_extrapolate,
                extrapolate_kwargs={"order": 2},
            )
            @qp.qnode(qp.device(device_name, wires=2), interface="jax")
            def qfunc(w1, w2):
                qp.SimplifiedTwoDesign(w1, w2, wires=range(2))
                return qp.expval(qp.Z(0))

            return qfunc

        n_wires = 2
        n_layers = 2
        shapes = qp.SimplifiedTwoDesign.shape(n_layers, n_wires)
        np.random.seed(0)
        w1, w2 = [np.random.random(s) for s in shapes]

        qnode_control = qnode_builder("default.qubit")
        qnode_backend = qnode_builder(backend)

        compiled = qjit(qnode_backend)
        observed = compiled(w1, w2)
        expected = qnode_control(w1, w2)
        assert np.allclose(expected, observed)

        jax_jit = jax.jit(qnode_control)
        expected = jax_jit(w1, w2)
        assert np.allclose(expected, observed)

        _, expected_shape = jax.tree_util.tree_flatten(expected)
        _, observed_shape = jax.tree_util.tree_flatten(observed)
        assert expected_shape == observed_shape


class TestQuantumMonteCarlo:
    """Test quantum Monte Carlo transforms"""

    def test_apply_controlled_Q(self, backend):
        """Test apply_controlled_Q"""

        def qnode_builder(device_name):
            n_wires = 3
            wires = range(n_wires)
            target_wire = n_wires - 1
            control_wire = n_wires
            a_mat = scipy.stats.unitary_group.rvs(2 ** (n_wires - 1), random_state=1967)
            r_mat = scipy.stats.unitary_group.rvs(2**n_wires, random_state=1967)

            @partial(
                qp.transforms.apply_controlled_Q,
                wires=wires,
                target_wire=target_wire,
                control_wire=control_wire,
                work_wires=None,
            )
            @qp.qnode(qp.device(device_name, wires=n_wires + 1), interface="jax")
            def qfunc():
                qp.QubitUnitary(a_mat, wires=wires[:-1])
                qp.QubitUnitary(r_mat, wires=wires)
                return qp.expval(qp.Z(0))

            return qfunc

        qnode_control = qnode_builder("default.qubit")
        qnode_backend = qnode_builder(backend)

        jax_jit = jax.jit(qnode_control)
        compiled = qjit(qnode_backend)

        expected = jax_jit()
        observed = compiled()
        assert np.allclose(expected, observed)

        _, expected_shape = jax.tree_util.tree_flatten(expected)
        _, observed_shape = jax.tree_util.tree_flatten(observed)
        assert expected_shape == observed_shape

    def test_quantum_monte_carlo(self, backend):
        """Test quantum_monte_carlo"""

        def qnode_builder(device_name):
            n_wires = 3
            wires = range(n_wires)
            target_wire = n_wires - 1
            estimation_wires = range(n_wires, 2 * n_wires)
            a_mat = scipy.stats.unitary_group.rvs(2 ** (n_wires - 1), random_state=1967)
            r_mat = scipy.stats.unitary_group.rvs(2**n_wires, random_state=1967)

            @partial(
                qp.transforms.quantum_monte_carlo,
                wires=wires,
                target_wire=target_wire,
                estimation_wires=estimation_wires,
            )
            @qp.qnode(qp.device(device_name, wires=2 * n_wires), interface="jax")
            def qfunc():
                qp.QubitUnitary(a_mat, wires=wires[:-1])
                qp.QubitUnitary(r_mat, wires=wires)
                return qp.expval(qp.Z(0))

            return qfunc

        qnode_control = qnode_builder("default.qubit")
        qnode_backend = qnode_builder(backend)

        jax_jit = jax.jit(qnode_control)
        compiled = qjit(qnode_backend)

        expected = jax_jit()
        observed = compiled()
        assert np.allclose(expected, observed)

        _, expected_shape = jax.tree_util.tree_flatten(expected)
        _, observed_shape = jax.tree_util.tree_flatten(observed)
        assert expected_shape == observed_shape


class RX_broadcasted(qp.RX):
    """A version of qp.RX that detects batching."""

    ndim_params = (0,)
    compute_decomposition = staticmethod(lambda theta, wires=None: [qp.RX(theta, wires=wires)])


class RZ_broadcasted(qp.RZ):
    """A version of qp.RZ that detects batching."""

    ndim_params = (0,)
    compute_decomposition = staticmethod(lambda theta, wires=None: [qp.RZ(theta, wires=wires)])


parameters = [
    (0.2, np.array([0.1, 0.8, 2.1]), -1.5),
    (0.2, np.array([0.1]), np.array([-0.3])),
    (
        0.2,
        pnp.array([0.1, 0.3], requires_grad=True),
        pnp.array([-0.3, 2.1], requires_grad=False),
    ),
]

coeffs0 = [0.3, -5.1]
H0 = qp.Hamiltonian(qp.math.array(coeffs0), [qp.PauliZ(0), qp.PauliY(1)])

observables = [
    [qp.PauliZ(0)],
    [qp.PauliZ(0) @ qp.PauliY(1)],
    [qp.PauliZ(0), qp.PauliY(1)],
    [H0],
]


class TestBroadcastExpand:
    """Test Broadcast Expand"""

    @pytest.mark.skip(reason="https://github.com/PennyLaneAI/pennylane/issues/2762")
    @pytest.mark.parametrize("params", parameters)
    @pytest.mark.parametrize("obs", observables)
    def test_expansion_qnode(self, backend, params, obs):
        """Test broadcast expand"""

        if obs[0] == H0:
            pytest.xfail(reason="https://github.com/PennyLaneAI/catalyst/issues/339")

        def qnode_builder(device_name):
            """Builder"""

            @qp.transforms.broadcast_expand
            @qp.qnode(qp.device(device_name, wires=2), interface="jax")
            def circuit(x, y, z, obs):
                """Example taken from PL tests"""
                qp.StatePrep(
                    np.array([complex(1, 0), complex(0, 0), complex(0, 0), complex(0, 0)]),
                    wires=[0, 1],
                )
                RX_broadcasted(x, wires=0)
                qp.PauliY(0)
                RX_broadcasted(y, wires=1)
                RZ_broadcasted(z, wires=1)
                qp.Hadamard(1)
                return [qp.expval(ob) for ob in obs]

            return circuit

        qnode_control = qnode_builder("default.qubit")
        qnode_backend = qnode_builder(backend)

        expected = jax.jit(qnode_control)(*params, obs)
        observed = qjit(qnode_backend)(*params, obs)

        assert np.allclose(expected, observed)
        _, expected_shape = jax.tree_util.tree_flatten(expected)
        _, observed_shape = jax.tree_util.tree_flatten(observed)
        assert expected_shape == observed_shape

    @pytest.mark.parametrize("params", parameters)
    @pytest.mark.parametrize("obs", observables)
    def test_expansion_qnode_no_cache(self, backend, params, obs):
        """Test broadcast expand.

        This test is used as an alternative to test_expansion_qnode which cannot succeed due to bug.
        The only difference here is that we specify cache=False on the qnode.

        Delete me once cache=False is not necessary.
        """

        if obs[0] == H0:
            pytest.xfail(reason="https://github.com/PennyLaneAI/catalyst/issues/339")

        def qnode_builder(device_name):
            """Builder"""

            @qp.transforms.broadcast_expand
            @qp.qnode(qp.device(device_name, wires=2), interface="jax", cache=False)
            def circuit(x, y, z, obs):
                """Example taken from PL tests"""
                qp.StatePrep(
                    np.array([complex(1, 0), complex(0, 0), complex(0, 0), complex(0, 0)]),
                    wires=[0, 1],
                )
                RX_broadcasted(x, wires=0)
                qp.PauliY(0)
                RX_broadcasted(y, wires=1)
                RZ_broadcasted(z, wires=1)
                qp.Hadamard(1)
                return [qp.expval(ob) for ob in obs]

            return circuit

        qnode_control = qnode_builder("default.qubit")
        qnode_backend = qnode_builder(backend)

        expected = jax.jit(qnode_control)(*params, obs)
        observed = qjit(qnode_backend)(*params, obs)

        assert np.allclose(expected, observed)
        _, expected_shape = jax.tree_util.tree_flatten(expected)
        _, observed_shape = jax.tree_util.tree_flatten(observed)
        # TODO: See https://github.com/PennyLaneAI/catalyst/issues/1099
        # assert expected_shape == observed_shape
        assert expected_shape.num_leaves == observed_shape.num_leaves


class TestCutCircuitMCTransform:
    """Test Cut Circuit MC Transform"""

    def test_cut_circuit_mc_sample(self, backend):
        """
        Tests that a circuit containing sampling measurements can be cut and
        postprocessed to return bitstrings of the original circuit size.
        """

        def qnode_builder(device_name):
            """Builder"""

            @qp.qnode(qp.device(device_name, wires=2))
            def qfunc(x):
                """Example taken from PL tests."""
                qp.RX(x, wires=0)
                qp.RY(0.543, wires=1)
                qp.WireCut(wires=0)
                qp.CNOT(wires=[0, 1])
                qp.RZ(0.240, wires=0)
                qp.RZ(0.133, wires=1)
                return qp.expval(qp.PauliZ(wires=[0]))

            return qfunc

        qnode_default = qnode_builder("default.qubit")
        qnode_backend = qnode_builder(backend)

        x = jnp.array(0.531)
        cut_circuit_jit = jax.jit(qcut.cut_circuit(qnode_default, use_opt_einsum=False))
        cut_circuit_qjit = qjit(qcut.cut_circuit(qnode_backend, use_opt_einsum=False))

        expected = cut_circuit_jit(x)
        observed = cut_circuit_qjit(x)

        assert_allclose(expected, observed)
        _, expected_shape = jax.tree_util.tree_flatten(expected)
        _, observed_shape = jax.tree_util.tree_flatten(observed)
        assert expected_shape == observed_shape


class TestSplitNonCommuting:
    """Test split_non_commuting"""

    def test_split_non_commuting_single_observable(self, backend):
        """Test split_non_commuting on a single, multi-term observable containing
        non-commuting terms."""

        H4 = (
            qp.PauliX(0) @ qp.PauliZ(2)
            + 3 * qp.PauliZ(2)
            - 2 * qp.PauliX(0)
            + qp.PauliZ(2)
            + qp.PauliZ(2)
        )
        H4 += qp.PauliZ(0) @ qp.PauliX(1) @ qp.PauliY(2)

        def qnode_builder(device_name):
            """Builder"""

            @qp.transforms.split_non_commuting
            @qp.qnode(qp.device(device_name, wires=3))
            def qfunc():
                """Example taken from PL tests."""
                qp.Hadamard(0)
                qp.Hadamard(1)
                qp.PauliZ(1)
                qp.PauliX(2)
                return qp.expval(H4)

            return qfunc

        qnode_backend = qnode_builder(backend)
        qnode_control = qnode_builder("default.qubit")
        expected = jax.jit(qnode_control)()
        observed = qjit(qnode_backend)()

        assert np.allclose(expected, observed)
        _, expected_shape = jax.tree_util.tree_flatten(expected)
        _, observed_shape = jax.tree_util.tree_flatten(observed)
        assert expected_shape == observed_shape

    def test_split_non_commuting_mulitiple_observables(self, backend):
        """Test split_non_commuting on two separate measurements with non-commuting
        observables"""

        def qnode_builder(device_name):
            """Builder"""

            @qp.transforms.split_non_commuting
            @qp.qnode(qp.device(device_name, wires=2))
            def qfunc():
                """Example taken from PL tests"""
                obs1 = qp.prod(qp.PauliX(0), qp.PauliX(1))
                obs2 = qp.prod(qp.PauliX(0), qp.PauliY(1))
                return qp.expval(obs1), qp.expval(obs2)

            return qfunc

        qnode_backend = qnode_builder(backend)
        qnode_control = qnode_builder("default.qubit")
        expected = jax.jit(qnode_control)()
        observed = qjit(qnode_backend)()
        assert np.allclose(expected, observed)
        _, expected_shape = jax.tree_util.tree_flatten(expected)
        _, observed_shape = jax.tree_util.tree_flatten(observed)
        assert expected_shape.num_leaves == observed_shape.num_leaves
        # TODO: See https://github.com/PennyLaneAI/catalyst/issues/1099
        # assert expected_shape == observed_shape


class TestQFuncTransforms:
    """Test QFunc Transforms"""

    @pytest.mark.parametrize(("theta_1", "theta_2"), [(0.3, -0.2)])
    def test_merge_rotations(self, backend, theta_1, theta_2):
        """Merge rotations"""

        def qnode_builder(device_name):
            """Builder"""

            @qp.qnode(qp.device(device_name, wires=3))
            @merge_rotations
            def qfunc(theta_1, theta_2):
                qp.RZ(theta_1, wires=0)
                qp.RZ(theta_2, wires=0)
                return qp.state()

            return qfunc

        qnode_backend = qnode_builder(backend)
        qnode_control = qnode_builder("default.qubit")
        expected = jax.jit(qnode_control)(theta_1, theta_2)
        compiled_function = qjit(qnode_backend)
        observed = compiled_function(theta_1, theta_2)
        assert np.allclose(expected, observed)
        _, expected_shape = jax.tree_util.tree_flatten(expected)
        _, observed_shape = jax.tree_util.tree_flatten(observed)
        assert expected_shape == observed_shape

        # Here we are asserting that there is only one RZ operation
        assert 1 == compiled_function.mlir.count('quantum.custom "RZ"')

    @pytest.mark.xfail(reason="qp.ctrl to HybridCtrl dispatch breaks the method of this transform")
    def test_unroll_ccrz(self, backend):
        """Test unroll_ccrz transform."""
        # TODO: Test by inspecting the circuit actually produced, testing the
        #       results does not verify the transform was applied correctly.

        @qp.transform
        def unroll_ccrz(tape):
            """Needed for lightning.qubit, as it does not natively support expansion of
            multi-controlled RZ."""

            for op in tape:
                if op.name == "C(RZ)":
                    qp.CNOT(wires=[op.control_wires[0], op.target_wires[0]])
                    qp.RZ(-op.data[0] / 4, wires=op.target_wires[0])
                    qp.CNOT(wires=[op.control_wires[1], op.target_wires[0]])
                    qp.RZ(op.data[0] / 4, wires=op.target_wires[0])
                    qp.CNOT(wires=[op.control_wires[0], op.target_wires[0]])
                    qp.RZ(-op.data[0] / 4, wires=op.target_wires[0])
                    qp.CNOT(wires=[op.control_wires[1], op.target_wires[0]])
                    qp.RZ(op.data[0] / 4, wires=op.target_wires[0])
                else:
                    qp.apply(op)

        def sub_circuit():
            """Just a controlled RZ operation."""
            qp.ctrl(qp.RZ, [0, 1], control_values=[0, 0])(jnp.pi, wires=[2])

        def qnode_builder(device_name):
            """Builder"""

            @qp.qnode(qp.device(device_name, wires=3))
            def circuit():
                """Example."""
                unroll_ccrz(sub_circuit)()  # pylint: disable=not-callable
                return qp.state()

            return circuit

        qnode_backend = qnode_builder(backend)
        qnode_control = qnode_builder("default.qubit")
        expected = jax.jit(qnode_control)()
        compiled = qjit(qnode_backend)
        observed = compiled()
        assert np.allclose(expected, observed)
        _, expected_shape = jax.tree_util.tree_flatten(expected)
        _, observed_shape = jax.tree_util.tree_flatten(observed)
        assert expected_shape == observed_shape


class TestTransformValidity:
    """Test validity of transforms."""

    @pytest.mark.parametrize("transform", (measurements_from_counts, measurements_from_samples))
    def test_invalid_modify_measurements_classical_return(self, backend, transform, monkeypatch):
        """Test verification for transforms that are non-batching but modify tape measurements
        while returning classical values."""

        def inject_device_transforms(self, execution_config=None):
            program, config = original_preprocess(self, execution_config)

            program.add_transform(transform, self.wires)

            return program, config

        # Simulate a Qrack-like device that requires measurement process transforms.
        # Qnode transforms raise this error anyway so we cannot use them directly.
        original_preprocess = QJITDevice.preprocess
        monkeypatch.setattr(QJITDevice, "preprocess", inject_device_transforms)

        dev = qp.device(backend, wires=2)

        @partial(transform, device_wires=dev.wires)
        @qp.set_shots(5)
        @qp.qnode(dev)
        def qfunc():
            qp.X(0)
            measurements = [measure(i) for i in range(2)]
            return measurements, qp.expval(qp.PauliZ(0))

        with pytest.raises(
            CompileError,
            match="Transforming MeasurementProcesses is unsupported with non-MeasurementProcess",
        ):
            qjit(qfunc)

    @pytest.mark.parametrize("transform", (measurements_from_counts, measurements_from_samples))
    def test_valid_modify_measurements_no_measurements(self, backend, transform, monkeypatch):
        """Test verification for transforms that are non-batching and in-principle can modify tape
        measurements but don't, while returning classical values."""

        def inject_device_transforms(self, execution_config=None):
            program, config = original_preprocess(self, execution_config)

            program.add_transform(transform, self.wires)

            return program, config

        # Simulate a Qrack-like device that requires meassurement process transforms.
        # Qnode transforms raise this error anyway so we cannot use them directly.
        original_preprocess = QJITDevice.preprocess
        monkeypatch.setattr(QJITDevice, "preprocess", inject_device_transforms)

        dev = qp.device(backend, wires=2)

        @qjit
        @qp.qnode(dev)
        def qfunc():
            qp.X(0)
            measurements = [measure(i) for i in range(2)]
            return measurements

        m1, m2 = qfunc()
        assert m1 == True and m2 == False

    def test_invalid_batch_return_classical_value(self, backend):
        """Test that there's an error raised if the users uses a transform and returns
        a classical value."""

        H4 = (
            qp.PauliX(0) @ qp.PauliZ(2)
            + 3 * qp.PauliZ(2)
            - 2 * qp.PauliX(0)
            + qp.PauliZ(2)
            + qp.PauliZ(2)
        )
        H4 += qp.PauliZ(0) @ qp.PauliX(1) @ qp.PauliY(2)

        @qp.transforms.split_non_commuting
        @qp.qnode(qp.device(backend, wires=3))
        def qfunc():
            """Example taken from PL tests."""
            qp.Hadamard(0)
            qp.Hadamard(1)
            qp.PauliZ(1)
            qp.PauliX(2)
            return [1, qp.expval(H4)]

        with pytest.raises(
            CompileError,
            match="Batch transforms are unsupported with MCMs or non-MeasurementProcess",
        ):
            qjit(qfunc)

    def test_invalid_batch_transform_due_to_measure(self, backend):
        """Test split non commuting"""

        def qnode_builder(device_name):
            """Builder"""

            @qp.transforms.split_non_commuting
            @qp.qnode(qp.device(device_name, wires=6), interface="jax")
            def qfunc():
                """Example taken from PL tests"""
                qp.Hadamard(1)
                qp.Hadamard(0)
                qp.PauliZ(0)
                # There is a measure which is a source of uncertainty!
                measure(0)
                qp.Hadamard(3)
                qp.Hadamard(5)
                qp.T(5)
                return (
                    qp.expval(qp.PauliZ(0) @ qp.PauliZ(1)),
                    qp.expval(qp.PauliX(0)),
                    qp.expval(qp.PauliZ(1)),
                    qp.expval(qp.PauliX(1) @ qp.PauliX(4)),
                    qp.expval(qp.PauliX(3)),
                    qp.expval(qp.PauliY(5)),
                )

            return qfunc

        with pytest.raises(
            CompileError,
            match="Batch transforms are unsupported with MCMs or non-MeasurementProcess",
        ):
            qjit(qnode_builder(backend))

    @pytest.mark.parametrize(("theta_1", "theta_2"), [(0.3, -0.2)])
    def test_valid_due_to_non_batch(self, backend, theta_1, theta_2):
        """This program is valid even in the presence of a mid circuit measurement.
        This is because it will not create multiple tapes, and therefore not
        non-deterministic behaviour across the execution of multiple tapes.
        """

        def qnode_builder(device_name):
            """Builder"""

            @qp.qnode(qp.device(device_name, wires=3))
            @merge_rotations
            def qfunc(theta_1, theta_2):
                measure(0)
                qp.RZ(theta_1, wires=0)
                qp.RZ(theta_2, wires=0)
                return qp.state()

            return qfunc

        qnode_backend = qnode_builder(backend)
        compiled_function = qjit(qnode_backend)
        compiled_function(theta_1, theta_2)

        # Here we are asserting that there is only one RZ operation
        assert 1 == compiled_function.mlir.count('quantum.custom "RZ"')

    def test_informative_transform(self, backend):
        """Informative transforms are not supported!"""

        # Just fake that it is informative
        @partial(qp.transforms.core.transform, is_informative=True)
        def id_transform(tape):
            return [tape], lambda res: res[0]

        @id_transform
        @qp.qnode(qp.device(backend, wires=1))
        def f():
            return qp.state()

        with pytest.raises(CompileError, match="Catalyst does not support informative transforms."):
            qjit(f)


@pytest.mark.xfail(reason="Fails due to use of numpy arrays in transform")
def test_clifford_t_decomposition(backend):
    """Test clifford_t_decomposition"""

    def qnode_builder(device_name):
        """Builder"""

        @qp.transforms.clifford_t_decomposition
        @qp.qnode(qp.device(device_name, wires=2), interface="jax")
        def qfunc(x, y):
            qp.RX(x, 0)
            qp.CNOT([0, 1])
            qp.RY(y, 0)
            return qp.expval(qp.Z(0))

        return qfunc

    x, y = 1.1, 2.2

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    expected = jax_jit(x, y)
    observed = compiled(x, y)
    assert np.allclose(expected, observed)

    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)
    assert expected_shape == observed_shape


@pytest.mark.xfail(reason="Catalyst does not support informative transforms.")
def test_commutation_dag(backend):
    """Test commutation DAG"""

    def qnode_builder(device_name):
        """Builder"""

        @qp.commutation_dag
        @qp.qnode(qp.device(device_name, wires=3), interface="jax")
        def qfunc(x, y, z):
            qp.RX(x, wires=0)
            qp.RX(y, wires=0)
            qp.CNOT(wires=[1, 2])
            qp.RY(y, wires=1)
            qp.Hadamard(wires=2)
            qp.CRZ(z, wires=[2, 0])
            qp.RY(-y, wires=1)
            return qp.expval(qp.Z(0))

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    expected = jax_jit(np.pi / 4, np.pi / 3, np.pi / 2)
    observed = compiled(np.pi / 4, np.pi / 3, np.pi / 2)

    assert expected.get_nodes() == observed.get_nodes()


@pytest.mark.xfail(reason="catalyst.cond cannot accept MeasurementValue as a conditional")
def test_defer_measurements(backend):
    """Test defer_measurements"""
    # The defer_measurements transform looks for MidMeasureMP.
    # Catalyst's `measure` is not a MidMeasureMP.
    # So, this transformation simply does nothing when
    # the program uses `catalyst.measure`.

    def qnode_builder(device_name):
        """Builder"""

        @qp.transforms.defer_measurements
        @qp.qnode(qp.device(device_name, wires=2), interface="jax")
        def qfunc():
            qp.RY(0.123, wires=0)
            qp.Hadamard(wires=1)
            m_0 = qp.measure(1)
            qp.cond(m_0, qp.RY)(np.pi / 4, wires=0)
            return qp.expval(qp.Z(0))

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    expected = jax_jit()
    observed = compiled()
    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)

    assert np.allclose(expected, observed)
    assert expected_shape == observed_shape


@pytest.mark.xfail(reason="catalyst.cond cannot accept MeasurementValue as a conditional")
def test_dynamic_one_shot(backend):
    """Test dynamic_one_shot"""
    # Catalyst has its own dynamic_one_shot transform
    # Applying PennyLane's transform will result in errors.

    def qnode_builder(device_name):
        """Builder"""

        @qp.transforms.dynamic_one_shot
        @qp.qnode(qp.device(device_name, wires=2), interface="jax")
        def qfunc(x, y):
            qp.RX(x, wires=0)
            m0 = qp.measure(0)
            qp.cond(m0, qp.RY)(y, wires=1)
            return qp.expval(op=m0)

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    x = np.pi / 4
    y = np.pi / 4
    expected = jax_jit(x, y)
    observed = compiled(x, y)
    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)

    assert np.allclose(expected, observed)
    assert expected_shape == observed_shape


def test_pattern_matching_optimization(backend):
    """Test pattern_matching_optimization"""

    def qnode_builder(device_name):
        """Builder"""

        ops = [qp.S(0), qp.S(0), qp.Z(0)]
        pattern = qp.tape.QuantumTape(ops)

        @partial(qp.transforms.pattern_matching_optimization, pattern_tapes=[pattern])
        @qp.qnode(qp.device(device_name, wires=5))
        def qfunc():
            qp.S(wires=0)
            qp.Z(0)
            qp.S(wires=1)
            qp.CZ(wires=[0, 1])
            qp.S(wires=1)
            qp.S(wires=2)
            qp.CZ(wires=[1, 2])
            qp.S(wires=2)
            return qp.expval(qp.X(0))

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    expected = jax_jit()
    observed = compiled()
    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)

    assert np.allclose(expected, observed)
    assert expected_shape == observed_shape


@pytest.mark.xfail(
    reason="QJIT error ValueError: Passed tape must end in `qp.expval(H)` or qp.var(H)`, where H is of type `qp.Hamiltonian`"
)
def test_sign_expand(backend):
    """Test sign_expand"""

    def qnode_builder(device_name):
        """Builder"""

        H = qp.Z(0) + 0.5 * qp.Z(2) + qp.Z(1)

        @qp.transforms.sign_expand
        @qp.qnode(qp.device(device_name, wires=2), interface="jax")
        def qfunc():
            qp.Hadamard(wires=0)
            qp.CNOT(wires=[0, 1])
            qp.X(2)
            return qp.expval(H)

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    expected = jax_jit()
    observed = compiled()
    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)

    assert np.allclose(expected, observed)
    assert expected_shape == observed_shape


def test_split_to_single_terms(backend):
    """Test split_to_single_terms"""

    def qnode_builder(device_name):
        """Builder"""

        @qp.transforms.split_to_single_terms
        @qp.qnode(qp.device(device_name, wires=2), interface="jax")
        def qfunc(x):
            qp.RY(x[0], wires=0)
            qp.RX(x[1], wires=1)
            return [
                qp.expval(qp.X(0) @ qp.Z(1) + 0.5 * qp.Y(1) + qp.Z(0)),
                qp.expval(qp.X(1) + qp.Y(1)),
            ]

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    x = [np.pi / 4, np.pi / 4]
    expected = jax_jit(x)
    observed = compiled(x)
    assert np.allclose(expected, observed)

    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)
    assert expected_shape == observed_shape


@pytest.mark.xfail(reason="Both JAX JIT and QJIT fail due to this transform's dependency on PyZX")
def test_to_zx(backend):
    """Test to_zx"""

    def qnode_builder(device_name):
        """Builder"""

        @qp.transforms.to_zx
        @qp.qnode(qp.device(device_name, wires=2), interface="jax")
        def qfunc(p):
            qp.RZ(p[0], wires=1)
            qp.RZ(p[1], wires=1)
            qp.RX(p[2], wires=0)
            qp.Z(0)
            qp.RZ(p[3], wires=1)
            qp.X(1)
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 0])
            return qp.expval(qp.Z(0) @ qp.Z(1))

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    jax_jit = jax.jit(qnode_control)
    compiled = qjit(qnode_backend)

    params = [5 / 4 * np.pi, 3 / 4 * np.pi, 0.1, 0.3]
    expected = jax_jit(params)
    observed = compiled(params)
    assert np.allclose(expected, observed)

    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)
    assert expected_shape == observed_shape


@pytest.mark.xfail(reason="QJIT result differs from PennyLane")
def test_unitary_to_rot(backend):
    """Test unitary_to_rot"""

    def qnode_builder(device_name):
        """Builder"""
        U = scipy.stats.unitary_group.rvs(4)

        @qp.transforms.unitary_to_rot
        @qp.qnode(qp.device(device_name, wires=2), interface="jax")
        def qfunc(angles):
            qp.QubitUnitary(U, wires=[0, 1])
            qp.RX(angles[0], wires=0)
            qp.RY(angles[1], wires=1)
            qp.CNOT(wires=[1, 0])
            return qp.expval(qp.Z(0))

        return qfunc

    qnode_control = qnode_builder("default.qubit")
    qnode_backend = qnode_builder(backend)

    params = [0.2, 0.3]

    compiled = qjit(qnode_backend)
    observed = compiled(params)
    expected = qnode_control(params)
    assert np.allclose(expected, observed)

    jax_jit = jax.jit(qnode_control)
    expected = jax_jit(params)
    assert np.allclose(expected, observed)

    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)
    assert expected_shape == observed_shape
