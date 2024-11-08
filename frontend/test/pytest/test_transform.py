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
from typing import Callable, Sequence

import jax
import numpy as np
import pennylane as qml
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
from catalyst.utils.exceptions import CompileError

# pylint: disable=too-many-lines,line-too-long


def test_add_noise(backend):
    """Test the add_noise transform on a simple circuit"""

    def qnode_builder(device_name):
        """Builder"""

        fcond1 = qml.noise.op_eq(qml.RX) & qml.noise.wires_in([0, 1])
        noise1 = qml.noise.partial_wires(qml.RX, 0.4)

        fcond2 = qml.noise.op_in([qml.RX, qml.RZ])

        def noise2(op, **_):
            qml.CRX(op.data[0], wires=[op.wires[0], (op.wires[0] + 1) % 2])

        noise_model = qml.NoiseModel({fcond1: noise1, fcond2: noise2}, t1=2.0, t2=0.2)

        @partial(qml.transforms.add_noise, noise_model=noise_model)
        @qml.qnode(qml.device(device_name, wires=2), interface="jax")
        def qfunc(w, x, y, z):
            qml.RX(w, wires=0)
            qml.RY(x, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=0)
            qml.RX(z, wires=1)
            return qml.expval(qml.Z(0) @ qml.Z(1))

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

        @partial(qml.batch_input, argnums=1)
        @qml.qnode(qml.device(device_name, wires=2), interface="jax", diff_method="parameter-shift")
        def qfunc(inputs, weights):
            """Example taken from tests"""
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

        @qml.batch_params
        @qml.qnode(qml.device(device_name, wires=3), interface="jax")
        def qfunc(data, x, weights):
            """Example taken from PL tests"""
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

        @qml.qnode(qml.device(device_name, wires=2), interface="jax")
        def qfunc(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(qml.Z(0) @ qml.Z(1))

        return qfunc

    batch_size = 4
    x = np.linspace(0.1, 0.5, batch_size)
    y = np.array(0.2)

    qnode_control = qml.batch_partial(qnode_builder("default.qubit"), all_operations=True, y=y)
    qnode_backend = qml.batch_partial(qnode_builder(backend), all_operations=True, y=y)

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

        @qml.transforms.cancel_inverses
        @qml.qnode(qml.device(device_name, wires=3), interface="jax")
        def qfunc(x, y, z):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.Hadamard(wires=0)
            qml.RX(x, wires=2)
            qml.RY(y, wires=1)
            qml.X(1)
            qml.RZ(z, wires=0)
            qml.RX(y, wires=2)
            qml.CNOT(wires=[0, 2])
            qml.X(1)
            return qml.expval(qml.Z(0))

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

        @partial(qml.transforms.commute_controlled, direction="right")
        @qml.qnode(qml.device(device_name, wires=3), interface="jax")
        def qfunc(theta):
            qml.CZ(wires=[0, 2])
            qml.X(2)
            qml.S(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.Y(1)
            qml.CRY(theta, wires=[0, 1])
            qml.PhaseShift(theta / 2, wires=0)
            qml.Toffoli(wires=[0, 1, 2])
            qml.T(wires=0)
            qml.RZ(theta / 2, wires=1)
            return qml.expval(qml.Z(0))

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

        @qml.transforms.convert_to_numpy_parameters
        @qml.qnode(qml.device(device_name, wires=1), interface="jax")
        def qfunc():
            qml.S(0)
            qml.RX(0.1234, 0)
            return qml.expval(qml.X(0))

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

        @qml.transforms.diagonalize_measurements
        @qml.qnode(qml.device(device_name, wires=3), interface="jax")
        def qfunc(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=1)
            return qml.expval(qml.X(0) @ qml.Z(1)), qml.var(0.5 * qml.Y(2) + qml.X(0))

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

        @partial(qml.transforms.insert, op=qml.X, op_args=(), position="end")
        @qml.qnode(qml.device(device_name, wires=2), interface="jax")
        def qfunc(w, x, y, z):
            qml.RX(w, wires=0)
            qml.RY(x, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=0)
            qml.RX(z, wires=1)
            return qml.expval(qml.Z(0) @ qml.Z(1))

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

        @qml.transforms.merge_amplitude_embedding
        @qml.qnode(qml.device(device_name, wires=4), interface="jax")
        def qfunc():
            qml.CNOT(wires=[0, 1])
            qml.AmplitudeEmbedding([0, 1], wires=2)
            qml.AmplitudeEmbedding([0, 1], wires=3)
            return qml.state()

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

        @qml.transforms.remove_barrier
        @qml.qnode(qml.device(device_name, wires=2))
        def qfunc():
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.Barrier(wires=[0, 1])
            qml.X(0)
            return qml.expval(qml.Z(0))

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

        @qml.transforms.single_qubit_fusion
        @qml.qnode(qml.device(device_name, wires=1), interface="jax")
        def qfunc(r1, r2):
            qml.Hadamard(wires=0)
            qml.Rot(*r1, wires=0)
            qml.Rot(*r2, wires=0)
            qml.RZ(r1[0], wires=0)
            qml.RZ(r2[0], wires=0)
            return qml.expval(qml.X(0))

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

        @qml.transforms.split_non_commuting
        @qml.qnode(qml.device(device_name, wires=6), interface="jax")
        def qfunc():
            """Example taken from PL tests"""
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

    _, expected_shape = jax.tree_util.tree_flatten(expected)
    _, observed_shape = jax.tree_util.tree_flatten(observed)
    assert expected_shape == observed_shape


def test_transpile(backend):
    """Test transpile"""

    def qnode_builder(device_name):
        """Builder"""

        @partial(qml.transforms.transpile, coupling_map=[(0, 1), (1, 3), (3, 2), (2, 0)])
        @qml.qnode(qml.device(device_name, wires=4), interface="jax")
        def qfunc():
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[1, 3])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[0, 3])
            return qml.probs(wires=[0, 1, 2, 3])

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

        @qml.transforms.undo_swaps
        @qml.qnode(qml.device(device_name, wires=3), interface="jax")
        def qfunc():
            qml.Hadamard(wires=0)
            qml.X(1)
            qml.SWAP(wires=[0, 1])
            qml.SWAP(wires=[0, 2])
            qml.Y(0)
            return qml.expval(qml.Z(0))

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

    @pytest.mark.xfail(reason="PennyLane and QJIT give different values")
    def test_fold_global(self, backend):
        """Test fold_global"""

        def qnode_builder(device_name):
            """Builder"""

            @partial(qml.transforms.fold_global, scale_factor=2)
            @qml.qnode(qml.device(device_name, wires=3), interface="jax")
            def qfunc(x):
                qml.RX(x[0], wires=0)
                qml.RY(x[1], wires=1)
                qml.RZ(x[2], wires=2)
                qml.CNOT(wires=(0, 1))
                qml.CNOT(wires=(1, 2))
                qml.RX(x[3], wires=0)
                qml.RY(x[4], wires=1)
                qml.RZ(x[5], wires=2)
                return qml.expval(qml.Z(0) @ qml.Z(1) @ qml.Z(2))

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

    @pytest.mark.xfail(reason="PennyLane and QJIT give different values")
    def test_mitigate_with_zne(self, backend):
        """Test mitigate_with_zne"""

        def qnode_builder(device_name):
            """Builder"""

            @partial(
                qml.transforms.mitigate_with_zne,
                scale_factors=[1.0, 2.0, 3.0],
                folding=qml.transforms.fold_global,
                extrapolate=qml.transforms.poly_extrapolate,
                extrapolate_kwargs={"order": 2},
            )
            @qml.qnode(qml.device(device_name, wires=2), interface="jax")
            def qfunc(w1, w2):
                qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
                return qml.expval(qml.Z(0))

            return qfunc

        n_wires = 2
        n_layers = 2
        shapes = qml.SimplifiedTwoDesign.shape(n_layers, n_wires)
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
                qml.transforms.apply_controlled_Q,
                wires=wires,
                target_wire=target_wire,
                control_wire=control_wire,
                work_wires=None,
            )
            @qml.qnode(qml.device(device_name, wires=n_wires + 1), interface="jax")
            def qfunc():
                qml.QubitUnitary(a_mat, wires=wires[:-1])
                qml.QubitUnitary(r_mat, wires=wires)
                return qml.expval(qml.Z(0))

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
                qml.transforms.quantum_monte_carlo,
                wires=wires,
                target_wire=target_wire,
                estimation_wires=estimation_wires,
            )
            @qml.qnode(qml.device(device_name, wires=2 * n_wires), interface="jax")
            def qfunc():
                qml.QubitUnitary(a_mat, wires=wires[:-1])
                qml.QubitUnitary(r_mat, wires=wires)
                return qml.expval(qml.Z(0))

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


class RX_broadcasted(qml.RX):
    """A version of qml.RX that detects batching."""

    ndim_params = (0,)
    compute_decomposition = staticmethod(lambda theta, wires=None: [qml.RX(theta, wires=wires)])


class RZ_broadcasted(qml.RZ):
    """A version of qml.RZ that detects batching."""

    ndim_params = (0,)
    compute_decomposition = staticmethod(lambda theta, wires=None: [qml.RZ(theta, wires=wires)])


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
H0 = qml.Hamiltonian(qml.math.array(coeffs0), [qml.PauliZ(0), qml.PauliY(1)])

observables = [
    [qml.PauliZ(0)],
    [qml.PauliZ(0) @ qml.PauliY(1)],
    [qml.PauliZ(0), qml.PauliY(1)],
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

            @qml.transforms.broadcast_expand
            @qml.qnode(qml.device(device_name, wires=2), interface="jax")
            def circuit(x, y, z, obs):
                """Example taken from PL tests"""
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

            @qml.transforms.broadcast_expand
            @qml.qnode(qml.device(device_name, wires=2), interface="jax", cache=False)
            def circuit(x, y, z, obs):
                """Example taken from PL tests"""
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

            @qml.qnode(qml.device(device_name, wires=2, shots=None))
            def qfunc(x):
                """Example taken from PL tests."""
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
            qml.PauliX(0) @ qml.PauliZ(2)
            + 3 * qml.PauliZ(2)
            - 2 * qml.PauliX(0)
            + qml.PauliZ(2)
            + qml.PauliZ(2)
        )
        H4 += qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliY(2)

        def qnode_builder(device_name):
            """Builder"""

            @qml.transforms.split_non_commuting
            @qml.qnode(qml.device(device_name, wires=3))
            def qfunc():
                """Example taken from PL tests."""
                qml.Hadamard(0)
                qml.Hadamard(1)
                qml.PauliZ(1)
                qml.PauliX(2)
                return qml.expval(H4)

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

            @qml.transforms.split_non_commuting
            @qml.qnode(qml.device(device_name, wires=2, shots=None))
            def qfunc():
                """Example taken from PL tests"""
                obs1 = qml.prod(qml.PauliX(0), qml.PauliX(1))
                obs2 = qml.prod(qml.PauliX(0), qml.PauliY(1))
                return qml.expval(obs1), qml.expval(obs2)

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

            @qml.qnode(qml.device(device_name, wires=3))
            @merge_rotations
            def qfunc(theta_1, theta_2):
                qml.RZ(theta_1, wires=0)
                qml.RZ(theta_2, wires=0)
                return qml.state()

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

    @pytest.mark.xfail(reason="qml.ctrl to HybridCtrl dispatch breaks the method of this transform")
    def test_unroll_ccrz(self, backend):
        """Test unroll_ccrz transform."""
        # TODO: Test by inspecting the circuit actually produced, testing the
        #       results does not verify the transform was applied correctly.

        @qml.transform
        def unroll_ccrz(tape):
            """Needed for lightning.qubit, as it does not natively support expansion of
            multi-controlled RZ."""

            for op in tape:
                if op.name == "C(RZ)":
                    qml.CNOT(wires=[op.control_wires[0], op.target_wires[0]])
                    qml.RZ(-op.data[0] / 4, wires=op.target_wires[0])
                    qml.CNOT(wires=[op.control_wires[1], op.target_wires[0]])
                    qml.RZ(op.data[0] / 4, wires=op.target_wires[0])
                    qml.CNOT(wires=[op.control_wires[0], op.target_wires[0]])
                    qml.RZ(-op.data[0] / 4, wires=op.target_wires[0])
                    qml.CNOT(wires=[op.control_wires[1], op.target_wires[0]])
                    qml.RZ(op.data[0] / 4, wires=op.target_wires[0])
                else:
                    qml.apply(op)

        def sub_circuit():
            """Just a controlled RZ operation."""
            qml.ctrl(qml.RZ, [0, 1], control_values=[0, 0])(jnp.pi, wires=[2])

        def qnode_builder(device_name):
            """Builder"""

            @qml.qnode(qml.device(device_name, wires=3))
            def circuit():
                """Example."""
                unroll_ccrz(sub_circuit)()
                return qml.state()

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

    def test_return_classical_value(self, backend):
        """Test that there's an error raised if the users uses a transform and returns
        a classical value."""

        H4 = (
            qml.PauliX(0) @ qml.PauliZ(2)
            + 3 * qml.PauliZ(2)
            - 2 * qml.PauliX(0)
            + qml.PauliZ(2)
            + qml.PauliZ(2)
        )
        H4 += qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliY(2)

        msg = (
            "A transformed quantum function must return either a single measurement, "
            "or a nonempty sequence of measurements."
        )
        with pytest.raises(CompileError, match=msg):

            @qjit
            @qml.transforms.split_non_commuting
            @qml.qnode(qml.device(backend, wires=3))
            def qfunc():
                """Example taken from PL tests."""
                qml.Hadamard(0)
                qml.Hadamard(1)
                qml.PauliZ(1)
                qml.PauliX(2)
                return [1, qml.expval(H4)]

    def test_invalid_batch_transform_due_to_measure(self, backend):
        """Test split non commuting"""

        def qnode_builder(device_name):
            """Builder"""

            @qml.transforms.split_non_commuting
            @qml.qnode(qml.device(device_name, wires=6), interface="jax")
            def qfunc():
                """Example taken from PL tests"""
                qml.Hadamard(1)
                qml.Hadamard(0)
                qml.PauliZ(0)
                # There is a measure which is a source of uncertainty!
                measure(0)
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

        with pytest.raises(CompileError, match="Multiple tapes are generated"):
            qjit(qnode_builder(backend))

    @pytest.mark.parametrize(("theta_1", "theta_2"), [(0.3, -0.2)])
    def test_valid_due_to_non_batch(self, backend, theta_1, theta_2):
        """This program is valid even in the presence of a mid circuit measurement.
        This is because it will not create multiple tapes, and therefore not
        non-deterministic behaviour across the execution of multiple tapes.
        """

        def qnode_builder(device_name):
            """Builder"""

            @qml.qnode(qml.device(device_name, wires=3))
            @merge_rotations
            def qfunc(theta_1, theta_2):
                measure(0)
                qml.RZ(theta_1, wires=0)
                qml.RZ(theta_2, wires=0)
                return qml.state()

            return qfunc

        qnode_backend = qnode_builder(backend)
        compiled_function = qjit(qnode_backend)
        compiled_function(theta_1, theta_2)

        # Here we are asserting that there is only one RZ operation
        assert 1 == compiled_function.mlir.count('quantum.custom "RZ"')

    def test_informative_transform(self, backend):
        """Informative transforms are not supported!"""

        def _id(tape: qml.tape.QuantumTape) -> (Sequence[qml.tape.QuantumTape], Callable):
            return [tape], lambda res: res[0]

        # Just fake that it is informative
        id_transform = qml.transforms.core.transform(_id, is_informative=True)

        with pytest.raises(CompileError, match="Catalyst does not support informative transforms."):

            @qjit
            @id_transform
            @qml.qnode(qml.device(backend, wires=1))
            def f():
                return qml.state()


@pytest.mark.xfail(reason="Fails due to use of numpy arrays in transform")
def test_clifford_t_decomposition(backend):
    """Test clifford_t_decomposition"""

    def qnode_builder(device_name):
        """Builder"""

        @qml.transforms.clifford_t_decomposition
        @qml.qnode(qml.device(device_name, wires=2), interface="jax")
        def qfunc(x, y):
            qml.RX(x, 0)
            qml.CNOT([0, 1])
            qml.RY(y, 0)
            return qml.expval(qml.Z(0))

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

        @qml.commutation_dag
        @qml.qnode(qml.device(device_name, wires=3), interface="jax")
        def qfunc(x, y, z):
            qml.RX(x, wires=0)
            qml.RX(y, wires=0)
            qml.CNOT(wires=[1, 2])
            qml.RY(y, wires=1)
            qml.Hadamard(wires=2)
            qml.CRZ(z, wires=[2, 0])
            qml.RY(-y, wires=1)
            return qml.expval(qml.Z(0))

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

        @qml.transforms.defer_measurements
        @qml.qnode(qml.device(device_name, wires=2), interface="jax")
        def qfunc():
            qml.RY(0.123, wires=0)
            qml.Hadamard(wires=1)
            m_0 = qml.measure(1)
            qml.cond(m_0, qml.RY)(np.pi / 4, wires=0)
            return qml.expval(qml.Z(0))

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

        @qml.transforms.dynamic_one_shot
        @qml.qnode(qml.device(device_name, wires=2), interface="jax")
        def qfunc(x, y):
            qml.RX(x, wires=0)
            m0 = qml.measure(0)
            qml.cond(m0, qml.RY)(y, wires=1)
            return qml.expval(op=m0)

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


@pytest.mark.xfail(
    reason="QJIT fails with ValueError: Eagerly computing the adjoint (lazy=False) is only supported on single operators."
)
def test_pattern_matching_optimization(backend):
    """Test pattern_matching_optimization"""

    def qnode_builder(device_name):
        """Builder"""

        ops = [qml.S(0), qml.S(0), qml.Z(0)]
        pattern = qml.tape.QuantumTape(ops)

        @partial(qml.transforms.pattern_matching_optimization, pattern_tapes=[pattern])
        @qml.qnode(qml.device(device_name, wires=5))
        def qfunc():
            qml.S(wires=0)
            qml.Z(0)
            qml.S(wires=1)
            qml.CZ(wires=[0, 1])
            qml.S(wires=1)
            qml.S(wires=2)
            qml.CZ(wires=[1, 2])
            qml.S(wires=2)
            return qml.expval(qml.X(0))

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
    reason="QJIT error ValueError: Passed tape must end in `qml.expval(H)` or qml.var(H)`, where H is of type `qml.Hamiltonian`"
)
def test_sign_expand(backend):
    """Test sign_expand"""

    def qnode_builder(device_name):
        """Builder"""

        H = qml.Z(0) + 0.5 * qml.Z(2) + qml.Z(1)

        @qml.transforms.sign_expand
        @qml.qnode(qml.device(device_name, wires=2), interface="jax")
        def qfunc():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.X(2)
            return qml.expval(H)

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


@pytest.mark.xfail(reason="JIT and QJIT return different shapes")
def test_split_to_single_terms(backend):
    """Test split_to_single_terms"""

    def qnode_builder(device_name):
        """Builder"""

        @qml.transforms.split_to_single_terms
        @qml.qnode(qml.device(device_name, wires=2), interface="jax")
        def qfunc(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=1)
            return [
                qml.expval(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
                qml.expval(qml.X(1) + qml.Y(1)),
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

        @qml.transforms.to_zx
        @qml.qnode(qml.device(device_name, wires=2), interface="jax")
        def qfunc(p):
            qml.RZ(p[0], wires=1)
            qml.RZ(p[1], wires=1)
            qml.RX(p[2], wires=0)
            qml.Z(0)
            qml.RZ(p[3], wires=1)
            qml.X(1)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 0])
            return qml.expval(qml.Z(0) @ qml.Z(1))

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

        @qml.transforms.unitary_to_rot
        @qml.qnode(qml.device(device_name, wires=2), interface="jax")
        def qfunc(angles):
            qml.QubitUnitary(U, wires=[0, 1])
            qml.RX(angles[0], wires=0)
            qml.RY(angles[1], wires=1)
            qml.CNOT(wires=[1, 0])
            return qml.expval(qml.Z(0))

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
