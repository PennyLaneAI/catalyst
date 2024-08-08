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
from jax import numpy as jnp
from numpy.testing import assert_allclose
from pennylane import numpy as pnp

try:
    from pennylane import qcut
except:  # pylint: disable=bare-except
    from pennylane.transforms import qcut

from pennylane.transforms import hamiltonian_expand, merge_rotations, sum_expand

from catalyst import measure, qjit
from catalyst.utils.exceptions import CompileError

# pylint: disable=unnecessary-lambda-assignment


@pytest.mark.skip(reason="Uses part of old API")
def test_batch_input(backend):
    """Test that batching works for a simple circuit"""

    def qnode_builder(device_name):
        """Builder"""

        @partial(qml.batch_input, argnum=1)
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
        #breakpoint()
        qj = qjit(qnode_backend)
        #breakpoint()
        observed = qj(*params, obs)
        #observed = qjit(qnode_backend)(*params, obs)
        breakpoint()
        assert np.allclose(expected, observed)
        _, expected_shape = jax.tree_util.tree_flatten(expected)
        _, observed_shape = jax.tree_util.tree_flatten(observed)
        breakpoint()
        # TODO: expected is tuple, observed is list
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


class TestHamiltonianExpand:
    """Test Hamiltonian Expand"""

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
            """Builder"""

            @hamiltonian_expand
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


class TestSumExpand:
    """Test Sum Expand"""

    def test_sum_expand(self, backend):
        """Test Sum Expand"""

        def qnode_builder(device_name):
            """Builder"""

            @sum_expand
            @qml.qnode(qml.device(device_name, wires=2, shots=None))
            def qfunc():
                """Example taken from PL tests"""
                obs1 = qml.prod(qml.PauliX(0), qml.PauliX(1))
                obs2 = qml.prod(qml.PauliX(0), qml.PauliY(1))
                return [qml.expval(obs1), qml.expval(obs2)]

            return qfunc

        qnode_backend = qnode_builder(backend)
        qnode_control = qnode_builder("default.qubit")
        expected = jax.jit(qnode_control)()
        observed = qjit(qnode_backend)()
        assert np.allclose(expected, observed)
        _, expected_shape = jax.tree_util.tree_flatten(expected)
        _, observed_shape = jax.tree_util.tree_flatten(observed)
        assert expected_shape == observed_shape


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
            @hamiltonian_expand
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
