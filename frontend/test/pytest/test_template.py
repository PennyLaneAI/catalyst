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
"""Tests for QJIT compatibility of PennyLane templates."""

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pytest
from scipy.stats import norm

from catalyst import for_loop, qjit

# pylint: disable=too-many-lines


def test_adder(backend):
    """Test Adder."""
    x = 8
    k = 5
    mod = 15

    x_wires = [0, 1, 2, 3]
    work_wires = [4, 5]

    def adder():
        qp.BasisEmbedding(x, wires=x_wires)
        qp.Adder(k, x_wires, mod, work_wires)
        return qp.sample(wires=x_wires)

    device = qp.device(backend, wires=6)
    interpreted_fn = qp.set_shots(qp.QNode(adder, device), shots=2)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(), jitted_fn())


def test_amplitude_embedding(backend):
    """Test amplitude embedding."""

    def amplitude_embedding(f: jax.core.ShapedArray([4], float)):
        qp.AmplitudeEmbedding(features=f, wires=range(2))
        return qp.expval(qp.PauliZ(0))

    device = qp.device(backend, wires=2)
    params = jax.numpy.array([1 / 2] * 4)
    interpreted_fn = qp.QNode(amplitude_embedding, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_angle_embedding(backend):
    """Test angle embedding."""

    def angle_embedding(f: jax.core.ShapedArray([3], int)):
        qp.AngleEmbedding(features=f, wires=[0, 1, 2], rotation="Z")
        qp.Hadamard(0)
        return qp.probs(wires=[0, 1, 2])

    device = qp.device(backend, wires=3)
    params = jnp.array([1, 2, 3])
    interpreted_fn = qp.QNode(angle_embedding, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_basis_embedding(backend):
    """Test basis embedding."""

    def basis_embedding(f: jax.core.ShapedArray([3], int)):
        qp.BasisEmbedding(features=f, wires=[0, 1, 2])
        return qp.state()

    device = qp.device(backend, wires=3)
    params = jax.numpy.array([1, 1, 1])
    interpreted_fn = qp.QNode(basis_embedding, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_cosine_window(backend):
    """Test cosine window."""

    def cosine_window():
        qp.CosineWindow(wires=[0, 1])
        return qp.probs(wires=[0, 1])

    device = qp.device(backend, wires=2)
    interpreted_fn = qp.QNode(cosine_window, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(), jitted_fn())


def test_iqp_embedding(backend):
    """Test iqp embedding."""

    def iqp_embedding(f: jax.core.ShapedArray([3], float)):
        qp.IQPEmbedding(f, wires=[0, 1, 2])
        return qp.state()

    device = qp.device(backend, wires=3)
    params = jnp.array([1.0, 2.0, 3.0])
    interpreted_fn = qp.QNode(iqp_embedding, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_qaoa_embedding(backend):
    """Test qaoa embedding."""

    def qaoa_embedding(
        weights: jax.core.ShapedArray([2, 3], float), f: jax.core.ShapedArray([2], float)
    ):
        qp.QAOAEmbedding(features=f, weights=weights, wires=range(2))
        return qp.state()

    device = qp.device(backend, wires=2)
    params = [jnp.array([[0.1, -0.3, 1.5], [3.1, 0.2, -2.8]]), jnp.array([1.0, 2.0])]
    interpreted_fn = qp.QNode(qaoa_embedding, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(*params), jitted_fn(*params))


def test_random_layers(backend):
    """Test random layers."""

    def randomlayers(weights: jax.core.ShapedArray([1, 3], float)):
        qp.RandomLayers(weights=weights, wires=range(2))
        return qp.state()

    device = qp.device(backend, wires=3)
    params = jnp.array([[1.0, 2.0, 3.0]])
    interpreted_fn = qp.QNode(randomlayers, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_strongly_entangled_layers(backend):
    """Test strongly entangled layers."""

    def strongly_entangled_layers(weights: jax.core.ShapedArray([2, 4, 3], float)):
        qp.StronglyEntanglingLayers(weights=weights, wires=range(4))
        return qp.state()

    n_layers = 2
    n_wires = 4
    device = qp.device(backend, wires=n_wires)
    size = qp.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
    params = jnp.array(np.random.random(size))
    interpreted_fn = qp.QNode(strongly_entangled_layers, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_simplified_two_design(backend):
    """Test simplified two design."""

    def simplified_two_design(init_weights, weights):
        qp.SimplifiedTwoDesign(initial_layer_weights=init_weights, weights=weights, wires=range(3))
        return qp.state()

    device = qp.device(backend, wires=3)
    init_weights = jnp.array([jnp.pi, jnp.pi, jnp.pi])
    weights = jax.numpy.array([[[0.0, jnp.pi], [0.0, jnp.pi]], [[jnp.pi, 0.0], [jnp.pi, 0.0]]])
    params = [init_weights, weights]
    interpreted_fn = qp.QNode(simplified_two_design, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(*params), jitted_fn(*params))


def test_basic_entangler_layers(backend):
    """Test basic entangler layers."""

    def basic_entangler_layers(weights):
        qp.BasicEntanglerLayers(weights=weights, wires=range(3))
        return [qp.expval(qp.PauliZ(wires=i)) for i in range(3)]

    device = qp.device(backend, wires=3)
    params = jnp.array([[jnp.pi, jnp.pi, jnp.pi]])
    interpreted_fn = qp.QNode(basic_entangler_layers, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_mottonen_state_preparation(backend):
    """Test mottonen state preparation."""

    def mottonen_state_prep(state: jax.core.ShapedArray([8], complex)):
        qp.MottonenStatePreparation(state_vector=state, wires=range(3))
        return qp.state()

    device = qp.device(backend, wires=3)
    params = jnp.array(
        [
            complex(1, 0),
            complex(0, 2),
            complex(3, 0),
            complex(0, 4),
            complex(5, 0),
            complex(0, 6),
            complex(7, 0),
            complex(0, 8),
        ]
    )
    state = params / jnp.linalg.norm(params)
    interpreted_fn = qp.QNode(mottonen_state_prep, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(state), jitted_fn(state))


def test_arbitrary_state_preparation(backend):
    """Test arbitrary state preparation."""

    def vqe(weights):
        qp.ArbitraryStatePreparation(weights, wires=[0, 1])
        return qp.state()

    device = qp.device(backend, wires=2)
    params = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    interpreted_fn = qp.QNode(vqe, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_all_single_doubles(backend):
    """Test all single doubles."""

    electrons = 2
    qubits = 4
    hf_state = qp.qchem.hf_state(electrons, qubits)
    singles, doubles = qp.qchem.excitations(electrons, qubits)

    # Refer to this issue as to why it isn't exactly the same as usage details:
    # https://github.com/PennyLaneAI/pennylane/issues/3226
    def all_single_doubles(weights):
        qp.templates.AllSinglesDoubles(weights, range(4), hf_state, singles, doubles)
        return qp.state()

    params = jnp.array(np.random.normal(0, np.pi, len(singles) + len(doubles)))
    device = qp.device(backend, wires=4)
    interpreted_fn = qp.QNode(all_single_doubles, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_gate_fabric(backend):
    """Test gate fabric."""

    symbols = ["H", "H"]
    coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
    H, qubits = qp.qchem.molecular_hamiltonian(symbols, coordinates)

    electrons = 2
    ref_state = qp.qchem.hf_state(electrons, qubits)

    device = qp.device(backend, wires=qubits)

    def ansatz(weights):
        qp.GateFabric(weights, wires=[0, 1, 2, 3], init_state=ref_state, include_pi=True)

        return qp.expval(H)

    layers = 2
    shape = qp.GateFabric.shape(n_layers=layers, n_wires=qubits)
    params = jnp.array(np.random.random(shape))

    interpreted_fn = qp.QNode(ansatz, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(qp.numpy.array(params)), jitted_fn(params))


def test_uccsd(backend):
    """Test UCCSD."""

    symbols = ["H", "H", "H"]
    geometry = np.array(
        [
            [0.01076341, 0.04449877, 0.0],
            [0.98729513, 1.63059094, 0.0],
            [1.87262415, -0.00815842, 0.0],
        ]
    )
    electrons = 2
    charge = 1

    H, qubits = qp.qchem.molecular_hamiltonian(symbols, geometry, charge=charge)

    hf_state = qp.qchem.hf_state(electrons, qubits)
    singles, doubles = qp.qchem.excitations(electrons, qubits)
    s_wires, d_wires = qp.qchem.excitations_to_wires(singles, doubles)
    wires = range(qubits)

    def circuit(params):
        qp.UCCSD(params, wires, s_wires, d_wires, hf_state)
        return qp.expval(H)

    device = qp.device(backend, wires=qubits)
    params = jax.numpy.array(np.zeros(len(singles) + len(doubles)))

    interpreted_fn = qp.QNode(circuit, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_kup(backend):
    """Test KUP."""

    def kup(weights):
        qp.kUpCCGSD(weights, wires=[0, 1, 2, 3], k=1, delta_sz=0, init_state=[1, 1, 0, 0])
        return qp.state()

    device = qp.device(backend, wires=4)
    params = jnp.array(np.random.random((1, 6)))
    interpreted_fn = qp.QNode(kup, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_particle_conserving_u1(backend):
    """Test particle conserving U1"""

    def particle_conserving_u1(weights):
        qp.ParticleConservingU1(weights, range(2), init_state=np.array([1, 1]))
        return qp.expval(qp.PauliZ(0))

    device = qp.device(backend, wires=2)
    weights = jnp.array(np.random.random((1, 1, 2)))
    interpreted_fn = qp.QNode(particle_conserving_u1, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(weights), jitted_fn(weights))


def test_particle_conserving_u2(backend):
    """Test particle conserving U2"""

    def particle_conserving_u2(weights):
        qp.ParticleConservingU2(weights, range(2), init_state=np.array([1, 1]))
        return qp.expval(qp.PauliZ(0))

    device = qp.device(backend, wires=2)
    weights = jnp.array(np.random.random((1, 3)))
    interpreted_fn = qp.QNode(particle_conserving_u2, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(weights), jitted_fn(weights))


def test_mps(backend):
    """Test MPS."""

    def block(weights, wires):
        qp.CNOT(wires=[wires[0], wires[1]])
        qp.RY(weights[0], wires=wires[0])
        qp.RY(weights[1], wires=wires[1])

    def mps(template_weights):
        n_wires = 4
        n_block_wires = 2
        n_params_block = 2
        qp.MPS(range(n_wires), n_block_wires, block, n_params_block, template_weights)
        return qp.state()

    device = qp.device(backend, wires=4)
    params = jnp.array([[0.1, -0.3]] * 3)
    interpreted_fn = qp.QNode(mps, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_ttn(backend):
    """Test TTN."""

    def block(weights, wires):
        qp.CNOT(wires=[wires[0], wires[1]])
        qp.RY(weights[0], wires=wires[0])
        qp.RY(weights[1], wires=wires[1])

    def ttn(template_weights):
        n_wires = 4
        n_block_wires = 2
        n_params_block = 2
        qp.TTN(range(n_wires), n_block_wires, block, n_params_block, template_weights)
        return qp.state()

    device = qp.device(backend, wires=4)
    params = jnp.array([[0.1, -0.3]] * 3)
    interpreted_fn = qp.QNode(ttn, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_mera(backend):
    """Test MERA."""

    def block(weights, wires):
        qp.CNOT(wires=[wires[0], wires[1]])
        qp.RY(weights[0], wires=wires[0])
        qp.RY(weights[1], wires=wires[1])

    def mera(template_weights):
        n_wires = 4
        n_block_wires = 2
        n_params_block = 2
        qp.MERA(range(n_wires), n_block_wires, block, n_params_block, template_weights)
        return qp.state()

    device = qp.device(backend, wires=4)
    params = jnp.array([[0.1, -0.3]] * 5)
    interpreted_fn = qp.QNode(mera, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_two_local_swap_network(backend):
    """Test TwoLocalSwapNetwork."""

    def two_local_swap_network(weights):
        qp.templates.TwoLocalSwapNetwork(
            wires=range(4),
            acquaintances=lambda index, wires, param: qp.CRX(param, index),
            weights=weights,
            fermionic=True,
            shift=False,
        )
        return qp.expval(qp.PauliZ(0))

    device = qp.device(backend, wires=4)
    weights = jnp.array(np.random.random(6))
    interpreted_fn = qp.QNode(two_local_swap_network, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(weights), jitted_fn(weights))


def test_grover(backend):
    """Test Grover."""

    n_wires = 3
    wires = list(range(n_wires))

    def oracle():
        qp.Hadamard(wires=2)
        qp.Toffoli(wires=range(3))
        qp.Hadamard(wires=2)

    def grover_interpreted(num_iterations=1):
        for wire in wires:
            qp.Hadamard(wire)

        for _ in range(num_iterations):
            oracle()
            qp.templates.GroverOperator(wires=wires)

        return qp.state()

    def grover_compiled(num_iterations: int):
        @for_loop(0, n_wires, 1)
        def loop_fn(i):
            qp.Hadamard(i)

        loop_fn()

        @for_loop(0, num_iterations, 1)
        def body(i):
            oracle()
            qp.templates.GroverOperator(wires=wires)

        body()
        return qp.state()

    device = qp.device(backend, wires=n_wires)
    interpreted_fn = qp.QNode(grover_interpreted, device)
    jitted_fn = qjit(qp.QNode(grover_compiled, device))
    # Outcome is considered equivalent since
    # -1 = e^i*phi
    positive = np.allclose(interpreted_fn(1), jitted_fn(1))
    negative = np.allclose(interpreted_fn(1), -jitted_fn(1))
    assert positive or negative


def test_reflection(backend):
    """Test Reflection."""

    @qp.prod
    def hadamards(wires):
        for wire in wires:
            qp.Hadamard(wires=wire)

    def reflection(alpha):
        """Test circuit"""
        qp.RY(1.2, wires=0)
        qp.RY(-1.4, wires=1)
        qp.RX(-2, wires=0)
        qp.CRX(1, wires=[0, 1])
        qp.Reflection(hadamards(range(3)), alpha)
        return qp.probs(wires=range(3))

    x = np.array(0.25)

    device = qp.device(backend, wires=3)
    interpreted_fn = qp.QNode(reflection, device)
    jitted_fn = qjit(qp.QNode(interpreted_fn, device))

    assert np.allclose(interpreted_fn(x), jitted_fn(x))


def test_amplitude_amplification(backend):
    """Test AmplitudeAmplification."""

    def amplitude_amplification(params):
        qp.RY(params[0], wires=0)
        qp.AmplitudeAmplification(
            qp.RY(params[0], wires=0),
            qp.RZ(params[1], wires=0),
            iters=3,
            fixed_point=True,
            work_wire=2,
        )

        return qp.expval(qp.PauliZ(0))

    params = jnp.array([0.9, 0.1])
    device = qp.device(backend, wires=3)
    interpreted_fn = qp.QNode(amplitude_amplification, device)
    jitted_fn = qjit(qp.QNode(interpreted_fn, device))

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_fermionic(backend):
    """Test Fermionic."""

    def fermionic(weight):
        qp.FermionicSingleExcitation(weight, wires=[0, 1, 2])
        return qp.state()

    device = qp.device(backend, wires=3)
    params = jnp.array(0.56)
    interpreted_fn = qp.QNode(fermionic, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_fermionic_double(backend):
    """Test Fermionic double."""

    def fermionic(weight):
        qp.FermionicDoubleExcitation(weight, wires1=[0, 1], wires2=[2, 3, 4])
        return qp.state()

    device = qp.device(backend, wires=5)
    weight = 1.34817
    interpreted_fn = qp.QNode(fermionic, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(weight), jitted_fn(weight))


def test_arbitrary_unitary(backend):
    """Test ArbitraryUnitary."""

    def arbitrary_unitary(weights):
        qp.ArbitraryUnitary(weights, wires=range(2))
        return qp.expval(qp.PauliZ(0))

    weights = jnp.array(np.random.random((15,)))
    device = qp.device(backend, wires=2)
    interpreted_fn = qp.QNode(arbitrary_unitary, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(weights), jitted_fn(weights))


def test_permute(backend):
    """Test Permute."""

    def permute():
        qp.templates.Permute([4, 2, 0, 1, 3], wires=[0, 1, 2, 3, 4])
        return qp.state()

    device = qp.device(backend, wires=5)
    interpreted_fn = qp.QNode(permute, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(), jitted_fn())


def test_qft(backend):
    """Test QFT."""

    def qft(basis_state):
        qp.BasisState(basis_state, wires=range(3))
        qp.QFT(wires=range(3))
        # TODO: investigate global phase
        # return qp.state()
        return qp.probs()

    device = qp.device(backend, wires=3)
    params = jnp.array([1, 0, 0])
    interpreted_fn = qp.QNode(qft, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_aqft(backend):
    """Test AQFT."""

    def aqft(order):
        qp.X(0)
        qp.Hadamard(1)
        qp.AQFT(order, wires=range(3))
        return qp.state()

    device = qp.device(backend, wires=3)
    interpreted_fn = qp.QNode(aqft, device)
    jitted_fn = qjit(interpreted_fn, static_argnames="order")

    assert np.allclose(interpreted_fn(1), jitted_fn(1))


@pytest.mark.xfail(reason="Takes quantum tape as a parameter")
def test_hilbert_schmidt(backend):
    """Test HilbertSchmidt."""
    with qp.QueuingManager.stop_recording():
        u_tape = qp.tape.QuantumTape([qp.Hadamard(0)])

    def v_function(params):
        qp.RZ(params[0], wires=1)

    def hilbert_test(v_params):
        qp.HilbertSchmidt(v_params, v_function=v_function, v_wires=[1], u_tape=u_tape)
        return qp.probs(u_tape.wires + [1])

    v_params = np.array([0])
    device = qp.device(backend, wires=2)
    interpreted_fn = qp.QNode(hilbert_test, device)
    jitted_fn = qjit(hilbert_test)

    assert np.allclose(interpreted_fn(v_params), jitted_fn(v_params))


@pytest.mark.xfail(reason="Takes quantum tape as a parameter")
def test_local_hilbert_schmidt(backend):
    """Test LocalHilbertSchmidt."""
    with qp.QueuingManager.stop_recording():
        u_tape = qp.tape.QuantumTape([qp.CZ(wires=(0, 1))])

    def v_function(params):
        qp.RZ(params[0], wires=2)
        qp.RZ(params[1], wires=3)
        qp.CNOT(wires=[2, 3])
        qp.RZ(params[2], wires=3)
        qp.CNOT(wires=[2, 3])

    def local_hilbert_test(v_params):
        qp.LocalHilbertSchmidt(v_params, v_function=v_function, v_wires=[2, 3], u_tape=u_tape)
        return qp.probs(u_tape.wires + [2, 3])

    v_params = np.array([3 * np.pi / 2, 3 * np.pi / 2, np.pi / 2])
    device = qp.device(backend, wires=4)
    interpreted_fn = qp.QNode(local_hilbert_test, device)
    jitted_fn = qjit(local_hilbert_test)

    assert np.allclose(interpreted_fn(v_params), jitted_fn(v_params))


def test_commuting_evolution(backend):
    """Test CommutingEvolution."""

    n_wires = 2
    device = qp.device(backend, wires=n_wires)

    def circuit(time):
        qp.PauliX(0)
        coeff = [1, -1]
        obs = [qp.PauliX(0) @ qp.PauliY(1), qp.PauliY(0) @ qp.PauliX(1)]
        hamiltonian = qp.Hamiltonian(coeff, obs)
        frequencies = (2, 4)
        qp.CommutingEvolution(hamiltonian, time, frequencies)
        return qp.expval(qp.PauliZ(0))

    interpreted_fn = qp.QNode(circuit, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(1), jitted_fn(1))


def test_flip_sign(backend):
    """Test FlipSign."""

    def flip_sign():
        qp.Hadamard(wires=0)
        qp.Hadamard(wires=1)
        qp.FlipSign([1, 0], wires=list(range(2)))
        return qp.state()

    device = qp.device(backend, wires=2)
    interpreted_fn = qp.QNode(flip_sign, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(), jitted_fn())


def test_qsvt(backend):
    """Test QSVT."""
    block_encoding = qp.Hadamard(wires=0)
    phase_shifts = [qp.RZ(-2 * theta, wires=0) for theta in (1.23, -0.5, 4)]

    def qsvt():
        qp.QSVT(block_encoding, phase_shifts)
        return qp.expval(qp.Z(0))

    device = qp.device(backend, wires=1)
    interpreted_fn = qp.QNode(qsvt, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(), jitted_fn())


def test_approx_time_evoluation(backend):
    """Test ApproxTimeEvolution."""

    def approx_time_evolution(time):
        coeffs = [1, 1]
        obs = [qp.PauliX(0), qp.PauliX(1)]
        hamiltonian = qp.Hamiltonian(coeffs, obs)
        qp.ApproxTimeEvolution(hamiltonian, time, 1)
        return [qp.expval(qp.PauliZ(wires=i)) for i in range(3)]

    device = qp.device(backend, wires=3)
    params = jnp.array(1)
    interpreted_fn = qp.QNode(approx_time_evolution, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_qdrift(backend):
    """Test QDrift."""
    coeffs = [1, 1, 1]
    ops = [qp.PauliX(0), qp.PauliY(0), qp.PauliZ(1)]
    time = jnp.array(0.5)
    seed = 1234

    def qdrift(time):
        hamiltonian = qp.sum(*(qp.s_prod(coeff, op) for coeff, op in zip(coeffs, ops)))
        qp.QDrift(hamiltonian, time, n=2, seed=seed)
        return qp.state()

    device = qp.device(backend, wires=2)
    interpreted_fn = qp.QNode(qdrift, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(time), jitted_fn(time))


def test_trotter_product(backend):
    """Test Trotter product."""
    time = jnp.array(0.5)
    c1 = jnp.array(1.23)
    c2 = jnp.array(-0.45)
    terms = [qp.PauliX(0), qp.PauliZ(0)]

    def trotter_product(time, c1, c2):
        h = qp.sum(
            qp.s_prod(c1, terms[0]),
            qp.s_prod(c2, terms[1]),
        )
        qp.TrotterProduct(h, time, n=2, order=2, check_hermitian=False)

        return qp.state()

    device = qp.device(backend, wires=2)
    interpreted_fn = qp.QNode(trotter_product, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(time, c1, c2), jitted_fn(time, c1, c2))


def test_quantum_phase_estimation(backend):
    """Test QuantumPhaseEstimation."""

    phase = 5
    target_wires = [0]
    unitary = qp.RX(phase, wires=0).matrix()
    n_estimation_wires = 5
    estimation_wires = range(1, n_estimation_wires + 1)

    def quantum_phase_estimation():
        qp.Hadamard(wires=target_wires)
        qp.QuantumPhaseEstimation(
            unitary, target_wires=target_wires, estimation_wires=estimation_wires
        )
        return qp.probs(estimation_wires)

    device = qp.device(backend, wires=6)
    interpreted_fn = qp.QNode(quantum_phase_estimation, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(), jitted_fn())


def test_quantum_montecarlo():
    """Test QuantumMonteCarlo."""

    m = 5
    M = 2**m
    xmax = np.pi
    xs = np.linspace(-xmax, xmax, M)
    probs = np.array([norm().pdf(x) for x in xs])
    probs /= np.sum(probs)
    func = lambda i: np.sin(xs[i]) ** 2
    n = 10
    N = 2**n
    target_wires = range(m + 1)
    estimation_wires = range(m + 1, n + m + 1)
    device = qp.device("lightning.qubit", n + m + 1)

    def circuit():
        qp.templates.QuantumMonteCarlo(
            probs, func, target_wires=target_wires, estimation_wires=estimation_wires
        )
        return qp.probs(estimation_wires)

    interpreted_fn = qp.QNode(circuit, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(), jitted_fn())


def test_qnn_ticket(backend):  # pylint: disable-next=line-too-long
    """https://discuss.pennylane.ai/t/error-faced-in-training-the-quantum-network-for-estimating-parameters/3624/22"""
    n_dset = 10
    n_qubits = 3
    layers = 2

    dev = qp.device(backend, wires=n_qubits)

    @qp.qnode(dev, diff_method="adjoint")
    def qnn(weights, inputs):
        qp.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with=0.5)
        qp.BasicEntanglerLayers(weights, wires=range(n_qubits))
        for i in range(n_qubits - 1):
            if i % 2 == 0:
                qp.CNOT(wires=[i, i + 1])
        return [qp.expval(qp.PauliZ(i)) for i in range(n_qubits)[1::2]]

    x_train = np.random.rand(n_dset, 2**n_qubits)
    weights = np.random.rand(layers, n_qubits)
    expected = qnn(weights, x_train[0])
    observed = qjit(qnn)(weights, x_train[0])
    assert np.allclose(expected, observed)


def test_select(backend):
    """Test Select"""

    def select():
        ops = [qp.X(2), qp.X(3), qp.Y(2), qp.SWAP([2, 3])]
        qp.Select(ops, control=[0, 1])
        return qp.state()

    device = qp.device(backend, wires=4)
    interpreted_fn = qp.QNode(select, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(), jitted_fn())


def test_controlled_sequence(backend):
    """Test ControlledSequence."""

    def controlled_sequence(x):
        """Test circuit"""
        qp.PauliX(2)
        qp.ControlledSequence(qp.RX(x, wires=3), control=[0, 1, 2])
        return qp.probs(wires=range(4))

    x = jnp.array(0.25)
    device = qp.device(backend, wires=4)
    interpreted_fn = qp.QNode(controlled_sequence, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(x), jitted_fn(x))


def test_fable(backend):
    """Test FABLE."""

    def fable(input_matrix):
        qp.FABLE(input_matrix, wires=range(5), tol=0)
        return qp.expval(qp.PauliZ(wires=0))

    input_matrix = np.array(
        [
            [-0.5, -0.4, 0.6, 0.7],
            [0.9, 0.9, 0.8, 0.9],
            [0.8, 0.7, 0.9, 0.8],
            [0.9, 0.7, 0.8, 0.3],
        ]
    )

    device = qp.device(backend, wires=5)
    interpreted_fn = qp.QNode(fable, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(input_matrix), jitted_fn(input_matrix))


def test_qubitization(backend):
    """Test Qubitization."""

    def qubitization(coeffs):
        H = qp.ops.LinearCombination(coeffs, [qp.Z(0), qp.Z(1), qp.Z(0) @ qp.Z(2)])
        qp.Hadamard(wires=0)
        qp.Qubitization(H, control=[3, 4])
        return qp.expval(qp.PauliZ(0) @ qp.PauliZ(4))

    coeffs = [0.1, 0.3, -0.3]
    device = qp.device(backend, wires=5)
    interpreted_fn = qp.QNode(qubitization, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(coeffs), jitted_fn(coeffs))


def test_qrom(backend):
    """Test QROM."""

    def qrom():
        qp.QROM(["1", "0", "0", "1"], control_wires=[0, 1], target_wires=[2], work_wires=[3])
        return qp.probs(wires=3)

    device = qp.device(backend, wires=4)
    interpreted_fn = qp.QNode(qrom, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(), jitted_fn())


def test_prepselprep(backend):
    """Test PrepSelPrep"""

    params = np.array([0.4, 0.5, 0.1, 0.3])

    def prepselprep(coeffs):
        H = qp.ops.LinearCombination(
            coeffs, [qp.Y(0), qp.Y(1) @ qp.Y(2), qp.X(0), qp.X(1) @ qp.X(2)]
        )
        qp.PrepSelPrep(H, control=(3, 4))
        return qp.expval(qp.PauliZ(3) @ qp.PauliZ(4))

    device = qp.device(backend, wires=5)
    interpreted_fn = qp.QNode(prepselprep, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(params), jitted_fn(params))


# This test passes on Catalyst version 0.8.0 but fails on 0.9.0 or greater
@pytest.mark.xfail(reason="QJIT gives an incorrect result")
def test_mod_exp(backend):
    """Test ModExp."""
    base = 2
    mod = 7

    x_wires = [0, 1]
    output_wires = [2, 3, 4]
    work_wires = [5, 6, 7, 8, 9]

    def mod_exp():
        qp.X(0)
        qp.X(1)
        qp.X(4)
        qp.ModExp(x_wires, output_wires, base, mod, work_wires)
        return qp.sample(wires=output_wires)

    device = qp.device(backend, wires=10)
    interpreted_fn = qp.QNode(mod_exp, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(), jitted_fn())


def test_multiplier(backend):
    """Test Multiplier."""
    x = 3
    k = 4
    mod = 7

    x_wires = [0, 1, 2]
    work_wires = [3, 4, 5, 6, 7]

    def multiplier():
        qp.BasisEmbedding(x, wires=x_wires)
        qp.Multiplier(k, x_wires, mod, work_wires)
        return qp.sample(wires=x_wires)

    device = qp.device(backend, wires=8)
    interpreted_fn = qp.set_shots(qp.QNode(multiplier, device), shots=2)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(), jitted_fn())


def test_out_adder(backend):
    """Test OutAdder."""
    mod = 7

    x_wires = [0, 1, 2]
    y_wires = [3, 4, 5]
    output_wires = [7, 8, 9]
    work_wires = [6, 10]

    def out_adder():
        qp.X(0)
        qp.X(2)
        qp.X(3)
        qp.X(4)
        qp.OutAdder(x_wires, y_wires, output_wires, mod, work_wires)
        return qp.sample(wires=output_wires)

    device = qp.device(backend, wires=11)
    interpreted_fn = qp.set_shots(qp.QNode(out_adder, device), shots=10000)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(), jitted_fn())


def test_out_multiplier(backend):
    """Test OutMultiplier."""
    mod = 12

    x_wires = [0, 1]
    y_wires = [2, 3, 4]
    output_wires = [6, 7, 8, 9]
    work_wires = [5, 10]

    def out_multiplier():
        qp.X(1)
        qp.X(2)
        qp.X(3)
        qp.X(4)
        qp.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
        return qp.sample(wires=output_wires)

    device = qp.device(backend, wires=11)
    interpreted_fn = qp.set_shots(qp.QNode(out_multiplier, device), shots=2)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(), jitted_fn())


def test_phase_adder(backend):
    """Test PhaseAdder."""
    x = 8
    k = 5
    mod = 15

    x_wires = [0, 1, 2, 3]
    work_wire = [4]

    def phase_adder():
        qp.BasisEmbedding(x, wires=x_wires)
        qp.QFT(wires=x_wires)
        qp.PhaseAdder(k, x_wires, mod, work_wire)
        qp.adjoint(qp.QFT)(wires=x_wires)
        return qp.sample(wires=x_wires)

    device = qp.device(backend, wires=range(5))
    interpreted_fn = qp.set_shots(qp.QNode(phase_adder, device), shots=2)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(), jitted_fn())


@pytest.mark.xfail(reason="Qutrit operators not supported on lightning.")
def test_qutrit_basis_state_preparation(backend):
    """Test QutritBasisStatePreparation."""
    basis_state = [0, 1]
    wires = [0, 1]
    obs = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

    def qutrit_basis_state_preparation(state, obs):
        qp.QutritBasisStatePreparation(state, wires)

        return [qp.expval(qp.THermitian(A=obs, wires=i)) for i in range(3)]

    device = qp.device(backend, wires=2)
    interpreted_fn = qp.QNode(qutrit_basis_state_preparation, device)
    jitted_fn = qjit(interpreted_fn)

    assert np.allclose(interpreted_fn(basis_state, obs), jitted_fn(basis_state, obs))


def test_basis_rotation(backend):
    """Test BasisRotation"""

    unitary_matrix = jnp.array(
        [
            [0.51378719 + 0.0j, 0.0546265 + 0.79145487j, -0.2051466 + 0.2540723j],
            [0.62651582 + 0.0j, -0.00828925 - 0.60570321j, -0.36704948 + 0.32528067j],
            [-0.58608928 + 0.0j, 0.03902657 + 0.04633548j, -0.57220635 + 0.57044649j],
        ]
    )

    def basis_rotation(unitary_matrix, check):
        qp.BasisState(qp.math.array([1, 1, 0]), wires=[0, 1, 2])
        qp.BasisRotation(
            wires=range(3),
            unitary_matrix=unitary_matrix,
            check=check,
        )
        return qp.expval(qp.PauliZ(0) @ qp.PauliZ(1))

    device = qp.device(backend, wires=3)
    interpreted_fn = qp.QNode(basis_rotation, device)
    jitted_fn = qjit(interpreted_fn, static_argnums=1)

    assert np.allclose(interpreted_fn(unitary_matrix, False), jitted_fn(unitary_matrix, False))


if __name__ == "__main__":
    pytest.main(["-x", __file__])
