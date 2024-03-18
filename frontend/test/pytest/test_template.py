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

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp
from scipy.stats import norm

from catalyst import for_loop, qjit


def test_amplitude_embedding(backend):
    """Test amplitude embedding."""

    def amplitude_embedding(f: jax.core.ShapedArray([4], float)):
        qml.AmplitudeEmbedding(features=f, wires=range(2))
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=2)
    params = jax.numpy.array([1 / 2] * 4)
    interpreted_fn = qml.QNode(amplitude_embedding, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_angle_embedding(backend):
    """Test angle embedding."""

    def angle_embedding(f: jax.core.ShapedArray([3], int)):
        qml.AngleEmbedding(features=f, wires=[0, 1, 2], rotation="Z")
        qml.Hadamard(0)
        return qml.probs(wires=[0, 1, 2])

    device = qml.device(backend, wires=3)
    params = jnp.array([1, 2, 3])
    interpreted_fn = qml.QNode(angle_embedding, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_basis_embedding(backend):
    """Test basis embedding."""

    def basis_embedding(f: jax.core.ShapedArray([3], int)):
        qml.BasisEmbedding(features=f, wires=[0, 1, 2])
        return qml.state()

    device = qml.device(backend, wires=3)
    params = jax.numpy.array([1, 1, 1])
    interpreted_fn = qml.QNode(basis_embedding, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_iqp_embedding(backend):
    """Test iqp embedding."""

    def iqp_embedding(f: jax.core.ShapedArray([3], float)):
        qml.IQPEmbedding(f, wires=[0, 1, 2])
        return qml.state()

    device = qml.device(backend, wires=3)
    params = jnp.array([1.0, 2.0, 3.0])
    interpreted_fn = qml.QNode(iqp_embedding, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_qaoa_embedding(backend):
    """Test qaoa embedding."""

    def qaoa_embedding(
        weights: jax.core.ShapedArray([2, 3], float), f: jax.core.ShapedArray([2], float)
    ):
        qml.QAOAEmbedding(features=f, weights=weights, wires=range(2))
        return qml.state()

    device = qml.device(backend, wires=2)
    params = [jnp.array([[0.1, -0.3, 1.5], [3.1, 0.2, -2.8]]), jnp.array([1.0, 2.0])]
    interpreted_fn = qml.QNode(qaoa_embedding, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(*params), jitted_fn(*params))


def test_random_layers(backend):
    """Test random layers."""

    def randomlayers(weights: jax.core.ShapedArray([1, 3], float)):
        qml.RandomLayers(weights=weights, wires=range(2))
        return qml.state()

    device = qml.device(backend, wires=3)
    params = jnp.array([[1.0, 2.0, 3.0]])
    interpreted_fn = qml.QNode(randomlayers, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_strongly_entangled_layers(backend):
    """Test strongly entangled layers."""

    def strongly_entangled_layers(weights: jax.core.ShapedArray([2, 4, 3], float)):
        qml.StronglyEntanglingLayers(weights=weights, wires=range(4))
        return qml.state()

    n_layers = 2
    n_wires = 4
    device = qml.device(backend, wires=n_wires)
    size = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
    params = jnp.array(np.random.random(size))
    interpreted_fn = qml.QNode(strongly_entangled_layers, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_simplified_two_design(backend):
    """Test simplified two design."""

    def simplified_two_design(init_weights, weights):
        qml.SimplifiedTwoDesign(initial_layer_weights=init_weights, weights=weights, wires=range(3))
        return qml.state()

    device = qml.device(backend, wires=3)
    init_weights = jnp.array([jnp.pi, jnp.pi, jnp.pi])
    weights = jax.numpy.array([[[0.0, jnp.pi], [0.0, jnp.pi]], [[jnp.pi, 0.0], [jnp.pi, 0.0]]])
    params = [init_weights, weights]
    interpreted_fn = qml.QNode(simplified_two_design, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(*params), jitted_fn(*params))


def test_basic_entangler_layers(backend):
    """Test basic entangler layers."""

    def basic_entangler_layers(weights):
        qml.BasicEntanglerLayers(weights=weights, wires=range(3))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(3)]

    device = qml.device(backend, wires=3)
    params = jnp.array([[jnp.pi, jnp.pi, jnp.pi]])
    interpreted_fn = qml.QNode(basic_entangler_layers, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_basis_state_preparation(backend):
    """Test basis state preparation."""

    def basis_state_preparation(basis_state):
        qml.BasisStatePreparation(basis_state, wires=range(4))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]

    device = qml.device(backend, wires=4)
    params = jnp.array([0, 1, 1, 0.0])
    interpreted_fn = qml.QNode(basis_state_preparation, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_mottonen_state_preparation(backend):
    """Test mottonen state preparation."""

    def mottonen_state_prep(state: jax.core.ShapedArray([8], complex)):
        qml.MottonenStatePreparation(state_vector=state, wires=range(3))
        return qml.state()

    device = qml.device(backend, wires=3)
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
    interpreted_fn = qml.QNode(mottonen_state_prep, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(state), jitted_fn(state))


def test_arbitrary_state_preparation(backend):
    """Test arbitrary state preparation."""

    def vqe(weights):
        qml.ArbitraryStatePreparation(weights, wires=[0, 1])
        return qml.state()

    device = qml.device(backend, wires=2)
    params = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    interpreted_fn = qml.QNode(vqe, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_all_single_doubles(backend):
    """Test all single doubles."""

    electrons = 2
    qubits = 4
    hf_state = qml.qchem.hf_state(electrons, qubits)
    singles, doubles = qml.qchem.excitations(electrons, qubits)

    # Refer to this issue as to why it isn't exactly the same as usage details:
    # https://github.com/PennyLaneAI/pennylane/issues/3226
    def all_single_doubles(weights):
        qml.templates.AllSinglesDoubles(weights, range(4), hf_state, singles, doubles)
        return qml.state()

    params = jnp.array(np.random.normal(0, np.pi, len(singles) + len(doubles)))
    device = qml.device(backend, wires=4)
    interpreted_fn = qml.QNode(all_single_doubles, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_gate_fabric(backend):
    """Test gate fabric."""

    symbols = ["H", "H"]
    coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

    electrons = 2
    ref_state = qml.qchem.hf_state(electrons, qubits)

    device = qml.device(backend, wires=qubits)

    def ansatz(weights):
        qml.GateFabric(weights, wires=[0, 1, 2, 3], init_state=ref_state, include_pi=True)

        return qml.expval(H)

    layers = 2
    shape = qml.GateFabric.shape(n_layers=layers, n_wires=qubits)
    params = jnp.array(np.random.random(size=shape))

    interpreted_fn = qml.QNode(ansatz, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(qml.numpy.array(params)), jitted_fn(params))


def test_uccsd(backend):
    """Test UCCSD."""

    symbols = ["H", "H", "H"]
    geometry = pnp.array(
        [
            [0.01076341, 0.04449877, 0.0],
            [0.98729513, 1.63059094, 0.0],
            [1.87262415, -0.00815842, 0.0],
        ],
        requires_grad=False,
    )
    electrons = 2
    charge = 1

    H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge)

    hf_state = qml.qchem.hf_state(electrons, qubits)
    singles, doubles = qml.qchem.excitations(electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    dev = qml.device(backend, wires=qubits)
    wires = qubits

    @qml.qnode(dev)
    def circuit(params):
        qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
        return qml.expval(H)

    params = jax.numpy.array(np.zeros(len(singles) + len(doubles)))


def test_kup(backend):
    """Test KUP."""

    def kup(weights):
        qml.kUpCCGSD(weights, wires=[0, 1, 2, 3], k=1, delta_sz=0, init_state=[1, 1, 0, 0])
        return qml.state()

    device = qml.device(backend, wires=4)
    params = jnp.array(np.random.random(size=(1, 6)))
    interpreted_fn = qml.QNode(kup, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_mps(backend):
    """Test MPS."""

    def block(weights, wires):
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RY(weights[0], wires=wires[0])
        qml.RY(weights[1], wires=wires[1])

    def mps(template_weights):
        n_wires = 4
        n_block_wires = 2
        n_params_block = 2
        qml.MPS(range(n_wires), n_block_wires, block, n_params_block, template_weights)
        return qml.state()

    device = qml.device(backend, wires=4)
    params = jnp.array([[0.1, -0.3]] * 3)
    interpreted_fn = qml.QNode(mps, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_ttn(backend):
    """Test TTN."""

    def block(weights, wires):
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RY(weights[0], wires=wires[0])
        qml.RY(weights[1], wires=wires[1])

    def ttn(template_weights):
        n_wires = 4
        n_block_wires = 2
        n_params_block = 2
        qml.TTN(range(n_wires), n_block_wires, block, n_params_block, template_weights)
        return qml.state()

    device = qml.device(backend, wires=4)
    params = jnp.array([[0.1, -0.3]] * 3)
    interpreted_fn = qml.QNode(ttn, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_mera(backend):
    """Test MERA."""

    def block(weights, wires):
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RY(weights[0], wires=wires[0])
        qml.RY(weights[1], wires=wires[1])

    def mera(template_weights):
        n_wires = 4
        n_block_wires = 2
        n_params_block = 2
        qml.MERA(range(n_wires), n_block_wires, block, n_params_block, template_weights)
        return qml.state()

    device = qml.device(backend, wires=4)
    params = jnp.array([[0.1, -0.3]] * 5)
    interpreted_fn = qml.QNode(mera, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_grover(backend):
    """Test Grover."""

    n_wires = 3
    wires = list(range(n_wires))

    def oracle():
        qml.Hadamard(wires=2)
        qml.Toffoli(wires=range(3))
        qml.Hadamard(wires=2)

    def grover_interpreted(num_iterations=1):
        for wire in wires:
            qml.Hadamard(wire)

        for _ in range(num_iterations):
            oracle()
            qml.templates.GroverOperator(wires=wires)

        return qml.state()

    def grover_compiled(num_iterations: int):
        @for_loop(0, n_wires, 1)
        def loop_fn(i):
            qml.Hadamard(i)

        loop_fn()

        @for_loop(0, num_iterations, 1)
        def body(i):
            oracle()
            qml.templates.GroverOperator(wires=wires)

        body()
        return qml.state()

    device = qml.device(backend, wires=n_wires)
    interpreted_fn = qml.QNode(grover_interpreted, device)
    jitted_fn = qjit(qml.QNode(grover_compiled, device))
    # Outcome is considered equivalent since
    # -1 = e^i*phi
    positive = np.allclose(interpreted_fn(1), jitted_fn(1))
    negative = np.allclose(interpreted_fn(1), -jitted_fn(1))
    assert positive or negative


def test_fermionic(backend):
    """Test Fermionic."""

    def fermionic(weight):
        qml.FermionicSingleExcitation(weight, wires=[0, 1, 2])
        return qml.state()

    device = qml.device(backend, wires=3)
    params = jnp.array(0.56)
    interpreted_fn = qml.QNode(fermionic, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_fermionic_double(backend):
    """Test Fermionic double."""

    def fermionic(weight):
        qml.FermionicDoubleExcitation(weight, wires1=[0, 1], wires2=[2, 3, 4])
        return qml.state()

    device = qml.device(backend, wires=5)
    weight = 1.34817
    interpreted_fn = qml.QNode(fermionic, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(weight), jitted_fn(weight))


def test_permute(backend):
    """Test Permute."""

    def permute():
        qml.templates.Permute([4, 2, 0, 1, 3], wires=[0, 1, 2, 3, 4])
        return qml.state()

    device = qml.device(backend, wires=5)
    interpreted_fn = qml.QNode(permute, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(), jitted_fn())


def test_qft(backend):
    """Test QFT."""

    def qft(basis_state):
        qml.BasisState(basis_state, wires=range(3))
        qml.QFT(wires=range(3))
        return qml.state()

    device = qml.device(backend, wires=3)
    params = jnp.array([1.0, 0.0, 0.0])
    interpreted_fn = qml.QNode(qft, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(interpreted_fn(params), jitted_fn(params))


def test_commuting_evolution(backend):
    """Test CommutingEvolution."""

    n_wires = 2
    device = qml.device(backend, wires=n_wires)

    def circuit(time):
        qml.PauliX(0)
        coeff = [1, -1]
        obs = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliY(0) @ qml.PauliX(1)]
        hamiltonian = qml.Hamiltonian(coeff, obs)
        frequencies = (2, 4)
        qml.CommutingEvolution(hamiltonian, time, frequencies)
        return qml.expval(qml.PauliZ(0))

    interpreted_fn = qml.QNode(circuit, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(jitted_fn(1), interpreted_fn(1))


def test_flip_sign(backend):
    """Test FlipSign."""

    def flip_sign():
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.FlipSign([1, 0], wires=list(range(2)))
        return qml.state()

    device = qml.device(backend, wires=2)
    interpreted_fn = qml.QNode(flip_sign, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(jitted_fn(), interpreted_fn())


def test_broadcast_single(backend):
    """Test broadcast single."""

    def broadcast_single(pars):
        qml.broadcast(unitary=qml.RX, pattern="single", wires=[0, 1, 2], parameters=pars)
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=3)
    params = jnp.array([1, 1, 2])
    interpreted_fn = qml.QNode(broadcast_single, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(jitted_fn(params), interpreted_fn(params))


def test_broadcast_double(backend):
    """Test broadcast double."""

    def broadcast_double(pars):
        qml.broadcast(unitary=qml.CRot, pattern="double", wires=[0, 1, 2, 3], parameters=pars)
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=4)
    params = jnp.array([[-1, 2.5, 3], [-1, 4, 2.0]])
    interpreted_fn = qml.QNode(broadcast_double, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(jitted_fn(params), interpreted_fn(params))


def test_broadcast_chain(backend):
    """Test broadcast chain."""

    def broadcast_chain(pars):
        qml.broadcast(unitary=qml.CRot, pattern="chain", wires=[0, 1, 2, 3], parameters=pars)
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=4)
    params = jnp.array([[1.8, 2, 3], [-1.0, 3, 1], [2, 1.2, 4]])
    interpreted_fn = qml.QNode(broadcast_chain, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(jitted_fn(params), interpreted_fn(params))


def test_broadcast_ring(backend):
    """Test broadcast ring."""

    def broadcast_ring(pars):
        qml.broadcast(unitary=qml.CRot, pattern="ring", wires=[0, 1, 2], parameters=pars)
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=3)
    params = jnp.array([[1, 2.2, 3], [-1, 3, 1.0], [2.6, 1, 4]])
    interpreted_fn = qml.QNode(broadcast_ring, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(jitted_fn(params), interpreted_fn(params))


def test_broadcast_pyramid(backend):
    """Test broadcast pyramid."""

    def broadcast_pyramid(pars):
        qml.broadcast(unitary=qml.CRot, pattern="pyramid", wires=[0, 1, 2, 3], parameters=pars)
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=4)
    params = jnp.array([[1, 2.2, 3]] * 3)
    interpreted_fn = qml.QNode(broadcast_pyramid, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(jitted_fn(params), interpreted_fn(params))


def test_broadcast_all_to_all(backend):
    """Test broadcast all to all."""

    def broadcast_all_to_all(pars):
        qml.broadcast(unitary=qml.CRot, pattern="all_to_all", wires=[0, 1, 2, 3], parameters=pars)
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=4)
    params = jnp.array([[1, 2.2, 3]] * 6)
    interpreted_fn = qml.QNode(broadcast_all_to_all, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(jitted_fn(params), interpreted_fn(params))


def test_approx_time_evoluation(backend):
    """Test ApproxTimeEvolution."""

    def approx_time_evolution(time):
        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliX(1)]
        hamiltonian = qml.Hamiltonian(coeffs, obs)
        qml.ApproxTimeEvolution(hamiltonian, time, 1)
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(3)]

    device = qml.device(backend, wires=3)
    params = jnp.array(1)
    interpreted_fn = qml.QNode(approx_time_evolution, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(jitted_fn(params), interpreted_fn(params))


def test_quantum_phase_estimation(backend):
    """Test QuantumPhaseEstimation."""

    phase = 5
    target_wires = [0]
    unitary = qml.RX(phase, wires=0).matrix()
    n_estimation_wires = 5
    estimation_wires = range(1, n_estimation_wires + 1)

    def quantum_phase_estimation():
        qml.Hadamard(wires=target_wires)
        qml.QuantumPhaseEstimation(
            unitary, target_wires=target_wires, estimation_wires=estimation_wires
        )
        return qml.probs(estimation_wires)

    device = qml.device(backend, wires=6)
    interpreted_fn = qml.QNode(quantum_phase_estimation, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(jitted_fn(), interpreted_fn())


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
    device = qml.device("lightning.qubit", n + m + 1)

    def circuit():
        qml.templates.QuantumMonteCarlo(
            probs, func, target_wires=target_wires, estimation_wires=estimation_wires
        )
        return qml.probs(estimation_wires)

    interpreted_fn = qml.QNode(circuit, device)
    jitted_fn = qjit(interpreted_fn)
    assert np.allclose(jitted_fn(), interpreted_fn())


def test_qnn_ticket(backend):  # pylint: disable-next=line-too-long
    """https://discuss.pennylane.ai/t/error-faced-in-training-the-quantum-network-for-estimating-parameters/3624/22"""

    n_dset = 10
    n_qubits = 3
    layers = 2

    dev = qml.device(backend, wires=n_qubits)

    @qml.qnode(dev, diff_method="adjoint")
    def qnn(weights, inputs):
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with=0.5)
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        for i in range(n_qubits - 1):
            if i % 2 == 0:
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)[1::2]]

    x_train = np.random.rand(n_dset, 2**n_qubits)
    weights = np.random.rand(layers, n_qubits)
    expected = qnn(weights, x_train[0])
    observed = qjit(qnn)(weights, x_train[0])
    assert np.allclose(expected, observed)


# Hilbert Schmidt templates take a quantum tape as a parameter.
# Therefore unsuitable for JIT compilation

if __name__ == "__main__":
    pytest.main(["-x", __file__])
