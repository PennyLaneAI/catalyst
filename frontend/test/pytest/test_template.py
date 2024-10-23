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
from scipy.stats import norm

from catalyst import for_loop, qjit


def test_adder(backend):
    """Test Adder."""
    x = 8
    k = 5
    mod = 15

    x_wires = [0, 1, 2, 3]
    work_wires = [4, 5]

    def adder():
        qml.BasisEmbedding(x, wires=x_wires)
        qml.Adder(k, x_wires, mod, work_wires)
        return qml.sample(wires=x_wires)

    device = qml.device(backend, wires=6, shots=2)
    interpreted_fn = qml.QNode(adder, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn()
    jitted_result = jitted_fn()

    assert np.allclose(interpreted_result, jitted_result)


def test_amplitude_embedding(backend):
    """Test amplitude embedding."""

    def amplitude_embedding(f: jax.core.ShapedArray([4], float)):
        qml.AmplitudeEmbedding(features=f, wires=range(2))
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=2)
    params = jax.numpy.array([1 / 2] * 4)
    interpreted_fn = qml.QNode(amplitude_embedding, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


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

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


def test_basis_embedding(backend):
    """Test basis embedding."""

    def basis_embedding(f: jax.core.ShapedArray([3], int)):
        qml.BasisEmbedding(features=f, wires=[0, 1, 2])
        return qml.state()

    device = qml.device(backend, wires=3)
    params = jax.numpy.array([1, 1, 1])
    interpreted_fn = qml.QNode(basis_embedding, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


@pytest.mark.xfail(reason="Displacement operator not supported on lightning.")
def test_displacement_embedding(backend):
    """Test displacement embedding."""

    def displacement_embedding(features):
        qml.DisplacementEmbedding(features, range(3))
        qml.Beamsplitter(0.5, 0, wires=[2, 1])
        qml.Beamsplitter(0.5, 0, wires=[1, 0])
        return qml.expval(qml.QuadX(0))

    features = jax.numpy.array([1.0, 1.0, 1.0])
    device = qml.device(backend, wires=3)
    interpreted_fn = qml.QNode(displacement_embedding, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(features)
    jitted_result = jitted_fn(features)

    assert np.allclose(interpreted_result, jitted_result)


@pytest.mark.xfail(reason="Squeezing operator not supported on lightning.")
def test_squeezing_embedding(backend):
    """Test squeezing embedding."""

    def displacement_embedding(features):
        qml.SqueezingEmbedding(features, range(3))
        qml.Beamsplitter(0.5, 0, wires=[2, 1])
        qml.Beamsplitter(0.5, 0, wires=[1, 0])
        return qml.expval(qml.QuadX(0))

    features = jax.numpy.array([1.0, 1.0, 1.0])
    device = qml.device(backend, wires=3)
    interpreted_fn = qml.QNode(displacement_embedding, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(features)
    jitted_result = jitted_fn(features)

    assert np.allclose(interpreted_result, jitted_result)


def test_iqp_embedding(backend):
    """Test iqp embedding."""

    def iqp_embedding(f: jax.core.ShapedArray([3], float)):
        qml.IQPEmbedding(f, wires=[0, 1, 2])
        return qml.state()

    device = qml.device(backend, wires=3)
    params = jnp.array([1.0, 2.0, 3.0])
    interpreted_fn = qml.QNode(iqp_embedding, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


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

    interpreted_result = interpreted_fn(*params)
    jitted_result = jitted_fn(*params)

    assert np.allclose(interpreted_result, jitted_result)


@pytest.mark.xfail(reason="Beamsplitter is not supported by lightning.")
def test_cv_layers(backend):
    """Test CVNeuralNetLayers."""

    def cv_layer(*weights):
        qml.CVNeuralNetLayers(*weights, range(2))
        return qml.expval(qml.X(0))

    def expected_shapes(n_layers, n_wires):
        # compute the expected shapes for a given number of wires
        n_if = n_wires * (n_wires - 1) // 2
        expected = (
            [(n_layers, n_if)] * 2
            + [(n_layers, n_wires)] * 3
            + [(n_layers, n_if)] * 2
            + [(n_layers, n_wires)] * 4
        )
        return expected

    shapes = expected_shapes(1, 2)
    weights = [np.random.random(shape) for shape in shapes]
    weights = [jnp.array(w) for w in weights]

    device = qml.device(backend, wires=2)
    interpreted_fn = qml.QNode(cv_layer, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(*weights)
    jitted_result = jitted_fn(*weights)

    assert np.allclose(interpreted_result, jitted_result)


def test_random_layers(backend):
    """Test random layers."""

    def randomlayers(weights: jax.core.ShapedArray([1, 3], float)):
        qml.RandomLayers(weights=weights, wires=range(2))
        return qml.state()

    device = qml.device(backend, wires=3)
    params = jnp.array([[1.0, 2.0, 3.0]])
    interpreted_fn = qml.QNode(randomlayers, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


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

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


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

    interpreted_result = interpreted_fn(*params)
    jitted_result = jitted_fn(*params)

    assert np.allclose(interpreted_result, jitted_result)


def test_basic_entangler_layers(backend):
    """Test basic entangler layers."""

    def basic_entangler_layers(weights):
        qml.BasicEntanglerLayers(weights=weights, wires=range(3))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(3)]

    device = qml.device(backend, wires=3)
    params = jnp.array([[jnp.pi, jnp.pi, jnp.pi]])
    interpreted_fn = qml.QNode(basic_entangler_layers, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


@pytest.mark.filterwarnings("ignore::pennylane.PennyLaneDeprecationWarning")
def test_basis_state_preparation(backend):
    """Test basis state preparation."""

    def basis_state_preparation(basis_state):
        qml.BasisStatePreparation(basis_state, wires=range(4))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]

    device = qml.device(backend, wires=4)
    params = jnp.array([0, 1, 1, 0.0])
    interpreted_fn = qml.QNode(basis_state_preparation, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


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

    interpreted_result = interpreted_fn(state)
    jitted_result = jitted_fn(state)

    assert np.allclose(interpreted_result, jitted_result)


def test_arbitrary_state_preparation(backend):
    """Test arbitrary state preparation."""

    def vqe(weights):
        qml.ArbitraryStatePreparation(weights, wires=[0, 1])
        return qml.state()

    device = qml.device(backend, wires=2)
    params = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    interpreted_fn = qml.QNode(vqe, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


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

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


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

    interpreted_result = interpreted_fn(qml.numpy.array(params))
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


def test_uccsd(backend):
    """Test UCCSD."""

    def uccsd(weights):
        qml.UCCSD(
            weights,
            wires=range(4),
            s_wires=[[0, 1]],
            d_wires=[[[0, 1], [2, 3]]],
            init_state=np.array([1, 0, 0, 0]),
        )
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=4)
    weights = jnp.array(np.random.random(size=(1, 2)))
    interpreted_fn = qml.QNode(uccsd, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(weights)
    jitted_result = jitted_fn(weights)

    assert np.allclose(interpreted_result, jitted_result)


def test_kup(backend):
    """Test KUP."""

    def kup(weights):
        qml.kUpCCGSD(weights, wires=[0, 1, 2, 3], k=1, delta_sz=0, init_state=[1, 1, 0, 0])
        return qml.state()

    device = qml.device(backend, wires=4)
    params = jnp.array(np.random.random(size=(1, 6)))
    interpreted_fn = qml.QNode(kup, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


def test_particle_conserving_u1(backend):
    """Test particle conserving U1"""

    def particle_conserving_u1(weights):
        qml.ParticleConservingU1(weights, range(2), init_state=np.array([1, 1]))
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=2)
    weights = jnp.array(np.random.random(size=(1, 1, 2)))
    interpreted_fn = qml.QNode(particle_conserving_u1, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(weights)
    jitted_result = jitted_fn(weights)

    assert np.allclose(interpreted_result, jitted_result)


def test_particle_conserving_u2(backend):
    """Test particle conserving U2"""

    def particle_conserving_u2(weights):
        qml.ParticleConservingU2(weights, range(2), init_state=np.array([1, 1]))
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=2)
    weights = jnp.array(np.random.random(size=(1, 3)))
    interpreted_fn = qml.QNode(particle_conserving_u2, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(weights)
    jitted_result = jitted_fn(weights)

    assert np.allclose(interpreted_result, jitted_result)


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

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


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

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


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

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


def test_two_local_swap_network(backend):
    """Test TwoLocalSwapNetwork."""

    def two_local_swap_network(weights):
        qml.templates.TwoLocalSwapNetwork(
            wires=range(4),
            acquaintances=lambda index, wires, param: qml.CRX(param, index),
            weights=weights,
            fermionic=True,
            shift=False,
        )
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=4)
    weights = jnp.array(np.random.random(size=6))
    interpreted_fn = qml.QNode(two_local_swap_network, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(weights)
    jitted_result = jitted_fn(weights)

    assert np.allclose(interpreted_result, jitted_result)


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


def test_reflection(backend):
    """Test reflection."""

    @qml.prod
    def hadamards(wires):
        for wire in wires:
            qml.Hadamard(wires=wire)

    def reflection(alpha):
        """Test circuit"""
        qml.RY(1.2, wires=0)
        qml.RY(-1.4, wires=1)
        qml.RX(-2, wires=0)
        qml.CRX(1, wires=[0, 1])
        qml.Reflection(hadamards(range(3)), alpha)
        return qml.probs(wires=range(3))

    x = np.array(0.25)

    device = qml.device(backend, wires=3)
    interpreted_fn = qml.QNode(reflection, device)
    jitted_fn = qjit(qml.QNode(interpreted_fn, device))

    interpreted_result = interpreted_fn(x)
    jitted_result = jitted_fn(x)

    assert np.allclose(interpreted_result, jitted_result)


def test_amplitude_amplification(backend):
    """Test amplitude amplification."""

    def amplitude_amplification(params):
        qml.RY(params[0], wires=0)
        qml.AmplitudeAmplification(
            qml.RY(params[0], wires=0),
            qml.RZ(params[1], wires=0),
            iters=3,
            fixed_point=True,
            work_wire=2,
        )

        return qml.expval(qml.PauliZ(0))

    params = jnp.array([0.9, 0.1])
    device = qml.device(backend, wires=3)
    interpreted_fn = qml.QNode(amplitude_amplification, device)
    jitted_fn = qjit(qml.QNode(interpreted_fn, device))

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


def test_fermionic(backend):
    """Test Fermionic."""

    def fermionic(weight):
        qml.FermionicSingleExcitation(weight, wires=[0, 1, 2])
        return qml.state()

    device = qml.device(backend, wires=3)
    params = jnp.array(0.56)
    interpreted_fn = qml.QNode(fermionic, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


def test_fermionic_double(backend):
    """Test Fermionic double."""

    def fermionic(weight):
        qml.FermionicDoubleExcitation(weight, wires1=[0, 1], wires2=[2, 3, 4])
        return qml.state()

    device = qml.device(backend, wires=5)
    weight = 1.34817
    interpreted_fn = qml.QNode(fermionic, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(weight)
    jitted_result = jitted_fn(weight)

    assert np.allclose(interpreted_result, jitted_result)


def test_arbitrary_unitary(backend):
    """Test arbitrary unitary."""

    def arbitrary_unitary(weights):
        qml.ArbitraryUnitary(weights, wires=range(2))
        return qml.expval(qml.PauliZ(0))

    weights = jnp.array(np.random.random(size=(15,)))
    device = qml.device(backend, wires=2)
    interpreted_fn = qml.QNode(arbitrary_unitary, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(weights)
    jitted_result = jitted_fn(weights)

    assert np.allclose(interpreted_result, jitted_result)


def test_permute(backend):
    """Test Permute."""

    def permute():
        qml.templates.Permute([4, 2, 0, 1, 3], wires=[0, 1, 2, 3, 4])
        return qml.state()

    device = qml.device(backend, wires=5)
    interpreted_fn = qml.QNode(permute, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn()
    jitted_result = jitted_fn()

    assert np.allclose(interpreted_result, jitted_result)


def test_qft(backend):
    """Test QFT."""

    def qft(basis_state):
        qml.BasisState(basis_state, wires=range(3))
        qml.QFT(wires=range(3))
        # TODO: investigate global phase
        # return qml.state()
        return qml.probs()

    device = qml.device(backend, wires=3)
    params = jnp.array([1, 0, 0])
    interpreted_fn = qml.QNode(qft, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


def test_aqft(backend):
    """Test AQFT."""

    def aqft():
        qml.X(0)
        qml.Hadamard(1)
        qml.AQFT(order=1, wires=range(3))
        return qml.state()

    device = qml.device(backend, wires=3)
    interpreted_fn = qml.QNode(aqft, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn()
    jitted_result = jitted_fn()

    assert np.allclose(interpreted_result, jitted_result)


# Hilbert Schmidt templates take a quantum tape as a parameter.
# Therefore unsuitable for JIT compilation


@pytest.mark.xfail
def test_hilbert_schmidt(backend):
    """Test Hilbert Schmidt."""
    with qml.QueuingManager.stop_recording():
        u_tape = qml.tape.QuantumTape([qml.Hadamard(0)])

    def v_function(params):
        qml.RZ(params[0], wires=1)

    def hilbert_test(v_params):
        qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=[1], u_tape=u_tape)
        return qml.probs(u_tape.wires + [1])

    v_params = np.array([0])
    device = qml.device(backend, wires=2)
    interpreted_fn = qml.QNode(hilbert_test, device)
    jitted_fn = qjit(hilbert_test)

    interpreted_result = interpreted_fn(v_params)
    jitted_result = jitted_fn(v_params)

    assert np.allclose(interpreted_result, jitted_result)


@pytest.mark.xfail
def test_local_hilbert_schmidt(backend):
    """Test Local Hilbert Schmidt."""
    with qml.QueuingManager.stop_recording():
        u_tape = qml.tape.QuantumTape([qml.CZ(wires=(0, 1))])

    def v_function(params):
        qml.RZ(params[0], wires=2)
        qml.RZ(params[1], wires=3)
        qml.CNOT(wires=[2, 3])
        qml.RZ(params[2], wires=3)
        qml.CNOT(wires=[2, 3])

    def local_hilbert_test(v_params):
        qml.LocalHilbertSchmidt(v_params, v_function=v_function, v_wires=[2, 3], u_tape=u_tape)
        return qml.probs(u_tape.wires + [2, 3])

    v_params = np.array([3 * np.pi / 2, 3 * np.pi / 2, np.pi / 2])
    device = qml.device(backend, wires=4)
    interpreted_fn = qml.QNode(local_hilbert_test, device)
    jitted_fn = qjit(local_hilbert_test)

    interpreted_result = interpreted_fn(v_params)
    jitted_result = jitted_fn(v_params)

    assert np.allclose(interpreted_result, jitted_result)


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

    interpreted_result = interpreted_fn(1)
    jitted_result = jitted_fn(1)

    assert np.allclose(interpreted_result, jitted_result)


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

    interpreted_result = interpreted_fn()
    jitted_result = jitted_fn()

    assert np.allclose(interpreted_result, jitted_result)


def test_qsvt(backend):
    """Test QSVT."""
    block_encoding = qml.Hadamard(wires=0)
    phase_shifts = [qml.RZ(-2 * theta, wires=0) for theta in (1.23, -0.5, 4)]

    def qsvt():
        qml.QSVT(block_encoding, phase_shifts)
        return qml.expval(qml.Z(0))

    device = qml.device(backend, wires=1)
    interpreted_fn = qml.QNode(qsvt, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn()
    jitted_result = jitted_fn()

    assert np.allclose(interpreted_result, jitted_result)


@pytest.mark.filterwarnings("ignore:qml.broadcast:pennylane.PennyLaneDeprecationWarning")
def test_broadcast_single(backend):
    """Test broadcast single."""

    def broadcast_single(pars):
        qml.broadcast(unitary=qml.RX, pattern="single", wires=[0, 1, 2], parameters=pars)
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=3)
    params = jnp.array([1, 1, 2])
    interpreted_fn = qml.QNode(broadcast_single, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


@pytest.mark.filterwarnings("ignore:qml.broadcast:pennylane.PennyLaneDeprecationWarning")
def test_broadcast_double(backend):
    """Test broadcast double."""

    def broadcast_double(pars):
        qml.broadcast(unitary=qml.CRot, pattern="double", wires=[0, 1, 2, 3], parameters=pars)
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=4)
    params = jnp.array([[-1, 2.5, 3], [-1, 4, 2.0]])
    interpreted_fn = qml.QNode(broadcast_double, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


@pytest.mark.filterwarnings("ignore:qml.broadcast:pennylane.PennyLaneDeprecationWarning")
def test_broadcast_chain(backend):
    """Test broadcast chain."""

    def broadcast_chain(pars):
        qml.broadcast(unitary=qml.CRot, pattern="chain", wires=[0, 1, 2, 3], parameters=pars)
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=4)
    params = jnp.array([[1.8, 2, 3], [-1.0, 3, 1], [2, 1.2, 4]])
    interpreted_fn = qml.QNode(broadcast_chain, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


@pytest.mark.filterwarnings("ignore:qml.broadcast:pennylane.PennyLaneDeprecationWarning")
def test_broadcast_ring(backend):
    """Test broadcast ring."""

    def broadcast_ring(pars):
        qml.broadcast(unitary=qml.CRot, pattern="ring", wires=[0, 1, 2], parameters=pars)
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=3)
    params = jnp.array([[1, 2.2, 3], [-1, 3, 1.0], [2.6, 1, 4]])
    interpreted_fn = qml.QNode(broadcast_ring, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


@pytest.mark.filterwarnings("ignore:qml.broadcast:pennylane.PennyLaneDeprecationWarning")
def test_broadcast_pyramid(backend):
    """Test broadcast pyramid."""

    def broadcast_pyramid(pars):
        qml.broadcast(unitary=qml.CRot, pattern="pyramid", wires=[0, 1, 2, 3], parameters=pars)
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=4)
    params = jnp.array([[1, 2.2, 3]] * 3)
    interpreted_fn = qml.QNode(broadcast_pyramid, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


@pytest.mark.filterwarnings("ignore:qml.broadcast:pennylane.PennyLaneDeprecationWarning")
def test_broadcast_all_to_all(backend):
    """Test broadcast all to all."""

    def broadcast_all_to_all(pars):
        qml.broadcast(unitary=qml.CRot, pattern="all_to_all", wires=[0, 1, 2, 3], parameters=pars)
        return qml.expval(qml.PauliZ(0))

    device = qml.device(backend, wires=4)
    params = jnp.array([[1, 2.2, 3]] * 6)
    interpreted_fn = qml.QNode(broadcast_all_to_all, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


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

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


def test_qdrift(backend):
    """Test QDrift."""
    coeffs = [1, 1, 1]
    ops = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(1)]
    time = jnp.array(0.5)
    seed = 1234

    def qdrift(time):
        hamiltonian = qml.sum(*(qml.s_prod(coeff, op) for coeff, op in zip(coeffs, ops)))
        qml.QDrift(hamiltonian, time, n=2, seed=seed)
        return qml.state()

    device = qml.device(backend, wires=2)
    interpreted_fn = qml.QNode(qdrift, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(time)
    jitted_result = jitted_fn(time)

    assert np.allclose(interpreted_result, jitted_result)


def test_trotter_product(backend):
    """Test Trotter product."""
    time = jnp.array(0.5)
    c1 = jnp.array(1.23)
    c2 = jnp.array(-0.45)
    terms = [qml.PauliX(0), qml.PauliZ(0)]

    def trotter_product(time, c1, c2):
        h = qml.sum(
            qml.s_prod(c1, terms[0]),
            qml.s_prod(c2, terms[1]),
        )
        qml.TrotterProduct(h, time, n=2, order=2, check_hermitian=False)

        return qml.state()

    device = qml.device(backend, wires=2)
    interpreted_fn = qml.QNode(trotter_product, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(time, c1, c2)
    jitted_result = jitted_fn(time, c1, c2)

    assert np.allclose(interpreted_result, jitted_result)


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

    interpreted_result = interpreted_fn()
    jitted_result = jitted_fn()

    assert np.allclose(interpreted_result, jitted_result)


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

    interpreted_result = interpreted_fn()
    jitted_result = jitted_fn()

    assert np.allclose(interpreted_result, jitted_result)


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


def test_select(backend):
    """Test Select"""

    def select():
        ops = [qml.X(2), qml.X(3), qml.Y(2), qml.SWAP([2, 3])]
        qml.Select(ops, control=[0, 1])
        return qml.state()

    device = qml.device(backend, wires=4)
    interpreted_fn = qml.QNode(select, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn()
    jitted_result = jitted_fn()

    assert np.allclose(interpreted_result, jitted_result)


def test_controlled_sequence(backend):
    """Test controlled sequence."""

    def controlled_sequence(x):
        """Test circuit"""
        qml.PauliX(2)
        qml.ControlledSequence(qml.RX(x, wires=3), control=[0, 1, 2])
        return qml.probs(wires=range(4))

    x = jnp.array(0.25)
    device = qml.device(backend, wires=4)
    interpreted_fn = qml.QNode(controlled_sequence, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(x)
    jitted_result = jitted_fn(x)

    assert np.allclose(interpreted_result, jitted_result)


def test_fable(backend):
    """Test FABLE."""

    def fable(input_matrix):
        qml.FABLE(input_matrix, wires=range(5), tol=0)
        return qml.expval(qml.PauliZ(wires=0))

    input_matrix = np.array(
        [
            [-0.5, -0.4, 0.6, 0.7],
            [0.9, 0.9, 0.8, 0.9],
            [0.8, 0.7, 0.9, 0.8],
            [0.9, 0.7, 0.8, 0.3],
        ]
    )

    device = qml.device(backend, wires=5)
    interpreted_fn = qml.QNode(fable, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(input_matrix)
    jitted_result = jitted_fn(input_matrix)

    assert np.allclose(interpreted_result, jitted_result)


def test_qubitization(backend):
    """Test Qubitization."""

    def qubitization():
        H = qml.ops.LinearCombination([0.1, 0.3, -0.3], [qml.Z(0), qml.Z(1), qml.Z(0) @ qml.Z(2)])
        qml.Hadamard(wires=0)
        qml.Qubitization(H, control=[3, 4])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(4))

    device = qml.device(backend, wires=5)
    interpreted_fn = qml.QNode(qubitization, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn()
    jitted_result = jitted_fn()

    assert np.allclose(interpreted_result, jitted_result)


def test_qrom(backend):
    """Test QROM."""

    def qrom():
        qml.QROM(["1", "0", "0", "1"], control_wires=[0, 1], target_wires=[2], work_wires=[3])
        return qml.probs(wires=3)

    device = qml.device(backend, wires=4)
    interpreted_fn = qml.QNode(qrom, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn()
    jitted_result = jitted_fn()

    assert np.allclose(interpreted_result, jitted_result)


def test_prepselprep(backend):
    """Test PrepSelPrep"""

    params = np.array([0.4, 0.5, 0.1, 0.3])

    def prepselprep(coeffs):
        H = qml.ops.LinearCombination(
            coeffs, [qml.Y(0), qml.Y(1) @ qml.Y(2), qml.X(0), qml.X(1) @ qml.X(2)]
        )
        qml.PrepSelPrep(H, control=(3, 4))
        return qml.expval(qml.PauliZ(3) @ qml.PauliZ(4))

    device = qml.device(backend, wires=5)
    interpreted_fn = qml.QNode(prepselprep, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(params)
    jitted_result = jitted_fn(params)

    assert np.allclose(interpreted_result, jitted_result)


def test_mod_exp(backend):
    """Test ModExp."""
    x, b = 3, 1
    base = 2
    mod = 7

    x_wires = [0, 1]
    output_wires = [2, 3, 4]
    work_wires = [5, 6, 7, 8, 9]

    def mod_exp():
        qml.BasisEmbedding(x, wires=x_wires)
        qml.BasisEmbedding(b, wires=output_wires)
        qml.ModExp(x_wires, output_wires, base, mod, work_wires)
        return qml.sample(wires=output_wires)

    device = qml.device(backend, wires=10, shots=2)
    interpreted_fn = qml.QNode(mod_exp, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn()
    jitted_result = jitted_fn()

    assert np.allclose(interpreted_result, jitted_result)


def test_multiplier(backend):
    """Test Multiplier."""
    x = 3
    k = 4
    mod = 7

    x_wires = [0, 1, 2]
    work_wires = [3, 4, 5, 6, 7]

    def multiplier():
        qml.BasisEmbedding(x, wires=x_wires)
        qml.Multiplier(k, x_wires, mod, work_wires)
        return qml.sample(wires=x_wires)

    device = qml.device(backend, wires=8, shots=2)
    interpreted_fn = qml.QNode(multiplier, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn()
    jitted_result = jitted_fn()

    assert np.allclose(interpreted_result, jitted_result)


@pytest.mark.xfail(reason="Multiple state preparations at beginning of circuit")
def test_out_adder(backend):
    """Test OutAdder."""
    x = 5
    y = 6
    mod = 7

    x_wires = [0, 1, 2]
    y_wires = [3, 4, 5]
    output_wires = [7, 8, 9]
    work_wires = [6, 10]

    def out_adder():
        qml.BasisEmbedding(x, wires=x_wires)
        qml.BasisEmbedding(y, wires=y_wires)
        qml.OutAdder(x_wires, y_wires, output_wires, mod, work_wires)
        return qml.sample(wires=output_wires)

    device = qml.device(backend, wires=11, shots=10000)
    interpreted_fn = qml.QNode(out_adder, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn()
    jitted_result = jitted_fn()

    assert np.allclose(interpreted_result, jitted_result)


@pytest.mark.xfail(reason="Multiple state preparations at beginning of circuit")
def test_out_multiplier(backend):
    """Test OutMultiplier."""
    x = 2
    y = 7
    mod = 12

    x_wires = [0, 1]
    y_wires = [2, 3, 4]
    output_wires = [6, 7, 8, 9]
    work_wires = [5, 10]

    def out_multiplier():
        qml.BasisEmbedding(x, wires=x_wires)
        qml.BasisEmbedding(y, wires=y_wires)
        qml.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
        return qml.sample(wires=output_wires)

    device = qml.device(backend, wires=11, shots=2)
    interpreted_fn = qml.QNode(out_multiplier, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn()
    jitted_result = jitted_fn()

    assert np.allclose(interpreted_result, jitted_result)


def test_phase_adder(backend):
    """Test PhaseAdder."""
    x = 8
    k = 5
    mod = 15

    x_wires = [0, 1, 2, 3]
    work_wire = [4]

    def phase_adder():
        qml.BasisEmbedding(x, wires=x_wires)
        qml.QFT(wires=x_wires)
        qml.PhaseAdder(k, x_wires, mod, work_wire)
        qml.adjoint(qml.QFT)(wires=x_wires)
        return qml.sample(wires=x_wires)

    device = qml.device(backend, wires=range(5), shots=2)
    interpreted_fn = qml.QNode(phase_adder, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn()
    jitted_result = jitted_fn()

    assert np.allclose(interpreted_result, jitted_result)


@pytest.mark.xfail(reason="Qutrit operators not supported on lightning.")
def test_qutrit_basis_state_preparation(backend):
    """Test QutritBasisStatePreparation."""
    basis_state = [0, 1]
    wires = [0, 1]
    obs = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

    def qutrit_basis_state_preparation(state, obs):
        qml.QutritBasisStatePreparation(state, wires)

        return [qml.expval(qml.THermitian(A=obs, wires=i)) for i in range(3)]

    device = qml.device(backend, wires=2)
    interpreted_fn = qml.QNode(qutrit_basis_state_preparation, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn(basis_state, obs)
    jitted_result = jitted_fn(basis_state, obs)

    assert np.allclose(interpreted_result, jitted_result)


def test_cosine_window(backend):
    """Test cosine window."""

    def cosine_window():
        qml.CosineWindow(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    device = qml.device(backend, wires=2)
    interpreted_fn = qml.QNode(cosine_window, device)
    jitted_fn = qjit(interpreted_fn)

    interpreted_result = interpreted_fn()
    jitted_result = jitted_fn()

    assert np.allclose(interpreted_result, jitted_result)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
