import jax
import jax.numpy as jnp
import pennylane as qml
import pennylane.numpy as pnp

import catalyst
from catalyst import for_loop

# jax.config.update("jax_platform_name", "cpu")

dev = qml.device("lightning.qubit", wires=4)


def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates."""

    @for_loop(0, nqubits, 1)
    def H(idx):
        qml.Hadamard(wires=idx)

    H()


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis."""

    @for_loop(0, len(w), 1)
    def loop_fn(idx):
        qml.RY(w[idx], wires=idx)

    loop_fn()


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT."""

    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    @for_loop(0, nqubits - 1, 2)  # Loop over even indices: i=0,2,...N-2
    def first_loop(i):
        qml.CNOT(wires=[i, i + 1])

    @for_loop(1, nqubits - 1, 2)  # Loop over odd indices:  i=1,3,...N-3
    def second_loop(i):
        qml.CNOT(wires=[i, i + 1])

    first_loop()
    second_loop()


n_qubits = 4
q_depth = 6


@qml.qnode(dev)
def quantum_net(q_input_features, q_weights_flat):
    """
    The variational quantum circuit.
    """

    # Reshape weights
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(q_input_features)

    # Sequence of trainable variational layers
    @for_loop(0, q_depth, 1)
    def layer(k):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k])

    layer()

    # Expectation values in the Z basis
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return tuple(exp_vals)


@catalyst.qjit
def transfer_learning(pre_net, q_params, post_net, input_features):
    """
    pre_net: f x n_qubits
    input_features: b x f
    post_net:
    """

    # obtain the input features for the quantum circuit
    # by reducing the feature dimension from 512 to 4
    pre_out = jnp.matmul(input_features, pre_net)
    q_in = jnp.tanh(pre_out) * jnp.pi / 2.0

    # Apply the quantum circuit to each element of the batch
    # @for_loop(0, q_in.shape[0], 1)
    # def qnet(iv, q_out):
    #     q_out = q_out.at[iv].set(quantum_net(q_in[iv], q_params))
    #     return q_out
    q_out = jnp.array([quantum_net(elem, q_params) for elem in q_in])

    # q_out = qnet(jnp.zeros((q_in.shape[0], n_qubits)))

    return jnp.matmul(q_out, post_net)


batch_size = 4
pre_net = jnp.ones((512, n_qubits))
input_features = jnp.ones((batch_size, 512))
q_params = jnp.ones(q_depth * n_qubits)
post_net = jnp.ones((n_qubits, 2))


jitted = catalyst.qjit(transfer_learning)
jitted(pre_net, q_params, post_net, input_features)


def compile():
    jitted.compile()


def work():
    # quantum_net(jnp.ones((n_qubits,)), jnp.ones((q_depth * n_qubits,)))
    transfer_learning(pre_net, q_params, post_net, input_features)
