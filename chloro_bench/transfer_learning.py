import pennylane as qml
from pennylane import numpy as np
from catalyst import qjit

mode = None

from catalyst.ast_parser import chlorophyll

n_qubits = 4  # Number of qubits
step = 0.0004  # Learning rate
batch_size = 4  # Number of samples for each training step
num_epochs = 3  # Number of training epochs
q_depth = 6  # Depth of the quantum circuit (number of variational layers)
gamma_lr_scheduler = 0.1  # Learning rate reduction applied every 10 epochs.
q_delta = 0.01  # Initial spread of random quantum weights

dev = qml.device("lightning.qubit", wires=n_qubits)


def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates."""
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis."""
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT."""
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


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
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k])

    # Expectation values in the Z basis
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return tuple(exp_vals)


import jax

# import jax.numpy as np


# @jax.jit
# @qjit
@chlorophyll
def transfer_learning(pre_net, q_params, post_net, input_features):
    """
    pre_net: f x n_qubits
    input_features: b x f
    post_net:
    """

    # obtain the input features for the quantum circuit
    # by reducing the feature dimension from 512 to 4
    pre_out = np.matmul(input_features, pre_net)
    q_in = np.tanh(pre_out) * np.pi / 2.0

    # Apply the quantum circuit to each element of the batch
    q_out = np.array([quantum_net(elem, q_params) for elem in q_in])

    return np.matmul(q_out, post_net)


# Must be JAX NumPy for Catalyst and JAX, must be PennyLane NumPy for Chlorophyll
pre_net = np.ones((512, n_qubits))
input_features = np.ones((batch_size, 512))
q_params = np.ones(q_depth * n_qubits)
post_net = np.ones((n_qubits, 2))

# Required for Chlorophyll to look at the concrete parameter types
transfer_learning(pre_net, q_params, post_net, input_features)


def compile():
    # Forces JAX to recompile the function
    # jax.jit(
    #     lambda: transfer_learning(pre_net, q_params, post_net, input_features)
    # ).lower().compile()

    # Re-compiles either Catalyst or Chlorophyll depending on the decorator
    transfer_learning.compile()


def work():
    transfer_learning(pre_net, q_params, post_net, input_features)
