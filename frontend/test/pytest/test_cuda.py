import jax
import pennylane as qml
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

import catalyst
from catalyst import measure, qjit
from catalyst.utils.exceptions import CompileError


from catalyst.cuda import SoftwareQQPP

from catalyst import for_loop, measure, qjit, while_loop


# @qml.qnode(SoftwareQQPP(wires=1))
# def circuit(a):
#     qml.RX(a / 2, wires=[0])
#     return qml.state()

# @qml.qnode(qml.device("lightning.qubit", wires=1))
# def circuit_lightning(a):
#     qml.RX(a / 2, wires=[0])
#     return qml.state()


@qml.qnode(qml.device("lightning.qubit", wires=6))
def circuit_lightning(n: int):
    # qml.Hadamard(wires=0)

    # @for_loop(0, n - 1, 1)
    # def loop_fn(i):
    #     qml.CNOT(wires=[i, i + 1])

    # loop_fn()
    # return qml.state()
    qml.Hadamard(wires=[0])
    qml.CNOT(wires=[0, 1])
    return qml.state()

@qml.qnode(SoftwareQQPP(wires=6))
def circuit(n: int):
    # qml.Hadamard(wires=0)

    # @for_loop(0, n - 1, 1)
    # def loop_fn(i):
    #     qml.CNOT(wires=[i, i + 1])

    # loop_fn()
    # return qml.state()

    qml.Hadamard(wires=[0])
    qml.CNOT(wires=[0, 1])
    return qml.state()

cuda_compiled = catalyst.cuda.qjit(circuit)
catalyst_compiled = qjit(circuit_lightning)
expected = catalyst_compiled(2)
observed = cuda_compiled(2)
assert_allclose(expected, observed)

print("works!")