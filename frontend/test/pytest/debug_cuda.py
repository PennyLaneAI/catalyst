import jax
import pennylane as qml
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

import numpy as np
import catalyst
from catalyst import measure, qjit
from catalyst.utils.exceptions import CompileError


from catalyst.cuda import SoftwareQQPP

from catalyst import for_loop, measure, qjit, while_loop


@qml.qnode(qml.device("lightning.qubit", wires=4))
def circuit_lightning(n):
    # QFT
    # @for_loop(0, n-2, 1)
    # def qft(i):
    #     @for_loop(i+1, n-1, 1)
    #     def inner(j):
    #         # qml.ControlledPhaseShift(np.pi / 2 ** (n - j + 1), [i, j])
    #         qml.CNOT(wires=[i, j])

    #     inner()
    # qft()

    # Expected output: |100...>
    qml.ControlledPhaseShift(np.pi / 2 ** (n - 1 + 1), [0, 1])
    return qml.state()

from catalyst.cuda import SoftwareQQPP
@qml.qnode(SoftwareQQPP(wires=4))
def circuit(n):
    # QFT
    # @for_loop(0, n-2, 1)
    # def qft(i):
    #     @for_loop(i+ 1, n - 1, 1)
    #     def inner(j):
    #         qml.CNOT(wires=[i, j])

    #     inner()
    # qft()

    # Expected output: |100...>
    qml.ControlledPhaseShift(np.pi / 2 ** (n - 1 + 1), [0, 1])
    return qml.state()

cuda_compiled = catalyst.cuda.qjit(circuit)
catalyst_compiled = qjit(circuit_lightning)
expected = catalyst_compiled(4)
observed = cuda_compiled(4)
assert_allclose(expected, observed)

print("works!")

# RX, Hadamard