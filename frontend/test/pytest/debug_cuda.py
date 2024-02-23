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
        # Input state: equal superposition
    @for_loop(0, n, 1)
    def init(i):
        # qml.Hadamard(wires=i)
        pass

    # QFT
    @for_loop(0, n, 1)
    def qft(i):
        # qml.Hadamard(wires=i)
        pass

        @for_loop(i + 1, n, 1)
        def inner(j):
            qml.ControlledPhaseShift(np.pi / 2 ** (n - j + 1), [i, j])
            pass

        inner()

    init()
    qft()

    # Expected output: |100...>
    return qml.state()

from catalyst.cuda import SoftwareQQPP
@qml.qnode(SoftwareQQPP(wires=4))
def circuit(n):
        # Input state: equal superposition
    @for_loop(0, n, 1)
    def init(i):
        # qml.Hadamard(wires=i)
        pass

    # QFT
    @for_loop(0, n, 1)
    def qft(i):
        # qml.Hadamard(wires=i)
        pass

        @for_loop(i + 1, n, 1)
        def inner(j):
            qml.ControlledPhaseShift(np.pi / 2 ** (n - j + 1), [i, j])
            pass

        inner()

    init()
    qft()

    # Expected output: |100...>
    return qml.state()

cuda_compiled = catalyst.cuda.qjit(circuit)
catalyst_compiled = qjit(circuit_lightning)
expected = catalyst_compiled(4)
observed = cuda_compiled(4)
assert_allclose(expected, observed)

print("works!")

# RX, Hadamard