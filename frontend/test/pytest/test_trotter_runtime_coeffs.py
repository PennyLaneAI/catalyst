# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for the IR-amplification fixes for runtime-coefficient
Hamiltonians: scalarize-tensor-extracts and elementwise fusion in the default
pipeline must preserve numerics for Trotterized workloads with runtime
coefficients."""

import numpy as np
import pennylane as qp
import pytest
from jax import numpy as jnp

from catalyst import qjit

# H2/STO-3G-like coefficients; the structure (15 terms, runtime values) is what
# exercises the pipeline, the values just need to be a valid Hamiltonian.
COEFFS = [
    -0.0996,
    0.1711,
    0.1711,
    -0.2225,
    -0.2225,
    0.1686,
    0.0453,
    -0.0453,
    -0.0453,
    0.0453,
    0.1205,
    0.1658,
    0.1658,
    0.1205,
    0.1743,
]


def make_ops():
    """Build a fresh list of the Hamiltonian's Pauli-word operators."""
    return [
        qp.Identity(0),
        qp.PauliZ(0),
        qp.PauliZ(1),
        qp.PauliZ(2),
        qp.PauliZ(3),
        qp.PauliZ(0) @ qp.PauliZ(1),
        qp.PauliY(0) @ qp.PauliX(1) @ qp.PauliX(2) @ qp.PauliY(3),
        qp.PauliY(0) @ qp.PauliY(1) @ qp.PauliX(2) @ qp.PauliX(3),
        qp.PauliX(0) @ qp.PauliX(1) @ qp.PauliY(2) @ qp.PauliY(3),
        qp.PauliX(0) @ qp.PauliY(1) @ qp.PauliY(2) @ qp.PauliX(3),
        qp.PauliZ(0) @ qp.PauliZ(2),
        qp.PauliZ(0) @ qp.PauliZ(3),
        qp.PauliZ(1) @ qp.PauliZ(2),
        qp.PauliZ(1) @ qp.PauliZ(3),
        qp.PauliZ(2) @ qp.PauliZ(3),
    ]


N_QUBITS = 4
N_EST = 2
N_TROTTER = 6


def make_qpe(dev, runtime: bool):
    """Controlled-power Trotterized QPE, with runtime or compile-time coeffs."""

    @qp.qnode(dev)
    def qpe_circuit(coeffs):
        qp.PauliX(0)
        qp.PauliX(1)
        for k in range(N_EST):
            qp.Hadamard(wires=N_QUBITS + k)
        H = qp.dot(coeffs if runtime else COEFFS, make_ops())
        for k in range(N_EST):
            t = 2 ** (N_EST - 1 - k)
            # For a symmetric second-order step, negating time gives its adjoint.
            qp.ctrl(
                qp.TrotterProduct(H, time=-t, n=N_TROTTER, order=2, check_hermitian=False),
                control=N_QUBITS + k,
            )
        qp.adjoint(qp.QFT)(wires=range(N_QUBITS, N_QUBITS + N_EST))
        return qp.probs(wires=range(N_QUBITS, N_QUBITS + N_EST))

    return qpe_circuit


class TestRuntimeCoefficientTrotter:
    """Numerical equivalence of runtime- and fixed-coefficient Trotterization
    through the default pipeline (which scalarizes and fuses)."""

    def test_runtime_matches_fixed(self, backend):
        """qp.dot with traced coefficients must produce the same distribution
        as the same Hamiltonian with Python-float coefficients."""
        dev = qp.device(backend, wires=N_QUBITS + N_EST)
        coeffs = jnp.array(COEFFS)

        dyn = qjit(make_qpe(dev, runtime=True), capture=False)(coeffs)
        fixed = qjit(make_qpe(dev, runtime=False), capture=False)(coeffs)

        assert np.allclose(np.asarray(dyn), np.asarray(fixed), atol=1e-9)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
