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
Hamiltonians: scalarize-tensor-extracts, elementwise fusion, and reroll-loops
in the default pipeline must preserve numerics for Trotterized workloads with
runtime coefficients."""

import numpy as np
import pennylane as qml
import pytest
from jax import numpy as jnp

from catalyst import qjit

# H2/STO-3G-like coefficients; the structure (15 terms, runtime values) is what
# exercises the pipeline, the values just need to be a valid Hamiltonian.
COEFFS = [
    -0.0996, 0.1711, 0.1711, -0.2225, -0.2225, 0.1686, 0.0453, -0.0453,
    -0.0453, 0.0453, 0.1205, 0.1658, 0.1658, 0.1205, 0.1743,
]
OPS_FACTORY = lambda: [
    qml.Identity(0),
    qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2), qml.PauliZ(3),
    qml.PauliZ(0) @ qml.PauliZ(1),
    qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliY(3),
    qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliX(3),
    qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliY(3),
    qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliX(3),
    qml.PauliZ(0) @ qml.PauliZ(2), qml.PauliZ(0) @ qml.PauliZ(3),
    qml.PauliZ(1) @ qml.PauliZ(2), qml.PauliZ(1) @ qml.PauliZ(3),
    qml.PauliZ(2) @ qml.PauliZ(3),
]

N_QUBITS = 4
N_EST = 2
N_TROTTER = 6  # enough repetitions for reroll-loops to fire


def make_qpe(dev, runtime: bool):
    """Controlled-power Trotterized QPE, with runtime or compile-time coeffs."""

    @qml.qnode(dev)
    def qpe_circuit(coeffs):
        qml.PauliX(0)
        qml.PauliX(1)
        for k in range(N_EST):
            qml.Hadamard(wires=N_QUBITS + k)
        H = qml.dot(coeffs if runtime else COEFFS, OPS_FACTORY())
        for k in range(N_EST):
            t = 2 ** (N_EST - 1 - k)
            qml.ctrl(
                qml.adjoint(
                    qml.TrotterProduct(H, time=t, n=N_TROTTER, order=2,
                                       check_hermitian=False)
                ),
                control=N_QUBITS + k,
            )
        qml.adjoint(qml.QFT)(wires=range(N_QUBITS, N_QUBITS + N_EST))
        return qml.probs(wires=range(N_QUBITS, N_QUBITS + N_EST))

    return qpe_circuit


class TestRuntimeCoefficientTrotter:
    """Numerical equivalence of runtime- and fixed-coefficient Trotterization
    through the default pipeline (which scalarizes, fuses, and rerolls)."""

    def test_runtime_matches_fixed(self):
        """qml.dot with traced coefficients must produce the same distribution
        as the same Hamiltonian with Python-float coefficients."""
        dev = qml.device("lightning.qubit", wires=N_QUBITS + N_EST)
        coeffs = jnp.array(COEFFS)

        dyn = qjit(make_qpe(dev, runtime=True))(coeffs)
        fixed = qjit(make_qpe(dev, runtime=False))(coeffs)

        assert np.allclose(np.asarray(dyn), np.asarray(fixed), atol=1e-9)

    def test_reroll_recovers_loops(self):
        """The compiled module must contain scf.for loops recovered from the
        unrolled Trotter steps (guards against silent regression of
        reroll-loops in the default pipeline)."""
        dev = qml.device("lightning.qubit", wires=N_QUBITS + N_EST)
        coeffs = jnp.array(COEFFS)

        compiled = qjit(make_qpe(dev, runtime=True), keep_intermediate=True)
        compiled(coeffs)
        try:
            workspace = str(compiled.workspace)
            import glob
            import os

            hlo_files = glob.glob(os.path.join(workspace, "*HLOLowering*.mlir"))
            assert hlo_files, "no post-HLO snapshot written"
            content = open(hlo_files[0], encoding="utf-8").read()
            assert "scf.for" in content, "reroll-loops did not fire"
        finally:
            compiled.workspace.cleanup()


if __name__ == "__main__":
    pytest.main(["-x", __file__])
