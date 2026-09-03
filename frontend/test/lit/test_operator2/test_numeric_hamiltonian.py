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
"""Tests that ``qp.TrotterCDF`` and ``qp.TrotterCGF`` lower their Hamiltonian tensor data
to MLIR as ranked tensor operands.

Two use cases are covered for each representation:

* **concrete** data closed over by the circuit;
* **abstract** data, where a Hamiltonian built from ``qp.typing.Float[...]`` supplies the
  ahead-of-time signature, so the tensors lower to function arguments of the declared
  shape.
"""

# pylint: disable = missing-function-docstring,line-too-long

# RUN: %PYTHON %s | FileCheck %s

import numpy as np
import pennylane as qp
from pennylane.typing import Float

# CDF: N orbitals -> 2N wires. CGF: M modes x K modals -> M*K wires.
L, N = 1, 2
M, K = 2, 2

rng = np.random.default_rng(42)


def random_orthogonal(dim):
    """A real orthogonal matrix, as the Trotter leaf tensors require."""
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))


CDF = qp.CDFHamiltonian(
    core_tensors=rng.standard_normal((L + 1, N, N)),
    leaf_tensors=np.stack([random_orthogonal(N) for _ in range(L + 1)]),
    nuc_constant=0.5,
)

CGF = qp.CGFHamiltonian(
    core_tensors=rng.standard_normal((L + 1, M, M, K, K)),
    leaf_tensors=np.stack(
        [np.stack([random_orthogonal(K) for _ in range(M)]) for _ in range(L + 1)]
    ),
    nuc_constant=0.5,
)

ABSTRACT_CDF = qp.CDFHamiltonian(Float[L + 1, N, N], Float[L + 1, N, N], Float)
ABSTRACT_CGF = qp.CGFHamiltonian(Float[L + 1, M, M, K, K], Float[L + 1, M, K, K], Float)


# CDF with concrete data: the two (L+1, N, N) tensors and the scalar constant become
# ranked tensor operands under a single ``hamiltonian`` entry in ``param_map``.
@qp.qjit(capture=True, target="mlir", collect_decomp_rules=False)
@qp.qnode(qp.device("null.qubit", wires=2 * N))
def cdf_concrete(t: float):
    # CHECK-LABEL: func.func public @cdf_concrete
    # CHECK: qref.operator "TrotterCDF"({{%.+}}: tensor<f64>, {{%.+}}: tensor<2x2x2xf64>, {{%.+}}: tensor<2x2x2xf64>, {{%.+}}: tensor<f64>)
    # CHECK-SAME: qubits({{.+}})
    # CHECK: UID
    # CHECK: param_map = {evolution_time = [0], hamiltonian = [1, 2, 3]}
    # CHECK: qubit_map = {wires = [0, 1, 2, 3]}
    qp.TrotterCDF(evolution_time=t, num_trotter_steps=2, hamiltonian=CDF, wires=range(2 * N))
    return qp.state()


print(cdf_concrete.mlir)


# CGF with concrete data: the same operator shape in the IR, but a (L+1, M, M, K, K) core
# and a (L+1, M, K, K) leaf. No per-representation special-casing in the lowering.
@qp.qjit(capture=True, target="mlir", collect_decomp_rules=False)
@qp.qnode(qp.device("null.qubit", wires=M * K))
def cgf_concrete(t: float):
    # CHECK-LABEL: func.func public @cgf_concrete
    # CHECK: qref.operator "TrotterCGF"({{%.+}}: tensor<f64>, {{%.+}}: tensor<2x2x2x2x2xf64>, {{%.+}}: tensor<2x2x2x2xf64>, {{%.+}}: tensor<f64>)
    # CHECK: UID
    # CHECK: param_map = {evolution_time = [0], hamiltonian = [1, 2, 3]}
    # CHECK: qubit_map = {wires = [0, 1, 2, 3]}
    qp.TrotterCGF(evolution_time=t, num_trotter_steps=2, hamiltonian=CGF, wires=range(M * K))
    return qp.state()


print(cgf_concrete.mlir)


# CDF with abstract data: the abstract Hamiltonian *is* the ahead-of-time signature, so the
# tensors arrive as function arguments of the declared shape rather than as constants.
@qp.qjit(capture=True, target="mlir", collect_decomp_rules=False)
@qp.qnode(qp.device("null.qubit", wires=2 * N))
def cdf_abstract(
    t: float,
    core: ABSTRACT_CDF.core_tensors,
    leaf: ABSTRACT_CDF.leaf_tensors,
    nuc: ABSTRACT_CDF.nuc_constant,
):
    # CHECK-LABEL: func.func public @cdf_abstract
    # CHECK-SAME: tensor<2x2x2xf64>
    # CHECK: qref.operator "TrotterCDF"({{%.+}}: tensor<f64>, {{%.+}}: tensor<2x2x2xf64>, {{%.+}}: tensor<2x2x2xf64>, {{%.+}}: tensor<f64>)
    # CHECK: param_map = {evolution_time = [0], hamiltonian = [1, 2, 3]}
    qp.TrotterCDF(
        evolution_time=t,
        num_trotter_steps=2,
        hamiltonian=qp.CDFHamiltonian(core, leaf, nuc),
        wires=range(2 * N),
    )
    return qp.state()


print(cdf_abstract.mlir)


# CGF with abstract data.
@qp.qjit(capture=True, target="mlir", collect_decomp_rules=False)
@qp.qnode(qp.device("null.qubit", wires=M * K))
def cgf_abstract(
    t: float,
    core: ABSTRACT_CGF.core_tensors,
    leaf: ABSTRACT_CGF.leaf_tensors,
    nuc: ABSTRACT_CGF.nuc_constant,
):
    # CHECK-LABEL: func.func public @cgf_abstract
    # CHECK-SAME: tensor<2x2x2x2x2xf64>
    # CHECK: qref.operator "TrotterCGF"({{%.+}}: tensor<f64>, {{%.+}}: tensor<2x2x2x2x2xf64>, {{%.+}}: tensor<2x2x2x2xf64>, {{%.+}}: tensor<f64>)
    # CHECK: param_map = {evolution_time = [0], hamiltonian = [1, 2, 3]}
    qp.TrotterCGF(
        evolution_time=t,
        num_trotter_steps=2,
        hamiltonian=qp.CGFHamiltonian(core, leaf, nuc),
        wires=range(M * K),
    )
    return qp.state()


print(cgf_abstract.mlir)


# Controlled evolution. Both variants live in one function so that FileCheck pattern
# variables can compare their UIDs: variables do not carry across ``CHECK-LABEL`` blocks.
@qp.qjit(capture=True, target="mlir", collect_decomp_rules=False)
@qp.qnode(qp.device("null.qubit", wires=2 * N + 1))
def cdf_controlled(t: float):
    # CHECK-LABEL: func.func public @cdf_controlled

    # The control qubit rides in ``ctrls``, so it is absent from ``qubit_map``.
    # CHECK: qref.operator "TrotterCDF"({{%.+}}: tensor<f64>, {{%.+}}: tensor<2x2x2xf64>, {{%.+}}: tensor<2x2x2xf64>, {{%.+}}: tensor<f64>)
    # CHECK: UID([[SINGLE_PHASE_UID:[0-9]+]])
    # CHECK: ctrls({{%.+}}) ctrl_vals({{%.+}})
    # CHECK: param_map = {evolution_time = [0], hamiltonian = [1, 2, 3]}
    # CHECK: qubit_map = {wires = [0, 1, 2, 3]}
    qp.ctrl(
        qp.TrotterCDF(evolution_time=t, num_trotter_steps=2, hamiltonian=CDF, wires=range(2 * N)),
        control=[2 * N],
    )

    # ``double_phase`` is static data on an operator that declares hybrid arguments, so it
    # is folded into the UID rather than emitted as a ``static_data`` attribute. Every
    # operand matches the operator above, so the UID is the only thing that can differ.
    # CHECK: qref.operator "TrotterCDF"({{%.+}}: tensor<f64>, {{%.+}}: tensor<2x2x2xf64>, {{%.+}}: tensor<2x2x2xf64>, {{%.+}}: tensor<f64>)
    # ``CHECK-NOT`` scans up to the next ``CHECK``, so it is anchored on ``ctrls`` rather
    # than on the UID line itself; anchoring on the UID would leave an empty region and
    # pass vacuously.
    # CHECK-NOT: UID([[SINGLE_PHASE_UID]])
    # CHECK: ctrls({{%.+}}) ctrl_vals({{%.+}})
    # CHECK: param_map = {evolution_time = [0], hamiltonian = [1, 2, 3]}
    qp.ctrl(
        qp.TrotterCDF(
            evolution_time=t,
            num_trotter_steps=2,
            hamiltonian=CDF,
            wires=range(2 * N),
            double_phase=True,
        ),
        control=[2 * N],
    )
    return qp.state()


print(cdf_controlled.mlir)


@qp.qjit(capture=True, target="mlir", collect_decomp_rules=True)
@qp.qnode(qp.device("null.qubit", wires=2 * N))
def cdf_with_rules(t: float):
    # CHECK-LABEL: func.func public @cdf_with_rules

    # The operator still lowers with its tensor operands...
    # CHECK: qref.operator "TrotterCDF"({{%.+}}: tensor<f64>, {{%.+}}: tensor<2x2x2xf64>, {{%.+}}: tensor<2x2x2xf64>, {{%.+}}: tensor<f64>)
    # CHECK: param_map = {evolution_time = [0], hamiltonian = [1, 2, 3]}
    qp.TrotterCDF(evolution_time=t, num_trotter_steps=2, hamiltonian=CDF, wires=range(2 * N))
    return qp.state()


# CHECK: func.func private @"__builtin__trotter_cdf_decomposition_TrotterCDF{
# CHECK-SAME: evolution_time:[tensor<f64>]
# CHECK-SAME: hamiltonian:[tensor<2x2x2xf64>,tensor<2x2x2xf64>,tensor<f64>]
# CHECK-SAME: {wires:4}
print(cdf_with_rules.mlir)
