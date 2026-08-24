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
"""Tests that CDF/CGF Hamiltonian tensor data lowers to MLIR as ranked tensor operands.

There are two use cases that we want to support:

* **concrete** data hamiltonians, and
* **abstract** data, where a Hamiltonian built from ``qp.typing.Float[...]``.

And there are two ways of accepting the Hamiltonian:

* A single ``hybrid`` argument, giving ``param_map = {hamiltonian = [1, 2, 3]}``, and
* as three ``dynamic`` arguments, naming each tensor individually in ``param_map``.

Note that the latter is the design to prefer for now, because Catalyst hands hybrid
arguments to decomposition-rule compilation as ``AbstractArray`` specifications rather
than traceable values, so a rule that computes with the tensors cannot be compiled.
"""

# pylint: disable = missing-function-docstring,line-too-long,unused-argument

# RUN: %PYTHON %s | FileCheck %s

import numpy as np
import pennylane as qp
from pennylane.numeric_hamiltonians import CDFHamiltonian, CGFHamiltonian, NumericHamiltonian
from pennylane.typing import Float

L, M, N = 2, 2, 3


class TrotterFragmented(qp.core.Operator2):
    """Takes the Hamiltonian as a single hybrid argument."""

    dynamic_argnames = ("evolution_time",)
    static_argnames = ("num_trotter_steps",)
    hybrid_argnames = ("hamiltonian",)
    wire_argnames = ("wires",)

    def __init__(self, evolution_time, num_trotter_steps, hamiltonian, wires):
        assert isinstance(hamiltonian, NumericHamiltonian)
        super().__init__(evolution_time, num_trotter_steps, hamiltonian, wires=wires)


class TrotterFragmentedFlat(qp.core.Operator2):
    """Takes the three tensors as dynamic arguments, so each is named in ``param_map``."""

    dynamic_argnames = ("evolution_time", "core_tensors", "leaf_tensors", "nuc_constant")
    static_argnames = ("num_trotter_steps",)
    wire_argnames = ("wires",)

    def __init__(
        self, evolution_time, core_tensors, leaf_tensors, nuc_constant, num_trotter_steps, wires
    ):
        super().__init__(
            evolution_time, core_tensors, leaf_tensors, nuc_constant, num_trotter_steps, wires=wires
        )

    @classmethod
    def from_hamiltonian(cls, evolution_time, num_trotter_steps, hamiltonian, wires):
        return cls(
            evolution_time, *hamiltonian.tensors, num_trotter_steps=num_trotter_steps, wires=wires
        )


@qp.register_resources({qp.RZ: 1, qp.GlobalPhase: 1})
def _flat_decomp(
    evolution_time, core_tensors, leaf_tensors, nuc_constant, num_trotter_steps, wires
):
    qp.RZ(evolution_time * core_tensors[0, ..., 0, 0].sum(), wires[0])
    qp.GlobalPhase(evolution_time * nuc_constant)


qp.add_decomps(TrotterFragmentedFlat, _flat_decomp)

rng = np.random.default_rng(42)
CGF = CGFHamiltonian(rng.random((L + 1, M, M, N, N)), rng.random((L + 1, M, N, N)), 0.5)
CDF = CDFHamiltonian(rng.random((L + 1, N, N)), rng.random((L + 1, N, N)), 0.5)

ABSTRACT_CGF = CGFHamiltonian(Float[L + 1, M, M, N, N], Float[L + 1, M, N, N], Float)
ABSTRACT_CDF = CDFHamiltonian(Float[L + 1, N, N], Float[L + 1, N, N], Float)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=M * N))
def cgf_concrete(t: float):
    # CHECK-LABEL: func.func public @cgf_concrete
    # CHECK: qref.operator "TrotterFragmentedFlat"({{%.+}}: tensor<f64>, {{%.+}}: tensor<3x2x2x3x3xf64>, {{%.+}}: tensor<3x2x3x3xf64>, {{%.+}}: tensor<f64>)
    # CHECK-SAME: qubits({{.+}})
    # CHECK: param_map = {core_tensors = [1], evolution_time = [0], leaf_tensors = [2], nuc_constant = [3]}
    # CHECK: qubit_map = {wires = [0, 1, 2, 3, 4, 5]}
    TrotterFragmentedFlat.from_hamiltonian(t, 10, CGF, wires=range(M * N))
    return qp.state()


print(cgf_concrete.mlir)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=2 * N))
def cdf_concrete(t: float):
    # CHECK-LABEL: func.func public @cdf_concrete
    # CHECK: qref.operator "TrotterFragmentedFlat"({{%.+}}: tensor<f64>, {{%.+}}: tensor<3x3x3xf64>, {{%.+}}: tensor<3x3x3xf64>, {{%.+}}: tensor<f64>)
    # CHECK: param_map = {core_tensors = [1], evolution_time = [0], leaf_tensors = [2], nuc_constant = [3]}
    # CHECK: qubit_map = {wires = [0, 1, 2, 3, 4, 5]}
    TrotterFragmentedFlat.from_hamiltonian(t, 10, CDF, wires=range(2 * N))
    return qp.state()


print(cdf_concrete.mlir)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=M * N))
def cgf_abstract(
    t: float,
    core: ABSTRACT_CGF.core_tensors,
    leaf: ABSTRACT_CGF.leaf_tensors,
    nuc: ABSTRACT_CGF.nuc_constant,
):
    # CHECK-LABEL: func.func public @cgf_abstract
    # CHECK-SAME: tensor<3x2x2x3x3xf64>
    # CHECK-SAME: tensor<3x2x3x3xf64>
    # CHECK: qref.operator "TrotterFragmentedFlat"({{%.+}}: tensor<f64>, {{%.+}}: tensor<3x2x2x3x3xf64>, {{%.+}}: tensor<3x2x3x3xf64>, {{%.+}}: tensor<f64>)
    # CHECK: param_map = {core_tensors = [1], evolution_time = [0], leaf_tensors = [2], nuc_constant = [3]}
    TrotterFragmentedFlat.from_hamiltonian(
        t, 10, CGFHamiltonian(core, leaf, nuc), wires=range(M * N)
    )
    return qp.state()


print(cgf_abstract.mlir)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=2 * N))
def cdf_abstract(
    t: float,
    core: ABSTRACT_CDF.core_tensors,
    leaf: ABSTRACT_CDF.leaf_tensors,
    nuc: ABSTRACT_CDF.nuc_constant,
):
    # CHECK-LABEL: func.func public @cdf_abstract
    # CHECK-SAME: tensor<3x3x3xf64>
    # CHECK: qref.operator "TrotterFragmentedFlat"({{%.+}}: tensor<f64>, {{%.+}}: tensor<3x3x3xf64>, {{%.+}}: tensor<3x3x3xf64>, {{%.+}}: tensor<f64>)
    # CHECK: param_map = {core_tensors = [1], evolution_time = [0], leaf_tensors = [2], nuc_constant = [3]}
    TrotterFragmentedFlat.from_hamiltonian(
        t, 10, CDFHamiltonian(core, leaf, nuc), wires=range(2 * N)
    )
    return qp.state()


print(cdf_abstract.mlir)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=M * N))
def cgf_hybrid_arg(t: float):
    # CHECK-LABEL: func.func public @cgf_hybrid_arg
    # CHECK: qref.operator "TrotterFragmented"({{%.+}}: tensor<f64>, {{%.+}}: tensor<3x2x2x3x3xf64>, {{%.+}}: tensor<3x2x3x3xf64>, {{%.+}}: tensor<f64>)
    # CHECK: UID
    # CHECK: param_map = {evolution_time = [0], hamiltonian = [1, 2, 3]}
    # CHECK: qubit_map = {wires = [0, 1, 2, 3, 4, 5]}
    TrotterFragmented(t, 10, CGF, wires=range(M * N))
    return qp.state()


print(cgf_hybrid_arg.mlir)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=M * N))
def cgf_hybrid_arg_abstract(
    t: float,
    core: ABSTRACT_CGF.core_tensors,
    leaf: ABSTRACT_CGF.leaf_tensors,
    nuc: ABSTRACT_CGF.nuc_constant,
):
    # CHECK-LABEL: func.func public @cgf_hybrid_arg_abstract
    # CHECK-SAME: tensor<3x2x2x3x3xf64>
    # CHECK: qref.operator "TrotterFragmented"({{%.+}}: tensor<f64>, {{%.+}}: tensor<3x2x2x3x3xf64>, {{%.+}}: tensor<3x2x3x3xf64>, {{%.+}}: tensor<f64>)
    # CHECK: param_map = {evolution_time = [0], hamiltonian = [1, 2, 3]}
    TrotterFragmented(t, 10, CGFHamiltonian(core, leaf, nuc), wires=range(M * N))
    return qp.state()


print(cgf_hybrid_arg_abstract.mlir)
