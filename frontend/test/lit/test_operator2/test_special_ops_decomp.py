# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test decomposition rules registered on the special operations are correctly generated.

There are a few special operators that do not lower to CustomOp or OperatorOp in MLIR, and instead
lower to their own operations:
- MultiRZ
- PauliRot
- GlobalPhase
- PCPhase
- QubitUnitary
- BasisState
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable = line-too-long,unused-argument

import pennylane as qp
from jax import numpy as jnp


def test_multirz():
    """
    Test that decomposing qp.MultiRZ works.
    """

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(qp.device("null.qubit", wires=3))
    def multirz():
        qp.MultiRZ(theta=0.1, wires=[0, 1])
        qp.MultiRZ(theta=0.2, wires=[2])
        return qp.state()

    print(multirz.mlir)


# CHECK: func.func public @multirz()
# CHECK: qref.multirz({{%.+}}) {{%.+}}, {{%.+}} : !qref.bit, !qref.bit
# CHECK: qref.multirz({{%.+}}) {{%.+}} : !qref.bit
# CHECK: func.func private @"__builtin__multi_rz_decomposition_MultiRZ{theta:[f64]}{wires:2}{}"
# CHECK-SAME:   target_gate = "MultiRZ{theta:[f64]}{wires:2}{}"
# CHECK: func.func private @"__builtin__multi_rz_decomposition_MultiRZ{theta:[f64]}{wires:1}{}"
# CHECK-SAME:   target_gate = "MultiRZ{theta:[f64]}{wires:1}{}"
test_multirz()


def test_paulirot():
    """
    Test that decomposing qp.PauliRot works.
    """

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(qp.device("null.qubit", wires=3))
    def paulirot():
        qp.PauliRot(theta=0.1, pauli_word="XX", wires=[0, 1])
        qp.PauliRot(theta=0.2, pauli_word="Z", wires=[2])
        qp.PauliRot(theta=0.2, pauli_word="YZX", wires=[0, 1, 2])
        return qp.state()

    print(paulirot.mlir)


# CHECK: func.func public @paulirot()
# CHECK: qref.paulirot ["X", "X"]({{%.+}}) {{%.+}}, {{%.+}} : !qref.bit, !qref.bit
# CHECK: qref.paulirot ["Z"]({{%.+}}) {{%.+}} : !qref.bit
# CHECK: qref.paulirot ["Y", "Z", "X"]({{%.+}}) {{%.+}}, {{%.+}}, {{%.+}} : !qref.bit, !qref.bit, !qref.bit
# CHECK: func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:2}{pauli_word:XX}"
# CHECK-SAME:   target_gate = "PauliRot{theta:[f64]}{wires:2}{pauli_word:XX}"
# CHECK: func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:1}{pauli_word:Z}"
# CHECK-SAME:   target_gate = "PauliRot{theta:[f64]}{wires:1}{pauli_word:Z}"
# CHECK: func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:3}{pauli_word:YZX}"
# CHECK-SAME:   target_gate = "PauliRot{theta:[f64]}{wires:3}{pauli_word:YZX}"
test_paulirot()


def test_pcphase():
    """
    Test that decomposing qp.PCPhase works.
    """

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(qp.device("null.qubit", wires=3))
    def pcphase():
        qp.PCPhase(0.27, dim=3, wires=range(3))
        qp.PCPhase(0.27, dim=0, wires=range(1))
        return qp.state()

    print(pcphase.mlir)


# CHECK: func.func public @pcphase()
# CHECK: qref.pcphase({{%.+}}, dim : 3) {{%.+}}, {{%.+}}, {{%.+}} : !qref.bit, !qref.bit, !qref.bit
# CHECK: qref.pcphase({{%.+}}, dim : 0) {{%.+}} : !qref.bit
# CHECK: func.func private @"__builtin__decompose_pcphase_PCPhase{phi:[f64]}{wires:3}{dim:3}"
# CHECK-SAME:   target_gate = "PCPhase{phi:[f64]}{wires:3}{dim:3}"
# CHECK: func.func private @"__builtin__decompose_pcphase_PCPhase{phi:[f64]}{wires:1}{dim:0}"
# CHECK-SAME:   target_gate = "PCPhase{phi:[f64]}{wires:1}{dim:0}"
test_pcphase()


def test_qubit_unitary():
    """
    Test that decomposing qp.QubitUnitary works.
    """

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(qp.device("null.qubit", wires=3))
    def unitary():
        qp.QubitUnitary(jnp.eye(4), wires=[0, 1])
        return qp.state()

    print(unitary.mlir)


# CHECK: func.func public @unitary
# CHECK: qref.unitary({{%.+}} : tensor<4x4xcomplex<f64>>)
# CHECK: func.func private @"__builtin_two_qubit_decomp_rule_QubitUnitary
# CHECK-SAME: {U:{{\[\[}}complex<f64>,complex<f64>,complex<f64>,complex<f64>],
# CHECK-SAME:     [complex<f64>,complex<f64>,complex<f64>,complex<f64>],
# CHECK-SAME:     [complex<f64>,complex<f64>,complex<f64>,complex<f64>],
# CHECK-SAME:     [complex<f64>,complex<f64>,complex<f64>,complex<f64>{{\]\]}}}{wires:2}{}"
# CHECK-SAME:   target_gate = "QubitUnitary
# CHECK-SAME:   {U:{{\[\[}}complex<f64>,complex<f64>,complex<f64>,complex<f64>],
# CHECK-SAME:       [complex<f64>,complex<f64>,complex<f64>,complex<f64>],
# CHECK-SAME:       [complex<f64>,complex<f64>,complex<f64>,complex<f64>],
# CHECK-SAME:       [complex<f64>,complex<f64>,complex<f64>,complex<f64>{{\]\]}}}{wires:2}{}"
test_qubit_unitary()


def test_gphase():
    """
    Test that decomposing qp.GlobalPhase works.
    Note that global phase does not have decomposition rules.
    """

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(qp.device("null.qubit", wires=3))
    def gphase():
        qp.GlobalPhase(0.27)
        return qp.state()

    print(gphase.mlir)


# CHECK: func.func public @gphase()
# CHECK: qref.gphase
test_gphase()


def test_basis_state():
    """
    Test that decomposing qp.BasisState works.
    """

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(qp.device("null.qubit", wires=3))
    def basisstate():
        qp.BasisState(jnp.array([0]), wires=[2])
        qp.BasisState(jnp.array([1, 1]), wires=[0, 1])
        return qp.state()

    print(basisstate.mlir)


# CHECK: func.func public @basisstate
# CHECK: qref.set_basis_state({{%.+}}) {{%.+}} : tensor<1xi1>, !qref.bit
# CHECK: qref.set_basis_state({{%.+}}) {{%.+}}, {{%.+}} : tensor<2xi1>, !qref.bit, !qref.bit
# CHECK: func.func private @"__builtin__basis_state_decomp_BasisState{state:[i1]}{wires:1}{}"
# CHECK-SAME:   target_gate = "BasisState{state:[i1]}{wires:1}{}"
# CHECK: func.func private @"__builtin__basis_state_decomp_BasisState{state:[i1,i1]}{wires:2}{}"
# CHECK-SAME:   target_gate = "BasisState{state:[i1,i1]}{wires:2}{}"
test_basis_state()
