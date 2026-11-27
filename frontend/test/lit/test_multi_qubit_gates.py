# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for multi-qubit gate compilation in Catalyst."""

# RUN: %PYTHON %s | FileCheck %s

import numpy as np
import pennylane as qp
from pennylane.devices.capabilities import OperatorProperties
from utils import get_custom_qjit_device

from catalyst import qjit


# CHECK-LABEL: public @jit_circuit
@qjit(target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=5))
def circuit(x: float):
    """Test circuit with various multi-qubit gates."""
    # CHECK: qref.custom "Identity"() {{.+}} : !qref.bit
    qp.Identity(0)
    # CHECK: qref.custom "CNOT"() {{.+}} : !qref.bit, !qref.bit
    qp.CNOT(wires=[0, 1])
    # CHECK: qref.custom "CSWAP"() {{.+}} : !qref.bit, !qref.bit, !qref.bit
    qp.CSWAP(wires=[0, 1, 2])
    # pylint: disable=line-too-long
    # CHECK: qref.multirz({{%.+}}) {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}} : !qref.bit, !qref.bit, !qref.bit, !qref.bit, !qref.bit
    qp.MultiRZ(x, wires=[0, 1, 2, 3, 4])

    # CHECK: qref.pcphase({{%.+}}, dim : 0) {{%.+}}, {{%.+}}, {{%.+}} : !qref.bit, !qref.bit, !qref.bit
    qp.PCPhase(x, dim=0, wires=[0, 1, 2])

    return qp.expval(qp.Z(wires=0))


print(circuit.mlir)


# CHECK-LABEL: public @jit_circuit_unitary
@qjit(target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=3))
def circuit_unitary():
    """Test circuit with unitary gates."""
    U1 = 1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
    # CHECK: qref.unitary({{%.+}} : tensor<2x2xcomplex<f64>>) {{%.+}} : !qref.bit
    qp.QubitUnitary(U1, wires=0)

    U2 = np.array(
        [
            [0.99500417 - 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 - 0.09983342j],
        ]
    )
    # pylint: disable=line-too-long
    # CHECK: qref.unitary({{%.+}} : tensor<4x4xcomplex<f64>>) {{%.+}}, {{%.+}} : !qref.bit, !qref.bit
    qp.QubitUnitary(U2, wires=[0, 2])

    return qp.expval(qp.Z(wires=0)), qp.expval(qp.Z(1))


print(circuit_unitary.mlir)


# CHECK-LABEL: public @jit_circuit_iswap_pswap
@qjit(target="mlir")
@qp.qnode(
    get_custom_qjit_device(2, (), {"ISWAP": OperatorProperties(), "PSWAP": OperatorProperties()})
)
def circuit_iswap_pswap(x: float):
    """Test circuit with ISWAP and PSWAP gates."""
    # CHECK: qref.custom "ISWAP"() {{.+}} : !qref.bit, !qref.bit
    qp.ISWAP(wires=[0, 1])
    # CHECK: qref.custom "PSWAP"({{%.+}}) {{.+}} : !qref.bit, !qref.bit
    qp.PSWAP(x, wires=[0, 1])
    return qp.probs()


print(circuit_iswap_pswap.mlir)


# CHECK-LABEL: public @jit_isingZZ_circuit
@qjit(target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=2))
def isingZZ_circuit(x: float):
    """Circuit that applies an IsingZZ gate to a pair of qubits."""
    # CHECK: qref.custom "IsingZZ"({{%.+}}) {{.+}} : !qref.bit, !qref.bit
    qp.IsingZZ(x, wires=[0, 1])
    return qp.state()


print(isingZZ_circuit.mlir)
