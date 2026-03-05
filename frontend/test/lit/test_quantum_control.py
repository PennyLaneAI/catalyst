# Copyright 2022-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RUN: %PYTHON %s | FileCheck %s
"""Test the lowering cases involving quantum control"""

import jax.numpy as jnp
import pennylane as qml
from pennylane.devices.capabilities import OperatorProperties
from utils import get_custom_qjit_device

from catalyst import qjit


def test_named_controlled():
    """Test that named-controlled operations are passed as-is."""
    dev = get_custom_qjit_device(2, set(), set())

    @qjit(target="mlir")
    @qml.qnode(dev)
    # CHECK-LABEL: public @jit_named_controlled
    def named_controlled():
        # CHECK: quantum.custom "CNOT"
        qml.CNOT(wires=[0, 1])
        # CHECK: quantum.custom "CY"
        qml.CY(wires=[0, 1])
        # CHECK: quantum.custom "CZ"
        qml.CZ(wires=[0, 1])
        return qml.state()

    print(named_controlled.mlir)


test_named_controlled()


def test_native_controlled_custom():
    """Test native control of a custom operation."""
    dev = get_custom_qjit_device(3, set(), {"Rot": OperatorProperties(True, True, False)})

    @qjit(target="mlir")
    @qml.qnode(dev)
    # CHECK-LABEL: public @jit_native_controlled
    def native_controlled():
        # CHECK: [[out:%.+]], [[out_ctrl:%.+]]:2 = quantum.custom "Rot"
        # CHECK-SAME: ctrls
        # CHECK-SAME: ctrlvals(%true, %true)
        qml.ctrl(qml.Rot(0.3, 0.4, 0.5, wires=[0]), control=[1, 2])
        return qml.state()

    print(native_controlled.mlir)


test_native_controlled_custom()


def test_native_controlled_unitary():
    """Test native control of the unitary operation."""
    dev = get_custom_qjit_device(4, set(), set())

    @qjit(target="mlir")
    @qml.qnode(dev)
    # CHECK-LABEL: public @jit_native_controlled_unitary
    def native_controlled_unitary():
        # CHECK: [[out:%.+]] = quantum.unitary
        qml.ctrl(
            qml.QubitUnitary(
                jnp.array(
                    [
                        [0.70710678 + 0.0j, 0.70710678 + 0.0j],
                        [0.70710678 + 0.0j, -0.70710678 + 0.0j],
                    ],
                    dtype=jnp.complex128,
                ),
                wires=[0],
            ),
            control=[1, 2, 3],
        )
        return qml.state()

    print(native_controlled_unitary.mlir)


test_native_controlled_unitary()


def test_native_controlled_multirz():
    """Test native control of the multirz operation."""
    dev = get_custom_qjit_device(3, set(), {"MultiRZ": OperatorProperties(True, True, True)})

    @qjit(target="mlir")
    @qml.qnode(dev)
    # CHECK-LABEL: public @jit_native_controlled_multirz
    def native_controlled_multirz():
        # CHECK: [[out:%.+]]:2, [[out_ctrl:%.+]] = quantum.multirz
        # CHECK-SAME: ctrls
        # CHECK-SAME: ctrlvals(%true)
        qml.ctrl(qml.MultiRZ(0.6, wires=[0, 2]), control=[1])
        return qml.state()

    print(native_controlled_multirz.mlir)


test_native_controlled_multirz()


def test_native_controlled_pcphase():
    """Test native control of the PCPhase operation."""
    dev = get_custom_qjit_device(3, set(), {"PCPhase": OperatorProperties(True, True, False)})

    @qjit(target="mlir")
    @qml.qnode(dev)
    # CHECK-LABEL: public @jit_native_controlled_pcphase
    def native_controlled_pcphase():
        # CHECK: [[out:%.+]]:2, [[out_ctrl:%.+]] = quantum.pcphase
        # CHECK-SAME: ctrls
        # CHECK-SAME: ctrlvals(%true)
        qml.ctrl(qml.PCPhase(0.5, dim=2, wires=[0, 2]), control=[1])
        return qml.state()

    print(native_controlled_pcphase.mlir)


test_native_controlled_pcphase()
