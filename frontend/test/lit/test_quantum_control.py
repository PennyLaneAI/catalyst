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
""" Test the lowering cases involving quantum control """

import jax.numpy as jnp
import pennylane as qml

from catalyst import qjit

# This is used just for internal testing
from catalyst.pennylane_extensions import qfunc
from catalyst.qjit_device import QJITDevice
from catalyst.utils.runtime import device_get_toml_config


def get_custom_qjit_device(num_wires, discarded_operations=None, added_operations=None):
    """Generate a custom device with the modified set of supported gates."""

    lightning = qml.device("lightning.qubit", wires=3)
    config = device_get_toml_config(lightning)
    operations_copy = lightning.operations.copy()
    observables_copy = lightning.observables.copy()
    for op in discarded_operations or []:
        operations_copy.discard(op)
    for op in added_operations or []:
        operations_copy.add(op)

    class CustomQJITDevice(QJITDevice):
        """Custom Device"""

        name = "Device without some operations"
        short_name = "dummy.device"
        pennylane_requires = "0.1.0"
        version = "0.0.1"
        author = "CV quantum"

        operations = operations_copy
        observables = observables_copy

        # pylint: disable=too-many-arguments
        def __init__(self, shots=None, wires=None, backend=None):
            super().__init__(config, wires=wires, shots=shots, backend=backend)

        def apply(self, operations, **kwargs):  # pylint: disable=missing-function-docstring
            pass

    return CustomQJITDevice(wires=num_wires)


def test_named_controlled():
    """Test that named-controlled operations are passed as-is."""
    dev = get_custom_qjit_device(2, set(), set())

    @qjit(target="mlir")
    @qfunc(device=dev)
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
    dev = get_custom_qjit_device(
        3, discarded_operations={"CRot"}, added_operations={"Rot", "C(Rot)"}
    )

    @qjit(target="mlir")
    @qfunc(device=dev)
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
    dev = get_custom_qjit_device(4, set(), added_operations={"C(QubitUnitary)"})

    @qjit(target="mlir")
    @qfunc(device=dev)
    # CHECK-LABEL: public @jit_native_controlled_unitary
    def native_controlled_unitary():
        # CHECK: [[out:%.+]], [[out_ctrl:%.+]]:3 = quantum.unitary
        # CHECK-SAME: ctrls
        # CHECK-SAME: ctrlvals(%true, %true, %true)
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
    dev = get_custom_qjit_device(3, set(), {"C(MultiRZ)"})

    @qjit(target="mlir")
    @qfunc(device=dev)
    # CHECK-LABEL: public @jit_native_controlled_multirz
    def native_controlled_multirz():
        # CHECK: [[out:%.+]]:2, [[out_ctrl:%.+]] = quantum.multirz
        # CHECK-SAME: ctrls
        # CHECK-SAME: ctrlvals(%true)
        qml.ctrl(qml.MultiRZ(0.6, wires=[0, 2]), control=[1])
        return qml.state()

    print(native_controlled_multirz.mlir)


test_native_controlled_multirz()
