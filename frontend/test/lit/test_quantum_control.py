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

import os
import tempfile

import jax.numpy as jnp
import pennylane as qml

from catalyst import qjit


def get_custom_qjit_device(num_wires, discards, additions):
    """Generate a custom device without gates in discards."""

    class CustomDevice(qml.QubitDevice):
        """Custom Gate Set Device"""

        name = "Custom Device"
        short_name = "lightning.qubit"
        pennylane_requires = "0.35.0"
        version = "0.0.2"
        author = "Tester"

        lightning_device = qml.device("lightning.qubit", wires=0)
        operations = lightning_device.operations.copy() - discards | additions
        observables = lightning_device.observables.copy()

        config = None
        backend_name = "default"
        backend_lib = "default"
        backend_kwargs = {}

        def __init__(self, shots=None, wires=None):
            super().__init__(wires=wires, shots=shots)
            self.toml_file = None

        def apply(self, operations, **kwargs):
            """Unused"""
            raise RuntimeError("Only C/C++ interface is defined")

        def __enter__(self, *args, **kwargs):
            lightning_toml = self.lightning_device.config
            with open(lightning_toml, mode="r", encoding="UTF-8") as f:
                toml_contents = f.readlines()

            # TODO: update once schema 2 is merged
            updated_toml_contents = []
            for line in toml_contents:
                if any(f'"{gate}",' in line for gate in discards):
                    continue

                updated_toml_contents.append(line)
                if "native = [" in line:
                    for gate in additions:
                        if not gate.startswith("C("):
                            updated_toml_contents.append(f'        "{gate}",\n')

            self.toml_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
            self.toml_file.writelines(updated_toml_contents)
            self.toml_file.close()  # close for now without deleting

            self.config = self.toml_file.name
            return self

        def __exit__(self, *args, **kwargs):
            os.unlink(self.toml_file.name)
            self.config = None

    return CustomDevice(wires=num_wires)


def test_named_controlled():
    """Test that named-controlled operations are passed as-is."""
    with get_custom_qjit_device(2, set(), set()) as dev:

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
    with get_custom_qjit_device(3, {"CRot"}, {"Rot", "C(Rot)"}) as dev:

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
    with get_custom_qjit_device(4, set(), set()) as dev:

        @qjit(target="mlir")
        @qml.qnode(dev)
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
    with get_custom_qjit_device(3, set(), {"C(MultiRZ)"}) as dev:

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
