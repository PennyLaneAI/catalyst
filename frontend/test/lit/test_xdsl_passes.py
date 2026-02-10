# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# REQUIRES: xdsl
# RUN: %PYTHON %s | FileCheck %s

"""
This file tests that the xDSL passes are detected and applied correctly.
"""

import pennylane as qml

from catalyst.python_interface.transforms import merge_rotations_pass


def test_mlir_pass_no_attribute():
    """Test that MLIR-only passes do NOT set uses_xdsl_passes and xdsl_pass attributes"""

    @qml.qjit(target="mlir")
    @qml.transforms.cancel_inverses
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit_with_mlir_pass():
        """Circuit using only MLIR pass"""
        qml.RX(0.5, 0)
        return qml.expval(qml.Z(0))

    print(circuit_with_mlir_pass.mlir)
    # CHECK-NOT: catalyst.uses_xdsl_passes
    # CHECK-NOT: catalyst.xdsl_pass


test_mlir_pass_no_attribute()


def test_xdsl_pass_with_attribute():
    """Test that xDSL passes set uses_xdsl_passes and xdsl_pass attributes"""
    qml.capture.enable()

    @qml.qjit(target="mlir")
    @merge_rotations_pass
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit_with_xdsl_pass():
        """Circuit using xDSL pass"""
        qml.RX(0.5, 0)
        return qml.expval(qml.Z(0))

    print(circuit_with_xdsl_pass.mlir)
    # CHECK: catalyst.uses_xdsl_passes
    # CHECK: catalyst.xdsl_pass
    qml.capture.disable()


test_xdsl_pass_with_attribute()


def test_mixed_passes_with_attribute():
    """Test that mixing MLIR and xDSL passes sets uses_xdsl_passes and xdsl_pass attributes"""
    qml.capture.enable()

    @qml.qjit(target="mlir")
    @qml.transforms.cancel_inverses
    @merge_rotations_pass
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit_with_mixed_passes():
        """Circuit using both MLIR and xDSL passes"""
        qml.RX(0.5, 0)
        return qml.expval(qml.Z(0))

    print(circuit_with_mixed_passes.mlir)
    # CHECK: catalyst.uses_xdsl_passes
    # CHECK: catalyst.xdsl_pass
    qml.capture.disable()


test_mixed_passes_with_attribute()
