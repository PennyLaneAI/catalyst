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

# RUN: %PYTHON %s | FileCheck %s

from functools import partial

import pennylane as qml


def test_empty():
    """Test a simple example to verify that the empty pass can be used
    with Catalyst."""

    @qml.qjit(target="mlir")
    @qml.transform(pass_name="empty")
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit():
        # CHECK: transform.apply_registered_pass "empty" to {{%.+}}
        return qml.state()

    print(circuit.mlir)


test_empty()


def test_empty_with_options():
    """Test a simple example to verify that the empty pass can be used
    with Catalyst."""

    @qml.qjit(target="mlir")
    @partial(qml.transform(pass_name="empty"), key="foo")
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit():
        # CHECK: transform.apply_registered_pass "empty" with options = {"key" = "foo"} to {{%.+}}
        return qml.state()

    print(circuit.mlir)


test_empty_with_options()
