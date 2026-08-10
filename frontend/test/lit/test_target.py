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

# RUN: %PYTHON %s | FileCheck %s
"""catalyst.target attaches catalyst.target/catalyst.dispatch to the nested module."""

import pennylane as qp

from catalyst import qjit, target


def test_full_target():
    """All target fields should appear in the emitted DictAttr."""

    dev = target(
        qp.device("null.qubit", wires=1),
        pipeline="my-pipeline",
        triple="my-triple",
    )

    @qjit(target="mlir")
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(0)
        return qp.state()

    # CHECK: module @module_circuit attributes {catalyst.target = {pipeline = "my-pipeline", triple = "my-triple"}}
    print(circuit.mlir)


test_full_target()


def test_single_field():
    """Only one field is set; the others must be absent from the dict."""

    dev = target(qp.device("null.qubit", wires=1), triple="my-triple")

    @qjit(target="mlir")
    @qp.qnode(dev)
    def circuit_min():
        return qp.state()

    # CHECK: module @module_circuit_min attributes {catalyst.target = {triple = "my-triple"}}
    print(circuit_min.mlir)


test_single_field()


def test_remote_dispatch():
    """An ``address`` additionally emits the catalyst.dispatch attribute."""

    dev = target(qp.device("null.qubit", wires=1), triple="my-triple", address="127.0.0.1:1373")

    @qjit(target="mlir")
    @qp.qnode(dev)
    def circuit_remote():
        return qp.state()

    # CHECK: module @module_circuit_remote attributes {catalyst.dispatch = {address = "127.0.0.1:1373"}, catalyst.target = {triple = "my-triple"}}
    print(circuit_remote.mlir)


test_remote_dispatch()


def test_untagged_device_has_no_attributes():
    """A device that was never tagged must not gain either attribute."""

    @qjit(target="mlir")
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit_plain():
        return qp.state()

    # CHECK: module @module_circuit_plain {
    print(circuit_plain.mlir)


test_untagged_device_has_no_attributes()
