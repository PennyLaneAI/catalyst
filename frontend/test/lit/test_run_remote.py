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
"""catalyst.run_remote tags an ordinary device for remote dispatch (catalyst.dispatch / target)."""

import pennylane as qp

import catalyst
from catalyst import qjit


def test_run_remote_address():
    """run_remote(device, address) attaches catalyst.dispatch with the executor address."""

    dev = catalyst.run_remote(qp.device("null.qubit", wires=1), "ADDR:PORT")

    @qjit(target="mlir")
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(0)
        return qp.state()

    # CHECK: catalyst.dispatch = {address = "ADDR:PORT"}
    print(circuit.mlir)


test_run_remote_address()


def test_run_remote_endpoint_triple():
    """run_remote from an Endpoint carries its triple into catalyst.target."""

    ep = qp.Endpoint("gpu.ip", role="qpu", local=False, attrs={"triple": "my-triple"})
    dev = catalyst.run_remote(qp.device("null.qubit", wires=1), ep)

    @qjit(target="mlir")
    @qp.qnode(dev)
    def circuit_ep():
        return qp.state()

    # CHECK: catalyst.dispatch = {address = "gpu.ip"}
    # CHECK-SAME: catalyst.target = {triple = "my-triple"}
    print(circuit_ep.mlir)


test_run_remote_endpoint_triple()
