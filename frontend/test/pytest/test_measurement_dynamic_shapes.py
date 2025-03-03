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
"""
This file contains tests for measurement primitives when the return shape is dynamic.
"""

# pylint: disable=line-too-long

import numpy as np
import pytest

import pennylane as qml

import catalyst
from catalyst.debug import get_compilation_stage, replace_ir


def test_dynamic_sample_backend_functionality():
    """Test that a `sample` program with dynamic shots can be executed correctly."""

    @catalyst.qjit(keep_intermediate=True)
    def workflow_dyn_sample(shots):  # pylint: disable=unused-argument
        # qml.device still needs concrete shots
        device = qml.device("lightning.qubit", wires=1, shots=10)

        @qml.qnode(device)
        def circuit():
            qml.RX(1.5, 0)
            return qml.sample()

        return circuit()

    workflow_dyn_sample(10)
    old_ir = get_compilation_stage(workflow_dyn_sample, "mlir")
    workflow_dyn_sample.workspace.cleanup()

    new_ir = old_ir.replace(
        "catalyst.launch_kernel @module_circuit::@circuit() : () -> tensor<10x1xi64>",
        "catalyst.launch_kernel @module_circuit::@circuit(%arg0) : (tensor<i64>) -> tensor<?x1xi64>",
    )
    new_ir = new_ir.replace(
        "func.func public @circuit() -> tensor<10x1xi64>",
        "func.func public @circuit(%arg0: tensor<i64>) -> tensor<?x1xi64>",
    )
    new_ir = new_ir.replace(
        "quantum.device shots(%extracted) [",
        """%shots = tensor.extract %arg0[] : tensor<i64>
      quantum.device shots(%shots) [""",
    )
    new_ir = new_ir.replace("tensor<10x1x", "tensor<?x1x")

    replace_ir(workflow_dyn_sample, "mlir", new_ir)
    res = workflow_dyn_sample(37)
    assert len(res) == 37

    workflow_dyn_sample.workspace.cleanup()


def test_dynamic_counts_backend_functionality():
    """Test that a `counts` program with dynamic shots can be executed correctly."""

    @catalyst.qjit(keep_intermediate=True)
    def workflow_dyn_counts(shots):  # pylint: disable=unused-argument
        # qml.device still needs concrete shots
        device = qml.device("lightning.qubit", wires=1, shots=10)

        @qml.qnode(device)
        def circuit():
            qml.RX(1.5, 0)
            return qml.counts()

        return circuit()

    workflow_dyn_counts(10)
    old_ir = get_compilation_stage(workflow_dyn_counts, "mlir")
    workflow_dyn_counts.workspace.cleanup()

    new_ir = old_ir.replace(
        "catalyst.launch_kernel @module_circuit::@circuit() : () ->",
        "catalyst.launch_kernel @module_circuit::@circuit(%arg0) : (tensor<i64>) ->",
    )
    new_ir = new_ir.replace(
        "func.func public @circuit() ->", "func.func public @circuit(%arg0: tensor<i64>) ->"
    )
    new_ir = new_ir.replace(
        "quantum.device shots(%extracted) [",
        """%shots = tensor.extract %arg0[] : tensor<i64>
      quantum.device shots(%shots) [""",
    )

    replace_ir(workflow_dyn_counts, "mlir", new_ir)
    res = workflow_dyn_counts(4000)
    assert res[1][0] + res[1][1] == 4000

    workflow_dyn_counts.workspace.cleanup()


@pytest.mark.parametrize("readout", [qml.expval, qml.var])
def test_dynamic_wires_scalar_readouts(readout, backend, capfd):
    """
    Test that a circuit with dynamic number of wires can be executed correctly.

    As a unit test for allocating a dynamic number of wires, we use measurements
    whose shape do not depend on the number of wires, i.e. expval and var
    """

    def ref(num_qubits):
        print("compiling...")
        dev = qml.device(backend, wires=num_qubits)

        @qml.qnode(dev)
        def circ():
            @catalyst.for_loop(0, num_qubits, 1)
            def loop_0(i):
                qml.RY(2.2, wires=i)

            loop_0()
            qml.RX(1.23, wires=num_qubits - 1)
            return readout(qml.Z(wires=num_qubits - 1))

        return circ()

    cat = catalyst.qjit(ref)

    assert np.allclose(ref(10), cat(10))
    assert np.allclose(ref(4), cat(4))
    out, _ = capfd.readouterr()
    assert out.count("compiling...") == 3


if __name__ == "__main__":
    pytest.main(["-x", __file__])
