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

import shutil
from typing import Callable, Sequence

import numpy as np
import pennylane as qml
import pytest
from pennylane import transform

import catalyst
from catalyst import qjit
from catalyst.debug import get_compilation_stage


def test_split_multiple_tapes():
    dev = qml.device("lightning.qubit", wires=2)

    def my_quantum_transform(
        tape: qml.tape.QuantumTape,
    ) -> (Sequence[qml.tape.QuantumTape], Callable):
        tape1 = tape
        tape2 = qml.tape.QuantumTape(
            [qml.RY(tape1.operations[1].parameters[0] + 0.4, wires=0)], [qml.expval(qml.X(0))]
        )

        def post_processing_fn(results):
            return results[0] + results[1]

        return [tape1, tape2], post_processing_fn

    dispatched_transform = transform(my_quantum_transform)

    @qml.qnode(dev)
    def circuit(x):
        qml.adjoint(qml.RY)(x[0], wires=0)
        qml.RX(x[1] + 0.8, wires=1)
        return qml.expval(qml.X(0))

    circuit = dispatched_transform(circuit)
    expected = circuit([0.1, 0.2])

    circuit = qjit(circuit, keep_intermediate=True)
    qjit_results = circuit([0.1, 0.2])

    assert np.allclose(expected, qjit_results)

    split_tape_mlir = """module @circuit {
  func.func public @jit_circuit(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
    %0 = call @circuit(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
  module attributes {llvm.linkage = #llvm.linkage<internal>, transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      transform.yield 
    }
  }
  func.func private @circuit(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0:2 = call @circuit_tape_0(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    %extracted = tensor.extract %0#0[] : tensor<f64>
    %1 = call @circuit_tape_1(%0#1) : (tensor<f64>) -> tensor<f64>
    %extracted_0 = tensor.extract %1[] : tensor<f64>
    %2 = arith.addf %extracted, %extracted_0 : f64
    %from_elements = tensor.from_elements %2 : tensor<f64>
    return %from_elements : tensor<f64>
  }
  func.func private @circuit_tape_0(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %cst = arith.constant 8.000000e-01 : f64
    %extracted = tensor.extract %arg1[] : tensor<f64>
    quantum.device["/home/paul.wang/catalyst_new/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_lightning.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %out_qubits = quantum.custom "RY"(%extracted_0) %1 {adjoint} : !quantum.bit
    %2 = quantum.namedobs %out_qubits[ PauliX] : !quantum.obs
    %3 = quantum.expval %2 : f64
    %from_elements = tensor.from_elements %3 : tensor<f64>
    %4 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
    %5 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %6 = arith.addf %extracted, %cst : f64
    %from_elements_1 = tensor.from_elements %6 : tensor<f64>
    %out_qubits_2 = quantum.custom "RX"(%6) %5 : !quantum.bit
    %7 = quantum.insert %4[ 1], %out_qubits_2 : !quantum.reg, !quantum.bit
    quantum.dealloc %7 : !quantum.reg
    quantum.device_release
    return %from_elements, %from_elements_1 : tensor<f64>, tensor<f64>
  }
  func.func private @circuit_tape_1(%arg0: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %cst = arith.constant 4.000000e-01 : f64
    %extracted = tensor.extract %arg0[] : tensor<f64>
    quantum.device["/home/paul.wang/catalyst_new/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_lightning.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = arith.addf %extracted, %cst : f64
    %out_qubits = quantum.custom "RY"(%2) %1 : !quantum.bit
    %3 = quantum.namedobs %out_qubits[ PauliX] : !quantum.obs
    %4 = quantum.expval %3 : f64
    %from_elements = tensor.from_elements %4 : tensor<f64>
    %5 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
    quantum.dealloc %5 : !quantum.reg
    quantum.device_release
    return %from_elements : tensor<f64>
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}"""
    assert get_compilation_stage(circuit, "HLOLoweringPass") == split_tape_mlir
    shutil.rmtree("circuit")


if __name__ == "__main__":
    pytest.main(["-x", __file__])
