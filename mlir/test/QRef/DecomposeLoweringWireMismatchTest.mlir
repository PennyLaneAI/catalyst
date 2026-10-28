// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: not quantum-opt --decompose-lowering %s 2>&1 | FileCheck %s

// CHECK: error: cannot build a 'tensor<1xi64>' operand for a decomposition rule from 2 value(s): the rule signature is inconsistent with the operator being decomposed.
module @wire_mismatch {
  func.func public @circuit() -> tensor<4xf64> attributes {quantum.node} {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %out:2 = quantum.custom "MismatchOp"() %1, %2 : !quantum.bit, !quantum.bit
    %3 = quantum.insert %0[ 0], %out#0 : !quantum.reg, !quantum.bit
    %4 = quantum.insert %3[ 1], %out#1 : !quantum.reg, !quantum.bit
    %5 = quantum.compbasis qreg %4 : !quantum.obs
    %6 = quantum.probs %5 : tensor<4xf64>
    quantum.dealloc %4 : !quantum.reg
    quantum.device_release
    return %6 : tensor<4xf64>
  }

  // The rule declares a single-wire grouped operand (tensor<1xi64>) for a two-wire target gate.
  func.func private @mismatch_rule(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg
      attributes {target_gate = "MismatchOp", llvm.linkage = #llvm.linkage<internal>} {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg0[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Terminal"() %2 : !quantum.bit
    %3 = quantum.insert %arg0[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }
}
