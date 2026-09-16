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

// RUN: quantum-opt --decompose-lowering --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func public @circuit
module @unused_param {
  func.func public @circuit() -> tensor<2xf64> attributes {quantum.node} {
    %cst = arith.constant 1.000000e+00 : f64
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom "UnusedParamOp"
    // CHECK: quantum.custom "Terminal"
    %out = quantum.custom "UnusedParamOp"(%cst) %1 : !quantum.bit
    %2 = quantum.insert %0[ 0], %out : !quantum.reg, !quantum.bit
    %3 = quantum.compbasis qreg %2 : !quantum.obs
    %4 = quantum.probs %3 : tensor<2xf64>
    quantum.dealloc %2 : !quantum.reg
    quantum.device_release
    return %4 : tensor<2xf64>
  }

  // CHECK: func.func private @unused_param_rule
  func.func private @unused_param_rule(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>)
      -> !quantum.reg attributes {target_gate = "UnusedParamOp", llvm.linkage = #llvm.linkage<internal>} {
    // %arg1 (the parameter) is intentionally unused by this rule body.
    %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg0[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Terminal"() %2 : !quantum.bit
    %3 = quantum.insert %arg0[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }
}
