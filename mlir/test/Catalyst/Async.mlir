// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --qnode-to-async-lowering --verify-diagnostics --split-input-file %s | FileCheck %s

module @workflow {
  // Test function signature
  // CHECK: async.func private @f() -> !async.value<tensor<2xcomplex<f64>>> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode}
  func.func private @f() -> tensor<2xcomplex<f64>> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    quantum.device["/home/erick.ochoalopez/Code/cataliist/frontend/catalyst/utils/../../../runtime/build/lib/librtd_lightning.so", "LightningSimulator", "{'shots': 0}"]
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.compbasis %1 : !quantum.obs
    %3 = quantum.state %2 : tensor<2xcomplex<f64>>
    quantum.dealloc %0 : !quantum.reg
    quantum.drelease
    return %3 : tensor<2xcomplex<f64>>
  }
}

// -----

// Test return
module @workflow {
  func.func private @f() -> tensor<2xcomplex<f64>> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    quantum.device["/home/erick.ochoalopez/Code/cataliist/frontend/catalyst/utils/../../../runtime/build/lib/librtd_lightning.so", "LightningSimulator", "{'shots': 0}"]
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.compbasis %1 : !quantum.obs
    %3 = quantum.state %2 : tensor<2xcomplex<f64>>
    quantum.dealloc %0 : !quantum.reg
    quantum.drelease
    // Weirdly enough, the return does not contain `async` in namespace.
    // And also the return value even if transformed remains as tensor<2xcomplex<f64>>
    // This test is here just for documentation.
    // CHECK: return
    return %3 : tensor<2xcomplex<f64>>
  }
}

