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

// -----

// Test function call

module @foo {
  func.func public @jit_foo() -> (tensor<2xcomplex<f64>>, tensor<2xf64>) attributes {llvm.emit_c_interface} {
    // CHECK: async.call @foo()
    %0:2 = call @foo() : () -> (tensor<2xcomplex<f64>>, tensor<2xf64>)
    return %0#0, %0#1 : tensor<2xcomplex<f64>>, tensor<2xf64>
  }
  func.func private @foo() -> (tensor<2xcomplex<f64>>, tensor<2xf64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    quantum.device["/home/erick.ochoalopez/Code/cataliist/frontend/catalyst/utils/../../../runtime/build/lib/librtd_lightning.so", "LightningSimulator", "{'shots': 0}"]
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.compbasis %1 : !quantum.obs
    %3 = quantum.state %2 : tensor<2xcomplex<f64>>
    %4 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %5 = quantum.compbasis %4 : !quantum.obs
    %6 = quantum.probs %5 : tensor<2xf64>
    quantum.dealloc %0 : !quantum.reg
    quantum.drelease
    return %3, %6 : tensor<2xcomplex<f64>>, tensor<2xf64>
  }
}

// -----

// Test await

module @foo {
  func.func public @jit_foo() -> (tensor<2xcomplex<f64>>, tensor<2xf64>) attributes {llvm.emit_c_interface} {
    // CHECK: [[promise:%.+]]:2 = async.call @foo()
    // CHECK: [[return_val_0:%.+]] = async.await [[promise]]#0
    // CHECK: [[memref_heap_0:%.+]] = memref.alloc
    // CHECK: [[memref_async_0:%.+]] = bufferization.to_memref [[return_val_0]]
    // CHECK: memref.copy [[memref_async_0]], [[memref_heap_0]]
    // CHECK: [[tensor_heap_0:%.+]] = bufferization.to_tensor [[memref_heap_0]]
    // CHECK: [[return_val_1:%.+]] = async.await [[promise]]#1
    // CHECK: [[memref_heap_1:%.+]] = memref.alloc
    // CHECK: [[memref_async_1:%.+]] = bufferization.to_memref [[return_val_1]]
    // CHECK: memref.copy [[memref_async_1]], [[memref_heap_1]]
    // CHECK: [[tensor_heap_1:%.+]] = bufferization.to_tensor [[memref_heap_1]]
    // CHECK: return [[tensor_heap_0]], [[tensor_heap_1]]
    %0:2 = call @foo() : () -> (tensor<2xcomplex<f64>>, tensor<2xf64>)
    return %0#0, %0#1 : tensor<2xcomplex<f64>>, tensor<2xf64>
  }
  func.func private @foo() -> (tensor<2xcomplex<f64>>, tensor<2xf64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    quantum.device["/home/erick.ochoalopez/Code/cataliist/frontend/catalyst/utils/../../../runtime/build/lib/librtd_lightning.so", "LightningSimulator", "{'shots': 0}"]
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.compbasis %1 : !quantum.obs
    %3 = quantum.state %2 : tensor<2xcomplex<f64>>
    %4 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %5 = quantum.compbasis %4 : !quantum.obs
    %6 = quantum.probs %5 : tensor<2xf64>
    quantum.dealloc %0 : !quantum.reg
    quantum.drelease
    return %3, %6 : tensor<2xcomplex<f64>>, tensor<2xf64>
  }
}

// -----

// Test drop_ref

module @foo {
  func.func public @jit_foo() -> (tensor<2xcomplex<f64>>, tensor<2xf64>) attributes {llvm.emit_c_interface} {
    %0:2 = call @foo() : () -> (tensor<2xcomplex<f64>>, tensor<2xf64>)
    // CHECK: async.runtime.drop_ref
    // CHECK: async.runtime.drop_ref
    return %0#0, %0#1 : tensor<2xcomplex<f64>>, tensor<2xf64>
  }
  func.func private @foo() -> (tensor<2xcomplex<f64>>, tensor<2xf64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    quantum.device["/home/erick.ochoalopez/Code/cataliist/frontend/catalyst/utils/../../../runtime/build/lib/librtd_lightning.so", "LightningSimulator", "{'shots': 0}"]
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.compbasis %1 : !quantum.obs
    %3 = quantum.state %2 : tensor<2xcomplex<f64>>
    %4 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %5 = quantum.compbasis %4 : !quantum.obs
    %6 = quantum.probs %5 : tensor<2xf64>
    quantum.dealloc %0 : !quantum.reg
    quantum.drelease
    return %3, %6 : tensor<2xcomplex<f64>>, tensor<2xf64>
  }
}
