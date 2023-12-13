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
  // CHECK: async.func private @f() -> !async.value<memref<2xcomplex<f64>>> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode}
  func.func private @f() -> memref<2xcomplex<f64>> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0 = memref.alloc() : memref<2xcomplex<f64>>
    return %0 : memref<2xcomplex<f64>>
  }
}

// -----

// Test return
module @workflow {
  func.func private @f() -> memref<2xcomplex<f64>> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    // Weirdly enough, the return does not contain `async` in namespace.
    // And also the return value even if transformed remains as memref<2xcomplex<f64>>
    // This test is here just for documentation.
    // CHECK: return
    %0 = memref.alloc() : memref<2xcomplex<f64>>
    return %0 : memref<2xcomplex<f64>>
  }
}

// -----

// Test function call

module @foo {
  func.func public @jit_foo() -> (memref<2xcomplex<f64>>, memref<2xf64>) attributes {llvm.emit_c_interface} {
    // CHECK: async.call @foo()
    %0:2 = call @foo() : () -> (memref<2xcomplex<f64>>, memref<2xf64>)
    return %0#0, %0#1 : memref<2xcomplex<f64>>, memref<2xf64>
  }
  func.func private @foo() -> (memref<2xcomplex<f64>>, memref<2xf64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0 = memref.alloc() : memref<2xcomplex<f64>>
    %1 = memref.alloc() : memref<2xf64>
    return %0, %1 : memref<2xcomplex<f64>>, memref<2xf64>
  }
}

// -----

// Test await and copy

module @foo {
  func.func public @jit_foo() -> (memref<2xcomplex<f64>>, memref<2xf64>) attributes {llvm.emit_c_interface} {
    // CHECK: [[promise:%.+]]:2 = async.call @foo()
    // CHECK: [[return_val_0:%.+]] = async.await [[promise]]#0
    // CHECK: [[alloc_0:%.+]] = memref.alloc
    // CHECK: memref.copy [[return_val_0]], [[alloc_0]]

    // CHECK: [[return_val_1:%.+]] = async.await [[promise]]#1
    // CHECK: [[alloc_1:%.+]] = memref.alloc
    // CHECK: memref.copy [[return_val_1]], [[alloc_1]]

    %0:2 = call @foo() : () -> (memref<2xcomplex<f64>>, memref<2xf64>)
    return %0#0, %0#1 : memref<2xcomplex<f64>>, memref<2xf64>
  }
  func.func private @foo() -> (memref<2xcomplex<f64>>, memref<2xf64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0 = memref.alloc() : memref<2xcomplex<f64>>
    %1 = memref.alloc() : memref<2xf64>
    return %0, %1 : memref<2xcomplex<f64>>, memref<2xf64>
  }
}

// -----

// Test drop_ref

module @foo {
  func.func public @jit_foo() -> (memref<2xcomplex<f64>>, memref<2xf64>) attributes {llvm.emit_c_interface} {
    %0:2 = call @foo() : () -> (memref<2xcomplex<f64>>, memref<2xf64>)
    // CHECK: async.runtime.drop_ref
    // CHECK: async.runtime.drop_ref
    return %0#0, %0#1 : memref<2xcomplex<f64>>, memref<2xf64>
  }
  func.func private @foo() -> (memref<2xcomplex<f64>>, memref<2xf64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0 = memref.alloc() : memref<2xcomplex<f64>>
    %1 = memref.alloc() : memref<2xf64>
    return %0, %1 : memref<2xcomplex<f64>>, memref<2xf64>
  }
}
