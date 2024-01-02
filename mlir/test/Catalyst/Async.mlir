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

// Test to find out if call op has been tagged with attribute transformed.

module @workflow {
  func.func private @f() -> memref<2xcomplex<f64>> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0 = memref.alloc() : memref<2xcomplex<f64>>
    return %0 : memref<2xcomplex<f64>>
  }

  func.func public @jit_foo() -> (memref<2xcomplex<f64>>) attributes {llvm.emit_c_interface} {
    // CHECK: transformed
    %0 = call @f() : () -> (memref<2xcomplex<f64>>)
    return %0 : memref<2xcomplex<f64>>
  }

}

// -----

// Test to find out if async.execute has been added.

module @workflow {
  func.func private @f() -> memref<2xcomplex<f64>> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0 = memref.alloc() : memref<2xcomplex<f64>>
    return %0 : memref<2xcomplex<f64>>
  }

  func.func public @jit_foo() -> (memref<2xcomplex<f64>>) attributes {llvm.emit_c_interface} {
    // CHECK: async.execute
    %0 = call @f() : () -> (memref<2xcomplex<f64>>)
    return %0 : memref<2xcomplex<f64>>
  }

}

// -----

// Test to find out if async.await has been added.

module @workflow {
  func.func private @f() -> memref<2xcomplex<f64>> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0 = memref.alloc() : memref<2xcomplex<f64>>
    return %0 : memref<2xcomplex<f64>>
  }

  func.func public @jit_foo() -> (memref<2xcomplex<f64>>) attributes {llvm.emit_c_interface} {
    // CHECK: async.await
    %0 = call @f() : () -> (memref<2xcomplex<f64>>)
    return %0 : memref<2xcomplex<f64>>
  }

}

// -----

// Test clone

module @workflow {
  func.func private @f() -> memref<2xcomplex<f64>> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0 = memref.alloc() : memref<2xcomplex<f64>>
    return %0 : memref<2xcomplex<f64>>
  }

  func.func public @jit_foo() -> (memref<2xcomplex<f64>>) attributes {llvm.emit_c_interface} {
    // CHECK: call @f
    // CHECK: async.yield
    // CHECK-NOT: call @f
    // CHECK: return
    %0 = call @f() : () -> (memref<2xcomplex<f64>>)
    return %0 : memref<2xcomplex<f64>>
  }
}

// -----

// Test the async.yield type

module @workflow {
  func.func private @f() -> memref<2xcomplex<f64>> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0 = memref.alloc() : memref<2xcomplex<f64>>
    return %0 : memref<2xcomplex<f64>>
  }

  func.func public @jit_foo() -> (memref<2xcomplex<f64>>) attributes {llvm.emit_c_interface} {
    // CHECK: async.yield
    // CHECK-SAME: memref<2xcomplex<f64>>
    %0 = call @f() : () -> (memref<2xcomplex<f64>>)
    return %0 : memref<2xcomplex<f64>>
  }
}

// -----

// Check that the return type is the one from inside the async.execute

module @workflow {
  func.func private @f() -> memref<2xcomplex<f64>> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0 = memref.alloc() : memref<2xcomplex<f64>>
    return %0 : memref<2xcomplex<f64>>
  }

  func.func public @jit_foo() -> (memref<2xcomplex<f64>>) attributes {llvm.emit_c_interface} {
    // CHECK: [[token:%.+]], [[bodyResults:%.+]] = async.execute
    // CHECK: [[value:%.+]] = async.await [[bodyResults]]
    // CHECK: return [[value:%.+]]
    %0 = call @f() : () -> (memref<2xcomplex<f64>>)
    return %0 : memref<2xcomplex<f64>>
  }
}

// -----

// Test for multiple returns
module @workflow {
  func.func private @f() -> (memref<2xcomplex<f64>>, memref<2xcomplex<f64>>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0 = memref.alloc() : memref<2xcomplex<f64>>
    return %0, %0 : memref<2xcomplex<f64>>, memref<2xcomplex<f64>>
  }

  func.func public @jit_foo() -> (memref<2xcomplex<f64>>, memref<2xcomplex<f64>>) attributes {llvm.emit_c_interface} {
    // CHECK: [[token:%.+]], [[bodyResults:%.+]]:2 = async.execute
    // CHECK: [[value0:%.+]] = async.await [[bodyResults]]#0
    // CHECK: [[value1:%.+]] = async.await [[bodyResults]]#1
    %0:2 = call @f() : () -> (memref<2xcomplex<f64>>, memref<2xcomplex<f64>>)
    return %0#0, %0#1 : memref<2xcomplex<f64>>, memref<2xcomplex<f64>>
  }
}
