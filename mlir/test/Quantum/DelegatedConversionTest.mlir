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

// RUN: quantum-opt %s --convert-quantum-to-llvm --split-input-file | FileCheck %s

// CHECK-LABEL: @delegate
module @delegate {
    // CHECK: llvm.mlir.global internal constant @device_extension_Delegated("Delegated\00")
    func.func @delegate() {
        // CHECK: [[name:%.+]] = llvm.getelementptr inbounds
        // CHECK: [[version:%.+]] = llvm.mlir.constant(1 : i32) : i32
        // CHECK: llvm.call @__catalyst__rt__device_require([[name]], [[version]])
        // CHECK: [[mode:%.+]] = llvm.mlir.constant(1 : i32) : i32
        // CHECK: llvm.call @__catalyst__qis__BeginDelegatedProgram([[mode]])
        quantum.delegate hw_like
        return
    }
}

// -----

// CHECK-LABEL: @result_ops
module @result_ops {
    func.func @result_ops(%q: !quantum.bit) -> !quantum.bit {
        // CHECK: [[m:%.+]] = llvm.call @__catalyst__qis__Measure
        // CHECK-NOT: llvm.load
        %m, %a = quantum.measure %q : !quantum.res, !quantum.bit
        // CHECK: [[true:%.+]] = llvm.mlir.constant(true) : i1
        // CHECK: [[t:%.+]] = llvm.call @__catalyst__qis__ResultConstant([[true]])
        %t = quantum.result.constant true
        // CHECK: [[x:%.+]] = llvm.call @__catalyst__qis__ResultXor([[m]], [[t]])
        %x = quantum.result.xor %m, %t
        // CHECK: [[n:%.+]] = llvm.call @__catalyst__qis__ResultNot([[x]])
        %n = quantum.result.not %x
        // CHECK: [[s:%.+]] = llvm.call @__catalyst__qis__ResultSelect([[n]], [[m]], [[t]])
        %s = quantum.result.select %n, %m, %t
        // CHECK: llvm.call @__catalyst__qis__ReleaseResult([[s]])
        quantum.result.release %s
        return %a : !quantum.bit
    }
}

// -----

// The then-region runs under the condition and the else-region under its negation; the
// returned handle is selected between them.

// CHECK-LABEL: @predicated
module @predicated {
    func.func @predicated(%c: !quantum.res, %q: !quantum.bit) -> (!quantum.bit, !quantum.res) {
        // CHECK: llvm.call @__catalyst__qis__PushPredicate(%arg0)
        // CHECK: [[m:%.+]] = llvm.call @__catalyst__qis__Measure
        // CHECK: llvm.call @__catalyst__qis__PopPredicate()
        // CHECK: [[not:%.+]] = llvm.call @__catalyst__qis__ResultNot(%arg0)
        // CHECK: llvm.call @__catalyst__qis__PushPredicate([[not]])
        // CHECK: llvm.call @__catalyst__qis__PauliX
        // CHECK: [[f:%.+]] = llvm.call @__catalyst__qis__ResultConstant
        // CHECK: llvm.call @__catalyst__qis__PopPredicate()
        // CHECK: llvm.call @__catalyst__qis__ReleaseResult([[not]])
        // CHECK: [[s:%.+]] = llvm.call @__catalyst__qis__ResultSelect(%arg0, [[m]], [[f]])
        // CHECK: llvm.call @__catalyst__qis__ReleaseResult([[m]])
        // CHECK: llvm.call @__catalyst__qis__ReleaseResult([[f]])
        %r:2 = quantum.predicated %c (%q) : (!quantum.bit) -> (!quantum.bit, !quantum.res) {
        ^bb0(%arg0: !quantum.bit):
            %m, %o = quantum.measure %arg0 : !quantum.res, !quantum.bit
            quantum.yield %o, %m : !quantum.bit, !quantum.res
        } else {
        ^bb0(%arg0: !quantum.bit):
            %x = quantum.custom "PauliX"() %arg0 : !quantum.bit
            %f = quantum.result.constant false
            quantum.yield %x, %f : !quantum.bit, !quantum.res
        }
        return %r#0, %r#1 : !quantum.bit, !quantum.res
    }
}

// -----

// A region without ops needs no predicate.

// CHECK-LABEL: @empty_else
module @empty_else {
    func.func @empty_else(%c: !quantum.res, %q: !quantum.bit) -> !quantum.bit {
        // CHECK: llvm.call @__catalyst__qis__PushPredicate(%arg0)
        // CHECK: llvm.call @__catalyst__qis__PauliX
        // CHECK: llvm.call @__catalyst__qis__PopPredicate()
        // CHECK-NOT: __catalyst__qis__ResultNot
        // CHECK-NOT: __catalyst__qis__PushPredicate
        // CHECK: llvm.return
        %r = quantum.predicated %c (%q) : (!quantum.bit) -> (!quantum.bit) {
        ^bb0(%arg0: !quantum.bit):
            %x = quantum.custom "PauliX"() %arg0 : !quantum.bit
            quantum.yield %x : !quantum.bit
        } else {
        ^bb0(%arg0: !quantum.bit):
            quantum.yield %arg0 : !quantum.bit
        }
        return %r : !quantum.bit
    }
}
