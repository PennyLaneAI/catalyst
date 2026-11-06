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

// The runtime receives an i1 argument as a C++ bool, so its declaration must mark the argument
// zeroext. Otherwise a computed i1, such as this negated measurement result, can reach the
// runtime with undefined upper bits and read as true.

// RUN: quantum-opt --convert-arith-to-llvm --convert-catalyst-to-llvm --convert-pbc-to-llvm \
// RUN:   --convert-quantum-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts %s \
// RUN:   | FileCheck %s
// RUN: quantum-opt --convert-arith-to-llvm --convert-catalyst-to-llvm --convert-pbc-to-llvm \
// RUN:   --convert-quantum-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts %s \
// RUN:   | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVMIR

// CHECK-DAG: llvm.func @__catalyst__qis__PauliRot(!llvm.ptr, f64, !llvm.ptr, i1 {llvm.zeroext}, i64, ...)
// CHECK-DAG: llvm.func @__catalyst__rt__assert_bool(i1 {llvm.zeroext}, !llvm.ptr)

// LLVMIR-DAG: declare void @__catalyst__qis__PauliRot(ptr, double, ptr, i1 zeroext, i64, ...)
// LLVMIR-DAG: declare void @__catalyst__rt__assert_bool(i1 zeroext, ptr)

// CHECK-LABEL: @negated_condition
func.func @negated_condition(%q0 : !quantum.bit, %theta : f64, %m : i1) -> !quantum.bit {
    %true = arith.constant true
    // CHECK: [[not_m:%.+]] = llvm.xor %arg2, {{%.+}} : i1
    %not_m = arith.xori %m, %true : i1
    // CHECK: llvm.call @__catalyst__qis__PauliRot({{%.+}}, {{%.+}}, {{%.+}}, [[not_m]], {{%.+}}, %arg0)
    %out = pbc.ppr.arbitrary ["Z"](%theta) %q0 cond(%not_m) : !quantum.bit
    return %out : !quantum.bit
}

// CHECK-LABEL: @negated_assert
func.func @negated_assert(%m : i1) {
    %true = arith.constant true
    // CHECK: [[not_m:%.+]] = llvm.xor %arg0, {{%.+}} : i1
    %not_m = arith.xori %m, %true : i1
    // CHECK: llvm.call @__catalyst__rt__assert_bool([[not_m]], {{%.+}})
    "catalyst.assert"(%not_m) <{error = "negated"}> : (i1) -> ()
    return
}
