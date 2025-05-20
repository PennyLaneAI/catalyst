// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s \
// RUN:   --convert-arith-to-llvm \
// RUN:   --convert-mbqc-to-llvm \
// RUN:   --convert-quantum-to-llvm \
// RUN:   --reconcile-unrealized-casts \
// RUN:   --split-input-file -verify-diagnostics \
// RUN: | FileCheck %s

// CHECK-DAG: llvm.func @__catalyst__mbqc__measure_in_basis(!llvm.ptr, i32, f64, i32)

// CHECK-LABEL: testXY
func.func @testXY(%q1 : !quantum.bit) {
    // CHECK: [[angle:%.+]] = llvm.mlir.constant(3.1415926535897931 : f64) : f64
    %angle = arith.constant 3.141592653589793 : f64

    // CHECK: [[plane:%.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[postselect:%.+]] = llvm.mlir.constant(-1 : i32) : i32
    // CHECK: llvm.call @__catalyst__mbqc__measure_in_basis(%arg0, [[plane]], [[angle]], [[postselect]])
    %res, %new_q = mbqc.measure_in_basis [XY, %angle] %q1 : i1, !quantum.bit
    func.return
}

// -----

// CHECK-DAG: llvm.func @__catalyst__mbqc__measure_in_basis(!llvm.ptr, i32, f64, i32)

// CHECK-LABEL: testYZ
func.func @testYZ(%q1 : !quantum.bit) {
    // CHECK: [[angle:%.+]] = llvm.mlir.constant(3.1415926535897931 : f64) : f64
    %angle = arith.constant 3.141592653589793 : f64

    // CHECK: [[plane:%.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[postselect:%.+]] = llvm.mlir.constant(-1 : i32) : i32
    // CHECK: llvm.call @__catalyst__mbqc__measure_in_basis(%arg0, [[plane]], [[angle]], [[postselect]])
    %res, %new_q = mbqc.measure_in_basis [YZ, %angle] %q1 : i1, !quantum.bit
    func.return
}

// -----

// CHECK-DAG: llvm.func @__catalyst__mbqc__measure_in_basis(!llvm.ptr, i32, f64, i32)

// CHECK-LABEL: testZX
func.func @testZX(%q1 : !quantum.bit) {
    // CHECK: [[angle:%.+]] = llvm.mlir.constant(3.1415926535897931 : f64) : f64
    %angle = arith.constant 3.141592653589793 : f64

    // CHECK: [[plane:%.+]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: [[postselect:%.+]] = llvm.mlir.constant(-1 : i32) : i32
    // CHECK: llvm.call @__catalyst__mbqc__measure_in_basis(%arg0, [[plane]], [[angle]], [[postselect]])
    %res, %new_q = mbqc.measure_in_basis [ZX, %angle] %q1 : i1, !quantum.bit
    func.return
}

// -----

// CHECK-DAG: llvm.func @__catalyst__mbqc__measure_in_basis(!llvm.ptr, i32, f64, i32)

// CHECK-LABEL: testXYPostSelect0
func.func @testXYPostSelect0(%q1 : !quantum.bit) {
    // CHECK: [[angle:%.+]] = llvm.mlir.constant(3.1415926535897931 : f64) : f64
    %angle = arith.constant 3.141592653589793 : f64

    // CHECK: [[plane:%.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[postselect:%.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.call @__catalyst__mbqc__measure_in_basis(%arg0, [[plane]], [[angle]], [[postselect]])
    %res, %new_q = mbqc.measure_in_basis [XY, %angle] %q1 postselect 0 : i1, !quantum.bit
    func.return
}

// -----

// CHECK-DAG: llvm.func @__catalyst__mbqc__measure_in_basis(!llvm.ptr, i32, f64, i32)

// CHECK-LABEL: testXYPostSelect1
func.func @testXYPostSelect1(%q1 : !quantum.bit) {
    // CHECK: [[angle:%.+]] = llvm.mlir.constant(3.1415926535897931 : f64) : f64
    %angle = arith.constant 3.141592653589793 : f64

    // CHECK: [[plane:%.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[postselect:%.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: llvm.call @__catalyst__mbqc__measure_in_basis(%arg0, [[plane]], [[angle]], [[postselect]])
    %res, %new_q = mbqc.measure_in_basis [XY, %angle] %q1 postselect 1 : i1, !quantum.bit
    func.return
}
