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

// RUN: quantum-opt \
// RUN:   --pass-pipeline="builtin.module(convert-quantum-to-llvm{use-array-backed-registers=true})" \
// RUN:   --split-input-file %s | FileCheck %s

// CHECK: llvm.func @__catalyst__rt__array_update_element_1d(!llvm.ptr, i64, !llvm.ptr)

// CHECK-LABEL: @insert
func.func @insert(%r : !quantum.reg, %q : !quantum.bit) -> !quantum.reg {

    // CHECK: [[c5:%.+]] = llvm.mlir.constant(5 : i64) : i64
    // CHECK: llvm.call @__catalyst__rt__array_update_element_1d(%arg0, [[c5]], %arg1) : (!llvm.ptr, i64, !llvm.ptr) -> ()
    %new_r = quantum.insert %r[5], %q : !quantum.reg, !quantum.bit

    // CHECK: llvm.return %arg0 : !llvm.ptr
    return %new_r : !quantum.reg
}
