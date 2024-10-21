// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: not qcc %s --tool=llc --verify-diagnostics 2>&1 | FileCheck %s -check-prefix=CHECK-LLC
// RUN: not qcc %s --tool=translate --verify-diagnostics 2>&1 | FileCheck %s -check-prefix=CHECK-TRANSLATE


func.func @foo() {
    return
}

// CHECK-LLC: Compilation failed:
// CHECK-LLC: Expected LLVM IR input but received MLIR

// CHECK-TRANSLATE: Failed to translate LLVM module
