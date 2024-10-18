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

// RUN: qcc %s --dump-pass-pipeline --verify-diagnostics 2>&1 | FileCheck %s
// RUN: qcc --tool=opt %s --catalyst-pipeline="pipe1(split-multiple-tapes;apply-transform-sequence),pipe2(inline-nested-module)" --dump-pass-pipeline --verify-diagnostics 2>&1 | FileCheck %s --check-prefix=CHECK-CUSTOM
// RUN: qcc --tool=opt %s -cse --dump-pass-pipeline --verify-diagnostics 2>&1 | FileCheck %s --check-prefix=CHECK-ONE-PASS
// RUN: not qcc --tool=opt %s -cse --catalyst-pipeline="pipe1(split-multiple-tapes;apply-transform-sequence),pipe2(inline-nested-module)" --dump-pass-pipeline --verify-diagnostics 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL

func.func @foo() {
    return
}

// CHECK: Pass Manager with
// CHECK: builtin.module

// CHECK-CUSTOM: Pass Manager with 2 passes
// CHECK-CUSTOM: Pass Manager with 1 passes

// CHECK-ONE-PASS: Pass Manager with 1 passes

// CHECK-FAIL: --catalyst-pipline option can't be used with individual pass options.
// CHECK-FAIL: Compilation failed