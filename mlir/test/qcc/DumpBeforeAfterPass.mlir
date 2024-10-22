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

// RUN: catalyst-cli --tool=opt %s --catalyst-pipeline="pipe1(split-multiple-tapes;apply-transform-sequence),pipe2(inline-nested-module)" --mlir-print-ir-before-all --verify-diagnostics 2>&1 | FileCheck %s --check-prefix=CHECK-BEFORE
// RUN: catalyst-cli --tool=opt %s --catalyst-pipeline="pipe1(split-multiple-tapes;apply-transform-sequence),pipe2(inline-nested-module)" --mlir-print-ir-after-all --verify-diagnostics 2>&1 | FileCheck %s --check-prefix=CHECK-AFTER
// RUN: catalyst-cli --tool=opt %s --catalyst-pipeline="pipe1(split-multiple-tapes;apply-transform-sequence),pipe2(inline-nested-module)" --mlir-print-ir-before=inline-nested-module --verify-diagnostics 2>&1 | FileCheck %s --check-prefix=CHECK-BEFORE-ONE
// RUN: catalyst-cli --tool=opt %s --catalyst-pipeline="pipe1(split-multiple-tapes;apply-transform-sequence),pipe2(inline-nested-module)" --mlir-print-ir-after=inline-nested-module --verify-diagnostics 2>&1 | FileCheck %s --check-prefix=CHECK-AFTER-ONE
// RUN: catalyst-cli --tool=opt %s --catalyst-pipeline="pipe1(split-multiple-tapes;apply-transform-sequence),pipe2(inline-nested-module)" --mlir-print-op-generic --verify-diagnostics 2>&1 | FileCheck %s --check-prefix=CHECK-GENERIC

func.func @foo() {
    return
}

// CHECK-BEFORE: IR Dump Before SplitMultipleTapesPass
// CHECK-BEFORE: IR Dump Before ApplyTransformSequencePass
// CHECK-BEFORE: IR Dump Before InlineNestedModulePass

// CHECK-AFTER: IR Dump After SplitMultipleTapesPass
// CHECK-AFTER: IR Dump After ApplyTransformSequencePass
// CHECK-AFTER: IR Dump After InlineNestedModulePass

// CHECK-BEFORE-ONE: IR Dump Before InlineNestedModulePass

// CHECK-AFTER-ONE: IR Dump After InlineNestedModulePass

// CHECK-GENERIC: "builtin.module"() ({
// CHECK-GENERIC-NEXT: "func.func"() <{function_type = () -> (), sym_name = "foo"}> ({