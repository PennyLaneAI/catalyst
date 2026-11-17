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

// Test that resource-estimate counts the ops outside the decomposition graph inside loops,
// branches and function calls, as resource-analysis does.

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(resource-estimate{output-json gate-set=RX=1.0 bytecode-rules="%PRECOMPILED_BYTECODE_PATH"})' %s -o /dev/null | FileCheck %s

// The worst case of the first branch counts the PPR, and the second branch is weighted by its
// probability. The loop body and the helper are counted once, and called 3 and 2 times.
// CHECK-LABEL: "circuit": {
// CHECK:       "static"
// CHECK-NEXT:    "for_loop_1": 3
// CHECK-NEXT:    "helper": 2
// CHECK:       "quantum_operations"
// CHECK-NEXT:    "1"
// CHECK-NEXT:      "PPR-pi/2": 0.5
// CHECK-NEXT:      "PPR-pi/8": 1
// CHECK-LABEL: "for_loop_1": {
// CHECK:       "quantum_operations"
// CHECK-NEXT:    "1"
// CHECK-NEXT:      "PPR-pi/4": 1
// CHECK-LABEL: "helper": {
// CHECK:       "quantum_operations"
// CHECK-NEXT:    "2"
// CHECK-NEXT:      "PPR-pi/4": 1

func.func private @helper(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %r:2 = pbc.ppr ["Z", "X"](4) %q0, %q1 : !quantum.bit, !quantum.bit
    return %r#0, %r#1 : !quantum.bit, !quantum.bit
}

func.func public @circuit(%cond: i1) attributes {quantum.node} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    %3 = scf.for %i = %c0 to %c3 step %c1 iter_args(%q = %1) -> (!quantum.bit) {
        %z = pbc.ppr ["Z"](4) %q : !quantum.bit
        scf.yield %z : !quantum.bit
    }

    %4 = scf.if %cond -> !quantum.bit {
        %y = pbc.ppr ["Y"](8) %3 : !quantum.bit
        scf.yield %y : !quantum.bit
    } else {
        scf.yield %3 : !quantum.bit
    }

    %40 = scf.if %cond -> !quantum.bit {
        %x = pbc.ppr ["X"](2) %4 : !quantum.bit
        scf.yield %x : !quantum.bit
    } else {
        scf.yield %4 : !quantum.bit
    } {catalyst.estimated_probability = 0.5 : f64}

    %5:2 = func.call @helper(%40, %2) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    %6:2 = func.call @helper(%5#0, %5#1) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)

    %7 = quantum.insert %0[ 0], %6#0 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %7[ 1], %6#1 : !quantum.reg, !quantum.bit
    quantum.dealloc %8 : !quantum.reg
    return
}
