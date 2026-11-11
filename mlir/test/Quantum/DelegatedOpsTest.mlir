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

// RUN: quantum-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @roundtrip
func.func @roundtrip(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    // CHECK: quantum.delegate hw_like
    quantum.delegate hw_like
    // CHECK: quantum.measure {{.*}} : !quantum.res, !quantum.bit
    %m0, %a = quantum.measure %q0 : !quantum.res, !quantum.bit
    // CHECK: pbc.ppm ["X", "Z"](-) {{.*}} postselect 1 : !quantum.res, !quantum.bit, !quantum.bit
    %m1, %b:2 = pbc.ppm ["X", "Z"](-) %a, %q1 postselect 1 : !quantum.res, !quantum.bit, !quantum.bit
    // CHECK: [[t:%.+]] = quantum.result.constant true
    %t = quantum.result.constant true
    // CHECK: [[x:%.+]] = quantum.result.xor
    %x = quantum.result.xor %m0, %m1
    // CHECK: [[n:%.+]] = quantum.result.not [[x]]
    %n = quantum.result.not %x
    // CHECK: [[s:%.+]] = quantum.result.select [[n]], [[t]],
    %s = quantum.result.select %n, %t, %m0
    // CHECK: quantum.predicated [[s]]({{.*}}) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.res) {
    // CHECK: } else {
    %c:3 = quantum.predicated %s (%b#0, %b#1) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.res) {
    ^bb0(%arg0: !quantum.bit, %arg1: !quantum.bit):
        %m2, %o = pbc.ppm ["Y"] %arg0 : !quantum.res, !quantum.bit
        quantum.yield %o, %arg1, %m2 : !quantum.bit, !quantum.bit, !quantum.res
    } else {
    ^bb0(%arg0: !quantum.bit, %arg1: !quantum.bit):
        %f = quantum.result.constant false
        quantum.yield %arg0, %arg1, %f : !quantum.bit, !quantum.bit, !quantum.res
    }
    // CHECK: quantum.result.release
    quantum.result.release %c#2
    return %c#0, %c#1 : !quantum.bit, !quantum.bit
}

// -----

func.func @captured_qubit(%r: !quantum.res, %q0: !quantum.bit, %q1: !quantum.bit) -> !quantum.bit {
    // expected-error@+1 {{regions may not use qubits or registers from above}}
    %c = quantum.predicated %r (%q0) : (!quantum.bit) -> (!quantum.bit) {
    ^bb0(%arg0: !quantum.bit):
        %o:2 = pbc.ppr ["X", "X"](2) %arg0, %q1 : !quantum.bit, !quantum.bit
        quantum.yield %o#0 : !quantum.bit
    } else {
    ^bb0(%arg0: !quantum.bit):
        quantum.yield %arg0 : !quantum.bit
    }
    return %c : !quantum.bit
}

// -----

func.func @qubits_first(%r: !quantum.res, %q0: !quantum.bit) -> !quantum.res {
    // expected-error@+1 {{must return its qubits first, in operand order}}
    %c:2 = quantum.predicated %r (%q0) : (!quantum.bit) -> (!quantum.res, !quantum.bit) {
    ^bb0(%arg0: !quantum.bit):
        quantum.yield %r, %arg0 : !quantum.res, !quantum.bit
    } else {
    ^bb0(%arg0: !quantum.bit):
        quantum.yield %r, %arg0 : !quantum.res, !quantum.bit
    }
    return %c#0 : !quantum.res
}

// -----

func.func @region_arguments(%r: !quantum.res, %q0: !quantum.bit) -> !quantum.bit {
    // expected-error@+1 {{region arguments must match the qubit operands}}
    %c = quantum.predicated %r (%q0) : (!quantum.bit) -> (!quantum.bit) {
    ^bb0:
        quantum.yield %q0 : !quantum.bit
    } else {
    ^bb0(%arg0: !quantum.bit):
        quantum.yield %arg0 : !quantum.bit
    }
    return %c : !quantum.bit
}

// -----

func.func @yield_types(%r: !quantum.res, %q0: !quantum.bit) -> !quantum.bit {
    // expected-error@+1 {{regions must yield values of the op's result types}}
    %c = quantum.predicated %r (%q0) : (!quantum.bit) -> (!quantum.bit) {
    ^bb0(%arg0: !quantum.bit):
        quantum.yield %arg0, %r : !quantum.bit, !quantum.res
    } else {
    ^bb0(%arg0: !quantum.bit):
        quantum.yield %arg0 : !quantum.bit
    }
    return %c : !quantum.bit
}
