// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --tag-classical-control-measurements --split-input-file %s | FileCheck %s --check-prefix=TAG
// RUN: quantum-opt --instrument-classical-control --split-input-file %s | FileCheck %s

// All measurements are tagged. Without tags, nothing is instrumented.

// TAG-LABEL: func.func @tag_measurements
// CHECK-LABEL: func.func @tag_measurements
// CHECK-NOT: __catalyst__rt__
// CHECK: // -----
func.func @tag_measurements(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit) -> (!quantum.bit, i1, i1) {
    // TAG: quantum.measure %arg0 {quantum.track_classical_control}
    // TAG: pbc.ppm ["Z"] %arg1 {quantum.track_classical_control}
    // TAG: pbc.select.ppm {{.+}} {quantum.track_classical_control}
    %m0, %q0_1 = quantum.measure %q0 : i1, !quantum.bit
    %m1, %q1_1 = pbc.ppm ["Z"] %q1 : i1, !quantum.bit
    %m2, %q2_1 = pbc.select.ppm (%m0 ? ["X"] : ["Z"]) %q2 : i1, !quantum.bit
    %q0_2 = scf.if %m0 -> !quantum.bit {
        %z = quantum.custom "PauliZ"() %q0_1 : !quantum.bit
        scf.yield %z : !quantum.bit
    } else {
        scf.yield %q0_1 : !quantum.bit
    }
    return %q0_2, %m1, %m2 : !quantum.bit, i1, i1
}

// -----

// Only tagged measurements are tracked, and their tags are removed.

// CHECK-LABEL: func.func @partially_tagged
func.func @partially_tagged(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit) -> !quantum.bit {
    // CHECK: quantum.measure %arg0 : i1, !quantum.bit
    // CHECK-NEXT: quantum.measure %arg1 : i1, !quantum.bit
    // CHECK-NEXT: [[TOKEN:%.+]] = call @__catalyst__rt__measurement_token()
    %m0, %q0_1 = quantum.measure %q0 : i1, !quantum.bit
    %m1, %q1_1 = quantum.measure %q1 {quantum.track_classical_control} : i1, !quantum.bit
    %c = arith.andi %m0, %m1 : i1

    // CHECK: scf.if
    // CHECK-NEXT: call @__catalyst__rt__begin_classical_control([[TOKEN]])
    %q2_1 = scf.if %c -> !quantum.bit {
        %z = quantum.custom "PauliZ"() %q2 : !quantum.bit
        scf.yield %z : !quantum.bit
    } else {
        scf.yield %q2 : !quantum.bit
    }
    return %q2_1 : !quantum.bit
}

// -----

// The runtime functions are declared in the nearest symbol table of their calls, e.g. a nested
// module compiled separately.

// CHECK: // -----
// CHECK-NOT: __catalyst__rt__
// CHECK: module @nested
// CHECK-DAG: func.func private @__catalyst__rt__measurement_token() -> i64
// CHECK-DAG: func.func private @__catalyst__rt__begin_classical_control(i64)
// CHECK-DAG: func.func private @__catalyst__rt__end_classical_control()
// CHECK: func.func @cond_in_nested_module
module {
    module @nested {
        func.func @cond_in_nested_module(%q0: !quantum.bit, %q1: !quantum.bit) -> !quantum.bit {
            // CHECK: quantum.measure
            // CHECK-NEXT: [[TOKEN:%.+]] = call @__catalyst__rt__measurement_token()
            %m, %q0_1 = quantum.measure %q0 {quantum.track_classical_control} : i1, !quantum.bit

            // CHECK: scf.if
            // CHECK-NEXT: call @__catalyst__rt__begin_classical_control([[TOKEN]])
            %q1_1 = scf.if %m -> !quantum.bit {
                %z = quantum.custom "PauliZ"() %q1 : !quantum.bit
                scf.yield %z : !quantum.bit
            } else {
                scf.yield %q1 : !quantum.bit
            }
            return %q1_1 : !quantum.bit
        }
    }
}
