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

// RUN: quantum-opt --unroll-conditional-ppr-ppm --split-input-file -verify-diagnostics %s | FileCheck %s

func.func @test_select_ppm(%q : !quantum.bit, %cond : i1) -> (i1, !quantum.bit) {
    // CHECK-DAG-NOT: qec.select.ppm({{.*}}, ["X"], ["Z"])
    // CHECK: scf.if {{.*}} -> (i1, !quantum.bit)
    // CHECK: [[m1:%.+]], [[q1:%.+]] = qec.ppm ["X"]
    // CHECK: scf.yield [[m1]], [[q1]]
    // CHECK: else
    // CHECK: [[m2:%.+]], [[q2:%.+]] = qec.ppm ["Z"]
    // CHECK: scf.yield [[m2]], [[q2]]
    %m, %out = qec.select.ppm (%cond, ["X"], ["Z"]) %q : i1, !quantum.bit
    func.return %m, %out : i1, !quantum.bit
}

// -----

func.func @test_ppr_cond(%q : !quantum.bit, %cond : i1) -> !quantum.bit {
    // CHECK-DAG-NOT: qec.ppr["X"](4) {{.*}} cond({{.*}})
    // CHECK: scf.if {{.*}} -> (!quantum.bit)
    // CHECK: [[q1:%.+]] = qec.ppr ["X"](4)
    // CHECK: scf.yield [[q1]] : !quantum.bit
    // CHECK: else
    // CHECK-NOT: qec.ppr
    // CHECK: scf.yield {{.*}} : !quantum.bit
    %q1 = qec.ppr ["X"](4) %q cond(%cond) : !quantum.bit
    func.return %q1 : !quantum.bit
}

// -----

func.func @test_no_cond(%q : !quantum.bit) -> (i1, !quantum.bit) {
    // CHECK: [[q1:%.+]] = qec.ppr ["X"](4)
    // CHECK: [[q2:%.+]] = qec.ppm ["Y"] [[q1]]
    %q1 = qec.ppr ["X"](4) %q : !quantum.bit
    %m, %q2 = qec.ppm ["Y"] %q1 : i1, !quantum.bit
    func.return %m, %q2 : i1, !quantum.bit
}   
