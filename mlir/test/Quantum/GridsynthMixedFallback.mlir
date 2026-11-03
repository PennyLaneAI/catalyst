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

// RUN: quantum-opt --pass-pipeline="builtin.module(gridsynth{epsilon=0.01 method=mixed-fallback})" %s | FileCheck %s --check-prefixes=CHECK,CLIFFORD
// RUN: quantum-opt --pass-pipeline="builtin.module(gridsynth{epsilon=0.01 ppr-basis=True method=mixed-fallback})" %s | FileCheck %s --check-prefixes=CHECK,PPR

// CHECK-DAG: func.func private @rs_fallback_projective_get_size(f64, f64, i1, f64) -> index
// CHECK-DAG: func.func private @rs_fallback_projective_get_gates(memref<?xindex>, f64, f64, i1, f64)
// CHECK-DAG: func.func private @rs_fallback_projective_get_phase(f64, f64, i1, f64) -> f64
// CHECK-DAG: func.func private @rs_fallback_recovery_angle(f64, f64, f64) -> f64
// CHECK-DAG: func.func private @rs_fallback_recovery_epsilon(f64, f64, f64) -> f64

// COM: The projective rotation acts on an ancilla between two CY gates controlled by the target,
// COM: and the fallback is applied if measuring the ancilla returns 1.

// CHECK-LABEL: func.func private @__catalyst_decompose_RZ_mixed_fallback
// CHECK-SAME:  ([[QBIT:%.+]]: !quantum.bit, [[ANGLE:%.+]]: f64)
// PPR-DAG:     [[CY_PHASES:%.+]] = arith.constant 1.5707963267948966 : f64
// CHECK:       [[SAMPLE:%.+]] = call @__catalyst__rt__random_double()
// CHECK:       [[ANCILLA:%.+]] = quantum.alloc_qb
// CLIFFORD:    quantum.custom "CY"() [[QBIT]], [[ANCILLA]]
// PPR:         pbc.ppr ["Z", "Y"](4) [[QBIT]], [[ANCILLA]]
// CHECK:       call @rs_fallback_projective_get_gates({{%.+}}, [[ANGLE]], {{%.+}}, {{%.+}}, [[SAMPLE]])
// CHECK:       [[PHASE:%.+]] = call @rs_fallback_projective_get_phase([[ANGLE]], {{%.+}}, {{%.+}}, [[SAMPLE]])
// CHECK:       [[ROTATED:%.+]] = scf.for
// CLIFFORD:    [[CY:%.+]]:2 = quantum.custom "CY"() {{%.+}}, [[ROTATED]]
// CLIFFORD:    [[MRES:%.+]], [[MEASURED:%.+]] = quantum.measure [[CY]]#1
// PPR:         pbc.ppr ["Z", "Y"](4) {{%.+}}, [[ROTATED]]
// PPR:         [[MRES:%.+]], [[MEASURED:%.+]] = pbc.ppm ["Z"]
// CHECK:       quantum.dealloc_qb [[MEASURED]]
// CHECK:       [[IF:%.+]]:2 = scf.if [[MRES]] -> (!quantum.bit, f64) {
// CHECK:         [[ANGLE2:%.+]] = func.call @rs_fallback_recovery_angle([[ANGLE]], {{%.+}}, [[SAMPLE]])
// CHECK:         [[EPS2:%.+]] = func.call @rs_fallback_recovery_epsilon([[ANGLE]], {{%.+}}, [[SAMPLE]])
// CHECK:         func.call @rs_mixed_decomposition_get_gates({{%.+}}, [[ANGLE2]], [[EPS2]], {{%.+}}, {{%.+}})
// CHECK:         scf.for
// CHECK:       } else {
// CHECK:       } {catalyst.estimated_probability = 2.440000e-02 : f64}
// CHECK:       [[TOTAL_PHASE:%.+]] = arith.addf [[PHASE]], [[IF]]#1
// CLIFFORD:    quantum.gphase([[TOTAL_PHASE]])
// PPR:         [[CY_TOTAL_PHASE:%.+]] = arith.addf [[TOTAL_PHASE]], [[CY_PHASES]]
// PPR:         quantum.gphase([[CY_TOTAL_PHASE]])
// CHECK:       return [[IF]]#0 : !quantum.bit

// CHECK-LABEL: @test_rz_mixed_fallback_decomposition
func.func @test_rz_mixed_fallback_decomposition(%arg0: !quantum.bit, %theta: f64) -> !quantum.bit {
    // CHECK: call @__catalyst_decompose_RZ_mixed_fallback
    %q_out = quantum.custom "RZ"(%theta) %arg0 : !quantum.bit
    return %q_out : !quantum.bit
}
