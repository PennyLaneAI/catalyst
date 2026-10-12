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

// CHECK-DAG: func.func private @__catalyst__rt__random_double() -> f64
// CHECK-DAG: func.func private @rs_fallback_projective_get_size(f64, f64, i1, f64) -> index
// CHECK-DAG: func.func private @rs_fallback_projective_get_gates(memref<?xindex>, f64, f64, i1, f64)
// CHECK-DAG: func.func private @rs_fallback_projective_get_phase(f64, f64, i1, f64) -> f64
// CHECK-DAG: func.func private @rs_fallback_recovery_angle(f64, f64, f64) -> f64
// CHECK-DAG: func.func private @rs_fallback_recovery_epsilon(f64, f64, f64) -> f64
// CHECK-DAG: func.func private @rs_mixed_decomposition_get_size(f64, f64, i1, f64) -> index

// COM: The projective rotation acts on an ancilla between two CY gates controlled by the target.
// COM: The ancilla is released after its measurement. If the measurement returns 1, the target
// COM: receives the mixed decomposition of an angle and epsilon given by the runtime, sampled by a
// COM: second uniform sample.

// CLIFFORD-LABEL: func.func private @__catalyst_decompose_RZ_mixed_fallback(
// PPR-LABEL:      func.func private @__catalyst_decompose_RZ_mixed_fallback_ppr_basis(
// CHECK-SAME:     [[ARG_QBIT:%.+]]: !quantum.bit, [[ARG_ANGLE:%.+]]: f64)
// PPR-DAG:        [[CY_PHASES:%.+]] = arith.constant 1.5707963267948966 : f64
// CHECK-DAG:      [[EPS:%.+]] = arith.constant 1.000000e-02 : f64
// CHECK:          [[SAMPLE:%.+]] = call @__catalyst__rt__random_double()
// CHECK:          [[ANCILLA:%.+]] = quantum.alloc_qb : !quantum.bit
// CLIFFORD:       [[CY:%.+]]:2 = quantum.custom "CY"() [[ARG_QBIT]], [[ANCILLA]]
// PPR:            [[CY:%.+]]:2 = pbc.ppr ["Z", "Y"](4) [[ARG_QBIT]], [[ANCILLA]]
// PPR:            [[CY_TARGET:%.+]] = pbc.ppr ["Z"](-4) [[CY]]#0
// PPR:            [[CY_ANCILLA:%.+]] = pbc.ppr ["Y"](-4) [[CY]]#1
// CHECK:          [[NUM_GATES:%.+]] = call @rs_fallback_projective_get_size([[ARG_ANGLE]], [[EPS]], {{%.+}}, [[SAMPLE]])
// CHECK:          [[MEM:%.+]] = memref.alloc([[NUM_GATES]]) : memref<?xindex>
// CHECK:          call @rs_fallback_projective_get_gates([[MEM]], [[ARG_ANGLE]], [[EPS]], {{%.+}}, [[SAMPLE]])
// CHECK:          [[PHASE:%.+]] = call @rs_fallback_projective_get_phase([[ARG_ANGLE]], [[EPS]], {{%.+}}, [[SAMPLE]])
// CLIFFORD:       [[ROTATED:%.+]] = scf.for {{.*}} iter_args({{%.+}} = [[CY]]#1)
// PPR:            [[ROTATED:%.+]] = scf.for {{.*}} iter_args({{%.+}} = [[CY_ANCILLA]])
// COM: expected T and other entries (0.5505 - 0.0008) * log2(1/0.01) + 3.7633 + 1.9597 (Clifford+T), (0.5505 + 0.2777) * log2(1/0.01) + 3.7633 + 5.8389 (PPR)
// CLIFFORD:       } {catalyst.estimated_iterations = 9.37{{[0-9]+}} : f64}
// PPR:            } {catalyst.estimated_iterations = 15.10{{[0-9]+}} : f64}
// CLIFFORD:       [[CY2:%.+]]:2 = quantum.custom "CY"() [[CY]]#0, [[ROTATED]]
// PPR:            [[CY2:%.+]]:2 = pbc.ppr ["Z", "Y"](4) [[CY_TARGET]], [[ROTATED]]
// PPR:            [[TARGET:%.+]] = pbc.ppr ["Z"](-4) [[CY2]]#0
// PPR:            [[CY2_ANCILLA:%.+]] = pbc.ppr ["Y"](-4) [[CY2]]#1
// CLIFFORD:       [[MRES:%.+]], [[MEASURED:%.+]] = quantum.measure [[CY2]]#1
// PPR:            [[MRES:%.+]], [[MEASURED:%.+]] = pbc.ppm ["Z"] [[CY2_ANCILLA]]
// CHECK:          quantum.dealloc_qb [[MEASURED]]
// CHECK:          [[IF:%.+]]:2 = scf.if [[MRES]] -> (!quantum.bit, f64) {
// CHECK:            [[ANGLE2:%.+]] = func.call @rs_fallback_recovery_angle([[ARG_ANGLE]], [[EPS]], [[SAMPLE]])
// CHECK:            [[EPS2:%.+]] = func.call @rs_fallback_recovery_epsilon([[ARG_ANGLE]], [[EPS]], [[SAMPLE]])
// CHECK:            [[SAMPLE2:%.+]] = func.call @__catalyst__rt__random_double()
// CHECK:            func.call @rs_mixed_decomposition_get_size([[ANGLE2]], [[EPS2]], {{%.+}}, [[SAMPLE2]])
// CHECK:            [[RECOVERY_PHASE:%.+]] = func.call @rs_mixed_decomposition_get_phase([[ANGLE2]], [[EPS2]], {{%.+}}, [[SAMPLE2]])
// CLIFFORD:         [[RECOVERED:%.+]] = scf.for {{.*}} iter_args({{%.+}} = [[CY2]]#0)
// PPR:              [[RECOVERED:%.+]] = scf.for {{.*}} iter_args({{%.+}} = [[TARGET]])
// COM: expected T and other entries (1.5753 + 0.0115) * log2(1/0.01) - 7.0269 + 2.3601 (Clifford+T), (1.5753 + 0.8159) * log2(1/0.01) - 7.0269 + 0.4499 (PPR)
// CLIFFORD:         } {catalyst.estimated_iterations = 5.87{{[0-9]+}} : f64}
// PPR:              } {catalyst.estimated_iterations = 9.30{{[0-9]+}} : f64}
// CHECK:            scf.yield [[RECOVERED]], [[RECOVERY_PHASE]]
// CHECK:          } else {
// CLIFFORD:         scf.yield [[CY2]]#0, {{%.+}}
// PPR:              scf.yield [[TARGET]], {{%.+}}
// CHECK:          } {catalyst.estimated_probability = 2.440000e-02 : f64}
// CHECK:          [[TOTAL_PHASE:%.+]] = arith.addf [[PHASE]], [[IF]]#1
// CLIFFORD:       return [[IF]]#0, [[TOTAL_PHASE]]
// PPR:            [[CY_TOTAL_PHASE:%.+]] = arith.addf [[TOTAL_PHASE]], [[CY_PHASES]]
// PPR:            return [[IF]]#0, [[CY_TOTAL_PHASE]]

// CHECK-LABEL: @test_rz_mixed_fallback_decomposition
// CHECK-SAME: ([[Q_IN:%.+]]: !quantum.bit, [[THETA:%.+]]: f64)
func.func @test_rz_mixed_fallback_decomposition(%arg0: !quantum.bit, %theta: f64) -> !quantum.bit {
    // CLIFFORD: [[RES:%.+]]:2 = call @__catalyst_decompose_RZ_mixed_fallback([[Q_IN]], [[THETA]])
    // PPR:      [[RES:%.+]]:2 = call @__catalyst_decompose_RZ_mixed_fallback_ppr_basis([[Q_IN]], [[THETA]])
    // CHECK: quantum.gphase([[RES]]#1)
    // CHECK: return [[RES]]#0 : !quantum.bit
    %q_out = quantum.custom "RZ"(%theta) %arg0 : !quantum.bit
    return %q_out : !quantum.bit
}
