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

// RUN: quantum-opt --pass-pipeline="builtin.module(gridsynth{epsilon=0.01 method=mixed})" --split-input-file %s | FileCheck %s --check-prefixes=CHECK,CLIFFORD
// RUN: quantum-opt --pass-pipeline="builtin.module(gridsynth{epsilon=0.01 ppr-basis=True method=mixed})" --split-input-file %s | FileCheck %s --check-prefixes=CHECK,PPR
// RUN: not quantum-opt --pass-pipeline="builtin.module(gridsynth{method=unknown})" --split-input-file %s 2>&1 | FileCheck %s --check-prefix=ERROR

// ERROR: gridsynth method must be 'deterministic' or 'mixed', got 'unknown'

// CHECK-DAG: func.func private @__catalyst__rt__random_double() -> f64
// CHECK-DAG: func.func private @rs_mixed_decomposition_get_size(f64, f64, i1, f64) -> index
// CHECK-DAG: func.func private @rs_mixed_decomposition_get_gates(memref<?xindex>, f64, f64, i1, f64)
// CHECK-DAG: func.func private @rs_mixed_decomposition_get_phase(f64, f64, i1, f64) -> f64

// COMMENT: One uniform sample is shared by the three runtime calls, which receive the diamond-norm
// COMMENT: error 2 * epsilon.

// CLIFFORD-LABEL: func.func private @__catalyst_decompose_RZ_mixed(
// PPR-LABEL:      func.func private @__catalyst_decompose_RZ_mixed_ppr_basis(
// CHECK-SAME:     [[ARG_QBIT:%.+]]: !quantum.bit, [[ARG_ANGLE:%.+]]: f64)
// CHECK:          [[EPS:%.+]] = arith.constant 2.000000e-02 : f64
// CHECK:          [[SAMPLE:%.+]] = call @__catalyst__rt__random_double()
// CHECK:          [[NUM_GATES:%.+]] = call @rs_mixed_decomposition_get_size([[ARG_ANGLE]], [[EPS]], {{%.+}}, [[SAMPLE]])
// CHECK:          [[MEM:%.+]] = memref.alloc([[NUM_GATES]]) : memref<?xindex>
// CHECK:          call @rs_mixed_decomposition_get_gates([[MEM]], [[ARG_ANGLE]], [[EPS]], {{%.+}}, [[SAMPLE]])
// CHECK:          [[PHASE:%.+]] = call @rs_mixed_decomposition_get_phase([[ARG_ANGLE]], [[EPS]], {{%.+}}, [[SAMPLE]])
// CHECK:          scf.for {{.*}} iter_args({{%.+}} = [[ARG_QBIT]])
// CHECK:            scf.index_switch {{%.+}} {catalyst.estimated_probabilities = [{{.*}}]}
// COM: expected T and other entries (1.5356 + 0.0010) * log2(1/0.02) + 3.2211 + 3.4275 (Clifford+T), (1.5356 + 0.7723) * log2(1/0.02) + 3.2211 + 6.9217 (PPR)
// CLIFFORD:       } {catalyst.estimated_iterations = 15.3{{[0-9]+}} : f64}
// PPR:            } {catalyst.estimated_iterations = 23.1{{[0-9]+}} : f64}
// CHECK:          return {{%.+}}, [[PHASE]]

// CHECK-LABEL: @test_rz_mixed_decomposition
// CHECK-SAME: ([[Q_IN:%.+]]: !quantum.bit, [[THETA:%.+]]: f64)
func.func @test_rz_mixed_decomposition(%arg0: !quantum.bit, %theta: f64) -> !quantum.bit {
    // CLIFFORD: [[RES:%.+]]:2 = call @__catalyst_decompose_RZ_mixed([[Q_IN]], [[THETA]])
    // PPR:      [[RES:%.+]]:2 = call @__catalyst_decompose_RZ_mixed_ppr_basis([[Q_IN]], [[THETA]])
    // CHECK: quantum.gphase([[RES]]#1)
    // CHECK: return [[RES]]#0 : !quantum.bit
    %q_out = quantum.custom "RZ"(%theta) %arg0 : !quantum.bit
    return %q_out : !quantum.bit
}

// -----

// CHECK-LABEL: @test_ppr_arbitrary_z_mixed_decomposition
// CHECK-SAME: ([[Q_IN:%.+]]: !quantum.bit, [[THETA:%.+]]: f64)
func.func @test_ppr_arbitrary_z_mixed_decomposition(%arg0: !quantum.bit, %theta: f64) -> !quantum.bit {
    // CHECK: [[PHI:%.+]] = arith.mulf [[THETA]]
    // CLIFFORD: [[RES:%.+]]:2 = call @__catalyst_decompose_RZ_mixed([[Q_IN]], [[PHI]])
    // PPR:      [[RES:%.+]]:2 = call @__catalyst_decompose_RZ_mixed_ppr_basis([[Q_IN]], [[PHI]])
    // CHECK: quantum.gphase([[RES]]#1)
    %q_out = pbc.ppr.arbitrary ["Z"](%theta) %arg0 : !quantum.bit
    return %q_out : !quantum.bit
}
