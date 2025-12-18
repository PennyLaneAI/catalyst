// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --pass-pipeline="builtin.module(gridsynth{epsilon=0.01 ppr-basis=False})" --split-input-file %s | FileCheck %s --check-prefixes=CHECK,CLIFFORD
// RUN: quantum-opt --pass-pipeline="builtin.module(gridsynth{epsilon=0.01 ppr-basis=True})" --split-input-file %s | FileCheck %s --check-prefixes=CHECK,PPR



// COMMENT: Check that the external helpers are declared in the module

// CHECK-DAG: func.func private @rs_decomposition_get_size(f64, f64, i1) -> index
// CHECK-DAG: func.func private @rs_decomposition_get_gates(memref<?xindex>, f64, f64, i1)
// CHECK-DAG: func.func private @rs_decomposition_get_phase(f64, f64, i1) -> f64

// COMMENT: Test the defined decomposition function

// CLIFFORD-LABEL: func.func private @__catalyst_decompose_RZ(
// CLIFFORD-SAME: [[ARG_QBIT:%.+]]: !quantum.bit, [[ARG_ANGLE:%.+]]: f64)
// CLIFFORD-DAG:  [[C1:%.+]] = arith.constant 1 : index
// CLIFFORD-DAG:  [[C0:%.+]] = arith.constant 0 : index
// CLIFFORD:      [[NUM_GATES:%.+]] = call @rs_decomposition_get_size
// CLIFFORD:      [[MEM:%.+]] = memref.alloc([[NUM_GATES]]) : memref<?xindex>
// CLIFFORD:      call @rs_decomposition_get_gates([[MEM]]
// CLIFFORD:      [[PHASE:%.+]] = call @rs_decomposition_get_phase
// CLIFFORD:      [[LOOP_RES:%.+]] = scf.for [[IV:%.+]] = [[C0]] to [[NUM_GATES]] step [[C1]] iter_args([[L_QBIT:%.+]] = [[ARG_QBIT]])
// CLIFFORD:        [[CASE_ID:%.+]] = memref.load [[MEM]]{{\[}}[[IV]]{{\]}}
// CLIFFORD:        [[SWITCH_RES:%.+]] = scf.index_switch [[CASE_ID]]
// CLIFFORD:        case 0 {
// CLIFFORD:          [[RES:%.+]] = quantum.custom "T"() [[L_QBIT]]
// CLIFFORD:          scf.yield [[RES]]
// CLIFFORD:        }
// CLIFFORD:        case 1 {
// CLIFFORD:          [[G1:%.+]] = quantum.custom "Hadamard"() [[L_QBIT]]
// CLIFFORD:          [[G2:%.+]] = quantum.custom "T"() [[G1]]
// CLIFFORD:          scf.yield [[G2]]
// CLIFFORD:        }
// CLIFFORD:        case 2 {
// CLIFFORD:          [[G1:%.+]] = quantum.custom "S"() [[L_QBIT]]
// CLIFFORD:          [[G2:%.+]] = quantum.custom "Hadamard"() [[G1]]
// CLIFFORD:          [[G3:%.+]] = quantum.custom "T"() [[G2]]
// CLIFFORD:          scf.yield [[G3]]
// CLIFFORD:        }
// CLIFFORD:        case 3 {
// CLIFFORD:          [[G1:%.+]] = quantum.custom "Identity"() [[L_QBIT]]
// CLIFFORD:          scf.yield [[G1]]
// CLIFFORD:        }
// CLIFFORD:        case 4 {
// CLIFFORD:          [[G1:%.+]] = quantum.custom "PauliX"() [[L_QBIT]]
// CLIFFORD:          scf.yield [[G1]]
// CLIFFORD:        }
// CLIFFORD:        case 5 {
// CLIFFORD:          [[G1:%.+]] = quantum.custom "PauliY"() [[L_QBIT]]
// CLIFFORD:          scf.yield [[G1]]
// CLIFFORD:        }
// CLIFFORD:        case 6 {
// CLIFFORD:          [[G1:%.+]] = quantum.custom "PauliZ"() [[L_QBIT]]
// CLIFFORD:          scf.yield [[G1]]
// CLIFFORD:        }
// CLIFFORD:        case 7 {
// CLIFFORD:          [[G1:%.+]] = quantum.custom "Hadamard"() [[L_QBIT]]
// CLIFFORD:          scf.yield [[G1]]
// CLIFFORD:        }
// CLIFFORD:        case 8 {
// CLIFFORD:          [[G1:%.+]] = quantum.custom "S"() [[L_QBIT]]
// CLIFFORD:          scf.yield [[G1]]
// CLIFFORD:        }
// CLIFFORD:        case 9 {
// CLIFFORD:          [[G1:%.+]] = quantum.custom "S"() [[L_QBIT]] adj
// CLIFFORD:          scf.yield [[G1]]
// CLIFFORD:        }
// CLIFFORD:      }
// CLIFFORD:      scf.yield [[SWITCH_RES]]
// CLIFFORD:      }
// CLIFFORD:      memref.dealloc [[MEM]]
// CLIFFORD:      return [[LOOP_RES]], [[PHASE]] : !quantum.bit, f64

// PPR-LABEL: func.func private @__catalyst_decompose_RZ_ppr_basis
// PPR-SAME:  [[ARG_QBIT:%.+]]: !quantum.bit
// PPR:       [[MEM:%.+]] = memref.alloc
// PPR:       [[PHASE:%.+]] = call @rs_decomposition_get_phase
// PPR:       [[LOOP_RES:%.+]] = scf.for {{.*}} iter_args([[LOOP_QBIT:%.+]] = [[ARG_QBIT]])
// PPR:       scf.index_switch
// PPR:       case 0 {
// PPR:         scf.yield [[LOOP_QBIT]]
// PPR:       }
// PPR:       case 1 {
// PPR:         [[RES:%.+]] = qec.ppr ["X"](2) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 2 {
// PPR:         [[RES:%.+]] = qec.ppr ["X"](4) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 3 {
// PPR:         [[RES:%.+]] = qec.ppr ["X"](8) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 4 {
// PPR:         [[RES:%.+]] = qec.ppr ["X"](-2) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 5 {
// PPR:         [[RES:%.+]] = qec.ppr ["X"](-4) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 6 {
// PPR:         [[RES:%.+]] = qec.ppr ["X"](-8) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 7 {
// PPR:         [[RES:%.+]] = qec.ppr ["Y"](2) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 8 {
// PPR:         [[RES:%.+]] = qec.ppr ["Y"](4) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 9 {
// PPR:         [[RES:%.+]] = qec.ppr ["Y"](8) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 10 {
// PPR:         [[RES:%.+]] = qec.ppr ["Y"](-2) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 11 {
// PPR:         [[RES:%.+]] = qec.ppr ["Y"](-4) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 12 {
// PPR:         [[RES:%.+]] = qec.ppr ["Y"](-8) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 13 {
// PPR:         [[RES:%.+]] = qec.ppr ["Z"](2) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 14 {
// PPR:         [[RES:%.+]] = qec.ppr ["Z"](4) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 15 {
// PPR:         [[RES:%.+]] = qec.ppr ["Z"](8) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 16 {
// PPR:         [[RES:%.+]] = qec.ppr ["Z"](-2) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 17 {
// PPR:         [[RES:%.+]] = qec.ppr ["Z"](-4) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       case 18 {
// PPR:         [[RES:%.+]] = qec.ppr ["Z"](-8) [[LOOP_QBIT]]
// PPR:         scf.yield [[RES]]
// PPR:       }
// PPR:       memref.dealloc [[MEM]]
// PPR:       return [[LOOP_RES]], [[PHASE]]

// CHECK-LABEL: @test_rz_decomposition
// CHECK-SAME: ([[Q_IN:%.+]]: !quantum.bit)
func.func @test_rz_decomposition(%arg0: !quantum.bit) -> !quantum.bit {
    %theta = arith.constant 0.4 : f64

    // CLIFFORD: [[RES:%.+]]:2 = call @__catalyst_decompose_RZ([[Q_IN]], [[THETA:%.+]])
    // PPR:      [[RES:%.+]]:2 = call @__catalyst_decompose_RZ_ppr_basis([[Q_IN]], [[THETA:%.+]])
    
    // COM: check the phase from RZ is used directly
    // CHECK: quantum.gphase([[RES]]#1)
    
    // CHECK: return [[RES]]#0 : !quantum.bit

    %q_out = quantum.custom "RZ"(%theta) %arg0 : !quantum.bit
    return %q_out : !quantum.bit
}


// -----

// Check that the external helpers are declared in the module 
// CHECK-DAG: func.func private @rs_decomposition_get_size(f64, f64, i1) -> index
// CHECK-DAG: func.func private @rs_decomposition_get_gates(memref<?xindex>, f64, f64, i1)
// CHECK-DAG: func.func private @rs_decomposition_get_phase(f64, f64, i1) -> f64

// CHECK-LABEL: @test_phaseshift_decomposition
// CHECK-SAME: ([[Q_IN:%.+]]: !quantum.bit, [[PHI:%.+]]: f64)
func.func @test_phaseshift_decomposition(%arg0: !quantum.bit, %phi: f64) -> !quantum.bit {
    
    // CHECK: [[C2:%.+]] = arith.constant 2.0{{.*}} : f64
    
    // CLIFFORD: [[RES:%.+]]:2 = call @__catalyst_decompose_RZ([[Q_IN]], [[PHI]])
    // PPR:      [[RES:%.+]]:2 = call @__catalyst_decompose_RZ_ppr_basis([[Q_IN]], [[PHI]])
    
    // CHECK: [[HALF_ANGLE:%.+]] = arith.divf [[PHI]], [[C2]]
    // CHECK: [[TOTAL_PHASE:%.+]] = arith.subf [[RES]]#1, [[HALF_ANGLE]]
    // CHECK: quantum.gphase([[TOTAL_PHASE]])
    // CHECK: return [[RES]]#0 : !quantum.bit

    %q_out = quantum.custom "PhaseShift"(%phi) %arg0 : !quantum.bit
    return %q_out : !quantum.bit
}


// -----

// Test that the helper function is only declared once

// The helper function definition appears first in the module
// CLIFFORD:      func.func private @__catalyst_decompose_RZ(
// PPR:           func.func private @__catalyst_decompose_RZ_ppr_basis(

// Ensure we don't see a duplicate definition immediately following it
// CLIFFORD-NOT:  func.func private @__catalyst_decompose_RZ
// PPR-NOT:       func.func private @__catalyst_decompose_RZ_ppr_basis

// CHECK-LABEL:   @test_deduplication
// CHECK-SAME:    ([[Q_IN:%.+]]: !quantum.bit)
func.func @test_deduplication(%arg0: !quantum.bit) -> !quantum.bit {
    %theta = arith.constant 0.5 : f64

    // CLIFFORD: call @__catalyst_decompose_RZ
    // CLIFFORD: call @__catalyst_decompose_RZ
    // PPR:      call @__catalyst_decompose_RZ_ppr_basis
    // PPR:      call @__catalyst_decompose_RZ_ppr_basis

    %q1 = quantum.custom "RZ"(%theta) %arg0 : !quantum.bit
    %q2 = quantum.custom "RZ"(%theta) %q1 : !quantum.bit
    
    return %q2 : !quantum.bit
}

// -----

// CHECK-LABEL: @test_ppr_arbitrary_z_decomposition
// CHECK-SAME: ([[Q_IN:%.+]]: !quantum.bit, [[THETA:%.+]]: f64)
func.func @test_ppr_arbitrary_z_decomposition(%arg0: !quantum.bit, %theta: f64) -> !quantum.bit {
    
    // CHECK: [[C_MINUS_2:%.+]] = arith.constant 2.0{{.*}} : f64
    
    // CHECK: [[PHI:%.+]] = arith.mulf [[THETA]], [[C_MINUS_2]]

    // CLIFFORD: [[RES:%.+]]:2 = call @__catalyst_decompose_RZ([[Q_IN]], [[PHI]])
    // PPR:      [[RES:%.+]]:2 = call @__catalyst_decompose_RZ_ppr_basis([[Q_IN]], [[PHI]])
    
    // CHECK: quantum.gphase([[RES]]#1)
    
    // CHECK: return [[RES]]#0 : !quantum.bit

    %q_out = qec.ppr.arbitrary ["Z"](%theta) %arg0 : !quantum.bit
    return %q_out : !quantum.bit
}

// -----

// CHECK-LABEL: @test_ppr_arbitrary_ignored
// CHECK-SAME: ([[Q_IN:%.+]]: !quantum.bit, [[THETA:%.+]]: f64)
func.func @test_ppr_arbitrary_ignored(%arg0: !quantum.bit, %theta: f64) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    
    %c_true = arith.constant true

    // CHECK: qec.ppr.arbitrary ["X"]
    %q1 = qec.ppr.arbitrary ["X"](%theta) %arg0 : !quantum.bit

    // CHECK: qec.ppr.arbitrary ["Y"]
    %q2 = qec.ppr.arbitrary ["Y"](%theta) %arg0 : !quantum.bit

    // CHECK: qec.ppr.arbitrary ["Z", "Z"]
    %q3:2 = qec.ppr.arbitrary ["Z", "Z"](%theta) %q1, %q2 : !quantum.bit, !quantum.bit

    // CHECK: qec.ppr.arbitrary ["Z"]{{.*}} cond
    %q4 = qec.ppr.arbitrary ["Z"](%theta) %q3#0 cond(%c_true) : !quantum.bit

    return %q1, %q3#1, %q4 : !quantum.bit, !quantum.bit, !quantum.bit
}
