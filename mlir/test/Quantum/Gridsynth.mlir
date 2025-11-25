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


// Test the defined decomposition function

// CLIFFORD-LABEL: func.func private @__catalyst_decompose_RZ(
// CLIFFORD-SAME: [[ARG_REG:%.+]]: !quantum.reg, [[ARG_IDX:%.+]]: i64, [[ARG_ANGLE:%.+]]: f64)
// CLIFFORD-DAG:  [[C1:%.+]] = arith.constant 1 : index
// CLIFFORD-DAG:  [[C0:%.+]] = arith.constant 0 : index
// CLIFFORD:      [[NUM_GATES:%.+]] = call @rs_decomposition_get_size
// CLIFFORD:      [[MEM:%.+]] = memref.alloca([[NUM_GATES]]) : memref<?xindex>
// CLIFFORD:      call @rs_decomposition_get_gates([[MEM]]
// CLIFFORD:      [[LOOP_RES:%.+]] = scf.for [[IV:%.+]] = [[C0]] to [[NUM_GATES]] step [[C1]] iter_args([[L_REG:%.+]] = [[ARG_REG]])
// CLIFFORD:        [[CASE_ID:%.+]] = memref.load [[MEM]]{{\[}}[[IV]]{{\]}}
// CLIFFORD:        [[SWITCH_RES:%.+]] = scf.index_switch [[CASE_ID]]
// CLIFFORD:        case 0 {
// CLIFFORD:          [[Q:%.+]] = quantum.extract [[L_REG]]
// CLIFFORD:          [[RES:%.+]] = quantum.custom "T"() [[Q]]
// CLIFFORD:          [[INS:%.+]] = quantum.insert [[L_REG]]{{.*}}, [[RES]]
// CLIFFORD:          scf.yield [[INS]]
// CLIFFORD:        }
// CLIFFORD:        case 1 {
// CLIFFORD:          [[Q:%.+]] = quantum.extract [[L_REG]]
// CLIFFORD:          [[G1:%.+]] = quantum.custom "Hadamard"() [[Q]]
// CLIFFORD:          [[G2:%.+]] = quantum.custom "T"() [[G1]]
// CLIFFORD:          [[INS:%.+]] = quantum.insert [[L_REG]]{{.*}}, [[G2]]
// CLIFFORD:          scf.yield [[INS]]
// CLIFFORD:        }
// CLIFFORD:        case 2 {
// CLIFFORD:          [[Q:%.+]] = quantum.extract [[L_REG]]
// CLIFFORD:          [[G1:%.+]] = quantum.custom "S"() [[Q]]
// CLIFFORD:          [[G2:%.+]] = quantum.custom "Hadamard"() [[G1]]
// CLIFFORD:          [[G3:%.+]] = quantum.custom "T"() [[G2]]
// CLIFFORD:          [[INS:%.+]] = quantum.insert [[L_REG]]{{.*}}, [[G3]]
// CLIFFORD:          scf.yield [[INS]]
// CLIFFORD:        }
// CLIFFORD:        case 3 {
// CLIFFORD:          [[Q:%.+]] = quantum.extract [[L_REG]]
// CLIFFORD:          [[G1:%.+]] = quantum.custom "Identity"() [[Q]]
// CLIFFORD:          [[INS:%.+]] = quantum.insert [[L_REG]]{{.*}}, [[G1]]
// CLIFFORD:          scf.yield [[INS]]
// CLIFFORD:        }
// CLIFFORD:        case 4 {
// CLIFFORD:          [[Q:%.+]] = quantum.extract [[L_REG]]
// CLIFFORD:          [[G1:%.+]] = quantum.custom "PauliX"() [[Q]]
// CLIFFORD:          [[INS:%.+]] = quantum.insert [[L_REG]]{{.*}}, [[G1]]
// CLIFFORD:          scf.yield [[INS]]
// CLIFFORD:        }
// CLIFFORD:        case 5 {
// CLIFFORD:          [[Q:%.+]] = quantum.extract [[L_REG]]
// CLIFFORD:          [[G1:%.+]] = quantum.custom "PauliY"() [[Q]]
// CLIFFORD:          [[INS:%.+]] = quantum.insert [[L_REG]]{{.*}}, [[G1]]
// CLIFFORD:          scf.yield [[INS]]
// CLIFFORD:        }
// CLIFFORD:        case 6 {
// CLIFFORD:          [[Q:%.+]] = quantum.extract [[L_REG]]
// CLIFFORD:          [[G1:%.+]] = quantum.custom "PauliZ"() [[Q]]
// CLIFFORD:          [[INS:%.+]] = quantum.insert [[L_REG]]{{.*}}, [[G1]]
// CLIFFORD:          scf.yield [[INS]]
// CLIFFORD:        }
// CLIFFORD:        case 7 {
// CLIFFORD:          [[Q:%.+]] = quantum.extract [[L_REG]]
// CLIFFORD:          [[G1:%.+]] = quantum.custom "Hadamard"() [[Q]]
// CLIFFORD:          [[INS:%.+]] = quantum.insert [[L_REG]]{{.*}}, [[G1]]
// CLIFFORD:          scf.yield [[INS]]
// CLIFFORD:        }
// CLIFFORD:        case 8 {
// CLIFFORD:          [[Q:%.+]] = quantum.extract [[L_REG]]
// CLIFFORD:          [[G1:%.+]] = quantum.custom "S"() [[Q]]
// CLIFFORD:          [[INS:%.+]] = quantum.insert [[L_REG]]{{.*}}, [[G1]]
// CLIFFORD:          scf.yield [[INS]]
// CLIFFORD:        }
// CLIFFORD:        case 9 {
// CLIFFORD:          [[Q:%.+]] = quantum.extract [[L_REG]]
// CLIFFORD:          [[G1:%.+]] = quantum.custom "S"() [[Q]] adj
// CLIFFORD:          [[INS:%.+]] = quantum.insert [[L_REG]]{{.*}}, [[G1]]
// CLIFFORD:          scf.yield [[INS]]
// CLIFFORD:        }

// PPR-LABEL: func.func private @__catalyst_decompose_RZ_ppr_basis
// PPR-SAME:  [[ARG_REG:%.+]]: !quantum.reg
// PPR:       memref.alloca
// PPR:       scf.for {{.*}} iter_args([[LOOP_REG:%.+]] = [[ARG_REG]])
// PPR:       scf.index_switch
// PPR:       case 0 {
// PPR:         scf.yield [[LOOP_REG]]
// PPR:       }
// PPR:       case 1 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["X"](2) [[Q]]
// PPR:         [[INS:%.+]] = quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:         scf.yield [[INS]]
// PPR:       }
// PPR:       case 2 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["X"](4) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }
// PPR:       case 3 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["X"](8) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }
// PPR:       case 4 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["X"](-2) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }
// PPR:       case 5 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["X"](-4) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }
// PPR:       case 6 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["X"](-8) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }
// PPR:       case 7 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["Y"](2) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }
// PPR:       case 8 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["Y"](4) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }
// PPR:       case 9 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["Y"](8) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }
// PPR:       case 10 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["Y"](-2) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }
// PPR:       case 11 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["Y"](-4) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }
// PPR:       case 12 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["Y"](-8) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }
// PPR:       case 13 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["Z"](2) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }
// PPR:       case 14 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["Z"](4) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }
// PPR:       case 15 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["Z"](8) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }
// PPR:       case 16 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["Z"](-2) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }
// PPR:       case 17 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["Z"](-4) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }
// PPR:       case 18 {
// PPR:         [[Q:%.+]] = quantum.extract [[LOOP_REG]]
// PPR:         [[RES:%.+]] = qec.ppr ["Z"](-8) [[Q]]
// PPR:         quantum.insert [[LOOP_REG]]{{.*}}, [[RES]]
// PPR:       }

func.func @test_rz_decomposition(%arg0: !quantum.reg) -> !quantum.reg {
    %idx = arith.constant 0 : i64
    %theta = arith.constant 0.4 : f64
    
    %q_in = quantum.extract %arg0[%idx] : !quantum.reg -> !quantum.bit

    // CHECK-LABEL: @test_rz_decomposition
    // CHECK-SAME: [[REG:%.+]]: !quantum.reg
    
    // CLIFFORD: [[RES:%.+]]:2 = call @__catalyst_decompose_RZ
    // PPR:      [[RES:%.+]]:2 = call @__catalyst_decompose_RZ_ppr_basis
    
    // RZ check: The phase is used directly, not assigned to a new variable
    // CHECK: quantum.gphase([[RES]]#1)

    %q_out = quantum.custom "RZ"(%theta) %q_in : !quantum.bit
    
    %reg_final = quantum.insert %arg0[%idx], %q_out : !quantum.reg, !quantum.bit
    return %reg_final : !quantum.reg
}

// -----

func.func @test_phaseshift_decomposition(%arg0: !quantum.reg) -> !quantum.reg {
    %idx = arith.constant 0 : i64
    %phi = arith.constant 0.7 : f64
    
    %q_in = quantum.extract %arg0[%idx] : !quantum.reg -> !quantum.bit

    // CHECK-LABEL: @test_phaseshift_decomposition
    // Use regex {{.*}} for the float value to avoid precision mismatch errors
    // CHECK: [[PHI:%.+]] = arith.constant {{.*}} : f64
    
    // CLIFFORD: [[RES:%.+]]:2 = call @__catalyst_decompose_RZ
    // PPR:      [[RES:%.+]]:2 = call @__catalyst_decompose_RZ_ppr_basis
    
    // PhaseShift check: The phase IS added to the angle
    // CHECK: [[TOTAL_PHASE:%.+]] = arith.addf [[RES]]#1, [[PHI]]
    // CHECK: quantum.gphase([[TOTAL_PHASE]])

    %q_out = quantum.custom "PhaseShift"(%phi) %q_in : !quantum.bit
    
    %reg_final = quantum.insert %arg0[%idx], %q_out : !quantum.reg, !quantum.bit
    return %reg_final : !quantum.reg
}

// -----

// Test that the helper function is only declared once

func.func @test_deduplication(%arg0: !quantum.reg) -> !quantum.reg {
    %idx = arith.constant 0 : i64
    %theta = arith.constant 0.5 : f64
    %q_in = quantum.extract %arg0[%idx] : !quantum.reg -> !quantum.bit

    // Verify we match the definition ONCE at the top
    // CLIFFORD: func.func private @__catalyst_decompose_RZ
    // PPR:      func.func private @__catalyst_decompose_RZ_ppr_basis
    
    // Verify we DO NOT see the definition again (search rest of file)
    // CLIFFORD-NOT: func.func private @__catalyst_decompose_RZ
    // PPR-NOT:      func.func private @__catalyst_decompose_RZ_ppr_basis

    // CHECK-LABEL: @test_deduplication
    // CLIFFORD: call @__catalyst_decompose_RZ
    // CLIFFORD: call @__catalyst_decompose_RZ
    // PPR:      call @__catalyst_decompose_RZ_ppr_basis
    // PPR:      call @__catalyst_decompose_RZ_ppr_basis

    %q1 = quantum.custom "RZ"(%theta) %q_in : !quantum.bit
    %q2 = quantum.custom "RZ"(%theta) %q1 : !quantum.bit
    
    %reg_final = quantum.insert %arg0[%idx], %q2 : !quantum.reg, !quantum.bit
    return %reg_final : !quantum.reg
}
