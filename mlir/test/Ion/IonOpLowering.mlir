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

// RUN: quantum-opt %s --convert-ion-to-llvm --split-input-file -verify-diagnostics | FileCheck %s

// CHECK: llvm.mlir.global internal constant @upstate("upstate\00") {addr_space = 0 : i32}
// CHECK: llvm.mlir.global internal constant @estate("estate\00") {addr_space = 0 : i32}
// CHECK: llvm.mlir.global internal constant @downstate("downstate\00") {addr_space = 0 : i32}
// CHECK: llvm.mlir.global internal constant @Yb171("Yb171\00") {addr_space = 0 : i32}
    
// CHECK-LABEL: ion_op
func.func public @ion_op(%arg0: tensor<f64>, %arg1: tensor<f64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    // Ensure constants are correctly defined
    // CHECK: %[[const_1:.*]] = llvm.mlir.constant(1 : i64)
    // CHECK: %[[addr_of_yb171:.*]] = llvm.mlir.addressof @Yb171 : !llvm.ptr
    // CHECK: %[[ptr_to_yb171:.*]] = llvm.getelementptr inbounds %[[addr_of_yb171]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<6 x i8>
    // CHECK: %[[const_mass:.*]] = llvm.mlir.constant(1.710000e+02 : f64)
    // CHECK: %[[const_charge:.*]] = llvm.mlir.constant(1.000000e+00 : f64)

    // Check level array initialization
    // CHECK: %[[level_array_undef:.*]] = llvm.mlir.undef : !llvm.array<3 x struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>>

    // Level 0
    // CHECK: %[[level_struct_0_undef:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    // CHECK: llvm.mlir.addressof @downstate

    // Level 1
    // CHECK: %[[level_struct_1_undef:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    // llvm.mlir.addressof @estate : !llvm.ptr
    
    // Level 2
    // CHECK: %[[level_struct_2_undef:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    // CHECK: llvm.mlir.addressof @upstate : !llvm.ptr

    // Levels Array
    // CHECK: %[[levels_array_size:.*]] = llvm.mlir.constant(3 : i64) : i64
    // CHECK: %[[levels_array_ptr:.*]] = llvm.alloca %[[levels_array_size:.*]] x !llvm.array<3 x struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>> : (i64) -> !llvm.ptr
    // CHECK: llvm.store %[[levels_array:.*]], %[[levels_array_ptr:.*]] : !llvm.array<3 x struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>>, !llvm.ptr
    
    // Transition array initialization
    // CHECK: llvm.mlir.undef : !llvm.array<3 x struct<(ptr, ptr, f64)>>

    // Transition 1
    // CHECK: %[[transition_1_struct_undef:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, f64)>
    // CHECK: llvm.mlir.addressof @downstate
    // CHECK: llvm.mlir.addressof @estate

    // Transition 2
    // CHECK: %[[transition_2_struct_undef:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, f64)>
    // CHECK: llvm.mlir.addressof @downstate
    // CHECK: llvm.mlir.addressof @upstate

    // Transition 3
    // CHECK: %[[transition_3_struct_undef:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, f64)>
    // CHECK: llvm.mlir.addressof @estate
    // CHECK: llvm.mlir.addressof @upstate

    // Final Ion Struct
    // CHECK: %[[ion_struct_undef:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, f64, f64, array<3 x f64>, ptr, ptr)>
    // CHECK: %[[ion_name_ptr:.*]] = llvm.insertvalue %[[ptr_to_yb171]], %[[ion_struct_undef]][0] : !llvm.struct<(ptr, f64, f64, array<3 x f64>, ptr, ptr)> 
    // CHECK: %[[ion_mass_ptr:.*]] = llvm.insertvalue %[[const_mass]], %[[ion_name_ptr]][1] : !llvm.struct<(ptr, f64, f64, array<3 x f64>, ptr, ptr)> 
    // CHECK: %[[ion_charge_ptr:.*]] = llvm.insertvalue %[[const_charge]], %[[ion_mass_ptr]][2] : !llvm.struct<(ptr, f64, f64, array<3 x f64>, ptr, ptr)> 
    // CHECK: %[[ion_ptr:.*]] = llvm.call @__catalyst_ion(%{{.*}}) : (!llvm.ptr) -> !llvm.ptr

    %0 = ion.ion {
        charge = 1.000000e+00 : f64, 
        mass = 1.710000e+02 : f64, 
        name = "Yb171", 
        position = array<f64: 1.0, 2.0, -1.0>,
        levels = [
            #ion.level<
                label="downstate",
                principal = 6 : i64, 
                spin = 4.000000e-01 : f64, 
                orbital = 5.000000e-01 : f64, 
                nuclear = 6.000000e-01 : f64, 
                spin_orbital = 8.000000e-01 : f64, 
                spin_orbital_nuclear = 9.000000e-01 : f64, 
                spin_orbital_nuclear_magnetization = 1.000000e+00 : f64, 
                energy = 0.000000e+00 : f64
            >, 
            #ion.level<
                label="estate",
                principal = 6 : i64, 
                spin = 1.400000e+00 : f64, 
                orbital = 1.500000e+00 : f64, 
                nuclear = 1.600000e+00 : f64, 
                spin_orbital = 1.800000e+00 : f64, 
                spin_orbital_nuclear = 1.900000e+00 : f64, 
                spin_orbital_nuclear_magnetization = 2.000000e+00 : f64, 
                energy = 1.264300e+10 : f64
            >, 
            #ion.level<
                label="upstate",
                principal = 5 : i64, 
                spin = 2.400000e+00 : f64, 
                orbital = 2.500000e+00 : f64, 
                nuclear = 2.600000e+00 : f64, 
                spin_orbital = 2.800000e+00 : f64, 
                spin_orbital_nuclear = 2.900000e+00 : f64, 
                spin_orbital_nuclear_magnetization = 3.000000e+00 : f64, 
                energy = 8.115200e+14 : f64
            >
        ], 
        transitions = [
            #ion.transition<
                level_0 = "downstate",
                level_1 = "estate",
                einstein_a = 2.200000e+00 : f64
            >, 
            #ion.transition<
                level_0 = "downstate", 
                level_1 = "upstate", 
                einstein_a = 1.100000e+00 : f64
            >, 
            #ion.transition<
                level_0 = "estate", 
                level_1 = "upstate", 
                einstein_a = 3.300000e+00 : f64
            >
        ]
    } : !ion.ion
    return
}
