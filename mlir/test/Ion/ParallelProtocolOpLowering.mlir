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

// CHECK: llvm.func @__catalyst_pulse(!llvm.ptr, f64, f64, !llvm.struct<(i64, f64, f64, array<2 x i64>, array<2 x i64>)>) -> !llvm.ptr
// CHECK: llvm.func @__catalyst_parallel_protocol(!llvm.array<2 x ptr>) -> !llvm.ptr
// CHECK: llvm.func @__catalyst_ion(!llvm.ptr) -> !llvm.ptr
// CHECK: llvm.mlir.global internal constant @upstate("upstate\00") {addr_space = 0 : i32}
// CHECK: llvm.mlir.global internal constant @estate("estate\00") {addr_space = 0 : i32}
// CHECK: llvm.mlir.global internal constant @downstate("downstate\00") {addr_space = 0 : i32}
// CHECK: llvm.mlir.global internal constant @Yb171("Yb171\00") {addr_space = 0 : i32}
    
// CHECK-LABEL: parallel_protocol_op
func.func public @parallel_protocol_op(%arg0: f64) -> !quantum.bit {
    // Ion
    // CHECK: %[[ion:.*]] = llvm.call @__catalyst_ion

    // Pulse 1
    // CHECK: %[[pulse_1:.*]] = llvm.call @__catalyst_pulse
    
    // Pulse 2
    // CHECK: %[[pulse_2:.*]] = llvm.call @__catalyst_pulse

    // Pulse array
    // CHECK: %[[pulse_array:.*]] = llvm.mlir.undef : !llvm.array<2 x ptr>
    // CHECK: %[[pulse_array_insert_0:.*]] = llvm.insertvalue %[[pulse_1_ptr:.*]], %[[pulse_array:.*]][0] : !llvm.array<2 x ptr> 
    // CHECK: %[[pulse_array_insert_1:.*]] = llvm.insertvalue %[[pulse_2_ptr:.*]], %[[pulse_array:.*]][1] : !llvm.array<2 x ptr> 

    // Parallel Protocol Stub
    // CHECK: %[[pp:.*]] = llvm.call @__catalyst_parallel_protocol(%[[pulse_array_insert_1:.*]]) : (!llvm.array<2 x ptr>) -> !llvm.ptr

    %ion = ion.ion {
        charge = 1.000000e+00 : f64, 
        mass = 1.710000e+02 : f64, 
        name = "Yb171", 
        position = array<f64: 1.000000e+00, 2.000000e+00, -1.000000e+00>, 
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

    %qreg = quantum.alloc( 1) : !quantum.reg
    %q0 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit

    %pp= ion.parallelprotocol(%q0) : !quantum.bit{
        ^bb0(%arg1: !quantum.bit):
          %p1 = ion.pulse(%arg0: f64) %arg1 {
              beam=#ion.beam<
                  transition_index=1,
                  rabi=10.10,
                  detuning=11.11,
                  polarization=[0, 1],
                  wavevector=[0, 1]
              >,
              phase=0.0
          } : !ion.pulse
          
          %p2 = ion.pulse(%arg0: f64) %arg1 {
              beam=#ion.beam<
                  transition_index=0,
                  rabi=10.10,
                  detuning=11.11,
                  polarization=[0, 1],
                  wavevector=[0, 1]
              >,
              phase=0.0
          } : !ion.pulse
          ion.yield %arg1: !quantum.bit
    }

    return %pp: !quantum.bit
}
