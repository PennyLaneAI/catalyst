// Copyright 2024 Xanadu Quantum Technologies Inc.

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
    
// CHECK-LABEL: pulse_op
func.func @pulse_op(%arg0: f64) -> !quantum.bit {
    // Exttract the qubit
    // CHECK: %[[qureg:.*]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: %[[qubit:.*]] = quantum.extract %[[qureg:.*]][ 0]
    // CHECK: %[[qubit_ptr:.*]] = builtin.unrealized_conversion_cast %[[qubit:.*]] : !quantum.bit to !llvm.ptr

    // Ensure constants are correctly defined
    // CHECK: %[[const_1:.*]] = llvm.mlir.constant(1 : i64)
    // CHECK: %[[phase:.*]] = llvm.mlir.constant(0.000000e+00 : f64) : f64

    // Create the beam struct
     
    // CHECK: %[[beam_struct_undef:.*]] = llvm.mlir.undef : !llvm.struct<(i64, f64, f64, vector<2xi64>, vector<2xi64>)>
    // CHECK: %[[transition_index:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: %[[beam_struct_transition_index:.*]] = llvm.insertvalue %[[transition_index:.*]], %[[beam_struct_undef:.*]][0] : !llvm.struct<(i64, f64, f64, vector<2xi64>, vector<2xi64>)>
    // CHECK: %[[rabi:.*]] = llvm.mlir.constant(1.010000e+01 : f64) : f64
    // CHECK: %[[beam_struct_rabi:.*]] = llvm.insertvalue %[[rabi:.*]], %[[beam_struct_transition_index:.*]][1] : !llvm.struct<(i64, f64, f64, vector<2xi64>, vector<2xi64>)> 
    // CHECK: %[[detuning:.*]] = llvm.mlir.constant(1.111000e+01 : f64) : f64
    // CHECK: %[[beam_struct_detuning:.*]] = llvm.insertvalue %[[detuning:.*]], %[[beam_struct_rabi:.*]][2] : !llvm.struct<(i64, f64, f64, vector<2xi64>, vector<2xi64>)>
    // CHECK: %[[polarization:.*]] = llvm.mlir.constant(dense<[0, 1]> : vector<2xi64>) : vector<2xi64>
    // CHECK: %[[beam_struct_polarization:.*]] = llvm.insertvalue %[[polarization:.*]], %[[beam_struct_detuning:.*]][3] : !llvm.struct<(i64, f64, f64, vector<2xi64>, vector<2xi64>)>
    // CHECK: %[[wavevector:.*]] = llvm.mlir.constant(dense<[0, 1]> : vector<2xi64>) : vector<2xi64>
    // CHECK: %[[beam_struct_wavevector:.*]] = llvm.insertvalue %[[wavevector:.*]], %[[beam_struct_polarization:.*]][4] : !llvm.struct<(i64, f64, f64, vector<2xi64>, vector<2xi64>)>
    // CHECK: llvm.call @__catalyst_pulse(%[[qubit_ptr:.*]], %[[phase:.*]], %4, %[[beam_struct_wavevector:.*]]) : (!llvm.ptr, f64, f64, !llvm.struct<(i64, f64, f64, vector<2xi64>, vector<2xi64>)>) -> ()

    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

    ion.pulse(%arg0: f64) %1 {
        beam=#ion.beam<
            transition_index=0,
            rabi=10.10,
            detuning=11.11,
            polarization=dense<[0, 1]>: vector<2xi64>,
            wavevector=dense<[0, 1]>: vector<2xi64>
        >,
        phase=0.0
    }

    return %1: !quantum.bit
}
