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
    
// CHECK-LABEL: pulse_op
func.func @pulse_op(%arg0: f64) -> !quantum.bit {
    // Exttract the qubit
    // CHECK: %[[qureg:.*]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: %[[qubit:.*]] = quantum.extract %[[qureg:.*]][ 0]
    // CHECK: %[[qubit_ptr:.*]] = builtin.unrealized_conversion_cast %[[qubit:.*]] : !quantum.bit to !llvm.ptr

    // Ensure constants are correctly defined
    // CHECK: %[[phase:.*]] = llvm.mlir.constant(0.000000e+00 : f64) : f64

    // Create the beam struct
     
    // CHECK: %[[beam_struct_undef:.*]] = llvm.mlir.undef : !llvm.struct<(i64, f64, f64, array<2 x i64>, array<2 x i64>)>
    // CHECK: %[[transition_index:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: %[[beam_struct_transition_index:.*]] = llvm.insertvalue %[[transition_index:.*]], %[[beam_struct_undef:.*]][0] : !llvm.struct<(i64, f64, f64, array<2 x i64>, array<2 x i64>)>
    // CHECK: %[[rabi:.*]] = llvm.mlir.constant(1.010000e+01 : f64) : f64
    // CHECK: %[[beam_struct_rabi:.*]] = llvm.insertvalue %[[rabi:.*]], %[[beam_struct_transition_index:.*]][1] : !llvm.struct<(i64, f64, f64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[detuning:.*]] = llvm.mlir.constant(1.111000e+01 : f64) : f64
    // CHECK: %[[beam_struct_detuning:.*]] = llvm.insertvalue %[[detuning:.*]], %[[beam_struct_rabi:.*]][2] : !llvm.struct<(i64, f64, f64, array<2 x i64>, array<2 x i64>)>
    // CHECK: %[[polarization_0:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: %[[beam_struct_polarization_0:.*]] = llvm.insertvalue %[[polarization_0:.*]], %[[beam_struct_detuning:.*]][3, 0] : !llvm.struct<(i64, f64, f64, array<2 x i64>, array<2 x i64>)>
    // CHECK: %[[polarization_1:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[beam_struct_polarization_1:.*]] = llvm.insertvalue %[[polarization_1:.*]], %[[beam_struct_detuning:.*]][3, 1] : !llvm.struct<(i64, f64, f64, array<2 x i64>, array<2 x i64>)>
    // CHECK: %[[wavevector_0:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: %[[beam_struct_wavevector_0:.*]] = llvm.insertvalue %[[wavevector_0:.*]], %[[beam_struct_polarization_1:.*]][4, 0] : !llvm.struct<(i64, f64, f64, array<2 x i64>, array<2 x i64>)>
    // CHECK: %[[wavevector_1:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[beam_struct_wavevector_1:.*]] = llvm.insertvalue %[[wavevector_1:.*]], %[[beam_struct_wavevector_0:.*]][4, 1] : !llvm.struct<(i64, f64, f64, array<2 x i64>, array<2 x i64>)>
    // CHECK: %[[c1:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[beam_ptr:.*]] = llvm.alloca %[[c1:.*]] x !llvm.struct<(i64, f64, f64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    // CHECK: llvm.store %[[beam_struct_wavevector_1:.*]], %[[beam_ptr:.*]] : !llvm.struct<(i64, f64, f64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    // CHECK: %[[pulse:.*]] = llvm.call @__catalyst__oqd__pulse(%[[qubit_ptr:.*]], %arg0, %[[phase:.*]], %[[beam_struct_wavevector_1:.*]]) : (!llvm.ptr, f64, f64, !llvm.ptr) -> !llvm.ptr

    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

    %2 = ion.pulse(%arg0: f64) %1 {
        beam=#ion.beam<
            transition_index=0,
            rabi=10.10,
            detuning=11.11,
            polarization=[0, 1],
            wavevector=[0, 1]
        >,
        phase=0.0
    } : !ion.pulse

    return %1: !quantum.bit
}
