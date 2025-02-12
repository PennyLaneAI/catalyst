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

// CHECK: llvm.func @__catalyst__oqd__pulse(!llvm.ptr, f64, f64, !llvm.ptr) -> !llvm.ptr
// CHECK: llvm.func @__catalyst__oqd__ParallelProtocol(!llvm.ptr, i64) -> !llvm.ptr

// CHECK-LABEL: parallel_protocol_op
func.func public @parallel_protocol_op(%arg0: f64) -> !quantum.bit {

    // Pulse 1
    // CHECK: %[[pulse_1:.*]] = llvm.call @__catalyst__oqd__pulse

    // Pulse 2
    // CHECK: %[[pulse_2:.*]] = llvm.call @__catalyst__oqd__pulse

    // Pulse array
    // CHECK: %[[pulse_array:.*]] = llvm.mlir.undef : !llvm.array<2 x ptr>
    // CHECK: %[[pulse_array_insert_0:.*]] = llvm.insertvalue %[[pulse_1_ptr:.*]], %[[pulse_array:.*]][0] : !llvm.array<2 x ptr>
    // CHECK: %[[pulse_array_insert_1:.*]] = llvm.insertvalue %[[pulse_2_ptr:.*]], %[[pulse_array:.*]][1] : !llvm.array<2 x ptr>

    // Store pulse array on stack
    // CHECK: %[[c1:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[pulse_array_ptr:.*]] = llvm.alloca %[[c1:.*]] x !llvm.array<2 x ptr> : (i64) -> !llvm.ptr
    // CHECK: llvm.store %[[pulse_array_insert_1:.*]], %[[pulse_array_ptr:.*]] : !llvm.array<2 x ptr>, !llvm.ptr

    // Parallel Protocol Stub
    // CHECK: %[[pulse_array_size:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %[[pp:.*]] = llvm.call @__catalyst__oqd__ParallelProtocol(%[[pulse_array_ptr:.*]], %[[pulse_array_size:.*]]) : (!llvm.ptr, i64) -> !llvm.ptr


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
