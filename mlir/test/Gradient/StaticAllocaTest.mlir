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

// RUN: quantum-opt --lower-gradients --split-input-file %s | FileCheck %s

// CHECK-LABEL: @static_alloca_qubit_grad
module @static_alloca_qubit_grad {

   // CHECK-LABEL: @structured_circuit.qgrad
   // CHECK-NOT: scf
   // CHECK: memref.alloca
   // CHECK-NOT: scf
   // CHECK: memref.alloca

   // CHECK-LABEL: @structured_circuit.pcount
   // CHECK-NOT: scf
   // CHECK: memref.alloca

   // CHECK-LABEL: @structured_circuit.quantum
   // CHECK-NOT: scf
   // CHECK: memref.alloca

   func.func @structured_circuit(%arg0: f64, %arg1: i1, %arg2: i1) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    %idx = arith.constant 0 : i64
    %r = quantum.alloc(1) : !quantum.reg
    %q_0 = quantum.extract %r[%idx] : !quantum.reg -> !quantum.bit
    %q_1 = quantum.custom "rx"(%arg0) %q_0 : !quantum.bit
    %q_2 = scf.if %arg1 -> !quantum.bit {
        %q_1_0 = quantum.custom "ry"(%arg0) %q_1 : !quantum.bit
        %q_1_1 = scf.if %arg2 -> !quantum.bit {
            %q_1_0_0 = quantum.custom "rz"(%arg0) %q_1_0 : !quantum.bit
            scf.yield %q_1_0_0 : !quantum.bit
        } else {
            %q_1_0_1 = quantum.custom "rz"(%arg0) %q_1_0 : !quantum.bit
            %q_1_0_2 = quantum.custom "rz"(%arg0) %q_1_0_1 : !quantum.bit
            scf.yield %q_1_0_2 : !quantum.bit
        }
        scf.yield %q_1_1 : !quantum.bit
    } else {
        scf.yield %q_1 : !quantum.bit
    }
    cf.br ^exit
  ^exit:
    %q_3 = quantum.custom "rx"(%arg0) %q_2 : !quantum.bit
    %obs = quantum.namedobs %q_3[PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64
    func.return %expval : f64
  }

  func.func @gradCall1(%arg0: f64, %b0: i1, %b1: i1) -> f64 {
    
    %0 = gradient.grad "auto" @structured_circuit(%arg0, %b0, %b1) : (f64, i1, i1) -> f64
    func.return %0 : f64
  }
}

// -----
