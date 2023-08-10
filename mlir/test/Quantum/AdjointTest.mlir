// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --adjoint-lowering --split-input-file %s | FileCheck %s

// CHECK-LABEL:      @workflow_plain
func.func private @workflow_plain() -> tensor<4xcomplex<f64>> attributes {} {
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant 4.000000e-01 : f64
  %c0_i64 = arith.constant 0 : i64
  quantum.device ["backend", "lightning.qubit"]
  %0 = quantum.alloc( 2) : !quantum.reg
  %1 = quantum.extract %0[%c0_i64] : !quantum.reg -> !quantum.bit
  // CHECK:        RX
  %2 = quantum.custom "RX"(%cst) %1 : !quantum.bit
  %3 = quantum.insert %0[%c0_i64], %2 : !quantum.reg, !quantum.bit
  %4 = quantum.adjoint(%3) : !quantum.reg {
  // CHECK:        PauliZ
  // CHECK-SAME:          adjoint
  // CHECK:        PauliY
  // CHECK-SAME:          adjoint
  // CHECK:        PauliX
  // CHECK-SAME:          adjoint
  ^bb0(%arg0: !quantum.reg):
    %10 = quantum.extract %arg0[%c0_i64] : !quantum.reg -> !quantum.bit
    %11 = quantum.custom "PauliX"() %10 : !quantum.bit
    %12 = quantum.custom "PauliY"() %11 : !quantum.bit
    %13 = quantum.insert %arg0[%c0_i64], %12 : !quantum.reg, !quantum.bit
    %15 = quantum.extract %13[%c1_i64] : !quantum.reg -> !quantum.bit
    %16 = quantum.custom "PauliZ"() %15 : !quantum.bit
    %17 = quantum.insert %13[%c1_i64], %16 : !quantum.reg, !quantum.bit
    quantum.yield %17 : !quantum.reg
  }
  %5 = quantum.extract %4[%c0_i64] : !quantum.reg -> !quantum.bit
  // CHECK:        RY
  %6 = quantum.custom "RY"(%cst) %5 : !quantum.bit
  %7 = quantum.extract %4[%c1_i64] : !quantum.reg -> !quantum.bit
  %8 = quantum.compbasis %6, %7 : !quantum.obs
  %9 = quantum.state %8 : tensor<4xcomplex<f64>>
  quantum.dealloc %0 : !quantum.reg
  return %9 : tensor<4xcomplex<f64>>
}


// CHECK-LABEL:      @workflow_nested
// CHECK:      OpC
// CHECK:      OpD
// CHECK:      OpF
// CHECK-SAME:      adjoint
// CHECK:      OpE
// CHECK-SAME:      adjoint
// CHECK:      OpB
// CHECK-SAME:      adjoint
// CHECK:      OpA
// CHECK-SAME:      adjoint
func.func private @workflow_nested() -> tensor<4xcomplex<f64>> attributes {} {
  %c1_i64 = arith.constant 1 : i64
  %c0_i64 = arith.constant 0 : i64
  quantum.device ["backend", "lightning.qubit"]
  %0 = quantum.alloc( 2) : !quantum.reg
  %1 = quantum.adjoint(%0) : !quantum.reg {
  ^bb0(%arg0: !quantum.reg):
    %6 = quantum.extract %arg0[%c1_i64] : !quantum.reg -> !quantum.bit
    %7 = quantum.custom "OpA"() %6 : !quantum.bit
    %8 = quantum.custom "OpB"() %7 : !quantum.bit
    %9 = quantum.insert %arg0[%c1_i64], %8 : !quantum.reg, !quantum.bit
    %10 = quantum.adjoint(%9) : !quantum.reg {
    ^bb0(%arg1: !quantum.reg):
      %11 = quantum.extract %arg1[%c1_i64] : !quantum.reg -> !quantum.bit
      %12 = quantum.custom "OpC"() %11 : !quantum.bit
      %13 = quantum.custom "OpD"() %12 : !quantum.bit
      %14 = quantum.insert %arg1[%c1_i64], %13 : !quantum.reg, !quantum.bit
      %15 = quantum.adjoint(%14) : !quantum.reg {
      ^bb0(%arg2: !quantum.reg):
        %16 = quantum.extract %arg2[%c1_i64] : !quantum.reg -> !quantum.bit
        %17 = quantum.custom "OpE"() %16 : !quantum.bit
        %18 = quantum.custom "OpF"() %17 : !quantum.bit
        %19 = quantum.insert %arg2[%c1_i64], %18 : !quantum.reg, !quantum.bit
        quantum.yield %19 : !quantum.reg
      }
      quantum.yield %15 : !quantum.reg
    }
    quantum.yield %10 : !quantum.reg
  }
  %2 = quantum.extract %1[%c0_i64] : !quantum.reg -> !quantum.bit
  %3 = quantum.extract %1[%c1_i64] : !quantum.reg -> !quantum.bit
  %4 = quantum.compbasis %2, %3 : !quantum.obs
  %5 = quantum.state %4 : tensor<4xcomplex<f64>>
  quantum.dealloc %0 : !quantum.reg
  return %5 : tensor<4xcomplex<f64>>
}

