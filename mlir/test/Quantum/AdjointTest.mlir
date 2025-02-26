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

// RUN: quantum-opt --adjoint-lowering --split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL:      @workflow_plain
func.func private @workflow_plain() -> tensor<4xcomplex<f64>> attributes {} {
  %val = "test.op" () : () -> (i64)
  %cst = arith.constant 4.000000e-01 : f64
  %c0_i64 = arith.constant 0 : i64
  quantum.device ["rtd_lightning.so", "LightningQubit", "{shots: 0}"]
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
    %15 = quantum.extract %13[%val] : !quantum.reg -> !quantum.bit
    %16 = quantum.custom "PauliZ"() %15 : !quantum.bit
    %17 = quantum.insert %13[%val], %16 : !quantum.reg, !quantum.bit
    quantum.yield %17 : !quantum.reg
  }
  %5 = quantum.extract %4[%c0_i64] : !quantum.reg -> !quantum.bit
  // CHECK:        RY
  %6 = quantum.custom "RY"(%cst) %5 : !quantum.bit
  %7 = quantum.extract %4[%val] : !quantum.reg -> !quantum.bit
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
  quantum.device ["rtd_lightning.so", "LightningQubit", "{shots: 0}"]
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

// -----

func.func @workflow_unhandled() {
  %0 = quantum.alloc(1) : !quantum.reg
  %1 = quantum.adjoint (%0) : !quantum.reg {
  ^bb0(%arg0: !quantum.reg):
    %qb = quantum.extract %arg0[0] : !quantum.reg -> !quantum.bit
    // expected-error@+1 {{Unhandled operation in adjoint region}}
    quantum.measure %qb : i1, !quantum.bit
    quantum.yield %arg0 : !quantum.reg
  }
  quantum.dealloc %1 : !quantum.reg
  return
}


// -----

func.func private @qubit_unitary_test(%arg0: tensor<4x4xcomplex<f64>>) -> tensor<4xcomplex<f64>> {
    quantum.device ["rtd_lightning.so", "LightningQubit", "{shots: 0}"]
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.adjoint(%0) : !quantum.reg {
    ^bb0(%arg1: !quantum.reg):
      %6 = quantum.extract %arg1[ 0] : !quantum.reg -> !quantum.bit
      %7 = quantum.extract %arg1[ 1] : !quantum.reg -> !quantum.bit

      // CHECK-DAG: [[idx0:%.+]] = index.constant 0
      // CHECK-DAG: [[idx1:%.+]] = index.constant 1
      // CHECK-DAG: [[idxN:%.+]] = index.constant
      // CHECK: scf.for [[i:%.+]] = [[idx0]] to [[idxN]] step [[idx1]]
      // CHECK:     scf.for [[j:%.+]] = [[idx0]] to [[idxN]] step [[idx1]]
      // CHECK:     [[element:%.+]] = tensor.extract {{.*}}[[[i]], [[j]]
      // CHECK:     [[real:%.+]] = complex.re [[element]]
      // CHECK:     [[imag:%.+]] = complex.im [[element]]
      // CHECK:     catalyst.list_push [[real]]
      // CHECK:     catalyst.list_push [[imag]]

      %8:2 = quantum.unitary(%arg0 : tensor<4x4xcomplex<f64>>) %6, %7 : !quantum.bit, !quantum.bit

      // CHECK-DAG: [[result:%.+]] = tensor.empty
      // CHECK: scf.for [[k:%.+]] = [[idx0]] to [[idxN]] step [[idx1]] iter_args([[curr_k:%.+]] = [[result]])
      // CHECK:   [[kplus1:%.+]] = index.add [[k]], [[idx1]]
      // CHECK:   [[k_idx:%.+]] = index.sub [[idxN]], [[kplus1]]
      // CHECK:   [[last_tensor:%.+]] = scf.for [[l:%.+]] = [[idx0]] to [[idxN]] step [[idx1]] iter_args([[curr_l:%.+]] = [[curr_k]])
      // CHECK:     [[imag2:%.+]] = catalyst.list_pop
      // CHECK:     [[real2:%.+]] = catalyst.list_pop
      // CHECK:     [[complex:%.+]] = complex.create [[real2]], [[imag2]]
      // CHECK:     [[lplus1:%.+]] = index.add [[l]], [[idx1]]
      // CHECK:     [[l_idx:%.+]] = index.sub [[idxN]], [[lplus1]]
      // CHECK:     [[new_tensor:%.+]] = tensor.insert [[complex]] into [[curr_l]][[[k_idx]], [[l_idx]]]
      // CHECK:     scf.yield [[new_tensor]]

      // CHECK:   scf.yield [[last_tensor]]


      %9 = quantum.insert %arg1[ 0], %8#0 : !quantum.reg, !quantum.bit
      %10 = quantum.insert %9[ 1], %8#1 : !quantum.reg, !quantum.bit
      quantum.yield %10 : !quantum.reg
    }
    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %1[ 1] : !quantum.reg -> !quantum.bit
    %4 = quantum.compbasis %2, %3 : !quantum.obs
    %5 = quantum.state %4 : tensor<4xcomplex<f64>>
    quantum.dealloc %0 : !quantum.reg
    return %5 : tensor<4xcomplex<f64>>
  }

// -----

func.func private @circuit(%arg0: f64, %arg1: !quantum.reg) -> !quantum.reg {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %1 = quantum.extract %arg1[%c0_i64] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "PauliX"() %1 : !quantum.bit
    %3 = quantum.custom "RX"(%arg0) %2 : !quantum.bit
    %4 = quantum.custom "PauliZ"() %3 : !quantum.bit
    %5 = quantum.insert %arg1[%c1_i64], %4 : !quantum.reg, !quantum.bit
    func.return %5: !quantum.reg
}

// CHECK:   func.func private @circuit.adjoint(%arg0: f64, %arg1: !quantum.reg) -> !quantum.reg {
// CHECK:   quantum.custom "PauliZ"() {{%.+}} {adjoint} : !quantum.bit
// CHECK:   quantum.custom "RX"({{%.+}}) {{%.+}} {adjoint} : !quantum.bit
// CHECK:   quantum.custom "PauliX"() {{%.+}} {adjoint} : !quantum.bit

// CHECK:   func.func private @workflow_adjoint(%arg0: f64) -> tensor<4xcomplex<f64>> {
// CHECK:   quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
// CHECK:   quantum.custom "RY"({{%.+}}) {{%.+}} {adjoint} : !quantum.bit
// CHECK:   call @circuit.adjoint(%arg0, {{%.+}}) : (f64, !quantum.reg) -> !quantum.reg
// CHECK:   quantum.custom "PauliZ"() {{%.+}} {adjoint} : !quantum.bit
// CHECK:   quantum.custom "RX"({{%.+}}) {{%.+}} {adjoint} : !quantum.bit
// CHECK:   quantum.custom "PauliX"() {{%.+}} {adjoint} : !quantum.bit
// CHECK:   quantum.custom "RY"({{%.+}}) {{%.+}} : !quantum.bit

func.func private @workflow_adjoint(%arg0: f64) -> tensor<4xcomplex<f64>> attributes {} {
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant 4.000000e-01 : f64
  %c0_i64 = arith.constant 0 : i64
  quantum.device ["rtd_lightning.so", "LightningQubit", "{shots: 0}"]
  %0 = quantum.alloc( 2) : !quantum.reg
  %1 = quantum.extract %0[%c0_i64] : !quantum.reg -> !quantum.bit
  %2 = quantum.custom "RX"(%cst) %1 : !quantum.bit
  %3 = quantum.insert %0[%c0_i64], %2 : !quantum.reg, !quantum.bit
  %4 = quantum.adjoint(%3) : !quantum.reg {
    ^bb0(%arg1: !quantum.reg):
      %5 = quantum.extract %arg1[%c0_i64] : !quantum.reg -> !quantum.bit
      %6 = quantum.custom "PauliX"() %5 : !quantum.bit
      %7 = quantum.custom "RX"(%arg0) %6 : !quantum.bit
      %8 = quantum.custom "PauliZ"() %7 : !quantum.bit
      %9 = quantum.insert %arg1[%c0_i64], %8 : !quantum.reg, !quantum.bit
      %10 = func.call @circuit(%arg0, %9): (f64, !quantum.reg) -> !quantum.reg
      %11 = quantum.extract %10[%c0_i64] : !quantum.reg -> !quantum.bit
      %12 = quantum.custom "RY"(%arg0) %11 : !quantum.bit
      %13 = quantum.insert %10[%c0_i64], %12 : !quantum.reg, !quantum.bit
      quantum.yield %13 : !quantum.reg
  }
  %6 = quantum.extract %4[%c0_i64] : !quantum.reg -> !quantum.bit
  %7 = quantum.custom "RY"(%cst) %6 : !quantum.bit
  %8 = quantum.extract %4[%c1_i64] : !quantum.reg -> !quantum.bit
  %9 = quantum.compbasis %8, %7 : !quantum.obs
  %10 = quantum.state %9 : tensor<4xcomplex<f64>>
  quantum.dealloc %0 : !quantum.reg
  return %10 : tensor<4xcomplex<f64>>
}

// -----

// CHECK-LABEL: @param_ordering
func.func private @param_ordering(%0: !quantum.reg) -> !quantum.reg {
  // CHECK-DAG: [[C1:%.+]] = arith.constant 1.000000e-01
  // CHECK-DAG: [[C2:%.+]] = arith.constant 2.000000e-01
  // CHECK-DAG: [[C3:%.+]] = arith.constant 3.000000e-01
  %c1 = arith.constant 0.1 : f64
  %c2 = arith.constant 0.2 : f64
  %c3 = arith.constant 0.3 : f64

  // CHECK:     catalyst.list_push [[C1]]
  // CHECK:     catalyst.list_push [[C2]]
  // CHECK:     catalyst.list_push [[C3]]

  // CHECK-NOT: quantum.adjoint
  %1 = quantum.adjoint(%0) : !quantum.reg {
  ^bb0(%r0: !quantum.reg):
    %q0 = quantum.extract %r0[ 0] : !quantum.reg -> !quantum.bit

    // CHECK:   [[A3:%.+]] = catalyst.list_pop
    // CHECK:   [[A2:%.+]] = catalyst.list_pop
    // CHECK:   [[A1:%.+]] = catalyst.list_pop
    // CHECK:   quantum.custom "Rot"([[A1]], [[A2]], [[A3]])
    %q1 = quantum.custom "Rot"(%c1, %c2, %c3) %q0 : !quantum.bit

    %r1 = quantum.insert %r0[ 0], %q1 : !quantum.reg, !quantum.bit
    quantum.yield %r1 : !quantum.reg
  }

  return %1 : !quantum.reg
}
