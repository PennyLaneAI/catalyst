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

// RUN: quantum-opt --adjoint-lowering --split-input-file -verify-diagnostics %s | FileCheck %s --implicit-check-not="!quantum.bit" --implicit-check-not="!quantum.reg"

// CHECK-LABEL:      @workflow_plain
// CHECK:        RX
// CHECK:        PauliZ
// CHECK-SAME:          adj
// CHECK:        PauliY
// CHECK-SAME:          adj
// CHECK:        PauliX
// CHECK-SAME:          adj
// CHECK:        RY
// CHECK-LABEL:      @workflow_nested
// CHECK:      OpC
// CHECK:      OpD
// CHECK:      OpF
// CHECK-SAME:      adj
// CHECK:      OpE
// CHECK-SAME:      adj
// CHECK:      OpB
// CHECK-SAME:      adj
// CHECK:      OpA
// CHECK-SAME:      adj
// CHECK-LABEL:      @workflow_many_args
// CHECK: [[q0:%.+]] = qref.get %0[ 0] : !qref.reg<{{.*}}> -> !qref.bit
// CHECK: qref.custom "RX"({{%.+}}) [[q0]] : !qref.bit
// CHECK: [[q1:%.+]] = qref.get %0[ 1] : !qref.reg<{{.*}}> -> !qref.bit
// CHECK: qref.custom "CNOT"() [[q0]], [[q1]] adj : !qref.bit, !qref.bit
// CHECK: qref.custom "PauliY"() [[q1]] adj : !qref.bit
// CHECK: qref.custom "PauliX"() [[q0]] adj : !qref.bit
// CHECK: qref.custom "RY"({{%.+}}) [[q0]] : !qref.bit

module {
  func.func private @workflow_plain() -> tensor<4xcomplex<f64>> {
    %0 = "test.op"() : () -> i64
    %cst = arith.constant 4.000000e-01 : f64
    %c0_i64 = arith.constant 0 : i64
    quantum.device ["rtd_lightning.so", "LightningQubit", "{shots: 0}"]
    %1 = qref.alloc( 2) : !qref.reg<2>
    %2 = qref.get %1[%c0_i64] : !qref.reg<2>, i64 -> !qref.bit
    qref.custom "RX"(%cst) %2 : !qref.bit
    qref.adjoint {
      %7 = qref.get %1[%c0_i64] : !qref.reg<2>, i64 -> !qref.bit
      qref.custom "PauliX"() %7 : !qref.bit
      qref.custom "PauliY"() %7 : !qref.bit
      %8 = qref.get %1[%0] : !qref.reg<2>, i64 -> !qref.bit
      qref.custom "PauliZ"() %8 : !qref.bit
    }
    %3 = qref.get %1[%c0_i64] : !qref.reg<2>, i64 -> !qref.bit
    qref.custom "RY"(%cst) %3 : !qref.bit
    %4 = qref.get %1[%0] : !qref.reg<2>, i64 -> !qref.bit
    %5 = qref.compbasis qubits %3, %4 : !quantum.obs
    %6 = quantum.state %5 : tensor<4xcomplex<f64>>
    qref.dealloc %1 : !qref.reg<2>
    return %6 : tensor<4xcomplex<f64>>
  }
  func.func private @workflow_nested() -> tensor<4xcomplex<f64>> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    quantum.device ["rtd_lightning.so", "LightningQubit", "{shots: 0}"]
    %0 = qref.alloc( 2) : !qref.reg<2>
    qref.adjoint {
      %5 = qref.get %0[%c1_i64] : !qref.reg<2>, i64 -> !qref.bit
      qref.custom "OpA"() %5 : !qref.bit
      qref.custom "OpB"() %5 : !qref.bit
      qref.adjoint {
        %6 = qref.get %0[%c1_i64] : !qref.reg<2>, i64 -> !qref.bit
        qref.custom "OpC"() %6 : !qref.bit
        qref.custom "OpD"() %6 : !qref.bit
        qref.adjoint {
          %7 = qref.get %0[%c1_i64] : !qref.reg<2>, i64 -> !qref.bit
          qref.custom "OpE"() %7 : !qref.bit
          qref.custom "OpF"() %7 : !qref.bit
        }
      }
    }
    %1 = qref.get %0[%c0_i64] : !qref.reg<2>, i64 -> !qref.bit
    %2 = qref.get %0[%c1_i64] : !qref.reg<2>, i64 -> !qref.bit
    %3 = qref.compbasis qubits %1, %2 : !quantum.obs
    %4 = quantum.state %3 : tensor<4xcomplex<f64>>
    qref.dealloc %0 : !qref.reg<2>
    return %4 : tensor<4xcomplex<f64>>
  }
  func.func private @workflow_many_args() -> tensor<4xcomplex<f64>> {
    %cst = arith.constant 4.000000e-01 : f64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    quantum.device ["rtd_lightning.so", "LightningQubit", "{shots: 0}"]
    %0 = qref.alloc( 2) : !qref.reg<2>
    %1 = qref.get %0[%c0_i64] : !qref.reg<2>, i64 -> !qref.bit
    qref.custom "RX"(%cst) %1 : !qref.bit
    %2 = qref.get %0[%c1_i64] : !qref.reg<2>, i64 -> !qref.bit
    qref.adjoint {
      qref.custom "PauliX"() %1 : !qref.bit
      qref.custom "PauliY"() %2 : !qref.bit
      qref.custom "CNOT"() %1, %2 : !qref.bit, !qref.bit
    }
    qref.custom "RY"(%cst) %1 : !qref.bit
    %3 = qref.compbasis(qreg %0 : !qref.reg<2>) : !quantum.obs
    %4 = quantum.state %3 : tensor<4xcomplex<f64>>
    qref.dealloc %0 : !qref.reg<2>
    return %4 : tensor<4xcomplex<f64>>
  }
}

// -----

module {
  func.func @workflow_unhandled() {
    %0 = qref.alloc( 1) : !qref.reg<1>
    qref.adjoint {
      %1 = qref.get %0[ 0] : !qref.reg<1> -> !qref.bit
    // expected-error@+1 {{Unhandled operation in adjoint region}}
      %2 = qref.measure %1 : i1
    }
    qref.dealloc %0 : !qref.reg<1>
    return
  }
}

// -----

// CHECK-LABEL: @workflow_non_float_param
// CHECK-NOT: qref.adjoint
// CHECK: [[BOOL_TENSOR:%.+]] = "test.op"() : () -> tensor<2xi1>
// CHECK-NOT: catalyst.list_push
// CHECK-NOT: catalyst.list_pop
// CHECK: qref.operator "MultiX"([[BOOL_TENSOR]]: tensor<2xi1>) adj

module {
  func.func @workflow_non_float_param(%arg0: !qref.reg<?>) {
    qref.adjoint {
      %0 = "test.op"() : () -> tensor<2xi1>
      %1 = qref.get %arg0[ 0] : !qref.reg<?> -> !qref.bit
      %2 = qref.get %arg0[ 1] : !qref.reg<?> -> !qref.bit
      qref.operator "MultiX"(%0: tensor<2xi1>) qubits(%1, %2)
        param_map = {bitstring = [0]} qubit_map = {wires = [0, 1]}
    }
    return
  }
}

// -----

// CHECK-LABEL: @qubit_unitary_test
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

module {
  func.func private @qubit_unitary_test() -> tensor<4xcomplex<f64>> {
    quantum.device ["rtd_lightning.so", "LightningQubit", "{shots: 0}"]
    %0 = qref.alloc( 2) : !qref.reg<2>
    qref.adjoint {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %5 = "test.op"() : () -> tensor<4x4xcomplex<f64>>
        %6 = qref.get %0[ 0] : !qref.reg<2> -> !qref.bit
        %7 = qref.get %0[ 1] : !qref.reg<2> -> !qref.bit
        qref.unitary(%5 : tensor<4x4xcomplex<f64>>) %6, %7 : !qref.bit, !qref.bit
      }
    }
    %1 = qref.get %0[ 0] : !qref.reg<2> -> !qref.bit
    %2 = qref.get %0[ 1] : !qref.reg<2> -> !qref.bit
    %3 = qref.compbasis qubits %1, %2 : !quantum.obs
    %4 = quantum.state %3 : tensor<4xcomplex<f64>>
    qref.dealloc %0 : !qref.reg<2>
    return %4 : tensor<4xcomplex<f64>>
  }
}

// -----

// CHECK:   func.func private @circuit.adjoint(%arg0: f64, %arg1: !qref.reg<{{.*}}>) {
// CHECK:   qref.custom "PauliZ"() {{%.+}} adj : !qref.bit
// CHECK:   qref.custom "RX"({{%.+}}) {{%.+}} adj : !qref.bit
// CHECK:   qref.custom "PauliX"() {{%.+}} adj : !qref.bit
// CHECK:   func.func private @workflow_adjoint(%arg0: f64) -> tensor<4xcomplex<f64>> {
// CHECK:   qref.custom "RX"({{%.+}}) {{%.+}} : !qref.bit
// CHECK:   qref.custom "RY"({{%.+}}) {{%.+}} adj : !qref.bit
// CHECK:   call @circuit.adjoint(%arg0, {{%.+}}) : (f64, !qref.reg<{{.*}}>) -> ()
// CHECK:   qref.custom "PauliZ"() {{%.+}} adj : !qref.bit
// CHECK:   qref.custom "RX"({{%.+}}) {{%.+}} adj : !qref.bit
// CHECK:   qref.custom "PauliX"() {{%.+}} adj : !qref.bit
// CHECK:   qref.custom "RY"({{%.+}}) {{%.+}} : !qref.bit

module {
  func.func private @circuit(%arg0: f64, %arg1: !qref.reg<2>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %0 = qref.get %arg1[%c0_i64] : !qref.reg<2>, i64 -> !qref.bit
    qref.custom "PauliX"() %0 : !qref.bit
    qref.custom "RX"(%arg0) %0 : !qref.bit
    qref.custom "PauliZ"() %0 : !qref.bit
    return
  }
  func.func private @workflow_adjoint(%arg0: f64) -> tensor<4xcomplex<f64>> {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 4.000000e-01 : f64
    %c0_i64 = arith.constant 0 : i64
    quantum.device ["rtd_lightning.so", "LightningQubit", "{shots: 0}"]
    %0 = qref.alloc( 2) : !qref.reg<2>
    %1 = qref.get %0[%c0_i64] : !qref.reg<2>, i64 -> !qref.bit
    qref.custom "RX"(%cst) %1 : !qref.bit
    qref.adjoint {
      %6 = qref.get %0[%c0_i64] : !qref.reg<2>, i64 -> !qref.bit
      qref.custom "PauliX"() %6 : !qref.bit
      qref.custom "RX"(%arg0) %6 : !qref.bit
      qref.custom "PauliZ"() %6 : !qref.bit
      func.call @circuit(%arg0, %0) : (f64, !qref.reg<2>) -> ()
      %7 = qref.get %0[%c0_i64] : !qref.reg<2>, i64 -> !qref.bit
      qref.custom "RY"(%arg0) %7 : !qref.bit
    }
    %2 = qref.get %0[%c0_i64] : !qref.reg<2>, i64 -> !qref.bit
    qref.custom "RY"(%cst) %2 : !qref.bit
    %3 = qref.get %0[%c1_i64] : !qref.reg<2>, i64 -> !qref.bit
    %4 = qref.compbasis qubits %3, %2 : !quantum.obs
    %5 = quantum.state %4 : tensor<4xcomplex<f64>>
    qref.dealloc %0 : !qref.reg<2>
    return %5 : tensor<4xcomplex<f64>>
  }
}

// -----

// CHECK:   func.func private @circuit.adjoint(%arg0: f64, %arg1: !qref.bit, %arg2: !qref.bit) {
// CHECK:   qref.custom "PauliZ"() %arg2 adj : !qref.bit
// CHECK:   qref.custom "RX"({{%.+}}) %arg2 adj : !qref.bit
// CHECK:   qref.custom "PauliX"() %arg1 adj : !qref.bit
// CHECK: return
// CHECK:   func.func private @workflow_adjoint(%arg0: f64) -> tensor<4xcomplex<f64>> {
// CHECK:   [[cst:%.+]] = arith.constant 4.000000e-01 : f64
// CHECK:   [[q0:%.+]] = qref.get {{%.+}}[ 0] : !qref.reg<{{.*}}> -> !qref.bit
// CHECK:   [[q1:%.+]] = qref.get {{%.+}}[ 1] : !qref.reg<{{.*}}> -> !qref.bit
// CHECK:   qref.custom "RY"({{%.+}}) [[q1]] adj : !qref.bit
// CHECK:   call @circuit.adjoint([[cst]], {{%.+}}, {{%.+}}) : (f64, !qref.bit, !qref.bit) -> ()
// CHECK:   qref.custom "PauliZ"() [[q1]] adj : !qref.bit
// CHECK:   qref.custom "RX"({{%.+}}) [[q0]] adj : !qref.bit
// CHECK:   qref.custom "PauliX"() [[q0]] adj : !qref.bit
// CHECK:   [[obs:%.+]] = qref.compbasis qubits [[q0]], [[q1]] : !quantum.obs
// CHECK:   quantum.state [[obs]] : tensor<4xcomplex<f64>>
// CHECK: qref.dealloc {{%.+}} : !qref.reg<2>

module {
  func.func private @circuit(%arg0: f64, %arg1: !qref.bit, %arg2: !qref.bit) {
    qref.custom "PauliX"() %arg1 : !qref.bit
    qref.custom "RX"(%arg0) %arg2 : !qref.bit
    qref.custom "PauliZ"() %arg2 : !qref.bit
    return
  }
  func.func private @workflow_adjoint(%arg0: f64) -> tensor<4xcomplex<f64>> {
    %0 = qref.alloc( 2) : !qref.reg<2>
    %1 = qref.get %0[ 0] : !qref.reg<2> -> !qref.bit
    %2 = qref.get %0[ 1] : !qref.reg<2> -> !qref.bit
    qref.adjoint {
      qref.custom "PauliX"() %1 : !qref.bit
      qref.custom "RX"(%arg0) %1 : !qref.bit
      qref.custom "PauliZ"() %2 : !qref.bit
      %cst = arith.constant 4.000000e-01 : f64
      func.call @circuit(%cst, %1, %2) : (f64, !qref.bit, !qref.bit) -> ()
      qref.custom "RY"(%arg0) %2 : !qref.bit
    }
    %3 = qref.compbasis qubits %1, %2 : !quantum.obs
    %4 = quantum.state %3 : tensor<4xcomplex<f64>>
    qref.dealloc %0 : !qref.reg<2>
    return %4 : tensor<4xcomplex<f64>>
  }
}

// -----

// CHECK-LABEL: @param_ordering
// CHECK-DAG: [[F1:%.+]] = arith.constant 1.000000e-01
// CHECK-DAG: [[F2:%.+]] = arith.constant 2.000000e-01
// CHECK-DAG: [[F3:%.+]] = arith.constant 3.000000e-01
// CHECK-NOT: qref.adjoint
// CHECK: scf.for [[i:%.+]] =
// CHECK: [[C1:%.+]] = "test.op"([[F1]], [[i]]) : (f64, index) -> f64
// CHECK: [[C2:%.+]] = "test.op"([[F2]], [[i]]) : (f64, index) -> f64
// CHECK: [[C3:%.+]] = "test.op"([[F3]], [[i]]) : (f64, index) -> f64
// CHECK:     catalyst.list_push [[C1]]
// CHECK:     catalyst.list_push [[C2]]
// CHECK:     catalyst.list_push [[C3]]
// CHECK:   [[A3:%.+]] = catalyst.list_pop
// CHECK:   [[A2:%.+]] = catalyst.list_pop
// CHECK:   [[A1:%.+]] = catalyst.list_pop
// CHECK:   qref.custom "Rot"([[A1]], [[A2]], [[A3]])
// CHECK-SAME: adj

module {
  func.func private @param_ordering(%arg0: !qref.reg<?>) {
    %cst = arith.constant 1.000000e-01 : f64
    %cst_0 = arith.constant 2.000000e-01 : f64
    %cst_1 = arith.constant 3.000000e-01 : f64
    qref.adjoint {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c4 step %c1 {
        %0 = "test.op"(%cst, %arg1) : (f64, index) -> f64
        %1 = "test.op"(%cst_0, %arg1) : (f64, index) -> f64
        %2 = "test.op"(%cst_1, %arg1) : (f64, index) -> f64
        %3 = qref.get %arg0[ 0] : !qref.reg<?> -> !qref.bit
        qref.custom "Rot"(%0, %1, %2) %3 : !qref.bit
      }
    }
    return
  }
}

// -----

// Test adjoint of scf.for
// CHECK-LABEL: @adjoint_for_loop_static
// CHECK-DAG: [[start:%.+]] = index.constant 0
// CHECK-DAG: [[stop:%.+]] = index.constant 4
// CHECK-DAG: [[step:%.+]] = index.constant 1
// CHECK: [[reg:%.+]] = qref.alloc( 2) : !qref.reg<{{.*}}>
// CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<{{.*}}> -> !qref.bit
// CHECK: [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<{{.*}}> -> !qref.bit
// CHECK-NOT: qref.adjoint
// CHECK-NOT: catalyst.list_push
// CHECK-NOT: catalyst.list_pop
// CHECK: scf.for {{%.+}} = [[start]] to [[stop]] step [[step]]
// CHECK:   qref.custom "gate2"() [[q0]], [[q1]] adj : !qref.bit, !qref.bit
// CHECK:   qref.custom "gate1"() [[q0]], [[q1]] adj : !qref.bit, !qref.bit
// CHECK: }
// CHECK: qref.dealloc [[reg]]

module {
  func.func public @adjoint_for_loop_static() {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = qref.alloc( 2) : !qref.reg<2>
    %1 = qref.get %0[ 0] : !qref.reg<2> -> !qref.bit
    %2 = qref.get %0[ 1] : !qref.reg<2> -> !qref.bit
    qref.adjoint {
      scf.for %arg0 = %c0 to %c4 step %c1 {
        qref.custom "gate1"() %1, %2 : !qref.bit, !qref.bit
        qref.custom "gate2"() %1, %2 : !qref.bit, !qref.bit
      }
    }
    qref.dealloc %0 : !qref.reg<2>
    return
  }
}

// -----

// Test adjoint of scf.for with dynamic bounds: only the non-constant bound is cached, while
// constant bounds (start, step) are still rematerialized.
// CHECK-LABEL: @adjoint_for_loop_mixed_static_dynamic
// CHECK-DAG: [[start:%.+]] = index.constant 0
// CHECK-DAG: [[step:%.+]] = index.constant 1
// CHECK: [[reg:%.+]] = qref.alloc( 2) : !qref.reg<{{.*}}>
// CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<{{.*}}> -> !qref.bit
// CHECK: [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<{{.*}}> -> !qref.bit
// CHECK-NOT: qref.adjoint
// CHECK: catalyst.list_push %arg0
// CHECK: [[stop:%.+]] = catalyst.list_pop
// CHECK: scf.for {{%.+}} = [[start]] to [[stop]] step [[step]]

module {
  func.func public @adjoint_for_loop_mixed_static_dynamic(%arg0: index) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = qref.alloc( 2) : !qref.reg<2>
    %1 = qref.get %0[ 0] : !qref.reg<2> -> !qref.bit
    %2 = qref.get %0[ 1] : !qref.reg<2> -> !qref.bit
    qref.adjoint {
      scf.for %arg1 = %c0 to %arg0 step %c1 {
        qref.custom "gate1"() %1, %2 : !qref.bit, !qref.bit
        qref.custom "gate2"() %1, %2 : !qref.bit, !qref.bit
      }
    }
    qref.dealloc %0 : !qref.reg<2>
    return
  }
}

// -----

// Test adjoint of scf.while
// CHECK-LABEL: @adjoint_while_loop
// CHECK: [[reg:%.+]] = qref.alloc( 2) : !qref.reg<{{.*}}>
// CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<{{.*}}> -> !qref.bit
// CHECK: [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<{{.*}}> -> !qref.bit
// CHECK-NOT: qref.adjoint
// COM: while loops are converted into a pre-run with empty `do` block to compute the number
// COM: of iterations in runtime, and then a for loop with the gates.
// CHECK: memref.alloca() : memref<index>
// CHECK: scf.while
// CHECK: memref.store
// CHECK: memref.load {{%.+}}[] : memref
// CHECK: scf.for
// CHECK:   qref.custom "gate2"() [[q0]], [[q1]] adj : !qref.bit, !qref.bit
// CHECK:   qref.custom "gate1"() [[q0]], [[q1]] adj : !qref.bit, !qref.bit
// CHECK: }
// CHECK: qref.dealloc [[reg]]

module {
  func.func public @adjoint_while_loop() {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = qref.alloc( 2) : !qref.reg<2>
    %1 = qref.get %0[ 0] : !qref.reg<2> -> !qref.bit
    %2 = qref.get %0[ 1] : !qref.reg<2> -> !qref.bit
    qref.adjoint {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        qref.custom "gate1"() %1, %2 : !qref.bit, !qref.bit
        qref.custom "gate2"() %1, %2 : !qref.bit, !qref.bit
        scf.yield
      }
    }
    qref.dealloc %0 : !qref.reg<2>
    return
  }
}

// -----

// Test adjoint of scf.if
// CHECK-LABEL: @adjoint_if
// CHECK: [[reg:%.+]] = qref.alloc( 2) : !qref.reg<{{.*}}>
// CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<{{.*}}> -> !qref.bit
// CHECK: [[q1:%.+]] = qref.get [[reg]][ 1] : !qref.reg<{{.*}}> -> !qref.bit
// CHECK-NOT: qref.adjoint
// CHECK: scf.if {{%.+}} {
// CHECK:   qref.custom "gate2"() [[q0]], [[q1]] adj : !qref.bit, !qref.bit
// CHECK:   qref.custom "gate1"() [[q0]], [[q1]] adj : !qref.bit, !qref.bit
// CHECK: } else {
// CHECK:   qref.custom "gate4"() [[q0]], [[q1]] adj : !qref.bit, !qref.bit
// CHECK:   qref.custom "gate3"() [[q0]], [[q1]] adj : !qref.bit, !qref.bit
// CHECK: }
// CHECK: qref.dealloc [[reg]]

module {
  func.func public @adjoint_if(%arg0: i1) {
    %0 = qref.alloc( 2) : !qref.reg<2>
    %1 = qref.get %0[ 0] : !qref.reg<2> -> !qref.bit
    %2 = qref.get %0[ 1] : !qref.reg<2> -> !qref.bit
    qref.adjoint {
      scf.if %arg0 {
        qref.custom "gate1"() %1, %2 : !qref.bit, !qref.bit
        qref.custom "gate2"() %1, %2 : !qref.bit, !qref.bit
      } else {
        qref.custom "gate3"() %1, %2 : !qref.bit, !qref.bit
        qref.custom "gate4"() %1, %2 : !qref.bit, !qref.bit
      }
    }
    qref.dealloc %0 : !qref.reg<2>
    return
  }
}

// -----

// Test adjoint of an scf.if whose branch body is itself an scf.if.
// CHECK-LABEL: @adjoint_nested_if
// CHECK: [[reg:%.+]] = qref.alloc( 1) : !qref.reg<{{.*}}>
// CHECK: [[q0:%.+]] = qref.get [[reg]][ 0] : !qref.reg<{{.*}}> -> !qref.bit
// CHECK-NOT: qref.adjoint
// CHECK: scf.if {{%.+}} {
// CHECK:   qref.custom "gate1"() [[q0]] adj : !qref.bit
// CHECK: } else {
// CHECK:   scf.if {{%.+}} {
// CHECK:     qref.custom "gate2"() [[q0]] adj : !qref.bit
// CHECK:   } else {
// CHECK:   }
// CHECK: }
// CHECK: qref.dealloc [[reg]]

module {
  func.func public @adjoint_nested_if(%arg0: i1, %arg1: i1) {
    %0 = qref.alloc( 1) : !qref.reg<1>
    %1 = qref.get %0[ 0] : !qref.reg<1> -> !qref.bit
    qref.adjoint {
      scf.if %arg0 {
        qref.custom "gate1"() %1 : !qref.bit
      } else {
        scf.if %arg1 {
          qref.custom "gate2"() %1 : !qref.bit
        }
      }
    }
    qref.dealloc %0 : !qref.reg<1>
    return
  }
}

// -----

// Test adjoint of scf.index_switch
// CHECK: func.func private @switch_branch_callee.adjoint(%arg0: !qref.reg<{{.*}}>) {
// CHECK:     qref.custom "PauliX"() {{%.+}} adj : !qref.bit
// CHECK-LABEL: @adjoint_index_switch
// CHECK:       catalyst.list_push
// CHECK:       [[popped:%.+]] = catalyst.list_pop
// CHECK:       scf.index_switch [[popped]]
// CHECK:       case 0 {
// CHECK:         "Hadamard"() {{%.+}} adj
// CHECK:         "PauliX"() {{%.+}} adj
// CHECK:       case 1 {
// CHECK:         "T"() {{%.+}} adj
// CHECK:         "PauliY"() {{%.+}} adj
// CHECK:       default {
// CHECK:         call @switch_branch_callee.adjoint({{%.+}}) : (!qref.reg<{{.*}}>) -> ()
// CHECK:         "S"() {{%.+}} adj
// CHECK:         "PauliZ"() {{%.+}} adj

module {
  func.func private @switch_branch_callee(%arg0: !qref.reg<1>) {
    %0 = qref.get %arg0[ 0] : !qref.reg<1> -> !qref.bit
    qref.custom "PauliX"() %0 : !qref.bit
    return
  }
  func.func private @adjoint_index_switch(%arg0: index) {
    %0 = qref.alloc( 1) : !qref.reg<1>
    qref.adjoint {
      scf.index_switch %arg0
      case 0 {
        %1 = qref.get %0[ 0] : !qref.reg<1> -> !qref.bit
        qref.custom "PauliX"() %1 : !qref.bit
        qref.custom "Hadamard"() %1 : !qref.bit
        scf.yield
      }
      case 1 {
        %1 = qref.get %0[ 0] : !qref.reg<1> -> !qref.bit
        qref.custom "PauliY"() %1 : !qref.bit
        qref.custom "T"() %1 : !qref.bit
        scf.yield
      }
      default {
        %1 = qref.get %0[ 0] : !qref.reg<1> -> !qref.bit
        qref.custom "PauliZ"() %1 : !qref.bit
        qref.custom "S"() %1 : !qref.bit
        func.call @switch_branch_callee(%0) : (!qref.reg<1>) -> ()
      }
    }
    return
  }
}

// -----

// Test adjoint lowering through a dynamic single-qubit allocation/deallocation pair.
// CHECK-LABEL: @adjoint_dynamic_qubit
// CHECK-NOT: qref.adjoint
// CHECK: [[aux:%.+]] = qref.alloc_qb : !qref.bit
// CHECK: qref.custom "CNOT"() {{%.+}}, [[aux]] adj : !qref.bit, !qref.bit
// CHECK: qref.dealloc_qb [[aux]] : !qref.bit

module {
  func.func private @adjoint_dynamic_qubit(%arg0: !qref.reg<?>) {
    qref.adjoint {
      %0 = qref.get %arg0[ 0] : !qref.reg<?> -> !qref.bit
      %1 = qref.alloc_qb : !qref.bit
      qref.custom "CNOT"() %0, %1 : !qref.bit, !qref.bit
      qref.dealloc_qb %1 : !qref.bit
    }
    return
  }
}

// -----

// Test adjoint lowering through a dynamic register allocation/deallocation pair.
// CHECK-LABEL: @adjoint_dynamic_register
// CHECK-NOT: qref.adjoint
// CHECK: [[reg:%.+]] = qref.alloc( 1) : !qref.reg<{{.*}}>
// CHECK: [[aux:%.+]] = qref.get [[reg]][ 0] : !qref.reg<{{.*}}> -> !qref.bit
// CHECK: qref.custom "CNOT"() {{%.+}}, [[aux]] adj : !qref.bit, !qref.bit
// CHECK: qref.dealloc [[reg]] : !qref.reg<{{.*}}>

module {
  func.func private @adjoint_dynamic_register(%arg0: !qref.reg<?>) {
    qref.adjoint {
      %0 = qref.get %arg0[ 0] : !qref.reg<?> -> !qref.bit
      %1 = qref.alloc( 1) : !qref.reg<1>
      %2 = qref.get %1[ 0] : !qref.reg<1> -> !qref.bit
      qref.custom "CNOT"() %0, %2 : !qref.bit, !qref.bit
      qref.dealloc %1 : !qref.reg<1>
    }
    return
  }
}

// -----

// Test adjoint lowering through a dynamic register allocation/deallocation pair with dyn size.
// CHECK-LABEL: @adjoint_dynamic_register_dynamic_size
// CHECK-NOT: qref.adjoint
// CHECK: [[reg:%.+]] = qref.alloc(%arg1) : !qref.reg<{{.*}}>
// CHECK: [[aux:%.+]] = qref.get [[reg]][ 0] : !qref.reg<{{.*}}> -> !qref.bit
// CHECK: qref.custom "CNOT"() {{%.+}}, [[aux]] adj : !qref.bit, !qref.bit
// CHECK: qref.dealloc [[reg]] : !qref.reg<{{.*}}>

module {
  func.func private @adjoint_dynamic_register_dynamic_size(%arg0: !qref.reg<?>, %arg1: i64) {
    qref.adjoint {
      %0 = qref.get %arg0[ 0] : !qref.reg<?> -> !qref.bit
      %1 = qref.alloc(%arg1) : !qref.reg<?>
      %2 = qref.get %1[ 0] : !qref.reg<?> -> !qref.bit
      qref.custom "CNOT"() %0, %2 : !qref.bit, !qref.bit
      qref.dealloc %1 : !qref.reg<?>
    }
    return
  }
}

// -----

// CHECK-LABEL: @adjoint_real_matrix_param
// Forward pass: flatten the 2x2 matrix and push every element into the f64 cache.
// CHECK: [[cache:%.+]] = catalyst.list_init : <f64>
// CHECK: scf.for
// CHECK: [[e:%.+]] = tensor.extract {{%.+}}[{{%.+}}, {{%.+}}] : tensor<2x2xf64>
// CHECK: catalyst.list_push [[e]], [[cache]] : <f64>
// Reverse pass: pop the elements and rebuild the 2x2 matrix for the adjointed op.
// CHECK: tensor.empty() : tensor<2x2xf64>
// CHECK: scf.for {{.*}} iter_args
// CHECK: catalyst.list_pop [[cache]] : <f64>
// CHECK: tensor.insert {{%.+}} into {{%.+}}[{{%.+}}, {{%.+}}] : tensor<2x2xf64>
// CHECK: qref.operator "BasisRotation"({{%.+}}: tensor<2x2xf64>) adj

module {
  func.func @adjoint_real_matrix_param(%arg0: !qref.reg<?>) {
    qref.adjoint {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c4 step %c1 {
        %0 = "test.op"() : () -> tensor<2x2xf64>
        %1 = qref.get %arg0[ 0] : !qref.reg<?> -> !qref.bit
        %2 = qref.get %arg0[ 1] : !qref.reg<?> -> !qref.bit
        qref.operator "BasisRotation"(%0: tensor<2x2xf64>) qubits(%1, %2)
          static_data = {}
          param_map = {unitary_matrix = [0]} qubit_map = {wires = [0, 1]}
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @adjoint_integer_tensor_param
// CHECK: catalyst.list_init : <f64>
// CHECK: [[cache:%.+]] = catalyst.list_init : <i64>
// CHECK: scf.for
// CHECK: [[e:%.+]] = tensor.extract {{%.+}}[{{%.+}}] : tensor<2xi1>
// CHECK: [[ei:%.+]] = arith.extui [[e]] : i1 to i64
// CHECK: catalyst.list_push [[ei]], [[cache]] : <i64>
// Reverse pass: pop the i64 elements, truncate back to i1 and rebuild the bitstring.
// CHECK: tensor.empty() : tensor<2xi1>
// CHECK: scf.for {{.*}} iter_args
// CHECK: [[p:%.+]] = catalyst.list_pop [[cache]] : <i64>
// CHECK: [[pi:%.+]] = arith.trunci [[p]] : i64 to i1
// CHECK: tensor.insert [[pi]] into {{%.+}}[{{%.+}}] : tensor<2xi1>
// CHECK: qref.operator "MultiX"({{%.+}}: tensor<2xi1>) adj

module {
  func.func @adjoint_integer_tensor_param(%arg0: !qref.reg<?>) {
    qref.adjoint {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c4 step %c1 {
        %0 = "test.op"() : () -> tensor<2xi1>
        %1 = qref.get %arg0[ 0] : !qref.reg<?> -> !qref.bit
        %2 = qref.get %arg0[ 1] : !qref.reg<?> -> !qref.bit
        qref.operator "MultiX"(%0: tensor<2xi1>) qubits(%1, %2)
          static_data = {}
          param_map = {bitstring = [0]} qubit_map = {wires = [0, 1]}
      }
    }
    return
  }
}

// -----

// Repeated calls to the same function in an adjoint region share one adjoint function.
// CHECK:       func.func private @rotate.adjoint(
// CHECK-NOT:   func.func private @rotate.adjoint(
// CHECK-LABEL: func.func @repeated_call
// CHECK-SAME:  ([[Q:%.+]]: !qref.bit, [[X:%.+]]: f64, [[Y:%.+]]: f64)
// CHECK:       call @rotate.adjoint([[Q]], [[Y]])
// CHECK:       call @rotate.adjoint([[Q]], [[X]])
// CHECK: return

module {
  func.func private @rotate(%arg0: !qref.bit, %arg1: f64) {
    qref.custom "RX"(%arg1) %arg0 : !qref.bit
    return
  }
  func.func @repeated_call(%arg0: !qref.bit, %arg1: f64, %arg2: f64) {
    qref.adjoint {
      func.call @rotate(%arg0, %arg1) : (!qref.bit, f64) -> ()
      func.call @rotate(%arg0, %arg2) : (!qref.bit, f64) -> ()
    }
    return
  }
}

// -----

// Resource hints on SCF ops must survive adjoint-lowering recreation.
// CHECK-LABEL: @adjoint_preserves_scf_hints
// CHECK-NOT: qref.adjoint
// CHECK-DAG: catalyst.estimated_iterations = 1.000000e+01 : f64
// CHECK-DAG: catalyst.estimated_iterations = 2.500000e+00 : f64
// CHECK-DAG: catalyst.estimated_probability = 7.500000e-01 : f64
// CHECK-DAG: catalyst.estimated_probabilities = [2.000000e-01, 3.000000e-01]

module {
  func.func public @adjoint_preserves_scf_hints(%arg0: i1, %arg1: index, %arg2: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = qref.alloc( 1) : !qref.reg<1>
    %1 = qref.get %0[ 0] : !qref.reg<1> -> !qref.bit
    qref.adjoint {
      scf.for %arg3 = %c0 to %arg1 step %c1 {
        qref.custom "Hadamard"() %1 : !qref.bit
      } {catalyst.estimated_iterations = 1.000000e+01 : f64}
      scf.while : () -> () {
        scf.condition(%arg0)
      } do {
        qref.custom "PauliX"() %1 : !qref.bit
        scf.yield
      } attributes {catalyst.estimated_iterations = 2.500000e+00 : f64}
      scf.if %arg0 {
        qref.custom "PauliY"() %1 : !qref.bit
      } {catalyst.estimated_probability = 7.500000e-01 : f64}
      scf.index_switch %arg2 {catalyst.estimated_probabilities = [2.000000e-01, 3.000000e-01]}
      case 0 {
        qref.custom "PauliZ"() %1 : !qref.bit
        scf.yield
      }
      case 1 {
        scf.yield
      }
      default {
      }
    }
    qref.dealloc %0 : !qref.reg<1>
    return
  }
}

// -----

// Every QuantumGate implementation must replay controls computed inside the reversed loop,
// including gates with no ordinary parameters.
// CHECK-LABEL: @adjoint_dynamic_controls
// CHECK: [[controls:%.+]] = catalyst.list_init : <i1>
// CHECK: scf.for
// CHECK: [[cv:%.+]] = arith.cmpi eq,
// CHECK-COUNT-8: catalyst.list_push [[cv]], [[controls]] : <i1>
// CHECK: scf.for
// CHECK: [[cv7:%.+]] = catalyst.list_pop [[controls]] : <i1>
// CHECK: qref.gphase({{%.+}}) adj ctrls({{%.+}}) ctrlvals([[cv7]])
// CHECK: [[cv6:%.+]] = catalyst.list_pop [[controls]] : <i1>
// CHECK: qref.unitary({{%.+}} : tensor<2x2xcomplex<f64>>) {{%.+}} adj ctrls({{%.+}}) ctrlvals([[cv6]])
// CHECK: [[cv5:%.+]] = catalyst.list_pop [[controls]] : <i1>
// CHECK: qref.pcphase({{%.+}}, dim : 1) {{%.+}} adj ctrls({{%.+}}) ctrlvals([[cv5]])
// CHECK: [[cv4:%.+]] = catalyst.list_pop [[controls]] : <i1>
// CHECK: qref.multirz({{%.+}}) {{%.+}} adj ctrls({{%.+}}) ctrlvals([[cv4]])
// CHECK: [[cv3:%.+]] = catalyst.list_pop [[controls]] : <i1>
// CHECK: qref.paulirot ["Z"]({{%.+}}) {{%.+}} adj ctrls({{%.+}}) ctrlvals([[cv3]])
// CHECK: [[cv2:%.+]] = catalyst.list_pop [[controls]] : <i1>
// CHECK: qref.operator "Probe"({{%.+}}: f64) adj qubits({{%.+}})
// CHECK: ctrls({{%.+}}) ctrl_vals([[cv2]])
// CHECK: [[cv1:%.+]] = catalyst.list_pop [[controls]] : <i1>
// CHECK: qref.custom "PauliX"() {{%.+}} adj ctrls({{%.+}}) ctrlvals([[cv1]])
// CHECK: [[cv0:%.+]] = catalyst.list_pop [[controls]] : <i1>
// CHECK: qref.custom "RX"({{%.+}}) {{%.+}} adj ctrls({{%.+}}) ctrlvals([[cv0]])
// CHECK: catalyst.list_dealloc [[controls]] : <i1>

module {
  func.func @adjoint_dynamic_controls(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: f64, %arg3: tensor<2x2xcomplex<f64>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    qref.adjoint {
      scf.for %arg4 = %c0 to %c2 step %c1 {
        %0 = arith.cmpi eq, %arg4, %c0 : index
        qref.custom "RX"(%arg2) %arg0 ctrls(%arg1) ctrlvals(%0) : !qref.bit ctrls !qref.bit
        qref.custom "PauliX"() %arg0 ctrls(%arg1) ctrlvals(%0) : !qref.bit ctrls !qref.bit
        qref.operator "Probe"(%arg2: f64) qubits(%arg0)
          ctrls(%arg1) ctrl_vals(%0)
        qref.paulirot ["Z"](%arg2) %arg0 ctrls(%arg1) ctrlvals(%0) : !qref.bit ctrls !qref.bit
        qref.multirz(%arg2) %arg0 ctrls(%arg1) ctrlvals(%0) : !qref.bit ctrls !qref.bit
        qref.pcphase(%arg2, dim : 1) %arg0 ctrls(%arg1) ctrlvals(%0) : !qref.bit ctrls !qref.bit
        qref.unitary(%arg3 : tensor<2x2xcomplex<f64>>) %arg0 ctrls(%arg1) ctrlvals(%0) : !qref.bit ctrls !qref.bit
        qref.gphase(%arg2) ctrls(%arg1) ctrlvals(%0) : ctrls !qref.bit
      }
    }
    return
  }
}

// -----

// Parameters and control values use independent typed stacks.
// External and constant controls must remain in their original positions without being cached.
// CHECK-LABEL: @adjoint_mixed_controls
// CHECK-SAME: [[external:%[^:, ]+]]: i1)
// CHECK: [[true:%.+]] = arith.constant true
// CHECK: [[params:%.+]] = catalyst.list_init : <f64>
// CHECK: [[integers:%.+]] = catalyst.list_init : <i64>
// CHECK: [[controls:%.+]] = catalyst.list_init : <i1>
// CHECK: scf.for
// CHECK: [[forward:%.+]] = arith.index_cast
// CHECK: [[theta:%.+]] = arith.sitofp [[forward]] : i64 to f64
// CHECK: [[first:%.+]] = arith.cmpi eq,
// CHECK: [[last:%.+]] = arith.cmpi ne,
// CHECK: catalyst.list_push [[theta]], [[params]] : <f64>
// CHECK: catalyst.list_push [[forward]], [[integers]] : <i64>
// CHECK: catalyst.list_push [[first]], [[controls]] : <i1>
// CHECK: catalyst.list_push [[last]], [[controls]] : <i1>
// CHECK-NOT: catalyst.list_push {{%.+}}, [[controls]]
// CHECK: scf.for
// CHECK: [[last:%.+]] = catalyst.list_pop [[controls]] : <i1>
// CHECK: [[first:%.+]] = catalyst.list_pop [[controls]] : <i1>
// CHECK: [[forward:%.+]] = catalyst.list_pop [[integers]] : <i64>
// CHECK: [[theta:%.+]] = catalyst.list_pop [[params]] : <f64>
// CHECK-NOT: catalyst.list_pop
// CHECK: qref.operator "Probe"([[theta]]: f64) adj qubits({{%.+}})
// CHECK: UID(1) forward([[forward]]: i64)
// CHECK: ctrls({{%.+}}) ctrl_vals([[first]], [[external]], [[last]], [[true]])

module {
  func.func @adjoint_mixed_controls(%arg0: !qref.reg<?>, %arg1: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %true = arith.constant true
    qref.adjoint {
      scf.for %arg2 = %c0 to %c2 step %c1 {
        %0 = arith.index_cast %arg2 : index to i64
        %1 = arith.sitofp %0 : i64 to f64
        %2 = arith.cmpi eq, %arg2, %c0 : index
        %3 = arith.cmpi ne, %arg2, %c0 : index
        %4 = qref.get %arg0[ 0] : !qref.reg<?> -> !qref.bit
        %5 = qref.get %arg0[ 1] : !qref.reg<?> -> !qref.bit
        %6 = qref.get %arg0[ 2] : !qref.reg<?> -> !qref.bit
        %7 = qref.get %arg0[ 3] : !qref.reg<?> -> !qref.bit
        %8 = qref.get %arg0[ 4] : !qref.reg<?> -> !qref.bit
        qref.operator "Probe"(%1: f64) qubits(%4)
          UID(1) forward(%0: i64)
          ctrls(%5, %6, %7, %8) ctrl_vals(%2, %arg1, %3, %true)
      }
    }
    return
  }
}

// -----

// Controls defined outside the adjoint or at its immediate top level remain visible to the
// reverse pass. Reuse them without pushing or popping the control-value stack.
// CHECK-LABEL: @adjoint_available_controls
// CHECK-SAME: [[external:%[^:, ]+]]: i1,
// CHECK: [[true:%.+]] = arith.constant true
// CHECK: [[controls:%.+]] = catalyst.list_init : <i1>
// CHECK-NOT: catalyst.list_push
// CHECK: [[local:%.+]] = arith.cmpi sgt,
// CHECK-NOT: catalyst.list_pop
// CHECK: qref.custom "PauliX"() {{%.+}} adj ctrls({{%.+}}) ctrlvals([[external]], [[local]], [[true]])
// CHECK: catalyst.list_dealloc [[controls]] : <i1>

module {
  func.func @adjoint_available_controls(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: !qref.bit, %arg3: !qref.bit, %arg4: i1, %arg5: index) {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    qref.adjoint {
      %0 = arith.cmpi sgt, %arg5, %c0 : index
      qref.custom "PauliX"() %arg0 ctrls(%arg1, %arg2, %arg3) ctrlvals(%arg4, %0, %true) : !qref.bit ctrls !qref.bit, !qref.bit, !qref.bit
    }
    return
  }
}
