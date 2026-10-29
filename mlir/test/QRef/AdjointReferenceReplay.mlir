// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --adjoint-lowering --split-input-file %s | FileCheck %s --implicit-check-not="qref.adjoint" --implicit-check-not="!quantum.bit" --implicit-check-not="!quantum.reg"
// RUN: quantum-opt --convert-to-value-semantics --modifiers-lowering --convert-to-reference-semantics --verify-no-quantum-use-after-free --split-input-file %s -o /dev/null

// A reference defined in a loop can be reused and captured by a conditional. Each reversed
// use must recover its own wire index, with controls popped before targets.
// CHECK-LABEL: @shared_dynamic_references
// CHECK-SAME: [[REG:%[^:]+]]: !qref.reg<?>
// CHECK: catalyst.list_init : <f64>
// CHECK: catalyst.list_init : <i64>
// CHECK: [[WIRES:%.+]] = catalyst.list_init : <i64>
// CHECK: scf.for
// CHECK: [[WIRE:%.+]] = arith.index_cast
// CHECK: [[CONTROL_WIRE:%.+]] = arith.addi [[WIRE]], [[WIRE]]
// CHECK: catalyst.list_push [[WIRE]], [[WIRES]]
// CHECK: scf.if
// CHECK: catalyst.list_push [[WIRE]], [[WIRES]]
// CHECK: catalyst.list_push [[CONTROL_WIRE]], [[WIRES]]
// CHECK: catalyst.list_push [[WIRE]], [[WIRES]]
// CHECK: scf.for
// CHECK: [[H_WIRE:%.+]] = catalyst.list_pop [[WIRES]]
// CHECK: [[H:%.+]] = qref.get [[REG]][[[H_WIRE]]]
// CHECK: qref.custom "Hadamard"() [[H]] adj
// CHECK: scf.if
// CHECK: [[C_WIRE:%.+]] = catalyst.list_pop [[WIRES]]
// CHECK: [[C:%.+]] = qref.get [[REG]][[[C_WIRE]]]
// CHECK: [[T_WIRE:%.+]] = catalyst.list_pop [[WIRES]]
// CHECK: [[T:%.+]] = qref.get [[REG]][[[T_WIRE]]]
// CHECK: qref.custom "T"() [[T]] adj ctrls([[C]])
// CHECK: [[S_WIRE:%.+]] = catalyst.list_pop [[WIRES]]
// CHECK: [[S:%.+]] = qref.get [[REG]][[[S_WIRE]]]
// CHECK: qref.custom "S"() [[S]] adj

func.func @shared_dynamic_references(%reg: !qref.reg<?>, %stop: index, %cond: i1) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %true = arith.constant true
  qref.adjoint {
    scf.for %iv = %zero to %stop step %one {
      %wire = arith.index_cast %iv : index to i64
      %next = arith.addi %wire, %wire : i64
      %q = qref.get %reg[%wire] : !qref.reg<?>, i64 -> !qref.bit
      %ctrl = qref.get %reg[%next] : !qref.reg<?>, i64 -> !qref.bit
      qref.custom "S"() %q : !qref.bit
      scf.if %cond {
        qref.custom "T"() %q ctrls(%ctrl) ctrlvals(%true) : !qref.bit ctrls !qref.bit
      }
      qref.custom "Hadamard"() %q : !qref.bit
    }
  }
  return
}

// -----

// Quantum callees have no quantum results in reference semantics. Classical arguments from
// inside a loop must be replayed from the cache instead of referencing the forward loop body.
// CHECK-LABEL: @rotate.adjoint
// CHECK: qref.custom "RX"({{%.+}}) {{%.+}} adj
// CHECK-LABEL: @cached_call_arguments
// CHECK-SAME: [[Q:%[^:]+]]: !qref.bit
// CHECK: [[PARAMS:%.+]] = catalyst.list_init : <f64>
// CHECK: scf.for
// CHECK: [[ANGLE:%.+]] = arith.sitofp
// CHECK: catalyst.list_push [[ANGLE]], [[PARAMS]]
// CHECK: scf.for
// CHECK: [[REPLAY:%.+]] = catalyst.list_pop [[PARAMS]]
// CHECK: func.call @rotate.adjoint([[Q]], [[REPLAY]]) : (!qref.bit, f64) -> ()

func.func private @rotate(%q: !qref.bit, %angle: f64) {
  qref.custom "RX"(%angle) %q : !qref.bit
  return
}
func.func @cached_call_arguments(%q: !qref.bit, %stop: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  qref.adjoint {
    scf.for %iv = %zero to %stop step %one {
      %i = arith.index_cast %iv : index to i64
      %angle = arith.sitofp %i : i64 to f64
      func.call @rotate(%q, %angle) : (!qref.bit, f64) -> ()
    }
  }
  return
}

// -----

// Allocation size is classical loop-dependent data. Reallocate before rematerializing a
// qubit reference into the scratch register, then release that same register.
// CHECK-LABEL: @nested_allocation_size
// CHECK-SAME: [[Q:%[^:]+]]: !qref.bit
// CHECK: catalyst.list_init : <i64>
// CHECK: [[WIRES:%.+]] = catalyst.list_init : <i64>
// CHECK: scf.for
// CHECK: [[N:%.+]] = arith.index_cast
// CHECK: catalyst.list_push [[N]], [[WIRES]]
// CHECK: scf.for
// CHECK: [[SIZE:%.+]] = catalyst.list_pop [[WIRES]]
// CHECK: [[SCRATCH:%.+]] = qref.alloc([[SIZE]]) : !qref.reg<?>
// CHECK: [[AUX:%.+]] = qref.get [[SCRATCH]][ 0]
// CHECK: qref.custom "CNOT"() [[Q]], [[AUX]] adj
// CHECK: qref.dealloc [[SCRATCH]] : !qref.reg<?>

func.func @nested_allocation_size(%q: !qref.bit, %stop: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  qref.adjoint {
    scf.for %iv = %zero to %stop step %one {
      %n = arith.index_cast %iv : index to i64
      %scratch = qref.alloc(%n) : !qref.reg<?>
      %aux = qref.get %scratch[0] : !qref.reg<?> -> !qref.bit
      qref.custom "CNOT"() %q, %aux : !qref.bit, !qref.bit
      qref.dealloc %scratch : !qref.reg<?>
    }
  }
  return
}

// -----

// The while condition region executes once more than the body. Reverse that final execution
// before replaying the body/condition pairs, even though neither region yields qubits.
// CHECK-LABEL: @quantum_while_condition
// CHECK-SAME: [[Q:%[^:]+]]: !qref.bit
// CHECK: scf.while
// CHECK-NOT: qref.custom
// CHECK: qref.custom "S"() [[Q]] adj
// CHECK: scf.for
// CHECK: qref.custom "T"() [[Q]] adj
// CHECK-NEXT: qref.custom "S"() [[Q]] adj

func.func @quantum_while_condition(%q: !qref.bit, %cond: i1) {
  qref.adjoint {
    scf.while : () -> () {
      qref.custom "S"() %q : !qref.bit
      scf.condition(%cond)
    } do {
      qref.custom "T"() %q : !qref.bit
      scf.yield
    }
  }
  return
}

// -----

// A reference-semantics if need not have an else region or results.
// CHECK-LABEL: @then_only_adjoint
// CHECK-SAME: [[Q:%[^:]+]]: !qref.bit
// CHECK: [[PARAMS:%.+]] = catalyst.list_init : <f64>
// CHECK: scf.if
// CHECK: [[ANGLE:%.+]] = arith.negf
// CHECK: catalyst.list_push [[ANGLE]], [[PARAMS]]
// CHECK: scf.if
// CHECK: [[ANGLE:%.+]] = catalyst.list_pop [[PARAMS]]
// CHECK: qref.custom "RX"([[ANGLE]]) [[Q]] adj

func.func @then_only_adjoint(%q: !qref.bit, %cond: i1, %theta: f64) {
  qref.adjoint {
    scf.if %cond {
      %angle = arith.negf %theta : f64
      qref.custom "RX"(%angle) %q : !qref.bit
    }
  }
  return
}

// -----

// Global phase is quantum even when a callee has no reference arguments or results.
// CHECK-LABEL: @phase.adjoint
// CHECK: qref.gphase({{%.+}}) adj
// CHECK-LABEL: @adjoint_phase_call
// CHECK-SAME: [[THETA:%[^:]+]]: f64
// CHECK-NOT: call @phase(
// CHECK: call @phase.adjoint([[THETA]]) : (f64) -> ()
func.func private @phase(%theta: f64) {
  qref.gphase(%theta)
  return
}

func.func @adjoint_phase_call(%theta: f64) {
  qref.adjoint {
    func.call @phase(%theta) : (f64) -> ()
  }
  return
}
