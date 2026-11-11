// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: split-file %s %t
// RUN: catalyst --tool=opt --empty --emit-bytecode %t/library.mlir -o %t/library.mlirbc
// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=RX=1.0 bytecode-rules="%t/library.mlirbc"})' %t/circuit.mlir | FileCheck %s
// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=RX=1.0 bytecode-rules="%t/library.mlirbc"})' %t/empty.mlir | FileCheck %s --check-prefix=EMPTY

// Importing a library must preserve calls to helpers with colliding names and types.
// Unused rules and helpers must not enter the circuit or undergo semantics conversion.
// CHECK-NOT: @jit_rules_module
// CHECK-NOT: @unused
// CHECK-LABEL: func.func @circuit
// CHECK: call @__builtin_collision
// CHECK-LABEL: func.func private @helper(
// CHECK-SAME: i64
// CHECK-LABEL: func.func private @helper_0(
// CHECK-SAME: f64
// CHECK-LABEL: func.func private @__builtin_collision(
// CHECK: call @helper_0
// CHECK: quantum.custom "RX"
// CHECK-NOT: @unused
// CHECK-NOT: @jit_rules_module

// EMPTY-LABEL: func.func @empty_circuit
// EMPTY-NEXT: return
// EMPTY-NEXT: }
// EMPTY-NEXT: }

//--- empty.mlir
func.func @empty_circuit() {
    return
}

//--- circuit.mlir
func.func @circuit(%angle: f64) {
    %reg = quantum.alloc(1) : !quantum.reg
    %q = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %out = quantum.custom "CollisionGate"(%angle) %q : !quantum.bit
    %result = quantum.insert %reg[0], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %result : !quantum.reg
    return
}

func.func private @helper(%n: i64) -> i64 {
    return %n : i64
}

//--- library.mlir
func.func @jit_rules_module() attributes {llvm.emit_c_interface} {
    return
}

func.func private @helper(%angle: f64) -> f64 {
    %neg = arith.negf %angle : f64
    return %neg : f64
}

func.func private @__builtin_collision(%angle: f64, %q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "CollisionGate{0:[f64]}{wires:1}{}",
                frontend_name = "collision_to_rx",
                resources = {operations = {"RX{0:[f64]}{wires:1}{}" = 1 : i64}}} {
    %neg = func.call @helper(%angle) : (f64) -> f64
    %out = quantum.custom "RX"(%neg) %q : !quantum.bit
    return %out : !quantum.bit
}

func.func private @unused(%q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "RX{0:[f64]}{wires:1}{}",
                frontend_name = "rx_to_collision",
                resources = {operations = {"CollisionGate{0:[f64]}{wires:1}{}" = 1 : i64}}} {
    %out = func.call @unused_helper(%q) : (!quantum.bit) -> !quantum.bit
    return %out : !quantum.bit
}

// A rejected alternative can contain control flow unsupported by reference-semantics conversion.
func.func private @unused_helper(%q: !quantum.bit) -> !quantum.bit {
    %result = scf.execute_region -> !quantum.bit {
        %angle = arith.constant 0.1 : f64
        %out = quantum.custom "CollisionGate"(%angle) %q : !quantum.bit
        scf.yield %out : !quantum.bit
    }
    return %result : !quantum.bit
}
