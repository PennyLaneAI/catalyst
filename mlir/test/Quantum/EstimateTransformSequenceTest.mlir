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

// RUN: quantum-opt -estimate-transform-sequence --split-input-file -verify-diagnostics %s -o /dev/null | FileCheck %s

// A graph-decomposition pass, first in the sequence, is projected without being applied.

// CHECK:      "level": 0
// CHECK-NEXT: "pass": null
// CHECK:        "decomposed": {
// CHECK:            "Toffoli": 1
// CHECK:      "level": 1
// CHECK-NEXT: "pass": "graph-decomposition"
// CHECK:        "decomposed": {
// CHECK:            "toffoli_to_clifford_t(Toffoli{}{wires:3}{})": 1
// CHECK:        "toffoli_to_clifford_t(Toffoli{}{wires:3}{})": {
// CHECK:            "CNOT": 6
module {
  func.func public @decomposed() attributes {quantum.node} {
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    %4:3 = quantum.custom "Toffoli"() %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit
    %5 = quantum.insert %0[ 0], %4#0 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 1], %4#1 : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[ 2], %4#2 : !quantum.reg, !quantum.bit
    quantum.dealloc %7 : !quantum.reg
    return
  }

  func.func private @toffoli_to_clifford_t(%q: !quantum.bit) -> !quantum.bit attributes {
      target_gate = "Toffoli{}{wires:3}{}",
      resources = {operations = {"Adjoint(T){}{wires:1}{}" = 3 : i64, "CNOT{}{wires:2}{}" = 6 : i64, "Hadamard{}{wires:1}{}" = 2 : i64, "T{}{wires:1}{}" = 4 : i64}}} {
    return %q : !quantum.bit
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      %0 = transform.apply_registered_pass "graph-decomposition" with options = {"gate-set" = ["CNOT=1.0", "Hadamard=1.0", "T=1.0", "Adjoint(T)=1.0"]} to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      transform.yield
    }
  }
}

// -----

// Loop and branch hints, and loops with an unknown number of iterations, are kept by the
// projection.

// CHECK:      "level": 1
// CHECK:        "for_loop_1": {
// CHECK:              "toffoli_to_clifford_t(Toffoli{}{wires:3}{})": 1
// CHECK:        "hinted": {
// CHECK:            "dynamic": {
// CHECK-NEXT:         "dyn_for_loop_1": "{{0x[0-9a-f]+}}"
// CHECK:            "static": {
// CHECK-NEXT:         "for_loop_1": 2.5
// CHECK-NEXT:         "toffoli_to_clifford_t(Toffoli{}{wires:3}{})": 0.25
module {
  func.func public @hinted(%a: !quantum.bit, %b: !quantum.bit, %c: !quantum.bit, %n: index, %m: index, %cond: i1) attributes {quantum.node} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %q:3 = scf.for %i = %c0 to %n step %c1 iter_args(%x = %a, %y = %b, %z = %c) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
      %t:3 = quantum.custom "Toffoli"() %x, %y, %z : !quantum.bit, !quantum.bit, !quantum.bit
      scf.yield %t#0, %t#1, %t#2 : !quantum.bit, !quantum.bit, !quantum.bit
    } {catalyst.estimated_iterations = 2.5 : f64}
    %p = scf.for %i = %c0 to %m step %c1 iter_args(%x = %q#0) -> (!quantum.bit) {
      %t = quantum.custom "T"() %x : !quantum.bit
      scf.yield %t : !quantum.bit
    }
    %r:3 = scf.if %cond -> (!quantum.bit, !quantum.bit, !quantum.bit) {
      %t:3 = quantum.custom "Toffoli"() %p, %q#1, %q#2 : !quantum.bit, !quantum.bit, !quantum.bit
      scf.yield %t#0, %t#1, %t#2 : !quantum.bit, !quantum.bit, !quantum.bit
    } else {
      scf.yield %p, %q#1, %q#2 : !quantum.bit, !quantum.bit, !quantum.bit
    } {catalyst.estimated_probability = 0.25 : f64}
    return
  }

  func.func private @toffoli_to_clifford_t(%q: !quantum.bit) -> !quantum.bit attributes {
      target_gate = "Toffoli{}{wires:3}{}",
      resources = {operations = {"Adjoint(T){}{wires:1}{}" = 3 : i64, "CNOT{}{wires:2}{}" = 6 : i64, "Hadamard{}{wires:1}{}" = 2 : i64, "T{}{wires:1}{}" = 4 : i64}}} {
    return %q : !quantum.bit
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      %0 = transform.apply_registered_pass "graph-decomposition" with options = {"gate-set" = ["CNOT=1.0", "Hadamard=1.0", "T=1.0", "Adjoint(T)=1.0"]} to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      transform.yield
    }
  }
}

// -----

// A pass with a registered resource effect, such as symbol-dce, is estimated by its effect.

// CHECK:      "level": 1
// CHECK-NEXT: "pass": "symbol-dce"
// CHECK:        "registered_effect": {
// CHECK:            "Hadamard": 1
module {
  func.func public @registered_effect(%q: !quantum.bit) -> !quantum.bit attributes {quantum.node} {
    %0 = quantum.custom "Hadamard"() %q : !quantum.bit
    return %0 : !quantum.bit
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      %0 = transform.apply_registered_pass "symbol-dce" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      transform.yield
    }
  }
}

// -----

// Without a transform sequence, only level 0 is written.

// CHECK:      "level": 0
// CHECK:        "no_sequence": {
// CHECK-NOT:  "level": 1
func.func public @no_sequence() attributes {quantum.node} {
  return
}

// -----

module {
  func.func public @unknown_pass() attributes {quantum.node} {
    return
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      // expected-error@+1 {{no resource effect is declared for the pass 'canonicalize'}}
      %0 = transform.apply_registered_pass "canonicalize" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      transform.yield
    }
  }
}

// -----

module {
  func.func public @second_decomposition() attributes {quantum.node} {
    return
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      %0 = transform.apply_registered_pass "graph-decomposition" with options = {"gate-set" = ["T=1.0"]} to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      // expected-error@+1 {{resource estimation only supports graph-decomposition as the first pass of the transform sequence}}
      %1 = transform.apply_registered_pass "graph-decomposition" with options = {"gate-set" = ["T=1.0"]} to %0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      transform.yield
    }
  }
}

// -----

module {
  func.func public @unregistered_pass() attributes {quantum.node} {
    return
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      // expected-error@+1 {{'not-a-pass' is not a registered pass}}
      %0 = transform.apply_registered_pass "not-a-pass" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      transform.yield
    }
  }
}

// -----

module {
  func.func public @unsupported_transform() attributes {quantum.node} {
    return
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      // expected-error@+1 {{resource estimation only supports transform.apply_registered_pass}}
      transform.print %arg0 : !transform.op<"builtin.module">
      transform.yield
    }
  }
}
