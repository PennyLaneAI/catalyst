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

// RUN: quantum-opt --decompose-lowering --split-input-file -verify-diagnostics %s | FileCheck %s


/// Self-adjoint basis gate: Adjoint(H) -> H (the modifier is dropped).
///
// CHECK-LABEL: func.func @self_adjoint(
// CHECK-SAME:  %[[Q:.*]]: !quantum.bit
func.func @self_adjoint(%q: !quantum.bit) -> !quantum.bit {
  // The adjoint is consumed: a plain Hadamard remains (note the absence of `adj` before the `:`).
  // CHECK: %[[O:.*]] = quantum.custom "Hadamard"() %[[Q]] : !quantum.bit
  // CHECK: return %[[O]]
  %out = quantum.custom "Hadamard"() %q adj : !quantum.bit
  return %out : !quantum.bit
}

func.func private @adj_h(%q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "Adjoint(Hadamard{}{wires:1}{})", llvm.linkage = #llvm.linkage<internal>} {
  %o = quantum.custom "Hadamard"() %q : !quantum.bit
  return %o : !quantum.bit
}

