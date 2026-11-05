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

// RUN: quantum-opt --annotate-invalid-gradient-functions --split-input-file --verify-diagnostics %s

// Generic measurements reject automatic differentiation, but permit finite differences.
func.func @measurement(%angle: f64) -> f64 {
  %reg = quantum.alloc(1) : !quantum.reg
  %q = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
  %m, %out = quantum.measurement "MidCircuit"(%angle: f64) qubits(%q) : f64, !quantum.bit
  %out_reg = quantum.insert %reg[0], %out : !quantum.reg, !quantum.bit
  quantum.dealloc %out_reg : !quantum.reg
  return %m : f64
}

%angle = arith.constant 0.0 : f64
gradient.grad "fd" @measurement(%angle) : (f64) -> f64
// expected-error@+1 {{An operation without a valid gradient was found}}
gradient.grad "auto" @measurement(%angle) : (f64) -> f64

// -----

// The annotation propagates to callers of a generic measurement.
func.func @measurement(%angle: f64) -> f64 {
  %reg = quantum.alloc(1) : !quantum.reg
  %q = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
  %m, %out = quantum.measurement "MidCircuit"(%angle: f64) qubits(%q) : f64, !quantum.bit
  %out_reg = quantum.insert %reg[0], %out : !quantum.reg, !quantum.bit
  quantum.dealloc %out_reg : !quantum.reg
  return %m : f64
}

func.func @wrapper(%angle: f64) -> f64 {
  %result = func.call @measurement(%angle) : (f64) -> f64
  return %result : f64
}

%angle = arith.constant 0.0 : f64
gradient.grad "fd" @wrapper(%angle) : (f64) -> f64
// expected-error@+1 {{An operation without a valid gradient was found}}
gradient.grad "auto" @wrapper(%angle) : (f64) -> f64

// -----

// The shared interface also covers generic measurements on quantum references.
func.func @measurement(%angle: f64) -> f64 {
  %q = qref.alloc_qb : !qref.bit
  %m = qref.measurement "MidCircuit"(%angle: f64) qubits(%q) : f64
  qref.dealloc_qb %q : !qref.bit
  return %m : f64
}

func.func @wrapper(%angle: f64) -> f64 {
  %result = func.call @measurement(%angle) : (f64) -> f64
  return %result : f64
}

%angle = arith.constant 0.0 : f64
gradient.grad "fd" @wrapper(%angle) : (f64) -> f64
// expected-error@+1 {{An operation without a valid gradient was found}}
gradient.grad "auto" @wrapper(%angle) : (f64) -> f64

// -----

// Terminal measurements remain eligible for automatic differentiation.
func.func @terminal(%angle: f64) -> f64 {
  %reg = quantum.alloc(1) : !quantum.reg
  %q = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
  %out = quantum.custom "RX"(%angle) %q : !quantum.bit
  %obs = quantum.namedobs %out[PauliZ] : !quantum.obs
  %result = quantum.expval %obs : f64
  %out_reg = quantum.insert %reg[0], %out : !quantum.reg, !quantum.bit
  quantum.dealloc %out_reg : !quantum.reg
  return %result : f64
}

func.func @reference_terminal(%angle: f64) -> f64 {
  %q = qref.alloc_qb : !qref.bit
  qref.custom "RX"(%angle) %q : !qref.bit
  %obs = qref.namedobs %q[PauliZ] : !quantum.obs
  %result = quantum.expval %obs : f64
  qref.dealloc_qb %q : !qref.bit
  return %result : f64
}

%angle = arith.constant 0.0 : f64
gradient.grad "auto" @terminal(%angle) : (f64) -> f64
gradient.grad "auto" @reference_terminal(%angle) : (f64) -> f64
