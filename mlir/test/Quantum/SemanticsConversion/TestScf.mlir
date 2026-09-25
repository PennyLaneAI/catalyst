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

// RUN: quantum-opt --convert-to-reference-semantics --split-input-file -verify-diagnostics %s

// Region-bearing control flow outside the supported set (scf.if / scf.for / scf.while /
// scf.index_switch) has no rule for controlling its body, so it is rejected rather than silently
// left uncontrolled. scf.execute_region stands in for any such op here.
func.func @ctrl_unsupported_scf(%ctrl: !quantum.bit, %q: !quantum.bit) -> !quantum.bit {
  %true = arith.constant true
  %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    // expected-error @+1 {{scf.execute_region is not supported}}
    %r = scf.execute_region -> !quantum.bit {
      %h = quantum.custom "Hadamard"() %arg0 : !quantum.bit
      scf.yield %h : !quantum.bit
    }
    quantum.yield %r : !quantum.bit
  }
  return %outc : !quantum.bit
}

// -----

// The same restriction applies outside any controlled region.
func.func @execute_region_in_body(%q: !quantum.bit) -> !quantum.bit {
  // expected-error @+1 {{scf.execute_region is not supported}}
  %r = scf.execute_region -> !quantum.bit {
    %h = quantum.custom "Hadamard"() %q : !quantum.bit
    scf.yield %h : !quantum.bit
  }
  return %r : !quantum.bit
}
