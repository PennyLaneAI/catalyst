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

// RUN: quantum-opt --convert-to-value-semantics --split-input-file -verify-diagnostics %s

// Region-bearing control flow outside the supported set (scf.if / scf.for / scf.while /
// scf.index_switch) has no conversion rule, so it is rejected rather than silently left
// unconverted. scf.execute_region stands in for any such op here.
func.func @execute_region_in_body() attributes {quantum.node} {
  %a = qref.alloc(1) : !qref.reg<1>
  %q0 = qref.get %a[0] : !qref.reg<1> -> !qref.bit

  // expected-error @+1 {{scf.execute_region is not supported}}
  scf.execute_region {
    qref.custom "Hadamard"() %q0 : !qref.bit
    scf.yield
  }

  qref.dealloc %a : !qref.reg<1>
  return
}
