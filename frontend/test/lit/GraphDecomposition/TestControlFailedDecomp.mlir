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

// RUN: not --crash catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=Foo=1.0})' %s 2>&1 | FileCheck %s

// CHECK: Decomposition rule not found for operator 'id: C(Bar){}{wires:1}{}'
func.func @circuit(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  %out, %outc = quantum.custom "Bar"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

// A decomposition for the base Bar op. It is never applicable to the controlled
// Bar above, illustrating that Bar and C(Bar) require separate rules.
func.func private @bar_to_foo(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "Bar{}{wires:1}{}",
    resources = {operations = {"Foo{}{wires:1}{}" = 1 : i64}} } {
  %o = quantum.custom "Foo"() %q : !quantum.bit
  return %o : !quantum.bit
}
