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

// RUN: quantum-opt --decompose-lowering --mlir-print-ir-before=convert-to-value-semantics %s -o /dev/null 2>&1 | FileCheck %s

// A static register must be cast to the rule's dynamic register argument type.
// Inspect the IR before conversion to value semantics removes the cast.
// CHECK-LABEL: func.func @static_qreg
// CHECK: [[REG:%.+]] = qref.alloc( 3) : !qref.reg<3>
// CHECK: [[CAST:%.+]] = qref.cast [[REG]] : !qref.reg<3> to !qref.reg<?>
// CHECK-NEXT: call @x_rule_0([[CAST]], {{%.+}}) : (!qref.reg<?>, i64) -> ()
func.func @static_qreg() attributes {quantum.node} {
  %reg = qref.alloc(3) : !qref.reg<3>
  %q = qref.get %reg[0] : !qref.reg<3> -> !qref.bit
  qref.custom "PauliX"() %q : !qref.bit
  qref.dealloc %reg : !qref.reg<3>
  return
}

func.func private @x_rule(%reg: !qref.reg<?>, %wire: i64) attributes {target_gate = "PauliX"} {
  %q = qref.get %reg[%wire] : !qref.reg<?>, i64 -> !qref.bit
  qref.custom "PauliZ"() %q : !qref.bit
  return
}
