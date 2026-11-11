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

// Tests that the IR is unchanged after resource-estimate pass.

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(resource-estimate{output-json gate-set=S=1.0,T=1.0,Adjoint(S)=1.0,Adjoint(T)=1.0 bytecode-rules="%BYTECODE_PATH"})' %s | FileCheck %s

// CHECK-LABEL: func.func public @adjoint_circuit() attributes {quantum.node} {
// CHECK-NEXT:    [[REG:%.+]] = quantum.alloc( 1) : !quantum.reg
// CHECK-NEXT:    [[Q0:%.+]] = quantum.extract [[REG]][ 0] : !quantum.reg -> !quantum.bit
// CHECK-NEXT:    [[Q1:%.+]] = quantum.adjoint([[Q0]]) : !quantum.bit {
// CHECK-NEXT:    ^bb0([[ARG:%.+]]: !quantum.bit):
// CHECK-NEXT:      [[Q2:%.+]] = func.call @outer([[ARG]]) : (!quantum.bit) -> !quantum.bit
// CHECK-NEXT:      quantum.yield [[Q2]] : !quantum.bit
// CHECK-NEXT:    }
// CHECK-NEXT:    [[Q3:%.+]] = call @outer([[Q1]]) : (!quantum.bit) -> !quantum.bit
// CHECK-NEXT:    [[REG1:%.+]] = quantum.insert [[REG]][ 0], [[Q3]] : !quantum.reg, !quantum.bit
// CHECK-NEXT:    quantum.dealloc [[REG1]] : !quantum.reg
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func public @adjoint_circuit() attributes {quantum.node} {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.adjoint(%1) : !quantum.bit {
    ^bb0(%arg0: !quantum.bit):
        %3 = func.call @outer(%arg0) : (!quantum.bit) -> !quantum.bit
        quantum.yield %3 : !quantum.bit
    }
    %4 = func.call @outer(%2) : (!quantum.bit) -> !quantum.bit
    %5 = quantum.insert %0[ 0], %4 : !quantum.reg, !quantum.bit
    quantum.dealloc %5 : !quantum.reg
    return
}

func.func private @outer(%q: !quantum.bit) -> !quantum.bit {
    %0 = func.call @inner(%q) : (!quantum.bit) -> !quantum.bit
    %1 = quantum.custom "T"() %0 : !quantum.bit
    return %1 : !quantum.bit
}

func.func private @inner(%q: !quantum.bit) -> !quantum.bit {
    %0 = quantum.custom "S"() %q : !quantum.bit
    return %0 : !quantum.bit
}
