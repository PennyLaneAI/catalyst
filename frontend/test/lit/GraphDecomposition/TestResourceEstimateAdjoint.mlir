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

// An adjoint applied to a subroutine also applies to the subroutines it calls.

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(resource-estimate{output-json gate-set=S=1.0,T=1.0,Adjoint(S)=1.0,Adjoint(T)=1.0 bytecode-rules="%BYTECODE_PATH"})' %s -o /dev/null | FileCheck %s

// CHECK-LABEL: "A_inner": {
// CHECK:         "Adjoint(S)": 1
// CHECK-LABEL: "A_outer": {
// CHECK:         "A_inner": 1
// CHECK:         "Adjoint(T)": 1
// CHECK-LABEL: "inner": {
// CHECK:         "S": 1
// CHECK-LABEL: "outer": {
// CHECK:         "inner": 1
// CHECK:         "T": 1
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
