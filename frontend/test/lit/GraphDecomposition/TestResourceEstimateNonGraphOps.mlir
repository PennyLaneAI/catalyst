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

// Test that resource-estimate keeps the ops outside the decomposition graph, such as Pauli
// product rotations, with their names and numbers of qubits, as resource-analysis does.

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(resource-estimate{output-json gate-set=RX=1.0 bytecode-rules="%PRECOMPILED_BYTECODE_PATH"})' %s -o /dev/null | FileCheck %s

// CHECK-LABEL: "circuit"
// CHECK:       "quantum_operations"
// CHECK-NEXT:    "-1"
// CHECK-NEXT:      "RX": 1
// CHECK-NEXT:    }
// CHECK-NEXT:    "1"
// CHECK-NEXT:      "PPR-pi/4": 2
// CHECK-NEXT:    }
// CHECK-NEXT:    "2"
// CHECK-NEXT:      "PPR-pi/4": 1
func.func public @circuit() attributes {quantum.node} {
    %angle = arith.constant 0.5 : f64
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = pbc.ppr ["Z"](4) %1 : !quantum.bit
    %4 = pbc.ppr ["X"](4) %3 : !quantum.bit
    %5:2 = pbc.ppr ["Z", "X"](4) %4, %2 : !quantum.bit, !quantum.bit
    %6 = quantum.custom "RX"(%angle) %5#0 : !quantum.bit
    %7 = quantum.insert %0[ 0], %6 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %7[ 1], %5#1 : !quantum.reg, !quantum.bit
    quantum.dealloc %8 : !quantum.reg
    return
}
