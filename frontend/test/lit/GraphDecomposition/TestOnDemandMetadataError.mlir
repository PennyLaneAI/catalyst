// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: not catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=PhaseShift=1.0})' %s 2>&1 | FileCheck %s

// Metadata conversion failures in a batch retain the offending graph ID.
// CHECK: error: Failed to load on-demand decomposition rules:
// CHECK-SAME: with args S{}{wires:1}{unsupported = @unused}
// CHECK-SAME: Cannot convert the MLIR attribute @unused
// CHECK-SAME: unknown attribute type
func.func @circuit(%qubit: !quantum.bit) -> !quantum.bit attributes {quantum.node} {
    %out = quantum.operator "S"() qubits(%qubit) static_data = {unsupported = @unused}
        param_map = {} qubit_map = {wires = [0]}
    return %out : !quantum.bit
}

func.func private @unused() {
    return
}
