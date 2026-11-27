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

// Inheriting `DecomposableOp` makes an operator graph solver aware, but not necessarily decomposable.
// This tests checks that `MeasurementOp` (inherits `DecomposableOp` but has no signature analyzer)
// asserts if decomposition is requested.

// RUN: not --crash catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=testGate=1.0})' %s 2>&1 | FileCheck %s

func.func @circuit() {
    %q0 = quantum.alloc_qb : !quantum.bit
    // CHECK: lowering to decomposition rule not supported for operator
    %mres, %out = quantum.measurement "SampleMeasurement"() qubits(%q0) qubit_map = {wires = [0]} : (i1, !quantum.bit)
    quantum.dealloc_qb %q0 : !quantum.bit
    return
}

// A decomposition rule that should not be attempted to be decomposed
func.func private @sample_measurement_rule(%q: !quantum.bit) -> (!quantum.bit) attributes {
    target_gate = "SampleMeasurement{}{wires:1}{}",
    resources = {operations = {"testGate{}{wires:1}{}" = 1 : i64}}
} {
    %0 = quantum.custom "testGate"() %q : !quantum.bit
    return %0 : !quantum.bit
}
