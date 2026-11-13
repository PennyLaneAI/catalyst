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

// should pass despite a missing QPD and libpython path (RX is in the gateset, unregistered is not collected)
// RUN: env CATALYST_QPD=/missing/plugin CATALYST_LIBPYTHON=/missing/python catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=RX=1.0})' %s | FileCheck %s

// should fail to decompose RX to RZ due to missing QPD support
// RUN: not env CATALYST_QPD=/missing/plugin CATALYST_LIBPYTHON=/missing/python catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=RZ=1.0})' %s 2>&1 | FileCheck %s --check-prefix=ERROR

// CHECK-LABEL: func.func @circuit
// CHECK: quantum.custom "RX"

// ERROR: error: failed to load the Python decomposition callback
func.func @circuit(%theta: f64, %qubit: !quantum.bit) -> !quantum.bit attributes {quantum.node} {
    %out = quantum.custom "RX"(%theta) %qubit : !quantum.bit
    return %out : !quantum.bit
}

func.func private @unused(%qubit: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "UnregisteredGate"() %qubit : !quantum.bit
    return %out : !quantum.bit
}
