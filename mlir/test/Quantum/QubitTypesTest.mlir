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

// RUN: quantum-opt %s --split-input-file --verify-diagnostics

func.func @test_default_type(%q0 : !quantum.bit) {
    %q1 = quantum.custom ""() %q0 : !quantum.bit
    return
}

// -----

func.func @test_abstract_type(%q0 : !quantum.bit<abstract>) {
    %q1 = quantum.custom ""() %q0 : !quantum.bit<abstract>
    return
}

// -----

func.func @test_logical_type(%q0 : !quantum.bit<logical>) {
    %q1 = quantum.custom ""() %q0 : !quantum.bit<logical>
    return
}

// -----

func.func @test_qec_type(%q0 : !quantum.bit<qec>) {
    %q1 = quantum.custom ""() %q0 : !quantum.bit<qec>
    return
}

// -----

func.func @test_physical_type(%q0 : !quantum.bit<physical>) {
    %q1 = quantum.custom ""() %q0 : !quantum.bit<physical>
    return
}

// -----