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

// ============================================================
// hotspot_count != len(hotspot_indices) → error
// ============================================================

func.func @bad_count_mismatch() {
    // expected-error@+1 {{hotspot_count (2) must equal the number of entries in hotspot_indices (1)}}
    %p = quantum.freeze_partition
             {hotspot_count = 2 : i32,
              hotspot_indices = array<i32: 0>}
             : !quantum.partition<6, 2>
    func.return
}

// -----

// ============================================================
// hotspot_count > numQubits → error
// ============================================================

func.func @bad_count_exceeds_qubits() {
    // expected-error@+1 {{hotspot_count (5) must not exceed the circuit qubit count (4)}}
    %p = quantum.freeze_partition
             {hotspot_count = 5 : i32,
              hotspot_indices = array<i32: 0, 1, 2, 3, 4>}
             : !quantum.partition<4, 5>
    func.return
}

// -----

// ============================================================
// hotspot_count != partition type m → error
// ============================================================

func.func @bad_count_vs_m() {
    // expected-error@+1 {{hotspot_count (1) must equal the partition type parameter m (3)}}
    %p = quantum.freeze_partition
             {hotspot_count = 1 : i32,
              hotspot_indices = array<i32: 0>}
             : !quantum.partition<6, 3>
    func.return
}
