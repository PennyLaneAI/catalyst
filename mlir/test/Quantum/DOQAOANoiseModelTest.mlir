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

// Tests for --doqaoa-noise-preserve (Phase 3, Task 8).
//
// NOISE prefix: default FakeBrisbane noise parameters.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-noise-preserve \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=NOISE

// CUSTOM prefix: custom T1/T2/fidelity values.
// RUN: quantum-opt %s \
// RUN:   '--doqaoa-noise-preserve=noise-t1-ns=200000 noise-t2-ns=300000 noise-cx-fidelity=0.999 noise-cx-time-ns=400' \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=CUSTOM

// BUDGET prefix: tight T1 budget (forces decoherence warning for large circuits).
// RUN: quantum-opt %s \
// RUN:   '--doqaoa-noise-preserve=noise-t1-ns=1000 noise-cx-time-ns=533' \
// RUN:   --split-input-file 2>&1 | FileCheck %s --check-prefix=BUDGET

// REGFAIL prefix: expected-max-cnots regression gate fires.
// RUN: quantum-opt %s \
// RUN:   '--doqaoa-noise-preserve=expected-max-cnots=2' \
// RUN:   --split-input-file 2>&1 | FileCheck %s --check-prefix=REGFAIL

// ============================================================
// Test 1: 4-cycle m=1, K=1.
//   Free qubits = {1,2,3}, edges: 1-2, 2-3, 1-0(frozen via hotspot=0 → skip 0-x).
//   Actual free-free edges: 1-2, 2-3 → |E_free|=2, cnot_count=4.
//   circuit_time = 4 × 533 = 2132 ns < T1=127000 ns → depth_ok=1.
// ============================================================

// NOISE-LABEL: func @cycle_noise
// NOISE:       quantum.freeze_partition
// NOISE-SAME:  noise_cnot_counts
// NOISE-SAME:  noise_cx_fidelity = 9.918000e-01 : f64
// NOISE-SAME:  noise_cx_time_ns = 5.330000e+02 : f64
// NOISE-SAME:  noise_depth_ok = 1 : i32
// NOISE-SAME:  noise_max_cnots
// NOISE-SAME:  noise_t1_ns = 1.270000e+05 : f64
// NOISE-SAME:  noise_t2_ns = 2.180000e+05 : f64

// CUSTOM-LABEL: func @cycle_noise
// CUSTOM:       quantum.freeze_partition
// CUSTOM-SAME:  noise_cx_fidelity = 9.990000e-01 : f64
// CUSTOM-SAME:  noise_cx_time_ns = 4.000000e+02 : f64
// CUSTOM-SAME:  noise_depth_ok = 1 : i32
// CUSTOM-SAME:  noise_t1_ns = 2.000000e+05 : f64
// CUSTOM-SAME:  noise_t2_ns = 3.000000e+05 : f64

// BUDGET-LABEL: func @cycle_noise
// BUDGET:  warning: doqaoa-noise-preserve: circuit_time=

func.func @cycle_noise() {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>,
        h_quad = #quantum.dense_graph<4, dense<[
            [ 0.0, -0.5,  0.0, -0.5],
            [-0.5,  0.0, -0.5,  0.0],
            [ 0.0, -0.5,  0.0, -0.5],
            [-0.5,  0.0, -0.5,  0.0]]> : tensor<4x4xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf64>
    } : !quantum.partition<4, 1>
    func.return
}

// -----

// ============================================================
// Test 2: K4 m=1, 6 edges total, 1 frozen qubit.
//   Free qubits = {1,2,3}, edges among free: 1-2, 1-3, 2-3 → |E_free|=3, cnots=6.
//   circuit_time = 6 × 533 = 3198 ns < T1 → depth_ok=1.
// ============================================================

// NOISE-LABEL: func @k4_noise
// NOISE:       quantum.freeze_partition
// NOISE-SAME:  noise_cnot_counts
// NOISE-SAME:  noise_depth_ok = 1 : i32
// NOISE-SAME:  noise_max_cnots = 6 : i32

// REGFAIL-LABEL: func @k4_noise
// REGFAIL: error: doqaoa-noise-preserve: max_cnots={{.*}} exceeds expected-max-cnots=2

func.func @k4_noise() {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>,
        h_quad = #quantum.dense_graph<4, dense<[
            [ 0.0, -1.0, -1.0, -1.0],
            [-1.0,  0.0, -1.0, -1.0],
            [-1.0, -1.0,  0.0, -1.0],
            [-1.0, -1.0, -1.0,  0.0]]> : tensor<4x4xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf64>
    } : !quantum.partition<4, 1>
    func.return
}

// -----

// ============================================================
// Test 3: Sparse path m=2.
//   Free qubits = {0,3}, only edge 0-3 exists if present; path graph: 0-1-2-3.
//   Hotspots 1,2 frozen. Free: {0,3}. Edge 0-3 absent in path → |E_free|=0, cnots=0.
//   noise_max_cnots = 0, depth_ok = 1 (0 × 533 = 0 < T1).
// ============================================================

// NOISE-LABEL: func @sparse_noise
// NOISE:       quantum.freeze_partition
// NOISE-SAME:  noise_cnot_counts = array<i32: 0, 0, 0, 0>
// NOISE-SAME:  noise_depth_ok = 1 : i32
// NOISE-SAME:  noise_max_cnots = 0 : i32

func.func @sparse_noise() {
    %p = quantum.freeze_partition {
        hotspot_count   = 2 : i32,
        hotspot_indices = array<i32: 1, 2>,
        h_quad = #quantum.sparse_graph<4, 3,
            [0, 1, 2], [1, 2, 3],
            dense<[-0.5, -0.5, -0.5]> : tensor<3xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf64>
    } : !quantum.partition<4, 2>
    func.return
}
