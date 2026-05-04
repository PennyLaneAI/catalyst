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

// Full 5-pass pipeline through doqaoa-shared-buffer.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-landscape-overlap \
// RUN:   --doqaoa-bias-shift \
// RUN:   --doqaoa-representative-selection \
// RUN:   --doqaoa-training-schedule \
// RUN:   --doqaoa-shared-buffer \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=BUF

// GUARD prefix: run without prior passes → warning fires.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-shared-buffer \
// RUN:   --split-input-file 2>&1 | FileCheck %s --check-prefix=GUARD

// ============================================================
// Test 1: 4-cycle m=1, K=1 (2 sub-problems).
//   buffer_slot_map = [0, 0]  (both sub-problems in cluster 0)
//   param_buffer_size = 2
//   use_atomic_guards = 1
//   init_params shape = 1 × 2
// ============================================================

// BUF-LABEL: func @cycle_shared_buffer
// BUF:       quantum.freeze_partition
// BUF-SAME:  param_buffer_size = 2 : i32
// BUF-SAME:  use_atomic_guards = 1 : i32
// BUF-SAME:  buffer_slot_map = array<i32: 0, 0>

// GUARD: warning: doqaoa-shared-buffer: missing training-schedule attributes

func.func @cycle_shared_buffer() {
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
// Test 2: Complete graph K4, m=1, K=1.
//   buffer_slot_map = [0, 0]
//   param_buffer_size = 2
//   use_atomic_guards = 1
// ============================================================

// BUF-LABEL: func @k4_shared_buffer
// BUF:       quantum.freeze_partition
// BUF-SAME:  param_buffer_size = 2 : i32
// BUF-SAME:  use_atomic_guards = 1 : i32

// GUARD: warning: doqaoa-shared-buffer: missing training-schedule attributes

func.func @k4_shared_buffer() {
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
// Test 3: 4-cycle m=2, K=2 (4 sub-problems, 2 clusters).
//   Cluster 0 → reps share slot 0; cluster 1 → slot 1.
//   buffer_slot_map[0,1,2,3] = some permutation of 0s and 1s.
//   param_buffer_size = 2
// ============================================================

// BUF-LABEL: func @cycle_m2_shared_buffer
// BUF:       quantum.freeze_partition
// BUF-SAME:  param_buffer_size = 2 : i32
// BUF-SAME:  use_atomic_guards = 1 : i32

// GUARD: warning: doqaoa-shared-buffer: missing training-schedule attributes

func.func @cycle_m2_shared_buffer() {
    %p = quantum.freeze_partition {
        hotspot_count   = 2 : i32,
        hotspot_indices = array<i32: 0, 2>,
        h_quad = #quantum.dense_graph<4, dense<[
            [ 0.0, -0.5,  0.0, -0.5],
            [-0.5,  0.0, -0.5,  0.0],
            [ 0.0, -0.5,  0.0, -0.5],
            [-0.5,  0.0, -0.5,  0.0]]> : tensor<4x4xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf64>
    } : !quantum.partition<4, 2>
    func.return
}

// -----

// ============================================================
// Test 4: Sparse path graph m=2, K=1 (4 sub-problems).
//   All sub-problems in cluster 0 → buffer_slot_map = [0, 0, 0, 0].
//   param_buffer_size = 2
// ============================================================

// BUF-LABEL: func @sparse_shared_buffer
// BUF:       quantum.freeze_partition
// BUF-SAME:  param_buffer_size = 2 : i32
// BUF-SAME:  buffer_slot_map = array<i32: 0, 0, 0, 0>
// BUF-SAME:  use_atomic_guards = 1 : i32

// GUARD: warning: doqaoa-shared-buffer: missing training-schedule attributes

func.func @sparse_shared_buffer() {
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
