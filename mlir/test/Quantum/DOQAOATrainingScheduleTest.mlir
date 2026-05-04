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

// Full 4-pass pipeline: landscape-overlap → bias-shift → representative-selection
//                       → training-schedule.  Check schedule attributes.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-landscape-overlap \
// RUN:   --doqaoa-bias-shift \
// RUN:   --doqaoa-representative-selection \
// RUN:   --doqaoa-training-schedule \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=SCHED

// WARM prefix: threshold=0.0 forces all non-reps to warm-start; check phase2 count.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-landscape-overlap \
// RUN:   --doqaoa-bias-shift \
// RUN:   '--doqaoa-representative-selection=bias-threshold=0.0' \
// RUN:   '--doqaoa-training-schedule=full-epochs=200 warmstart-epochs=30' \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=WARM

// GUARD prefix: run training-schedule without prior passes → warning fires.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-training-schedule \
// RUN:   --split-input-file 2>&1 | FileCheck %s --check-prefix=GUARD

// ============================================================
// Test 1: 4-cycle m=1, K=1.
//   2 sub-problems: sp0=rep, sp1=direct copy.
//   Phase 1: [0], Phase 2: [], Phase 3: [1]
//   schedule_phase_ends = [1, 1, 2]
//   training_schedule   = [0, 1]
//   schedule_epochs     = [100, 0]    (default full=100)
//   schedule_sources    = [0, 0]      (sp1 copies from sp0)
// ============================================================

// SCHED-LABEL: func @cycle_training_schedule
// SCHED:       quantum.freeze_partition
// SCHED-SAME:  schedule_epochs = array<i32: 100, 0>
// SCHED-SAME:  schedule_phase_ends = array<i32: 1, 1, 2>
// SCHED-SAME:  schedule_sources = array<i32: 0, 0>
// SCHED-SAME:  training_schedule = array<i32: 0, 1>

// WARM-LABEL: func @cycle_training_schedule
// WARM:       quantum.freeze_partition
// WARM-SAME:  schedule_epochs = array<i32: 200, 30>
// WARM-SAME:  schedule_phase_ends = array<i32: 1, 2, 2>
// WARM-SAME:  training_schedule = array<i32: 0, 1>

// GUARD: warning: doqaoa-training-schedule: missing representative-selection attributes

func.func @cycle_training_schedule() {
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
//   sp0=rep (mode 0), sp1=direct copy (mode 1).
//   schedule_phase_ends = [1, 1, 2]
//   training_schedule   = [0, 1]
// ============================================================

// SCHED-LABEL: func @complete_k4_training_schedule
// SCHED:       quantum.freeze_partition
// SCHED-SAME:  schedule_epochs = array<i32: 100, 0>
// SCHED-SAME:  schedule_phase_ends = array<i32: 1, 1, 2>
// SCHED-SAME:  training_schedule = array<i32: 0, 1>

// WARM-LABEL: func @complete_k4_training_schedule
// WARM:       quantum.freeze_partition
// WARM-SAME:  schedule_phase_ends = array<i32: 1, 2, 2>
// WARM-SAME:  training_schedule = array<i32: 0, 1>

// GUARD: warning: doqaoa-training-schedule: missing representative-selection attributes

func.func @complete_k4_training_schedule() {
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
// Test 3: 4-cycle m=2, K=2.
//   4 sub-problems.  representatives=[1, 0] (cluster 0 rep=sp1, cluster 1 rep=sp0).
//   is_representative=[1,1,0,0]; transfer_modes=[0,0,1,1].
//   Phase 1: [0, 1] (both reps), Phase 2: [], Phase 3: [2, 3]
//   schedule_phase_ends = [2, 2, 4]
//   training_schedule   = [0, 1, 2, 3]  (reps first, copies after)
//   schedule_epochs     = [100, 100, 0, 0]
// ============================================================

// SCHED-LABEL: func @cycle_m2_training_schedule
// SCHED:       quantum.freeze_partition
// SCHED-SAME:  schedule_epochs = array<i32: 100, 100, 0, 0>
// SCHED-SAME:  schedule_phase_ends = array<i32: 2, 2, 4>
// SCHED-SAME:  training_schedule = array<i32: 0, 1, 2, 3>

// WARM-LABEL: func @cycle_m2_training_schedule
// WARM:       quantum.freeze_partition
// WARM-SAME:  schedule_epochs = array<i32: 200, 200, 30, 30>
// WARM-SAME:  schedule_phase_ends = array<i32: 2, 4, 4>
// WARM-SAME:  training_schedule = array<i32: 0, 1, 2, 3>

// GUARD: warning: doqaoa-training-schedule: missing representative-selection attributes

func.func @cycle_m2_training_schedule() {
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
// Test 4: Sparse path graph m=2, K=1.
//   4 sub-problems: sp0=rep, sp1-3=direct copy.
//   schedule_phase_ends = [1, 1, 4]
//   training_schedule   = [0, 1, 2, 3]
//   schedule_epochs     = [100, 0, 0, 0]
//   schedule_sources    = [0, 0, 0, 0]  (all inherit from sp0)
// ============================================================

// SCHED-LABEL: func @sparse_training_schedule
// SCHED:       quantum.freeze_partition
// SCHED-SAME:  schedule_epochs = array<i32: 100, 0, 0, 0>
// SCHED-SAME:  schedule_phase_ends = array<i32: 1, 1, 4>
// SCHED-SAME:  schedule_sources = array<i32: 0, 0, 0, 0>
// SCHED-SAME:  training_schedule = array<i32: 0, 1, 2, 3>

// WARM-LABEL: func @sparse_training_schedule
// WARM:       quantum.freeze_partition
// WARM-SAME:  schedule_epochs = array<i32: 200, 30, 30, 30>
// WARM-SAME:  schedule_phase_ends = array<i32: 1, 4, 4>

// GUARD: warning: doqaoa-training-schedule: missing representative-selection attributes

func.func @sparse_training_schedule() {
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

// -----

// ============================================================
// Test 5: Custom epoch budget via pass options.
//   Same 4-cycle m=1 as Test 1, but full-epochs=50.
//   schedule_epochs = [50, 0] with default threshold.
// ============================================================

// SCHED-LABEL: func @custom_epochs_training_schedule
// SCHED:       quantum.freeze_partition
// SCHED-SAME:  schedule_phase_ends = array<i32: 1, 1, 2>
// SCHED-SAME:  training_schedule = array<i32: 0, 1>

// WARM-LABEL: func @custom_epochs_training_schedule
// WARM:       quantum.freeze_partition
// WARM-SAME:  schedule_epochs = array<i32: 200, 30>
// WARM-SAME:  schedule_phase_ends = array<i32: 1, 2, 2>

// GUARD: warning: doqaoa-training-schedule: missing representative-selection attributes

func.func @custom_epochs_training_schedule() {
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
