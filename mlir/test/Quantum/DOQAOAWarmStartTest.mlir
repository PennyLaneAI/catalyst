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

// Full 6-pass pipeline through doqaoa-warmstart-scheduler.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-landscape-overlap \
// RUN:   --doqaoa-bias-shift \
// RUN:   --doqaoa-representative-selection \
// RUN:   --doqaoa-training-schedule \
// RUN:   --doqaoa-shared-buffer \
// RUN:   --doqaoa-warmstart-scheduler \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=WS

// EPOCHS prefix: custom epoch/lr options.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-landscape-overlap \
// RUN:   --doqaoa-bias-shift \
// RUN:   --doqaoa-representative-selection \
// RUN:   --doqaoa-training-schedule \
// RUN:   --doqaoa-shared-buffer \
// RUN:   '--doqaoa-warmstart-scheduler=warmstart-epochs=5 learning-rate=0.05' \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=EPOCHS

// WARM2 prefix: threshold=0.0 forces all non-reps to warm-start.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-landscape-overlap \
// RUN:   --doqaoa-bias-shift \
// RUN:   '--doqaoa-representative-selection=bias-threshold=0.0' \
// RUN:   --doqaoa-training-schedule \
// RUN:   --doqaoa-shared-buffer \
// RUN:   '--doqaoa-warmstart-scheduler=warmstart-epochs=3' \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=WARM2

// GUARD prefix: run without prior passes → warning fires.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-warmstart-scheduler \
// RUN:   --split-input-file 2>&1 | FileCheck %s --check-prefix=GUARD

// ============================================================
// Test 1: 4-cycle m=1, K=1.
//   2 sub-problems: sp0=rep (mode 0), sp1=direct copy (mode 1).
//   No warm-start sub-problems → warmstart_converged = [-1, -1].
//   warmstart_epochs_used = [0, 0].
// ============================================================

// WS-LABEL: func @cycle_warmstart
// WS:       quantum.freeze_partition
// WS-SAME:  warmstart_converged = array<i32: -1, -1>
// WS-SAME:  warmstart_epochs_used = array<i32: 0, 0>
// WS-SAME:  warmstart_params

// EPOCHS-LABEL: func @cycle_warmstart
// EPOCHS:   quantum.freeze_partition
// EPOCHS-SAME: warmstart_converged = array<i32: -1, -1>

// WARM2-LABEL: func @cycle_warmstart
// WARM2:    quantum.freeze_partition
// WARM2-SAME: warmstart_converged
// WARM2-SAME: warmstart_epochs_used
// WARM2-SAME: warmstart_params

// GUARD: warning: doqaoa-warmstart-scheduler: missing shared-buffer or

func.func @cycle_warmstart() {
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
//   sp0=rep, sp1=copy. No warm-start sub-problems.
//   warmstart_converged = [-1, -1], warmstart_params present.
// ============================================================

// WS-LABEL: func @k4_warmstart
// WS:       quantum.freeze_partition
// WS-SAME:  warmstart_converged = array<i32: -1, -1>
// WS-SAME:  warmstart_params

// GUARD: warning: doqaoa-warmstart-scheduler: missing shared-buffer or

func.func @k4_warmstart() {
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
// Test 3: Sparse path graph m=2, K=1 (4 sub-problems, all direct copy).
//   warmstart_converged = [-1, -1, -1, -1].
//   warmstart_epochs_used = [0, 0, 0, 0].
// ============================================================

// WS-LABEL: func @sparse_warmstart
// WS:       quantum.freeze_partition
// WS-SAME:  warmstart_converged = array<i32: -1, -1, -1, -1>
// WS-SAME:  warmstart_epochs_used = array<i32: 0, 0, 0, 0>

// WARM2-LABEL: func @sparse_warmstart
// WARM2:    quantum.freeze_partition
// WARM2-SAME: warmstart_epochs_used
// WARM2-SAME: warmstart_params

// GUARD: warning: doqaoa-warmstart-scheduler: missing shared-buffer or

func.func @sparse_warmstart() {
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
// Test 4: Biased graph — linear terms force warm-start sub-problems.
//   h_lin with large values pushes some sub-problems to mode-2
//   when bias-threshold=0.0 (WARM2 prefix).
//   WS prefix (default threshold=0.3): some may still be copies.
//   Key invariant: warmstart_params tensor always present.
// ============================================================

// WS-LABEL: func @biased_warmstart
// WS:       quantum.freeze_partition
// WS-SAME:  warmstart_converged
// WS-SAME:  warmstart_epochs_used
// WS-SAME:  warmstart_params

// WARM2-LABEL: func @biased_warmstart
// WARM2:    quantum.freeze_partition
// WARM2-SAME: warmstart_epochs_used
// WARM2-SAME: warmstart_params

// EPOCHS-LABEL: func @biased_warmstart
// EPOCHS:   quantum.freeze_partition
// EPOCHS-SAME: warmstart_params

// GUARD: warning: doqaoa-warmstart-scheduler: missing shared-buffer or

func.func @biased_warmstart() {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>,
        h_quad = #quantum.dense_graph<4, dense<[
            [ 0.0, -0.5,  0.0,  0.0],
            [-0.5,  0.0, -0.5,  0.0],
            [ 0.0, -0.5,  0.0, -0.5],
            [ 0.0,  0.0, -0.5,  0.0]]> : tensor<4x4xf64>>,
        h_lin  = dense<[0.8, -0.8, 0.8, -0.8]> : tensor<4xf64>
    } : !quantum.partition<4, 1>
    func.return
}
