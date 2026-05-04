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

// RUN: quantum-opt %s --split-input-file | FileCheck %s

// ============================================================
// freeze_partition — m=1
// ============================================================

// CHECK-LABEL: func.func @freeze_m1
func.func @freeze_m1() {
    // CHECK: quantum.freeze_partition
    // CHECK-SAME: hotspot_count = 1
    // CHECK-SAME: hotspot_indices = array<i32: 2>
    %p = quantum.freeze_partition
             {hotspot_count = 1 : i32,
              hotspot_indices = array<i32: 2>}
             : !quantum.partition<6, 1>
    func.return
}

// -----

// ============================================================
// freeze_partition — m=3 (8 sub-problems)
// ============================================================

// CHECK-LABEL: func.func @freeze_m3
func.func @freeze_m3() {
    // CHECK: quantum.freeze_partition
    // CHECK-SAME: hotspot_count = 3
    // CHECK-SAME: hotspot_indices = array<i32: 1, 4, 7>
    %p = quantum.freeze_partition
             {hotspot_count = 3 : i32,
              hotspot_indices = array<i32: 1, 4, 7>}
             : !quantum.partition<10, 3>
    func.return
}

// -----

// ============================================================
// landscape_cluster — K=1 (sparse graph s > sc)
// ============================================================

// CHECK-LABEL: func.func @cluster_k1
func.func @cluster_k1() {
    %p = quantum.freeze_partition
             {hotspot_count = 2 : i32,
              hotspot_indices = array<i32: 0, 5>}
             : !quantum.partition<8, 2>
    // CHECK: quantum.landscape_cluster
    // CHECK-SAME: k = 1
    %cmap = quantum.landscape_cluster(%p : !quantum.partition<8, 2>)
                {k = 1 : i32} : !quantum.cluster_map<1>
    func.return
}

// -----

// ============================================================
// landscape_cluster — K=2
// ============================================================

// CHECK-LABEL: func.func @cluster_k2
func.func @cluster_k2() {
    %p = quantum.freeze_partition
             {hotspot_count = 3 : i32,
              hotspot_indices = array<i32: 0, 2, 4>}
             : !quantum.partition<12, 3>
    // CHECK: quantum.landscape_cluster
    // CHECK-SAME: k = 2
    %cmap = quantum.landscape_cluster(%p : !quantum.partition<12, 3>)
                {k = 2 : i32} : !quantum.cluster_map<2>
    func.return
}

// -----

// ============================================================
// select_representative
// ============================================================

// CHECK-LABEL: func.func @select_rep
func.func @select_rep() {
    %p = quantum.freeze_partition
             {hotspot_count = 1 : i32,
              hotspot_indices = array<i32: 0>}
             : !quantum.partition<4, 1>
    %cmap = quantum.landscape_cluster(%p : !quantum.partition<4, 1>)
                {k = 1 : i32} : !quantum.cluster_map<1>
    // CHECK: quantum.select_representative
    // CHECK-SAME: !quantum.circuit_ref
    %ref = quantum.select_representative(%cmap : !quantum.cluster_map<1>)
               : !quantum.circuit_ref
    func.return
}

// -----

// ============================================================
// bias_transfer — direct copy (delta_B < threshold)
// ============================================================

// CHECK-LABEL: func.func @bias_transfer_direct
func.func @bias_transfer_direct(%p_rep : !quantum.params) {
    // CHECK: quantum.bias_transfer
    // CHECK-SAME: B_rep = 2.500000e-01
    // CHECK-SAME: B_target = 2.800000e-01
    // CHECK-SAME: threshold = 3.000000e-01
    %p_out = quantum.bias_transfer(%p_rep : !quantum.params)
                 {B_rep = 0.25 : f64,
                  B_target = 0.28 : f64,
                  threshold = 0.3 : f64}
                 : !quantum.params
    func.return
}

// -----

// ============================================================
// bias_transfer — warm-start (delta_B > threshold)
// ============================================================

// CHECK-LABEL: func.func @bias_transfer_warmstart
func.func @bias_transfer_warmstart(%p_rep : !quantum.params) {
    // CHECK: quantum.bias_transfer
    // CHECK-SAME: B_rep = 1.000000e-01
    // CHECK-SAME: B_target = 5.000000e-01
    // CHECK-SAME: threshold = 3.000000e-01
    %p_out = quantum.bias_transfer(%p_rep : !quantum.params)
                 {B_rep = 0.1 : f64,
                  B_target = 0.5 : f64,
                  threshold = 0.3 : f64}
                 : !quantum.params
    func.return
}

// -----

// ============================================================
// aggregate_min — 4 sub-circuits (m=2)
// ============================================================

// CHECK-LABEL: func.func @aggregate_min_test
func.func @aggregate_min_test(%p0 : !quantum.params,
                               %p1 : !quantum.params,
                               %p2 : !quantum.params,
                               %p3 : !quantum.params) {
    // CHECK: quantum.aggregate_min
    // CHECK-SAME: !quantum.bitstring
    %best = quantum.aggregate_min(%p0, %p1, %p2, %p3
                : !quantum.params, !quantum.params,
                  !quantum.params, !quantum.params)
                : !quantum.bitstring
    func.return
}

// -----

// ============================================================
// Full pipeline: freeze -> cluster -> select_rep -> bias_transfer -> aggregate_min
// ============================================================

// CHECK-LABEL: func.func @full_pipeline
func.func @full_pipeline(%p_rep : !quantum.params,
                          %p1   : !quantum.params,
                          %p2   : !quantum.params,
                          %p3   : !quantum.params) {
    // CHECK: quantum.freeze_partition
    %part = quantum.freeze_partition
                {hotspot_count = 2 : i32,
                 hotspot_indices = array<i32: 0, 3>}
                : !quantum.partition<4, 2>
    // CHECK: quantum.landscape_cluster
    %cmap = quantum.landscape_cluster(%part : !quantum.partition<4, 2>)
                {k = 1 : i32} : !quantum.cluster_map<1>
    // CHECK: quantum.select_representative
    %ref = quantum.select_representative(%cmap : !quantum.cluster_map<1>)
               : !quantum.circuit_ref
    // CHECK: quantum.bias_transfer
    %p_out1 = quantum.bias_transfer(%p_rep : !quantum.params)
                  {B_rep = 0.25 : f64, B_target = 0.28 : f64, threshold = 0.3 : f64}
                  : !quantum.params
    %p_out2 = quantum.bias_transfer(%p_rep : !quantum.params)
                  {B_rep = 0.25 : f64, B_target = 0.45 : f64, threshold = 0.3 : f64}
                  : !quantum.params
    %p_out3 = quantum.bias_transfer(%p_rep : !quantum.params)
                  {B_rep = 0.25 : f64, B_target = 0.50 : f64, threshold = 0.3 : f64}
                  : !quantum.params
    // CHECK: quantum.aggregate_min
    // CHECK-SAME: !quantum.bitstring
    %best = quantum.aggregate_min(%p_rep, %p_out1, %p_out2, %p_out3
                : !quantum.params, !quantum.params,
                  !quantum.params, !quantum.params)
                : !quantum.bitstring
    func.return
}

// -----

// ============================================================
// freeze_partition: large circuit (100 qubits, m=5)
// ============================================================

// CHECK-LABEL: func @freeze_large
// CHECK:       quantum.freeze_partition
// CHECK-SAME:  hotspot_count = 5
// CHECK-SAME:  hotspot_indices = array<i32: 10, 20, 30, 40, 50>
func.func @freeze_large() {
    %p = quantum.freeze_partition
             {hotspot_count = 5 : i32,
              hotspot_indices = array<i32: 10, 20, 30, 40, 50>}
             : !quantum.partition<100, 5>
    func.return
}

// -----

// ============================================================
// landscape_cluster: K=4 clusters (dense graph with s < sc)
// ============================================================

// CHECK-LABEL: func @cluster_k4
// CHECK:       quantum.landscape_cluster
// CHECK-SAME:  k = 4
func.func @cluster_k4() {
    %p = quantum.freeze_partition
             {hotspot_count = 2 : i32,
              hotspot_indices = array<i32: 0, 3>}
             : !quantum.partition<6, 2>
    %c = quantum.landscape_cluster(%p : !quantum.partition<6, 2>) {k = 4 : i32}
             : !quantum.cluster_map<4>
    func.return
}

// -----

// ============================================================
// aggregate_min: single input (degenerate case)
// ============================================================

// CHECK-LABEL: func @aggregate_min_single
// CHECK:       quantum.aggregate_min
// CHECK-SAME:  !quantum.bitstring
func.func @aggregate_min_single(%p0: !quantum.params) {
    %b = quantum.aggregate_min(%p0 : !quantum.params) : !quantum.bitstring
    func.return
}

// -----

// ============================================================
// bias_transfer: threshold boundary value (exactly 0.5)
// ============================================================

// CHECK-LABEL: func @bias_transfer_boundary
// CHECK:       B_rep = 2.500000e-01
// CHECK-SAME:  B_target = 7.500000e-01
// CHECK-SAME:  threshold = 5.000000e-01
func.func @bias_transfer_boundary(%p: !quantum.params) {
    %out = quantum.bias_transfer(%p : !quantum.params)
               {B_rep = 0.25 : f64, B_target = 0.75 : f64, threshold = 0.5 : f64}
               : !quantum.params
    func.return
}

// -----

// ============================================================
// Two independent freeze_partition ops (different m values)
// ============================================================

// CHECK-LABEL: func @two_pipelines
// CHECK:       hotspot_count = 1
// CHECK:       hotspot_count = 2
func.func @two_pipelines() {
    %p1 = quantum.freeze_partition
              {hotspot_count = 1 : i32, hotspot_indices = array<i32: 0>}
              : !quantum.partition<4, 1>
    %p2 = quantum.freeze_partition
              {hotspot_count = 2 : i32, hotspot_indices = array<i32: 1, 3>}
              : !quantum.partition<8, 2>
    %c1 = quantum.landscape_cluster(%p1 : !quantum.partition<4, 1>) {k = 1 : i32}
              : !quantum.cluster_map<1>
    %c2 = quantum.landscape_cluster(%p2 : !quantum.partition<8, 2>) {k = 2 : i32}
              : !quantum.cluster_map<2>
    func.return
}
