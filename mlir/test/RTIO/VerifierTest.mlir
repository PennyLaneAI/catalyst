// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --split-input-file --verify-diagnostics | FileCheck %s

////////////////////////
// Channel Operations //
////////////////////////

// CHECK-LABEL: func.func @channel_good()
func.func @channel_good() {
    // Smoke test for valid channel operations
    // CHECK: rtio.channel : !rtio.channel<"dds", 0>
    %ch0 = rtio.channel : !rtio.channel<"dds", 0>
    // CHECK: rtio.channel : !rtio.channel<"dds", [0 : i64], 0>
    %ch0_q1 = rtio.channel : !rtio.channel<"dds", [0], 0>
    // CHECK: rtio.channel : !rtio.channel<"dds", [0 : i64, "t0"], 0>
    %ch0_q2 = rtio.channel : !rtio.channel<"dds", [0, "t0"], 0>
    // CHECK: rtio.channel : !rtio.channel<"dds", ?>
    %ch_explict_dyn = rtio.channel : !rtio.channel<"dds", ?>
    // CHECK: rtio.channel : !rtio.channel<"dds", [0 : i64], ?>
    %ch_explict_dyn_q1 = rtio.channel : !rtio.channel<"dds", [0], ?>
    // CHECK: rtio.channel : !rtio.channel<"dds", [0 : i64, "t0"], ?>
    %ch_explict_dyn_q2 = rtio.channel : !rtio.channel<"dds", [0, "t0"], ?>

    // Implicit dynamic channel ID (omitted = dynamic)
    // CHECK: rtio.channel : !rtio.channel<"dds", ?>
    %ch0_implicit_dyn = rtio.channel : !rtio.channel<"dds">
    // CHECK: rtio.channel : !rtio.channel<"dds", [0 : i64], ?>
    %ch_implicit_dyn_q1 = rtio.channel : !rtio.channel<"dds", [0]>
    // CHECK: rtio.channel : !rtio.channel<"dds", [0 : i64, "t0"], ?>
    %ch_implicit_dyn_q2 = rtio.channel : !rtio.channel<"dds", [0, "t0"]>
    return
}

// -----

func.func @channel_negative_id() {
    // expected-error@+1 {{static channel ID must be non-negative}}
    %ch0_negative = rtio.channel : !rtio.channel<"dds", -1>
}

// -----

// CHECK-LABEL: func.func @qubit_to_channel_good
func.func @qubit_to_channel_good(%qubit: !ion.qubit) {
    // Smoke test for qubit_to_channel
    // CHECK: rtio.qubit_to_channel %{{.*}} : !ion.qubit -> !rtio.channel<"dds", ["transition_0"], ?>
    %ch0 = rtio.qubit_to_channel %qubit : !ion.qubit -> !rtio.channel<"dds", ["transition_0"], ?>
    return
}

// -----

////////////////////////////
// Event-Based Operations //
////////////////////////////

// CHECK-LABEL: func.func @pulse_basic
func.func @pulse_basic(%dur: f64, %freq: f64, %phase: f64) {
    // CHECK: %[[CH:.*]] = rtio.channel : !rtio.channel<"dds", 0>
    %ch0 = rtio.channel : !rtio.channel<"dds", 0>

    // CHECK: rtio.pulse %[[CH]] duration(%{{.*}}) frequency(%{{.*}}) phase(%{{.*}}) : <"dds", 0> -> !rtio.event
    %event = rtio.pulse %ch0 duration(%dur) frequency(%freq) phase(%phase)
        : !rtio.channel<"dds", 0> -> !rtio.event

    return
}

// -----

// CHECK-LABEL: func.func @pulse_with_wait
func.func @pulse_with_wait(%dur: f64, %freq: f64, %phase: f64) {
    // CHECK: %[[CH:.*]] = rtio.channel : !rtio.channel<"dds", 0>
    %ch0 = rtio.channel : !rtio.channel<"dds", 0>

    // CHECK: %[[E0:.*]] = rtio.pulse %[[CH]] duration(%{{.*}}) frequency(%{{.*}}) phase(%{{.*}}) : <"dds", 0> -> !rtio.event
    %event0 = rtio.pulse %ch0 duration(%dur) frequency(%freq) phase(%phase)
        : !rtio.channel<"dds", 0> -> !rtio.event

    // CHECK: rtio.pulse %[[CH]] duration(%{{.*}}) frequency(%{{.*}}) phase(%{{.*}}) wait(%[[E0]]) : <"dds", 0> -> !rtio.event
    %event1 = rtio.pulse %ch0 duration(%dur) frequency(%freq) phase(%phase) wait(%event0)
        : !rtio.channel<"dds", 0> -> !rtio.event

    return
}


// -----

// CHECK-LABEL: func.func @sync_basic
func.func @sync_basic(%dur: f64, %freq: f64, %phase: f64) {
    // CHECK: %[[CH0:.*]] = rtio.channel : !rtio.channel<"dds", 0>
    %ch0 = rtio.channel : !rtio.channel<"dds", 0>
    // CHECK: %[[CH1:.*]] = rtio.channel : !rtio.channel<"dds", 1>
    %ch1 = rtio.channel : !rtio.channel<"dds", 1>
    // CHECK: %[[CH2:.*]] = rtio.channel : !rtio.channel<"dds", 2>
    %ch2 = rtio.channel : !rtio.channel<"dds", 2>
    // CHECK: %[[CH3:.*]] = rtio.channel : !rtio.channel<"dds", 3>
    %ch3 = rtio.channel : !rtio.channel<"dds", 3>

    // sync single event
    // CHECK: %[[E0:.*]] = rtio.pulse %[[CH0]] duration(%{{.*}}) frequency(%{{.*}}) phase(%{{.*}}) : <"dds", 0> -> !rtio.event
    %event0 = rtio.pulse %ch0 duration(%dur) frequency(%freq) phase(%phase)
        : !rtio.channel<"dds", 0> -> !rtio.event

    // CHECK: %[[E1:.*]] = rtio.pulse %[[CH1]] duration(%{{.*}}) frequency(%{{.*}}) phase(%{{.*}}) : <"dds", 1> -> !rtio.event
    %event1 = rtio.pulse %ch1 duration(%dur) frequency(%freq) phase(%phase)
        : !rtio.channel<"dds", 1> -> !rtio.event

    // sync multiple events
    // CHECK: %[[SYNC1:.*]] = rtio.sync %[[E0]], %[[E1]] : !rtio.event
    %sync1 = rtio.sync %event0, %event1 : !rtio.event

    // CHECK: %[[E2:.*]] = rtio.pulse %[[CH2]] duration(%{{.*}}) frequency(%{{.*}}) phase(%{{.*}}) : <"dds", 2> -> !rtio.event
    %event2 = rtio.pulse %ch2 duration(%dur) frequency(%freq) phase(%phase)
        : !rtio.channel<"dds", 2> -> !rtio.event

    // CHECK: %[[E3:.*]] = rtio.pulse %[[CH3]] duration(%{{.*}}) frequency(%{{.*}}) phase(%{{.*}}) : <"dds", 3> -> !rtio.event
    %event3 = rtio.pulse %ch3 duration(%dur) frequency(%freq) phase(%phase)
        : !rtio.channel<"dds", 3> -> !rtio.event

    // CHECK: rtio.sync %[[SYNC1]], %[[E2]], %[[E3]] : !rtio.event
    %sync2 = rtio.sync %sync1, %event2, %event3 : !rtio.event

    return
}

// -----

func.func @sync_no_events() {
    // expected-error@+1 {{requires at least one event to synchronize}}
    %sync = rtio.sync : !rtio.event

    return
}

// -----

// CHECK-LABEL: func.func @empty_good()
func.func @empty_good() {
    // CHECK: rtio.empty : !rtio.event
    %empty = rtio.empty : !rtio.event
    return
}

// -----

// CHECK: module @config_test attributes {rtio.config = #rtio.config<{config1 = 1 : i32, config2 = "test", nested = {a = 0 : i32, b = "test"}}>}
module @config_test attributes {
    rtio.config = #rtio.config<{
        config1 = 1 : i32,
        config2 = "test",
        nested = {a = 0 : i32, b = "test"}
    }>
} {
    // CHECK: func.func @kernel()
    func.func @kernel() {
        return
    }
}

// -----

// CHECK: module @empty_config attributes {rtio.config = #rtio.config<{}>}
module @empty_config attributes {
    rtio.config = #rtio.config<{}>
} {
    // CHECK: func.func @kernel()
    func.func @kernel() {
        return
    }
}
