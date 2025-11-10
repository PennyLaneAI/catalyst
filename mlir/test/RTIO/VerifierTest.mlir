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

// RUN: quantum-opt %s --split-input-file --verify-diagnostics

////////////////////////
// Channel Operations //
////////////////////////

func.func @channel_good() {
    // Smoke test for valid channel operations
    %ch0 = rtio.channel : !rtio.channel<"dds", 0>
    %ch0_q1 = rtio.channel : !rtio.channel<"dds", [0], 0>
    %ch0_q2 = rtio.channel : !rtio.channel<"dds", [0, "t0"], 0>
    %ch_explict_dyn = rtio.channel : !rtio.channel<"dds", ?>
    %ch_explict_dyn_q1 = rtio.channel : !rtio.channel<"dds", [0], ?>
    %ch_explict_dyn_q2 = rtio.channel : !rtio.channel<"dds", [0, "t0"], ?>

    %ch0_implicit_dyn = rtio.channel : !rtio.channel<"dds">
    %ch_implicit_dyn_q1 = rtio.channel : !rtio.channel<"dds", [0]>
    %ch_implicit_dyn_q2 = rtio.channel : !rtio.channel<"dds", [0, "t0"]>
    return
}

// -----

func.func @channel_negative_id() {
    // expected-error@+1 {{static channel ID must be non-negative}}
    %ch0_negative = rtio.channel : !rtio.channel<"dds", -1>
}

// -----

func.func @qubit_to_channel_good(%qubit: !ion.qubit) {
    // Smoke test for qubit_to_channel
    %ch0 = rtio.qubit_to_channel %qubit : !ion.qubit -> !rtio.channel<"dds", ["transition_0"], ?>
    return
}

// -----

////////////////////////////
// Event-Based Operations //
////////////////////////////

func.func @pulse_basic(%dur: i64, %freq: f64, %phase: f64) {
    %ch0 = rtio.channel : !rtio.channel<"dds", 0>

    %event = rtio.pulse %ch0 duration(%dur) frequency(%freq) phase(%phase)
        : !rtio.channel<"dds", 0> -> !rtio.event

    return
}

// -----

func.func @pulse_with_wait(%dur: i64, %freq: f64, %phase: f64) {
    %ch0 = rtio.channel : !rtio.channel<"dds", 0>

    %event0 = rtio.pulse %ch0 duration(%dur) frequency(%freq) phase(%phase)
        : !rtio.channel<"dds", 0> -> !rtio.event

    %event1 = rtio.pulse %ch0 duration(%dur) frequency(%freq) phase(%phase) wait(%event0)
        : !rtio.channel<"dds", 0> -> !rtio.event

    return
}


// -----

func.func @sync_basic(%dur: i64, %freq: f64, %phase: f64) {
    %ch0 = rtio.channel : !rtio.channel<"dds", 0>
    %ch1 = rtio.channel : !rtio.channel<"dds", 1>
    %ch2 = rtio.channel : !rtio.channel<"dds", 2>
    %ch3 = rtio.channel : !rtio.channel<"dds", 3>

    // sync single event
    %event0 = rtio.pulse %ch0 duration(%dur) frequency(%freq) phase(%phase)
        : !rtio.channel<"dds", 0> -> !rtio.event

    %event1 = rtio.pulse %ch1 duration(%dur) frequency(%freq) phase(%phase)
        : !rtio.channel<"dds", 1> -> !rtio.event

    // sync multiple events
    %sync1 = rtio.sync %event0, %event1 : !rtio.event

    %event2 = rtio.pulse %ch2 duration(%dur) frequency(%freq) phase(%phase)
        : !rtio.channel<"dds", 2> -> !rtio.event

    %event3 = rtio.pulse %ch3 duration(%dur) frequency(%freq) phase(%phase)
        : !rtio.channel<"dds", 3> -> !rtio.event

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

///////////////////////////////
// Timeline-Based Operations //
///////////////////////////////

func.func @timeline_now() {
    %t = rtio.now : i64
    return
}

// -----

func.func @timeline_at() {
    %t = arith.constant 1000 : i64
    %delay = arith.constant 500 : i64
    %now = rtio.now : i64
    rtio.at %t : i64
    rtio.delay %delay : i64

    // rewind to the start time
    rtio.at %now : i64
    return
}

// -----

func.func @timeline_dalay() {
    %delay = arith.constant 500 : i64
    rtio.delay %delay : i64
    return
}

// -----

func.func @set_frequency_dds_good(%freq: f64) {
    %ch = rtio.channel : !rtio.channel<"dds", 0>
    rtio.set_frequency %ch, %freq : !rtio.channel<"dds", 0>, f64
    return
}


// -----

func.func @set_phase_dds_good(%phase: f64) {
    %ch = rtio.channel : !rtio.channel<"dds", 0>
    rtio.set_phase %ch, %phase : !rtio.channel<"dds", 0>, f64

    return
}

// -----

func.func @set_amplitude_dds_good(%amp: f64) {
    %ch = rtio.channel : !rtio.channel<"dds", 0>
    rtio.set_amplitude %ch, %amp : !rtio.channel<"dds", 0>, f64
    return
}

// -----

func.func @ttl_on_dds_good() {
    %ch = rtio.channel : !rtio.channel<"dds", 0>
    rtio.on %ch : !rtio.channel<"dds", 0>
    return
}

// -----

func.func @ttl_off_dds_good() {
    %ch = rtio.channel : !rtio.channel<"dds", 0>
    rtio.off %ch : !rtio.channel<"dds", 0>
    return
}
