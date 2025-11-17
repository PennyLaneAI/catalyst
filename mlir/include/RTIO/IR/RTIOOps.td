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

#ifndef RTIO_OPS
#define RTIO_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "RTIO/IR/RTIODialect.td"

//===----------------------------------------------------------------------===//
// Event-Based API
//===----------------------------------------------------------------------===//

def RTIOChannelOp : RTIO_Op<"channel"> {
    let summary = "Define a static channel";
    let description = [{
        The channel's identity (kind, qualifiers, and channel id) is
        encoded in its result type.

        The channel ID must be explicitly specified in the type as either:
        - `?` for dynamic channel (to be resolved by channel resolution stage)
        - Non-negative integer for static channel id

        Example:
        ```mlir
        // DDS on hardware channel 0
        %ch0_dds = rtio.channel : !rtio.channel<"dds", 0>

        // DDS on hardware channel 0 for qualifiers [0]
        %ch0_t0 = rtio.channel : !rtio.channel<"dds", [0], 0>

        // DDS for dynamic channel with qualifiers [0]
        %ch_t0 = rtio.channel : !rtio.channel<"dds", [0], ?>
        ```
    }];

    let arguments = (ins);
    let results = (outs RTIOChannelType:$channel);

    let assemblyFormat = [{
        attr-dict `:` type(results)
    }];
}

def RTIOQubitToChannelOp : RTIO_Op<"qubit_to_channel"> {
    let summary = "Map a qubit to an RTIO channel";
    let description = [{
        It's a temporary operation that will be lowered to a static `rtio.channel` operation during
        the channel resolution stage. The purpose of this operation is to allow the qubit from a
        from a high-level dialect to be mapped to an `!rtio.channel`.

        Example:
        ```mlir
        // Map ion qubit to dds channel with dynamic channel id
        %ch = rtio.qubit_to_channel %ion_qubit : !ion.qubit -> !rtio.channel<"dds", ?>
        ```

        During the channel resolution stage, this operation will be replaced by a
        static `rtio.channel` operation once the qubit identity and channel mapping
        are determined at compile time.
    }];

    let arguments = (ins AnyType:$qubit);
    let results = (outs RTIOChannelType:$channel);

    let assemblyFormat = [{
        $qubit attr-dict `:` type(operands) `->` type(results)
    }];
}

def RTIOPulseOp : RTIO_Op<"pulse"> {
    let summary = "Generate an event-based pulse on a channel";
    let description = [{
        Generate a pulse on the channel. Returns an `!rtio.event` handle that can be used with
        `rtio.sync` to wait for completion or to be used by other pulse operations with the `wait`
        operand.

        Parameters:
        - channel: Target output channel
        - duration: Pulse duration in seconds (f64)
        - frequency: frequency in Hz
        - phase: Phase in radians
        - wait: Optional events to wait for before starting this pulse (for sequencing)

        Returns:
        - event: Handle to the pending pulse `!rtio.event`

        Example:
        ```mlir
        %dur = arith.constant 1.66e-07 : f64
        %freq = arith.constant 1.266e10 : f64
        %phase = arith.constant 0.0 : f64

        %event0 = rtio.pulse %ch0 duration(%dur) frequency(%freq) phase(%phase)
            : !rtio.channel<"dds", 0> -> !rtio.event
        %event1 = rtio.pulse %ch1 duration(%dur) frequency(%freq) phase(%phase)
            : !rtio.channel<"dds", 1> -> !rtio.event

        %sync = rtio.sync %event0, %event1 : !rtio.event

        %event2 = rtio.pulse %ch2 duration(%dur) frequency(%freq) phase(%phase) wait(%sync)
            : !rtio.channel<"dds", 0> -> !rtio.event
        ```
    }];

    let arguments = (ins
        RTIOChannelType:$channel,
        F64:$duration,
        F64:$frequency,
        F64:$phase,
        Optional<RTIOEventType>:$wait
    );

    let results = (outs RTIOEventType:$event);

    let assemblyFormat = [{
        $channel `duration` `(` $duration `)` `frequency` `(` $frequency `)` `phase` `(` $phase `)`
        (`wait` `(` $wait^ `)`)? attr-dict `:` type($channel) `->` type($event)
    }];
}

def RTIOSyncOp : RTIO_Op<"sync"> {
    let summary = "Synchronization barrier for event-based operations";
    let description = [{
        Wait for all specified events to complete before proceeding.
        Returns a new event handle

        Example:
        ```mlir
        // Three pulses start in parallel
        %event0 = rtio.pulse %ch0 duration(%dur0) frequency(%freq0) phase(%phase0)
            : !rtio.channel<"dds", 0> -> !rtio.event
        %event1 = rtio.pulse %ch1 duration(%dur1) frequency(%freq1) phase(%phase1)
            : !rtio.channel<"dds", 1> -> !rtio.event

        // Wait for both to complete, get sync point
        %sync = rtio.sync %event0, %event1 : !rtio.event
        ```

        The sync operation tracks dependencies between events, making it easier for the
        event scheduling.
    }];

    let arguments = (ins Variadic<RTIOEventType>:$events);
    let results = (outs RTIOEventType:$sync_event);
    let assemblyFormat = "$events attr-dict `:` type($sync_event)";

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Timeline-Based IR (ARTIQ compatible)
//===----------------------------------------------------------------------===//

def RTIONowOp : RTIO_Op<"now", [Pure]> {
    let summary = "Read the current timeline cursor";
    let description = [{
        Returns the current value of the timeline cursor in machine units (mu).

        Example:
        ```mlir
        %t0 = rtio.now : i64
        rtio.delay 100 : i64
        %t1 = rtio.now : i64
        // %t1 = %t0 + 100
        ```
    }];

    let results = (outs I64:$time);
    let assemblyFormat = "attr-dict `:` type($time)";
}

def RTIODelayOp : RTIO_Op<"delay"> {
    let summary = "Advance the timeline cursor by a relative duration";
    let description = [{
        Advances the timeline cursor by the specified duration in machine units.
        Equivalent to `rtio.at(rtio.now() + duration)`.

        Example:
        ```mlir
        // case1
        rtio.delay 1000 : i64

        // case2
        rtio.delay %dur : i64
        ```
    }];

    let arguments = (ins I64:$duration);
    let assemblyFormat = "$duration attr-dict `:` type($duration)";
}

def RTIOAtOp : RTIO_Op<"at"> {
    let summary = "Set the timeline cursor to an absolute timestamp";
    let description = [{
        Moves the timeline cursor to a specific absolute timestamp.

        Pre-condition: The new time must satisfy `t >= rtio_counter_mu()` (hardware time),
        otherwise an RTIO underflow error occurs at runtime. Note that `t` can be less than
        the current timeline cursor value (from `rtio.now`), which is how parallel execution
        is implemented.

        Example:
        ```mlir
        %t_start = rtio.now : i64  // e.g. t_start = 1000 mu

        // First pulse lane
        rtio.on %ch0
        rtio.delay 500 : i64
        rtio.off %ch0
        // rtio.now would return 1500 mu (1000 mu + 500 mu)
        %t0 = rtio.now : i64

        // Second pulse lane (parallel with first lane)

        // Rewind to t_start = 1000 mu (< 1500 mu, but >= hardware counter)
        rtio.at %t_start : i64
        rtio.on %ch1
        rtio.delay 300 : i64
        rtio.off %ch1
        // rtio.now would return 1300 mu (1000 mu + 300 mu)
        %t1 = rtio.now : i64

        // Sync: advance to the maximum end time (1500 mu)
        %t_max = arith.maxui %t0, %t1 : i64
        rtio.at %t_max : i64  // advance to 1500 mu
        ```
    }];

    let arguments = (ins I64:$time);
    let assemblyFormat = "$time attr-dict `:` type($time)";
}

def RTIOSetFrequencyOp : RTIO_Op<"set_frequency"> {
    let summary = "Configure the frequency of a DDS channel";
    let description = [{
        Programs the DDS frequency tuning word (ftw) to generate a specific frequency.
        The frequency is specified in Hz as a f64.

        Example:
        ```mlir
        %freq = arith.constant 1.266300000e10 : f64  // 12.663 GHz
        rtio.set_frequency %ch0, %freq : !rtio.channel<"dds", 0>, f64
        ```

        Assume phase is 0 and amplitude is 1.
        It's equivalent to call the `AD9910.set` function with the following arguments:

        ```llvm
        %set_func = load ptr, ptr @F.artiq.coredevice.ad9910.AD9910.set
        call double %set_func(
            ptr %env,
            ptr %ch0,
            double 1.266300000e+10,  ; frequency
            double 0.000000e+00,     ; phase
            double 1.000000e+00      ; amplitude
        )
        ```
    }];

    let arguments = (ins
        RTIOChannelType:$channel,
        F64:$frequency
    );
    let assemblyFormat = "$channel `,` $frequency attr-dict `:` type($channel) `,` type($frequency)";
}

def RTIOSetPhaseOp : RTIO_Op<"set_phase"> {
    let summary = "Configure the phase of a DDS channel";
    let description = [{
        Programs the DDS phase offset register to set the carrier phase.
        The phase is specified in radians as a f64.

        Example:
        ```mlir
        // pi/2 phase shift
        %pi_2 = arith.constant 1.5707963267948966 : f64
        rtio.set_phase %ch0, %pi_2 : !rtio.channel<"dds", 0>, f64
        ```
    }];

    let arguments = (ins
        RTIOChannelType:$channel,
        F64:$phase
    );
    let assemblyFormat = "$channel `,` $phase attr-dict `:` type($channel) `,` type($phase)";
}

def RTIOSetAmplitudeOp : RTIO_Op<"set_amplitude"> {
    let summary = "Configure the amplitude of a DDS channel";
    let description = [{
        Programs the DDS amplitude

        Example:
        ```mlir
        %amp = arith.constant 1.0 : f64
        rtio.set_amplitude %ch0, %amp : !rtio.channel<"dds", 0>, f64
        ```
    }];

    let arguments = (ins
        RTIOChannelType:$channel,
        F64:$amplitude
    );
    let assemblyFormat = "$channel `,` $amplitude attr-dict `:` type($channel) `,` type($amplitude)";
}

def RTIOOnOp : RTIO_Op<"on"> {
    let summary = "Turn on a channel output";
    let description = [{
        Activates the output of a channel at the current cursor.
        For DDS channels, this typically enables the RF output switch.

        Example:
        ```mlir
        rtio.at %t_start : !rtio.time
        rtio.on %ch
        rtio.delay 1000 : i64
        rtio.off %ch
        ```
    }];

    let arguments = (ins RTIOChannelType:$channel);
    let assemblyFormat = "$channel attr-dict `:` type($channel)";
}

def RTIOOffOp : RTIO_Op<"off"> {
    let summary = "Turn off a channel output";
    let description = [{
        Deactivates the output of a channel at the current cursor.
        For DDS channels, this typically disables the RF output switch.

        Example:
        ```mlir
        rtio.off %ch : !rtio.channel<"dds", 0>
        ```
    }];

    let arguments = (ins RTIOChannelType:$channel);
    let assemblyFormat = "$channel attr-dict `:` type($channel)";
}

#endif // RTIO_OPS

