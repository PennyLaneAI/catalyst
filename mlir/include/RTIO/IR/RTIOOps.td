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

def RTIOChannelOp : RTIO_Op<"channel", [Pure]> {
    let summary = "Define a channel";
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

def RTIOQubitToChannelOp : RTIO_Op<"qubit_to_channel", [Pure]> {
    let summary = "Map a qubit to an RTIO channel";
    let description = [{
        It's a temporary operation that will be lowered to a static `rtio.channel` operation during
        the channel resolution stage. The purpose of this operation is to allow a qubit from a
        high-level dialect to be mapped to an `!rtio.channel`.

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

    let extraClassDeclaration = [{
        void setWait(mlir::Value waitEvent) {
            if (waitEvent) {
                getWaitMutable().assign(waitEvent);
            } else {
                getWaitMutable().clear();
            }
        }
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

def RTIOEmptyOp : RTIO_Op<"empty", [Pure]> {
    let summary = "Create an empty event for sequencing";
    let description = [{
        Creates an empty event that can be used for initializing the first event
        or when you need an event handle without performing any actual hardware operation.

        The empty event acts as a no-op synchronization point that completes immediately,
        allowing to establish event chains without triggering actions.

        Example:
        ```mlir
        // Initialize with empty event for loop
        %init_event = rtio.empty : !rtio.event
        %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%event = %init_event) -> (!rtio.event) {
            // First pulse waits on event from previous iteration (or init_event in first iteration)
            %e0 = rtio.pulse %ch duration(%dur) frequency(%freq) phase(%phase) wait(%event)
                : !rtio.channel<"dds", 0> -> !rtio.event
            %e1 = rtio.pulse %ch duration(%dur) frequency(%freq) phase(%phase) wait(%event)
                : !rtio.channel<"dds", 0> -> !rtio.event
            %sync = rtio.sync %e0, %e1 : !rtio.event
            scf.yield %sync : !rtio.event
        }
        ```
    }];

    let results = (outs RTIOEventType:$event);
    let assemblyFormat = "attr-dict `:` type($event)";
}

#endif // RTIO_OPS

