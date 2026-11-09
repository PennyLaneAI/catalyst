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

/**
 * @file DeviceExtensions.h
 * Optional, versioned device extensions, as plain C function tables.
 *
 * A device library built against this header exports `<Name>DeviceAbiVersion`
 * (emitted by `GENERATE_DEVICE_FACTORY`). A device without that symbol is
 * version 0, and the runtime never queries it for tables.
 *
 * From version 1, `QuantumDevice::QueryInterface(name, min_version)` returns a
 * table, or null. Every table opens with `CatalystDeviceTableHeader`. Rules for
 * evolving a table:
 *
 * - Fields are only appended. A later field is optional: null keeps the
 *   earlier behaviour. A field never changes meaning and is never removed.
 * - Appending a field bumps the table's version. The runtime reads a field
 *   only when `header.abi_version` is at least the version that added it.
 * - A breaking change is a new table name, such as "Delegated2".
 *
 * Every table describes semantics any target can implement, hardware included.
 */

#pragma once
#ifndef DEVICE_EXTENSIONS_H
#define DEVICE_EXTENSIONS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The device ABI this header describes.
#define CATALYST_DEVICE_ABI_VERSION 1u

typedef struct {
    /// The version of this table the device fills, at least 1.
    uint32_t abi_version;
    /// Zero.
    uint32_t reserved;
    /// Passed back to every function of the table. Valid for the device's lifetime.
    void *self;
} CatalystDeviceTableHeader;

/// How postselection is applied on a delegated program.
typedef enum {
    /// Project the state and renormalise; fail on zero probability.
    CATALYST_POSTSELECT_FILL_SHOTS = 0,
    /// Keep every shot and mark the ones that fail postselection dead.
    CATALYST_POSTSELECT_HW_LIKE = 1,
} CatalystPostselectMode;

/// Passed as `postselect` when a measurement has no postselection.
#define CATALYST_NO_POSTSELECT (-1)

/**
 * "Delegated": feed-forward handed to the device as data.
 *
 * After `begin_program`, a measurement returns a result handle: a device-owned
 * reference to that measurement's outcome in every shot. Compiled code never
 * reads it as a bit; the device may defer simulation until an output is
 * requested, and may simulate a recorded prefix at any time.
 *
 * Ops between `push_predicate(c)` and the matching `pop_predicate` apply only
 * on shots where every enclosing predicate holds. A measurement under a false
 * predicate does not collapse the state and yields 0. Allocation and release
 * ignore predicates.
 *
 * Version 1: every field below.
 */
typedef struct {
    CatalystDeviceTableHeader header;
    void (*begin_program)(void *self, uint32_t postselect_mode);
    void (*push_predicate)(void *self, RESULT *condition);
    void (*pop_predicate)(void *self);
    RESULT *(*result_constant)(void *self, bool value);
    RESULT *(*result_xor)(void *self, RESULT *a, RESULT *b);
    RESULT *(*result_not)(void *self, RESULT *a);
    /// `if_true` on shots where `condition` holds, `if_false` on the others.
    RESULT *(*result_select)(void *self, RESULT *condition, RESULT *if_true, RESULT *if_false);
    /// Compiled code holds `result` no longer.
    void (*release_result)(void *self, RESULT *result);
    /// Measure the Pauli word, negated if asked. `postselect` is 0, 1 or CATALYST_NO_POSTSELECT.
    RESULT *(*pauli_measure)(void *self, const char *word, const QubitIdType *wires,
                             size_t num_wires, bool negated, int32_t postselect);
    /// Outcomes of `results` in every shot, bit-packed: shot s is row s of `row_bytes` bytes, and
    /// result i is bit (i % 8) of byte (i / 8). May be null.
    void (*result_samples)(void *self, RESULT *const *results, size_t num_results, uint8_t *out,
                           size_t num_shots, size_t row_bytes);
} CatalystDelegatedTable;

/// The compiler's verdict on a gadget.
typedef enum {
    /// The compiler did not check the gadget.
    CATALYST_GADGET_UNCHECKED = 0,
    /// The gadget implements its source rotation exactly.
    CATALYST_GADGET_VERIFIED = 1,
    /// The gadget does not implement its source rotation.
    CATALYST_GADGET_MISMATCH = 2,
} CatalystGadgetVerdict;

/**
 * "Gadget": brackets around a span of ops that implements one rotation
 * exp(-i theta/2 P).
 *
 * A bracket informs scheduling, accounting and verification. By default a
 * target runs the span as given. It may substitute its own implementation of
 * the source rotation only after checking the substitute itself, or under an
 * explicit, reported user opt-in. Outcome bits read after the span keep their
 * distribution. Brackets do not nest.
 *
 * Version 1: every field below.
 */
typedef struct {
    CatalystDeviceTableHeader header;
    void (*begin_gadget)(void *self, const char *word, const QubitIdType *wires, size_t num_wires,
                         double theta, uint32_t verdict);
    void (*end_gadget)(void *self);
} CatalystGadgetTable;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // DEVICE_EXTENSIONS_H
