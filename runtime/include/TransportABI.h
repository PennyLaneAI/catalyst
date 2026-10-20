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

#pragma once
#ifndef TRANSPORTABI_H
#define TRANSPORTABI_H

#include <stddef.h>
#include <stdint.h>

// How many bytes a `str` argument occupies in either layout: a fixed, NUL-padded field.
#define CATALYST_TRANSPORT_STR_BYTES 256

#ifdef __cplusplus
extern "C" {
#endif

// LLVM ORC's CWrapperFunctionResult, exported so the adapters need not link LLVM.
typedef struct {
    union {
        char *value_ptr;
        char value[8];
    } data;
    size_t size;
} CatalystWrapperResult;

/// The highest CatalystCoprocessorFnInfo version this runtime knows the fields of.
#define CATALYST_COPROCESSOR_FN_ABI_VERSION 1

/// CatalystCoprocessorFnInfo::flags: the function takes the CatalystCoprocessorFrameFn signature.
#define CATALYST_COPROCESSOR_FN_FRAME_FIELDS (1u << 0)

/**
 * A per-message coprocessor function that is handed the decoded fields of each message, rather
 * than the message's frame. `payload` points at the `len` payload bytes, and `fn_id` is the id the
 * controller staged the message with, so one function can serve several decoders. It writes its
 * reply into `out` (capacity `out_cap`) and returns the bytes written, or SIZE_MAX
 * (COPROCESSOR_FN_ERROR) on failure. On the 16 B frame, `len` is 8 and the payload is zero-padded.
 * A library declares that its function takes this signature with
 * CATALYST_COPROCESSOR_FN_FRAME_FIELDS in its info's `flags`.
 */
typedef size_t (*CatalystCoprocessorFrameFn)(const void *payload, size_t len, uint32_t fn_id,
                                             void *out, size_t out_cap, void *ctx);

/**
 * The lifecycle hooks of a coprocessor function or launcher. The library defining the function
 * may export them as `const CatalystCoprocessorFnInfo *<symbol>_info(void)`, where `<symbol>` is
 * the function's name. The runtime looks `<symbol>_info` up only in the library that defines
 * `<symbol>`. A function without it is bound with a null ctx.
 *
 * - `abi_version`: the version whose fields the library fills, at least 1. The struct only grows
 *   at its end, and the runtime reads only the fields of the versions it knows. A field added by
 *   a later version must be optional: null or zero keeps the behaviour of the earlier versions.
 * - `flags`: CATALYST_COPROCESSOR_FN_* bits, zero for none. CATALYST_COPROCESSOR_FN_FRAME_FIELDS
 *   declares that a per-message function takes the CatalystCoprocessorFrameFn signature. Bits
 *   this runtime does not know are refused.
 * - `init` (may be null): called once, before the function is bound, with the node's
 *   `fn.`-prefixed config keys, prefix removed, as `key=value;...`. Returns the ctx the function
 *   is called with, or null if the function cannot be configured. Catalyst's frontend adds the
 *   controller's message sizes as `in_bytes=<n>` and `out_bytes=<n>`, so `init` must accept both
 *   keys. A function may check them against what it processes and return null on a mismatch.
 * - `fini` (may be null): releases the ctx `init` returned, once the session using it has stopped.
 *
 * The returned pointer must stay valid for the life of the library.
 */
typedef struct {
    uint32_t abi_version;
    uint32_t flags;
    void *(*init)(const char *config);
    void (*fini)(void *ctx);
} CatalystCoprocessorFnInfo;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TRANSPORTABI_H
