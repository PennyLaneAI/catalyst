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

// What the compiler passes for each operand of an in-process external call.
typedef struct {
    int64_t rank;
    void *data_aligned;
    int8_t dtype;
} CatalystEncodedMemref;

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

/**
 * The lifecycle hooks of a coprocessor function or launcher. The library defining the function
 * may export them as `const CatalystCoprocessorFnInfo *<symbol>_info(void)`, where `<symbol>` is
 * the function's name. The runtime looks `<symbol>_info` up only in the library that defines
 * `<symbol>`. A function without it is bound with a null ctx.
 *
 * - `abi_version`: the version whose fields the library fills, at least 1. The struct only grows
 *   at its end, and the runtime reads only the fields of the versions it knows. A field added by
 *   a later version must be optional: null or zero keeps the behaviour of the earlier versions.
 * - `reserved`: zero.
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
    uint32_t reserved;
    void *(*init)(const char *config);
    void (*fini)(void *ctx);
} CatalystCoprocessorFnInfo;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TRANSPORTABI_H
