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

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TRANSPORTABI_H
