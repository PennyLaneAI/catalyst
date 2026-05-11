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
#ifndef REMOTECAPI_H
#define REMOTECAPI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Remote Runtime API. This is expected to be called by the host program to establish a remote
// session. Including open a session, send the binary to the remote, launch the kernel on the remote
// and close the session.
int __catalyst__remote__open(const char *addr);
int __catalyst__remote__send_binary(const char *addr, const char *path, uint32_t format);
void __catalyst__remote__launch(const char *addr, const char *entry_symbol, size_t num_inputs,
                                void *const *input_descs, const size_t *input_ranks,
                                const size_t *input_elem_sizes, size_t num_outputs,
                                void *const *output_descs, const size_t *output_ranks,
                                const size_t *output_elem_sizes);
int __catalyst__remote__close();
const char *__catalyst__remote__last_error();

#ifdef __cplusplus
} // extern "C"
#endif

#endif
