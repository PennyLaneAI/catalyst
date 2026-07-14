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
#ifndef EXECUTORCAPI_H
#define EXECUTORCAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Executor Runtime API. This is expected to be called by the host program to establish a executor
// session. Including open a session, send the binary to the executor, launch the kernel on the
// executor and close the session.
int __catalyst__executor__open(const char *addr);
int __catalyst__executor__send_binary(const char *addr, const char *path, uint32_t format);
void __catalyst__executor__launch(const char *addr, const char *entry_symbol, size_t num_inputs,
                                  void *const *input_descs, const size_t *input_ranks,
                                  const size_t *input_elem_sizes, size_t num_outputs,
                                  void *const *output_descs, const size_t *output_ranks,
                                  const size_t *output_elem_sizes);
int __catalyst__executor__call_wrapper(const char *addr, const char *symbol, const char *args_buf,
                                       size_t args_size, void **out_buf, size_t *out_size);
void __catalyst__executor__free_result(void *buf);

int __catalyst__executor__close();

#ifdef __cplusplus
} // extern "C"
#endif

#endif
