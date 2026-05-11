// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <cstdint>

namespace catalyst::remote {

// Opaque session handle. Created by open(), released by close().
struct RemoteSession;

// Open a TCP session to a `host:port` executor. Returns nullptr on error.
RemoteSession *open(const char *remote_addr);

void close(RemoteSession *s);

// Load an object file into the remote JIT. Returns 0 on success, -1 on error.
int load_object_path(RemoteSession *s, const char *path);

// Look up a symbol address on the remote. Returns 0 on error.
uint64_t lookup(RemoteSession *s, const char *name);

// Run a remote function as `main(argc, argv)`.
int32_t run_as_main(RemoteSession *s, uint64_t entry_addr, int argc, const char *const *argv);

// Invoke a remote kernel. Returns 0 on success, -1 on error.
int invoke_kernel(RemoteSession *s, uint64_t entry_addr, size_t num_inputs,
                  void *const *input_descs, const size_t *input_ranks,
                  const size_t *input_elem_sizes, size_t num_outputs, void *const *output_descs,
                  const size_t *output_ranks, const size_t *output_elem_sizes);

// Last error message.
const char *last_error();

} // namespace catalyst::remote
