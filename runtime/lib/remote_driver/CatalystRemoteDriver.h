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
//
// C ABI for the catalyst remote driver.
#ifndef CATALYST_REMOTE_DRIVER_H
#define CATALYST_REMOTE_DRIVER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CatalystRemoteSession_ CatalystRemoteSession;

/// Connect to a `catalyst-executor` and JIT-link `kernel_path` into its address space.
/// `error_buf`/`error_buf_size` receive a NUL-terminated error message on failure
CatalystRemoteSession *catalyst_remote_open(const char *kernel_path, const char *remote_address,
                                            char *error_buf, size_t error_buf_size);

/// Close the session and release all remote allocated resources
void catalyst_remote_close(CatalystRemoteSession *s);

/// Look up an exported symbol by name in the JIT module.
/// Sets `*out_addr` to the symbol's address on success.
int catalyst_remote_lookup(CatalystRemoteSession *s, const char *name, uint64_t *out_addr);

/// Allocate `size` bytes of writable memory
/// Sets `*out_addr` to the address on success.
int catalyst_remote_alloc(CatalystRemoteSession *s, size_t size, uint64_t *out_addr);

/// Free memory previously allocated by `catalyst_remote_alloc`
int catalyst_remote_free(CatalystRemoteSession *s, uint64_t addr, size_t size);

/// Copy `size` bytes from host buffer `host_data` to `remote_addr`
int catalyst_remote_write(CatalystRemoteSession *s, uint64_t remote_addr, const void *host_data,
                          size_t size);

/// Copy `size` bytes from `remote_addr` to `host_data`
int catalyst_remote_read(CatalystRemoteSession *s, uint64_t remote_addr, void *host_data,
                         size_t size);

/// Run a function with `argc` arguments `argv`, sets `*out_rc` to the return value on success.
int catalyst_remote_run_as_main(CatalystRemoteSession *s, uint64_t entry_addr, int argc,
                                const char *const *argv, int32_t *out_rc);

/// Invoke a `void(void*, void*, ...)` function with up to 8 pointer arguments
int catalyst_remote_invoke_pyface(CatalystRemoteSession *s, uint64_t entry_addr,
                                  const uint64_t *arg_addrs, int n_args);

#ifdef __cplusplus
}
#endif
#endif // CATALYST_REMOTE_DRIVER_H
