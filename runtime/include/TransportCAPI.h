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

// TransportCAPI.h - C entry points the Catalyst compiler emits to drive a transport session.
//
// The backend is a separate plugin `.so` the runtime dlopen's at session create (see
// TransportBackend.h)

#pragma once
#ifndef TRANSPORTCAPI_H
#define TRANSPORTCAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque transport session handle
typedef struct CatalystTransportSession CatalystTransportSession;

// Return codes: 0 == success; negative == error
enum {
    CATALYST_TRANSPORT_OK = 0,
    CATALYST_TRANSPORT_ERR = -1,         // Generic exception
    CATALYST_TRANSPORT_ERR_MEMORY = -2,  // Memory error
    CATALYST_TRANSPORT_ERR_TIMEOUT = -3, // Timeout error
    CATALYST_TRANSPORT_ERR_STUCK = -4,   // Something got stuck
};

// Session role (mirrors catalyst::transport::Role in the dialect).
enum {
    CATALYST_TRANSPORT_ROLE_CONTROLLER = 0,
    CATALYST_TRANSPORT_ROLE_COPROCESSOR = 1,
};

// Create a session from a named backend plugin `.so` (dlopen'd by the runtime). `role` selects
// which factory symbol is looked up (controller vs coprocessor). `config` is the backend's string.
// Returns NULL on failure.
// `key` registers the session under (role, key) for later get_session; empty = not registered.
CatalystTransportSession *__catalyst__transport__create(const char *backend_lib, const char *config,
                                                        int32_t role, const char *key);

// Resolve the live session registered at create under (`role`, `key`).
// Lets a session brought up in one function be used in another.
CatalystTransportSession *__catalyst__transport__get_session(int32_t role, const char *key);

// Bring-up. The peer region learned in exchange_keys is kept inside the session, so
// establish_channel takes no peer handle. The *_async variants run on a worker thread and return a
// token to await with barrier.
int __catalyst__transport__connect(CatalystTransportSession *s, const char *peer,
                                   uint16_t oob_port);
int64_t __catalyst__transport__connect_async(CatalystTransportSession *s, const char *peer,
                                             uint16_t oob_port);
int __catalyst__transport__exchange_keys(CatalystTransportSession *s);
int64_t __catalyst__transport__exchange_keys_async(CatalystTransportSession *s);
int __catalyst__transport__barrier(int64_t token);
int __catalyst__transport__establish_channel(CatalystTransportSession *s, const char *data_path);

// Coprocessor-only: bind the function run per received message, resolved by runtime symbol name.
int __catalyst__transport__set_coprocessor_fn(CatalystTransportSession *s, const char *symbol);

// Controller-only: work items + kick.
int __catalyst__transport__commit_work_item(CatalystTransportSession *s, uint32_t work_item_idx,
                                            uint64_t in_bytes, uint64_t out_bytes);
void *__catalyst__transport__data_slot(CatalystTransportSession *s);
int __catalyst__transport__kick(CatalystTransportSession *s, uint32_t work_item_idx);

// Run / collect / teardown.
void __catalyst__transport__start(CatalystTransportSession *s);
int __catalyst__transport__collect(CatalystTransportSession *s, void *reply, uint64_t reply_bytes);
uint64_t __catalyst__transport__last_rtt_ns(CatalystTransportSession *s);
void __catalyst__transport__stop(CatalystTransportSession *s);
void __catalyst__transport__destroy(CatalystTransportSession *s);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TRANSPORTCAPI_H
