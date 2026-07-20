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

// DataPath enum (mirrors catalyst::transport::DataPath)
enum {
    CATALYST_TRANSPORT_PATH_CPU_VERBS = 0,
    CATALYST_TRANSPORT_PATH_GPU_ENGINE = 1,
    CATALYST_TRANSPORT_PATH_OTHER = 2,
};

// MemKind enum (mirrors catalyst::transport::MemKind)
enum {
    CATALYST_TRANSPORT_MEM_CPU_RAM = 0,
    CATALYST_TRANSPORT_MEM_GPU_HBM = 1,
    CATALYST_TRANSPORT_MEM_DDR = 2,
    CATALYST_TRANSPORT_MEM_OTHER = 3,
};

// ibverbs access flags for the advertised reply region
enum {
    CATALYST_TRANSPORT_ACCESS_REPLY = 7,
};

// Registered memory region handed back to the caller
typedef struct {
    void *addr;
    uint64_t size;
    uint32_t lkey;
    uint32_t rkey;
    int32_t kind; // one of MemKind enum values
} CatalystTransportMemRegion;

// Remote peer region descriptor
typedef struct {
    uint32_t rkey;
    uint64_t remote_addr;
    uint64_t size;
} CatalystTransportPeerRef;

// Create a controller session from a named backend plugin `.so` (dlopen'd by the runtime).
// `config` is the backend's "key=value;..." string. Returns NULL on failure.
CatalystTransportSession *__catalyst__transport__controller_create(const char *backend_lib,
                                                                   const char *config);

void __catalyst__transport__close(CatalystTransportSession *s);
int __catalyst__transport__connect(CatalystTransportSession *s, const char *peer,
                                   uint16_t oob_port);
int __catalyst__transport__alloc_reply(CatalystTransportSession *s, uint64_t size, int32_t mem_kind,
                                       uint32_t access, CatalystTransportMemRegion *out);

int __catalyst__transport__exchange_keys(CatalystTransportSession *s,
                                         CatalystTransportPeerRef *out);
int __catalyst__transport__establish_channel(CatalystTransportSession *s, int32_t data_path,
                                             const CatalystTransportPeerRef *peer);
int __catalyst__transport__commit_work_item(CatalystTransportSession *s, uint32_t work_item_idx,
                                            uint64_t in_bytes, uint64_t out_bytes);
void *__catalyst__transport__data_slot(CatalystTransportSession *s);
int __catalyst__transport__kick(CatalystTransportSession *s, uint32_t work_item_idx);
int __catalyst__transport__collect(CatalystTransportSession *s, void *correction, uint64_t bytes);
uint64_t __catalyst__transport__last_rtt_ns(CatalystTransportSession *s);
void __catalyst__transport__stop(CatalystTransportSession *s);
void __catalyst__transport__destroy(CatalystTransportSession *s);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TRANSPORTCAPI_H
