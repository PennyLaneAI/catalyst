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

// The plugin ABI for out-of-tree transport backends.
//
// A transport backend is a shared library that implements a TransportSession role (controller or
// coprocessor) and exports the matching factory symbol.

#pragma once
#ifndef TRANSPORTBACKEND_H
#define TRANSPORTBACKEND_H

#include <cstdint>
#include <exception>
#include <iostream>

#include "Transport.hpp"

#define CATALYST_TRANSPORT_CONTROLLER_FACTORY_SYMBOL "CatalystTransportControllerFactory"
#define CATALYST_TRANSPORT_COPROCESSOR_FACTORY_SYMBOL "CatalystTransportCoprocessorFactory"

// Backend ABI version and capabilities.
//
// A backend exports `uint32_t CatalystTransportAbiVersion(void)` and
// `const CatalystTransportCaps *CatalystTransportQueryCaps(void)`; the
// CATALYST_TRANSPORT_BACKEND_CAPS macro below defines both. The loader checks them before calling
// a factory:
//   - no CatalystTransportAbiVersion: a backend built before versioning, loaded as version 1, which
//     reports no capabilities and whose sessions the runtime deletes;
//   - a version newer than CATALYST_TRANSPORT_ABI_VERSION: refused, naming both versions;
//   - a role missing from `roles`: refused;
//   - a coprocessor from a backend older than version 3: refused, since version 3 changed the
//     CoprocessorFn signature the coprocessor calls.
// Otherwise a backend built against an older version keeps working, because the session interface
// in Transport.hpp follows these rules:
//   - virtual methods are only appended, never removed or reordered;
//   - the session base classes gain no data members;
//   - the runtime never calls a method added in version N on a backend reporting a lower version.
#define CATALYST_TRANSPORT_ABI_VERSION 3u
#define CATALYST_TRANSPORT_ABI_VERSION_SYMBOL "CatalystTransportAbiVersion"
#define CATALYST_TRANSPORT_QUERY_CAPS_SYMBOL "CatalystTransportQueryCaps"

// CatalystTransportCaps::roles
#define CATALYST_TRANSPORT_ROLE_BIT_CONTROLLER (1u << 0)
#define CATALYST_TRANSPORT_ROLE_BIT_COPROCESSOR (1u << 1)
// CatalystTransportCaps::frame_modes
#define CATALYST_TRANSPORT_FRAME_BL16 (1u << 0) // 16 B frame: 8 B payload, fn_id, seq
#define CATALYST_TRANSPORT_FRAME_BLV (1u << 1)  // variable length: 32 B header, payload, completion
// CatalystTransportCaps::ordering
#define CATALYST_TRANSPORT_ORDER_IN_ORDER 0u // a payload lands before the word written after it
// CatalystTransportCaps::mem_kinds and ::conventions, e.g. CATALYST_TRANSPORT_MEM_BIT(CpuRam).
#define CATALYST_TRANSPORT_MEM_BIT(KIND)                                                           \
    (1u << static_cast<unsigned>(catalyst::transport::MemKind::KIND))
#define CATALYST_TRANSPORT_CONV_BIT(CONVENTION)                                                    \
    (1u << static_cast<unsigned>(catalyst::transport::CoprocConvention::CONVENTION))

extern "C" {
/// What a backend library provides. Fields are only ever appended; a backend fills those of the
/// ABI version it reports.
typedef struct {
    uint32_t abi_version;       // CATALYST_TRANSPORT_ABI_VERSION the backend was built with
    uint32_t roles;             // CATALYST_TRANSPORT_ROLE_BIT_*
    uint32_t mem_kinds;         // CATALYST_TRANSPORT_MEM_BIT(...)
    uint32_t frame_modes;       // CATALYST_TRANSPORT_FRAME_*
    uint32_t max_payload_bytes; // largest request payload per message, whatever the frame mode
    uint32_t ordering;          // CATALYST_TRANSPORT_ORDER_*
    const char *name;           // e.g. "cpu_verbs"
    uint32_t max_reply_bytes;   // largest reply per message
    uint32_t conventions;       // CATALYST_TRANSPORT_CONV_BIT(...); 0 for controllers
    uint32_t max_work_items;    // work items a controller can commit
    uint32_t reserved;          // zero
    void (*destroy_session)(void *session); // deletes a session this library created
} CatalystTransportCaps;

using CatalystTransportAbiVersionFn = uint32_t();
using CatalystTransportQueryCapsFn = const CatalystTransportCaps *();
}

// Defines CatalystTransportAbiVersion and CatalystTransportQueryCaps for an in-tree backend.
// e.g. CATALYST_TRANSPORT_BACKEND_CAPS("cpu_verbs", CATALYST_TRANSPORT_ROLE_BIT_CONTROLLER,
//          CATALYST_TRANSPORT_MEM_BIT(CpuRam),
//          CATALYST_TRANSPORT_FRAME_BL16 | CATALYST_TRANSPORT_FRAME_BLV, 0, 1024, 1024, 1)
#define CATALYST_TRANSPORT_BACKEND_CAPS(NAME, ROLES, MEM_KINDS, FRAME_MODES, CONVENTIONS,          \
                                        MAX_PAYLOAD, MAX_REPLY, WORK_ITEMS)                        \
    extern "C" uint32_t CatalystTransportAbiVersion() { return CATALYST_TRANSPORT_ABI_VERSION; }   \
    extern "C" const CatalystTransportCaps *CatalystTransportQueryCaps() {                         \
        static const CatalystTransportCaps caps = {                                                \
            CATALYST_TRANSPORT_ABI_VERSION,                                                        \
            (ROLES),                                                                               \
            (MEM_KINDS),                                                                           \
            (FRAME_MODES),                                                                         \
            (MAX_PAYLOAD),                                                                         \
            CATALYST_TRANSPORT_ORDER_IN_ORDER,                                                     \
            (NAME),                                                                                \
            (MAX_REPLY),                                                                           \
            (CONVENTIONS),                                                                         \
            (WORK_ITEMS),                                                                          \
            0,                                                                                     \
            [](void *s) { delete static_cast<catalyst::transport::TransportSession *>(s); }};      \
        return &caps;                                                                              \
    }

// The factory signatures backends must export with C linkage.
extern "C" {
using CatalystTransportControllerFactoryFn = catalyst::transport::ControllerSession *(const char *);
using CatalystTransportCoprocessorFactoryFn =
    catalyst::transport::CoprocessorSession *(const char *);
}

// A helper template macro to generate the <IDENTIFIER>Factory function.
// e.g. GENERATE_TRANSPORT_CONTROLLER_FACTORY(CatalystTransportController, make_controller)
// where `make_controller(const std::string &config) -> ControllerSession*`.
#define GENERATE_TRANSPORT_CONTROLLER_FACTORY(IDENTIFIER, BUILDER)                                 \
    extern "C" catalyst::transport::ControllerSession *IDENTIFIER##Factory(const char *config) {   \
        try {                                                                                      \
            return (BUILDER)(config ? std::string(config) : std::string());                        \
        } catch (const std::exception &e) {                                                        \
            std::cerr << "[transport] controller factory failed: " << e.what() << "\n";            \
            return nullptr;                                                                        \
        } catch (...) {                                                                            \
            std::cerr << "[transport] controller factory failed: unknown exception\n";             \
            return nullptr;                                                                        \
        }                                                                                          \
    }

// e.g. GENERATE_TRANSPORT_COPROCESSOR_FACTORY(CatalystTransportCoprocessor, make_coprocessor)
// where `make_coprocessor(const std::string &config) -> CoprocessorSession*`.
#define GENERATE_TRANSPORT_COPROCESSOR_FACTORY(IDENTIFIER, BUILDER)                                \
    extern "C" catalyst::transport::CoprocessorSession *IDENTIFIER##Factory(const char *config) {  \
        try {                                                                                      \
            return (BUILDER)(config ? std::string(config) : std::string());                        \
        } catch (const std::exception &e) {                                                        \
            std::cerr << "[transport] coprocessor factory failed: " << e.what() << "\n";           \
            return nullptr;                                                                        \
        } catch (...) {                                                                            \
            std::cerr << "[transport] coprocessor factory failed: unknown exception\n";            \
            return nullptr;                                                                        \
        }                                                                                          \
    }

#endif // TRANSPORTBACKEND_H
