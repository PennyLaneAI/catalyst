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
#include <functional>
#include <memory>
#include <mutex>
#include <string_view>

#include "Transport.hpp"

namespace catalyst::transport::memcpy {

class CpuControllerSession;

// Rendezvous point for a memcpy controller and its paired coprocessor. Both sides look up
// the same MemcpyLink from a process-global registry keyed on the session pair key.
//
// The pair key is emitted by the compiler (inject-transport-session's `key` attribute on
// transport.create) and folded into the backend config as `pair=<key>` by TransportCAPI, so
// pairing is decided entirely at the MLIR level; memcpy's transport.connect intentionally
// carries no peer or oob_port and ConnectInfo is ignored by memcpy sessions.
//
// The coprocessor binds `process_message` on connect(); the controller's kick() invokes it
// inline. `mu` guards both fields across connect / kick / teardown.
struct MemcpyLink {
    // TODOs: Update this once ControllerSession::kick() is renamed to post() to match the dialect
    // and CAPI Invoked once per controller kick(). Reads `in_len` bytes from `in` (a wire-shaped
    // `common::Payload` frame synthesized by the controller: value bytes at offset 0,
    // decoder_id at offset PAYLOAD_DATA_BYTES, seq_num right after), writes at most
    // `out_cap` bytes into `out`, and returns the number of bytes actually written. A
    // return value greater than `out_cap` means the fn overran the caller's buffer.
    // Signature mirrors `CoprocessorFn` (Transport.hpp) minus the `ctx` slot; decoder_id
    // travels inside `in` as it does over cpu_verbs.
    using ProcessMessage = std::function<std::size_t(const void *in, std::size_t in_len, void *out,
                                                     std::size_t out_cap)>;

    std::mutex mu;
    // Duplicate-binding sentinels: connect() throws if the field for its role is already set,
    // and each session clears its own field in its destructor.
    CpuControllerSession *controller = nullptr;
    ProcessMessage process_message;
};

// Look up (or create) the MemcpyLink for a pair key. The backing registry lives in the shared
// `libmemcpy_cpu_impl.so`, so every memcpy plugin dlopen'd into the process shares one map;
// putting this in a header would give each plugin its own private static and stop controller
// and coprocessor plugins from pairing. Throws if `pair_key` is empty, since a blank key would
// silently cross-pair unrelated sessions.
auto acquire_memcpy_link(std::string_view pair_key) -> std::shared_ptr<MemcpyLink>;

// Extract the value of `pair=<key>` from a backend config string ("k1=v1;k2=v2;..."). The CAPI
// folds the compiler-emitted key into config on every create; this reads it back on the
// backend side. Returns an empty string when `pair=` is absent, which `acquire_memcpy_link`
// rejects.
auto parse_pair_key(std::string_view config) -> std::string;

} // namespace catalyst::transport::memcpy
