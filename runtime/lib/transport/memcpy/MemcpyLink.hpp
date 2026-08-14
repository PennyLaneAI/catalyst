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

#include "Transport.hpp"

namespace catalyst::transport::memcpy {

class CpuControllerSession;

// In-process memcpy link keyed by (peer, oob_port). The coprocessor binds `process_message`;
// the controller's kick() drives it inline. `mu` guards `controller` and `process_message`
// across connect / kick / teardown.
struct MemcpyLink {
    // Invoked once per controller kick(). Reads `in_len` bytes from `in` (a wire-shaped
    // `common::Payload` frame synthesized by the controller: value bytes at offset 0,
    // decoder_id at offset PAYLOAD_DATA_BYTES, seq_num right after), writes at most
    // `out_cap` bytes into `out`, and returns the number of bytes actually written. A
    // return value greater than `out_cap` means the fn overran the caller's buffer.
    // Signature mirrors `CoprocessorFn` (Transport.hpp) minus the `ctx` slot; decoder_id
    // travels inside `in` as it does over cpu_verbs.
    using ProcessMessage = std::function<std::size_t(const void *in, std::size_t in_len, void *out,
                                                     std::size_t out_cap)>;

    std::mutex mu;
    CpuControllerSession *controller = nullptr;
    ProcessMessage process_message;
};

// The registry backing `acquire_memcpy_link` lives in the dedicated
// `libcatalyst_transport_memcpy_registry.so` so every memcpy plugin loaded into the process
// sees the same (peer, oob_port) → MemcpyLink map. Compiling this function inline (per-plugin
// TU) would give each dlopen'd plugin its own private copy of the static map, and controller +
// coprocessor plugins would fail to pair.
auto acquire_memcpy_link(const ConnectInfo &info) -> std::shared_ptr<MemcpyLink>;

} // namespace catalyst::transport::memcpy
