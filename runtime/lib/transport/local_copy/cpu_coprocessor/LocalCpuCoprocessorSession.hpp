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
#include <memory>
#include <vector>

#include "LocalRegistry.hpp"
#include "Transport.hpp"

namespace catalyst::transport::local_copy {

class LocalCpuCoprocessorSession : public CoprocessorSession {
  public:
    explicit LocalCpuCoprocessorSession(std::string = {}) {}
    ~LocalCpuCoprocessorSession() override;

    // TransportSession
    int connect(const ConnectInfo &info) override;
    MemRegion alloc_memory(std::size_t size, MemKind kind) override;
    PeerRef exchange_keys(const MemRegion &local) override;
    void establish_channel(const ChannelDesc &desc, const MemRegion &local,
                           const PeerRef &peer) override;
    void start() override;
    int collect(void *const *replies, const std::uint64_t *replies_bytes, std::size_t n) override;
    void stop() override;

    // CoprocessorSession
    void set_coprocessor_fn(CoprocessorFn fn, void *ctx) override;

    // Called synchronously from the paired controller's kick(); returns bytes written to `out`.
    std::size_t process_message(const void *in, std::size_t in_len, std::uint32_t decoder_id,
                                void *out, std::size_t out_cap);

  private:
    std::shared_ptr<MemcpyLink> link_;

    /// Owns the buffers backing MemRegions handed out by alloc_memory().
    std::vector<std::unique_ptr<std::byte[]>> caller_memory_regions_;

    CoprocessorFn fn_ = nullptr;
    void *ctx_ = nullptr;
};

} // namespace catalyst::transport::local_copy
