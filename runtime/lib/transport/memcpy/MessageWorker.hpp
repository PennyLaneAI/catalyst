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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <stop_token>
#include <thread>
#include <vector>

#include "Frame.hpp"
#include "Transport.hpp"

namespace catalyst::transport::memcpy {

/**
 * @brief A request ring, a reply ring, and the worker thread that drains one into the other
 * through a per-message CoprocessorFn.
 *
 * Both rings hold codec frames laid out by the session's RingGeometry, the same frames an RDMA
 * backend writes into its peer's ring. process_message() runs on the controller thread, called
 * from its kick(): it stages and seals the request frame in the next request slot, waits for the
 * worker to seal the paired reply frame, and copies the reply's payload out. The worker decodes
 * each request and calls the bound function once per message, in order, as
 *
 *     fn(payload, len, fn_id, reply, reply_cap, ctx)
 *
 * where `reply_cap` is the frame's payload capacity, and the reply area is zero-filled before the
 * call. A function that returns COPROCESSOR_FN_ERROR fails that message only: process_message()
 * throws for it, and the worker goes on to the next message.
 *
 * Bind the function before start(). process_message() is not reentrant: one controller drives the
 * worker, one message at a time.
 */
class MessageWorker {
  public:
    /// Runs on the worker thread before its first message, e.g. to select a GPU.
    using ThreadInit = std::function<void()>;

    explicit MessageWorker(common::frame::RingGeometry ring = common::frame::BL16_RING)
        : ring_(ring) {}
    ~MessageWorker();
    MessageWorker(const MessageWorker &) = delete;
    MessageWorker &operator=(const MessageWorker &) = delete;

    /// Bind the function run per message. Null selects an echo of the payload.
    void bind(CoprocessorFn fn, void *ctx);

    /// Allocate the rings and start the worker thread, which runs `init` first, stopping a running
    /// worker first.
    void start(ThreadInit init = {});

    /// Stop and join the worker thread. A no-op when it is not running.
    void stop();

    /// Whether start() was called since the last stop().
    bool running() const { return started_; }

    /// Run one message through the worker, and copy `out_cap` bytes of its reply's payload to
    /// `out`. Returns `out_cap`. Rethrows the worker's exception if it failed.
    std::size_t process_message(const void *payload, std::size_t len, std::uint32_t fn_id,
                                void *out, std::size_t out_cap);

  private:
    void run(std::stop_token st);
    std::uint8_t *request_slot(std::uint64_t cursor) {
        return request_ring_ + ring_.slot_offset(cursor);
    }
    std::uint8_t *reply_slot(std::uint64_t cursor) {
        return reply_ring_ + ring_.slot_offset(cursor);
    }

    common::frame::RingGeometry ring_;
    std::vector<std::uint8_t> storage_; // both rings, aligned to their stride
    std::uint8_t *request_ring_ = nullptr;
    std::uint8_t *reply_ring_ = nullptr;
    // Per reply slot, whether the function failed that message. Written before the reply is
    // sealed, so it is visible once the seal is.
    std::vector<std::uint8_t> failed_slot_;
    CoprocessorFn fn_ = nullptr;
    void *ctx_ = nullptr;
    bool started_ = false;
    // Advanced by process_message only, on the controller thread.
    std::uint64_t process_cursor_ = 0;
    // If run() throws, error_ is set and failed_ published (release). process_message
    // acquire-loads failed_ and rethrows.
    std::atomic<bool> failed_{false};
    std::exception_ptr error_;
    std::jthread engine_;
};

} // namespace catalyst::transport::memcpy
