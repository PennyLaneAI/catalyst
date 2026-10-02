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

#include "Transport.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::memcpy {

/**
 * @brief A request ring, a reply ring, and the worker thread that drains one into the other
 * through a per-message CoprocessorFn.
 *
 * process_message() runs on the controller thread, called from its kick(). It copies the request
 * frame into the next request slot, waits for the worker to publish the paired reply, and copies
 * the reply out. The worker runs the bound function once per request, in order.
 *
 * Messages have no size limit. The rings are sized by the session's first message, which also
 * starts the worker thread: each request slot holds that message's frame, and each reply slot its
 * reply size, raised to at least common::PAYLOAD_DATA_BYTES. A controller commits its message
 * sizes once, so later messages fit. A larger one is rejected. The function is called as
 *
 *     fn(frame, frame_len, reply, reply_cap, ctx)
 *
 * where `frame` is the controller's frame as common::frame_bytes describes it, and `reply_cap` is
 * the controller's reply size, raised to at least common::PAYLOAD_DATA_BYTES. The reply is
 * zero-filled before the call, and the controller receives exactly its reply size. A function that
 * returns COPROCESSOR_FN_ERROR fails that message only: process_message() throws for it, and the
 * worker goes on to the next message.
 *
 * Bind the function before start(). process_message() is not reentrant: one controller drives the
 * worker, one message at a time.
 */
class MessageWorker {
  public:
    /// Runs on the worker thread before its first message, e.g. to select a GPU.
    using ThreadInit = std::function<void()>;

    MessageWorker() = default;
    ~MessageWorker();
    MessageWorker(const MessageWorker &) = delete;
    MessageWorker &operator=(const MessageWorker &) = delete;

    /// Bind the function run per message. Null selects an echo of the frame's leading bytes.
    void bind(CoprocessorFn fn, void *ctx);

    /// Accept messages, stopping a running worker first. The first message sizes the rings and
    /// starts the worker thread, which runs `init` before it.
    void start(ThreadInit init = {});

    /// Stop and join the worker thread. A no-op when it is not running.
    void stop();

    /// Whether start() was called since the last stop().
    bool running() const { return started_; }

    /// Run one message through the worker. Returns `out_cap`, the number of bytes written to
    /// `out`. Rethrows the worker's exception if it failed.
    std::size_t process_message(const void *in, std::size_t in_len, void *out, std::size_t out_cap);

  private:
    /// Reply `bytes` value marking a message the function failed to process.
    static constexpr std::uint64_t FAILED_REPLY = UINT64_MAX;

    // A ring of common::K_RING_SLOTS slots. Each slot is a 64 B header, then `capacity` data bytes,
    // padded so every slot, and its data, stays 64-B aligned.
    class Ring {
      public:
        struct alignas(64) Header {
            std::uint64_t bytes; // request: frame length; reply: bytes the function wrote
            std::uint64_t cap;   // request: reply capacity handed to the function
            std::uint32_t seq;   // cursor + 1 once the slot is fully written
        };

        /// Bytes from the start of a slot to its data: the 64 B header.
        static constexpr std::size_t DATA_OFFSET = sizeof(Header);

        /// Allocate zeroed slots of `capacity` data bytes. Throws std::bad_alloc.
        void allocate(std::size_t capacity);
        void release();
        std::size_t capacity() const { return capacity_; }
        Header &header(std::size_t i) { return *reinterpret_cast<Header *>(slot(i)); }
        std::byte *data(std::size_t i) { return slot(i) + DATA_OFFSET; }

      private:
        std::byte *slot(std::size_t i) { return base_ + i * stride_; }

        std::vector<std::byte> storage_;
        std::byte *base_ = nullptr;
        std::size_t stride_ = 0;
        std::size_t capacity_ = 0;
    };

    // Size the rings for frames of `frame_bytes` and replies of `reply_bytes`, and start the
    // worker thread.
    void launch(std::size_t frame_bytes, std::size_t reply_bytes);
    void run(std::stop_token st);

    Ring request_ring_;
    Ring reply_ring_;
    CoprocessorFn fn_ = nullptr;
    void *ctx_ = nullptr;
    ThreadInit init_;
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
