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
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>

// Backend-agnostic error helpers shared by every transport backend. Header-only, STL-only,
// so plugin dylibs stay self-contained (no dependency on catalyst runtime glue).

namespace catalyst::transport::common {
class TransportError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/**
 * Throw TransportError with a preformatted message.
 */
[[noreturn]] inline void transport_throw(const char *msg) { throw TransportError(msg); }
} // namespace catalyst::transport::common

// Unconditionally fail with "file:line: msg" context.
#define TP_FAIL(...)                                                                               \
    do {                                                                                           \
        char tp_msg_[256];                                                                         \
        std::snprintf(tp_msg_, sizeof(tp_msg_), __VA_ARGS__);                                      \
        char tp_full_[512];                                                                        \
        std::snprintf(tp_full_, sizeof(tp_full_), "%s:%d: %s", __FILE__, __LINE__, tp_msg_);       \
        ::catalyst::transport::common::transport_throw(tp_full_);                                  \
    } while (0)

// Throw TransportError with file:line when cond is false.
#define TP_CHECK(cond, ...)                                                                        \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            TP_FAIL(__VA_ARGS__);                                                                  \
        }                                                                                          \
    } while (0)

// Same as TP_FAIL, appending "(errno=..)". Use only where the failing call is documented to
// set errno; elsewhere errno holds an unrelated stale value. errno is captured before
// formatting, which may itself modify it.
#define TP_FAIL_ERRNO(...)                                                                         \
    do {                                                                                           \
        const int tp_errno_ = errno;                                                               \
        char tp_msg_[256];                                                                         \
        std::snprintf(tp_msg_, sizeof(tp_msg_), __VA_ARGS__);                                      \
        char tp_full_[512];                                                                        \
        std::snprintf(tp_full_, sizeof(tp_full_), "%s:%d: %s (errno=%d: %s)", __FILE__, __LINE__,  \
                      tp_msg_, tp_errno_, std::strerror(tp_errno_));                               \
        ::catalyst::transport::common::transport_throw(tp_full_);                                  \
    } while (0)

// As TP_CHECK, appending "(errno=..)". See TP_FAIL_ERRNO.
#define TP_CHECK_ERRNO(cond, ...)                                                                  \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            TP_FAIL_ERRNO(__VA_ARGS__);                                                            \
        }                                                                                          \
    } while (0)
