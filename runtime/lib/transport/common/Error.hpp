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

namespace catalyst::transport::common {
class RdmaError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/**
 * Throw RdmaError with a preformatted message.
 */
[[noreturn]] inline void rdma_throw(const char *msg) { throw RdmaError(msg); }
} // namespace catalyst::transport::common

// Unconditionally fail with "file:line: msg" context.
#define RDMA_FAIL(...)                                                                             \
    do {                                                                                           \
        char rdma_msg_[256];                                                                       \
        std::snprintf(rdma_msg_, sizeof(rdma_msg_), __VA_ARGS__);                                  \
        char rdma_full_[512];                                                                      \
        std::snprintf(rdma_full_, sizeof(rdma_full_), "%s:%d: %s", __FILE__, __LINE__, rdma_msg_); \
        ::catalyst::transport::common::rdma_throw(rdma_full_);                                     \
    } while (0)

// Throw RdmaError with file:line when cond is false.
#define RDMA_CHECK(cond, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            RDMA_FAIL(__VA_ARGS__);                                                                \
        }                                                                                          \
    } while (0)

// Same as RDMA_FAIL, appending "(errno=..)". Use only where the failing call is
// documented to set errno; elsewhere errno holds an unrelated stale value.
// errno is captured before formatting, which may itself modify it.
#define RDMA_FAIL_ERRNO(...)                                                                       \
    do {                                                                                           \
        const int rdma_errno_ = errno;                                                             \
        char rdma_msg_[256];                                                                       \
        std::snprintf(rdma_msg_, sizeof(rdma_msg_), __VA_ARGS__);                                  \
        char rdma_full_[512];                                                                      \
        std::snprintf(rdma_full_, sizeof(rdma_full_), "%s:%d: %s (errno=%d: %s)", __FILE__,        \
                      __LINE__, rdma_msg_, rdma_errno_, std::strerror(rdma_errno_));               \
        ::catalyst::transport::common::rdma_throw(rdma_full_);                                     \
    } while (0)

// As RDMA_CHECK, appending "(errno=..)". See RDMA_FAIL_ERRNO.
#define RDMA_CHECK_ERRNO(cond, ...)                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            RDMA_FAIL_ERRNO(__VA_ARGS__);                                                          \
        }                                                                                          \
    } while (0)
