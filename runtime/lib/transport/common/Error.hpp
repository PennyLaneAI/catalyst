#pragma once
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace rdma::devices::common {
class RdmaError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/**
 * Throw RdmaError with a preformatted message.
 */
[[noreturn]] inline void rdma_throw(const char *msg) { throw RdmaError(msg); }
} // namespace rdma::devices::common

// Unconditionally fail with "file:line: msg (errno=..)" context.
#define RDMA_FAIL(...)                                                                             \
    do {                                                                                           \
        char rdma_msg_[256];                                                                       \
        std::snprintf(rdma_msg_, sizeof(rdma_msg_), __VA_ARGS__);                                  \
        char rdma_full_[512];                                                                      \
        std::snprintf(rdma_full_, sizeof(rdma_full_), "%s:%d: %s (errno=%d: %s)", __FILE__,        \
                      __LINE__, rdma_msg_, errno, std::strerror(errno));                           \
        ::rdma::devices::common::rdma_throw(rdma_full_);                                           \
    } while (0)

// Throw RdmaError with file:line + errno when cond is false.
#define RDMA_CHECK(cond, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            RDMA_FAIL(__VA_ARGS__);                                                                \
        }                                                                                          \
    } while (0)
