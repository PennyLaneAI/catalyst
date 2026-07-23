#pragma once
#include <cstddef>
#include <cstdint>
#include <unistd.h>
#include <utility>

namespace rdma::devices::common {

// RAII handle for a socket file descriptor.
class FdGuard {
  public:
    FdGuard() noexcept = default;
    explicit FdGuard(int fd) noexcept : fd_(fd) {}
    ~FdGuard() { reset(); }

    FdGuard(FdGuard &&other) noexcept : fd_(other.release()) {}
    FdGuard &operator=(FdGuard &&o) noexcept
    {
        if (this != &o)
            reset(o.release());
        return *this;
    }
    FdGuard(const FdGuard &) = delete;
    FdGuard &operator=(const FdGuard &) = delete;

    [[nodiscard]] int get() const noexcept { return fd_; }
    [[nodiscard]] bool valid() const noexcept { return fd_ >= 0; }
    explicit operator bool() const noexcept { return valid(); }

    /**
     * @brief Closes the current socket and takes ownership of a new one.
     * @param new_fd The new socket descriptor to manage.
     */
    void reset(int new_fd = -1) noexcept
    {
        if (fd_ >= 0)
            ::close(fd_);
        fd_ = new_fd;
    }

    /**
     * @brief Releases ownership of the socket without closing it.
     * @return The raw socket file descriptor.
     */
    [[nodiscard]] int release() noexcept { return std::exchange(fd_, -1); }

  private:
    int fd_ = -1;
};

// OOB TCP handshake helpers.
FdGuard tcp_listen_accept(std::uint16_t port);
FdGuard tcp_connect(const char *host, std::uint16_t port);

// Blocking exact-length IO over an OOB socket.
void send_exact(int fd, const void *buf, std::size_t n);
void recv_exact(int fd, void *buf, std::size_t n);

} // namespace rdma::devices::common
