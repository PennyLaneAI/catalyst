#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

#include <infiniband/verbs.h>

#include "ProtectionDomain.hpp"

namespace rdma::devices::common {

/**
 * @enum MemAccess flag.
 * @brief Type-safe hardware access permissions for registered Memory Regions.
 */
enum class MemAccess : int {
    LOCAL_WRITE = IBV_ACCESS_LOCAL_WRITE,
    REMOTE_WRITE = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE,
    REMOTE_READ = IBV_ACCESS_REMOTE_READ,
};

/**
 * @brief Bitwise OR operator for combining MemAccess flags.
 */
constexpr MemAccess operator|(MemAccess a, MemAccess b)
{
    return static_cast<MemAccess>(static_cast<int>(a) | static_cast<int>(b));
}

/**
 * @class MemoryRegion
 * @brief RAII wrapper for an `ibv_mr`, managing hardware registration and
 * backing storage.
 */
class MemoryRegion {
  public:
    // register caller-provided memory; region does not own the buffer.
    MemoryRegion(std::shared_ptr<ProtectionDomain> pd, void *addr, std::size_t length,
                 MemAccess access);
    // register caller-provided memory and keep a shared_ptr for lifetime.
    MemoryRegion(std::shared_ptr<ProtectionDomain> pd, void *addr, std::size_t length,
                 MemAccess access, std::shared_ptr<void> backing);
    // register a dma-buf (does not own the buffer).
    MemoryRegion(std::shared_ptr<ProtectionDomain> pd, std::uint64_t offset, std::size_t length,
                 std::uint64_t iova, int fd, MemAccess access);
    // allocate + own an aligned host buffer, then register it.
    static MemoryRegion alloc_host(std::shared_ptr<ProtectionDomain> pd, std::size_t length,
                                   std::size_t alignment, MemAccess access);
    ~MemoryRegion();

    MemoryRegion(const MemoryRegion &) = delete;
    MemoryRegion &operator=(const MemoryRegion &) = delete;
    MemoryRegion(MemoryRegion &&o) noexcept;
    MemoryRegion &operator=(MemoryRegion &&o) noexcept;

    ibv_mr *get() const noexcept;

    /// @brief Returns the base virtual address of the registered region.
    void *addr() const noexcept;

    /// @brief Returns the total capacity of the memory region in bytes.
    std::size_t length() const noexcept;

    /// @brief Returns the local key required for posting local work requests.
    std::uint32_t lkey() const noexcept;

    /// @brief Returns the remote key required by peers for one-sided RDMA
    /// operations.
    std::uint32_t rkey() const noexcept;

  private:
    std::shared_ptr<ProtectionDomain> pd_; // keeps PD alive
    ibv_mr *mr_ = nullptr;
    // Keeps the backing storage alive until after the MR is deregistered;
    // null for borrowed / dma-buf regions that own nothing.
    std::shared_ptr<void> backing_buffer_;
};
} // namespace rdma::devices::common
