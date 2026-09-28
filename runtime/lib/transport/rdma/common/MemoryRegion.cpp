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

#include "MemoryRegion.hpp"

#include <bit>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "Error.hpp"

namespace catalyst::transport::common {

/**
 * @brief Register caller-owned host memory (borrowed; the region does not own
 * it).
 */
MemoryRegion::MemoryRegion(std::shared_ptr<ProtectionDomain> pd, void *addr, std::size_t length,
                           MemAccess access)
    : MemoryRegion(std::move(pd), addr, length, access, nullptr) {}

/**
 * @brief Register caller-provided memory, keeping `backing` alive for the MR's
 *        lifetime.
 */
MemoryRegion::MemoryRegion(std::shared_ptr<ProtectionDomain> pd, void *addr, std::size_t length,
                           MemAccess access, std::shared_ptr<void> backing)
    : pd_(std::move(pd)), backing_buffer_(std::move(backing)) {
    mr_ = ibv_reg_mr(pd_->get(), addr, length, static_cast<int>(access));
    TP_CHECK_ERRNO(mr_, "ibv_reg_mr");
}

/**
 * @brief Register a dma-buf region (e.g. exported GPU memory); does not own the
 *        buffer.
 */
MemoryRegion::MemoryRegion(std::shared_ptr<ProtectionDomain> pd, std::uint64_t offset,
                           std::size_t length, std::uint64_t iova, int fd, MemAccess access)
    : pd_(std::move(pd)) {
    mr_ = ibv_reg_dmabuf_mr(pd_->get(), offset, length, iova, fd, static_cast<int>(access));
    TP_CHECK_ERRNO(mr_, "ibv_reg_dmabuf_mr");
}

/**
 * @brief Allocate + own an aligned host buffer, then register it.
 */
MemoryRegion MemoryRegion::alloc_host(std::shared_ptr<ProtectionDomain> pd, std::size_t length,
                                      std::size_t alignment, MemAccess access) {
    // aligned_alloc requires a power-of-two alignment and a size that is a
    // multiple of it.
    TP_CHECK(std::has_single_bit(alignment), "alignment must be a power of two, got %zu",
             alignment);
    std::size_t rounded = ((length + alignment - 1) / alignment) * alignment;
    void *buf = std::aligned_alloc(alignment, rounded);
    TP_CHECK_ERRNO(buf, "aligned_alloc(%zu)", rounded);
    std::memset(buf, 0, rounded);
    return MemoryRegion(std::move(pd), buf, length, access, std::shared_ptr<void>(buf, std::free));
}

MemoryRegion::~MemoryRegion() {
    if (mr_) {
        ibv_dereg_mr(mr_);
    }
}

MemoryRegion::MemoryRegion(MemoryRegion &&other) noexcept
    : pd_(std::move(other.pd_)), mr_(std::exchange(other.mr_, nullptr)),
      backing_buffer_(std::move(other.backing_buffer_)) {}

MemoryRegion &MemoryRegion::operator=(MemoryRegion &&other) noexcept {
    if (this != &other) {
        if (mr_) {
            ibv_dereg_mr(mr_); // release our current MR before taking other's
        }
        pd_ = std::move(other.pd_);
        mr_ = std::exchange(other.mr_, nullptr);
        backing_buffer_ = std::move(other.backing_buffer_);
    }
    return *this;
}

ibv_mr *MemoryRegion::get() const noexcept { return mr_; }
void *MemoryRegion::addr() const noexcept { return mr_ ? mr_->addr : nullptr; }
std::size_t MemoryRegion::length() const noexcept { return mr_ ? mr_->length : 0; }
std::uint32_t MemoryRegion::lkey() const noexcept { return mr_ ? mr_->lkey : 0; }
std::uint32_t MemoryRegion::rkey() const noexcept { return mr_ ? mr_->rkey : 0; }

} // namespace catalyst::transport::common
