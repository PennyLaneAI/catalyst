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

#include <cstdint>
#include <dlfcn.h>

#include <infiniband/verbs.h>

namespace catalyst::transport::hwhs {

class HwhsMem {
  public:
    HwhsMem() {
        lib_ = dlopen("libumm.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (lib_ != nullptr) {
            alloc_chunk_ = reinterpret_cast<AllocChunk>(dlsym(lib_, "xib_umem_alloc_chunk"));
            alloc_mem_ = reinterpret_cast<AllocMem>(dlsym(lib_, "xib_umem_alloc_mem"));
            get_phy_ = reinterpret_cast<GetPhy>(dlsym(lib_, "xib_umem_get_phy_addr"));
            free_mem_ = reinterpret_cast<FreeMem>(dlsym(lib_, "xib_umem_free_mem"));
            free_chunk_ = reinterpret_cast<FreeChunk>(dlsym(lib_, "xib_umem_free_chunk"));
        }
        // ERNIC™ extension
        reg_mr_ex_ = reinterpret_cast<RegMrEx>(dlsym(RTLD_DEFAULT, "ibv_reg_mr_ex"));
    }
    ~HwhsMem() {
        if (lib_ != nullptr) {
            dlclose(lib_);
        }
    }
    HwhsMem(const HwhsMem &) = delete;
    HwhsMem &operator=(const HwhsMem &) = delete;

    [[nodiscard]] bool loaded() const {
        return lib_ != nullptr && alloc_chunk_ != nullptr && alloc_mem_ != nullptr &&
               get_phy_ != nullptr && free_mem_ != nullptr && free_chunk_ != nullptr;
    }

    int alloc_chunk(void *ctx, int mem_type, std::uint64_t block, std::uint64_t total, bool proc) {
        return alloc_chunk_(ctx, mem_type, block, total, proc);
    }

    std::uint64_t alloc_mem(void *ctx, int chunk, std::uint64_t size) {
        return alloc_mem_(ctx, chunk, size);
    }

    int get_phy_addr(void *ctx, unsigned chunk, std::uint64_t va, std::uint64_t *pa) {
        return get_phy_(ctx, chunk, va, pa);
    }

    int free_mem(void *ctx, unsigned chunk, std::uint64_t uva, std::uint64_t size) {
        return free_mem_(ctx, chunk, uva, size);
    }

    int free_chunk(void *ctx, int chunk) { return free_chunk_(ctx, chunk); }

    // Register an MR by explicit device VA
    ibv_mr *reg_mr(ibv_pd *pd, std::uint64_t addr, std::uint64_t size, int access) {
        if (reg_mr_ex_ != nullptr) {
            return reg_mr_ex_(pd, addr, size, access);
        }
        return ibv_reg_mr(pd, reinterpret_cast<void *>(addr), size, access);
    }

  private:
    using AllocChunk = int (*)(void *, int, std::uint64_t, std::uint64_t, bool);
    using AllocMem = std::uint64_t (*)(void *, int, std::uint64_t);
    using GetPhy = int (*)(void *, unsigned, std::uint64_t, std::uint64_t *);
    using FreeMem = int (*)(void *, unsigned, std::uint64_t, std::uint64_t);
    using FreeChunk = int (*)(void *, int);
    using RegMrEx = ibv_mr *(*)(ibv_pd *, std::uint64_t, std::uint64_t, int);

    void *lib_ = nullptr;
    AllocChunk alloc_chunk_ = nullptr;
    AllocMem alloc_mem_ = nullptr;
    GetPhy get_phy_ = nullptr;
    FreeMem free_mem_ = nullptr;
    FreeChunk free_chunk_ = nullptr;
    RegMrEx reg_mr_ex_ = nullptr;
};

} // namespace catalyst::transport::hwhs
