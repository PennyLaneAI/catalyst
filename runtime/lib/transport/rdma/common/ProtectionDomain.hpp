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
#include <memory>

#include "Context.hpp"

#include <infiniband/verbs.h>

namespace catalyst::transport::common {
/**
 * @class ProtectionDomain
 * @brief RAII wrapper for an `ibv_pd` resource, managing memory protection
 * domains.
 */
class ProtectionDomain {
  public:
    /**
     * @brief Allocates an InfiniBand protection domain tied to the lifecycle of
     * the context.
     * @param ctx Shared pointer to the underlying hardware context.
     */
    explicit ProtectionDomain(std::shared_ptr<Context> ctx);
    ProtectionDomain() = delete;

    /**
     * @brief Automatically releases the underlying `ibv_pd` resource.
     */
    ~ProtectionDomain();
    ProtectionDomain(const ProtectionDomain &) = delete;
    ProtectionDomain &operator=(const ProtectionDomain &) = delete;

    /**
     * @brief Returns a raw pointer to the underlying verbs protection domain.
     * @return Raw pointer to `ibv_pd`.
     */
    ibv_pd *get() const;

  private:
    std::shared_ptr<Context> ctx_; // keeps the Context alive
    ibv_pd *pd_ = nullptr;         // Low-level verbs protection domain handle.
};
} // namespace catalyst::transport::common
