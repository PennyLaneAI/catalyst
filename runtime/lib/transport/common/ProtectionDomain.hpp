#pragma once
#include <memory>

#include <infiniband/verbs.h>

#include "Context.hpp"

namespace rdma::devices::common {
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
} // namespace rdma::devices::common
