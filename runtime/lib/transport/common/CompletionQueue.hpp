#pragma once
#include <memory>

#include <infiniband/verbs.h>

#include "Context.hpp"

namespace rdma::devices::common {

/**
 * @class CompletionQueue class.
 *
 * @brief An RAII wrapper for an RDMA Completion Queue (`ibv_cq`).
 */
class CompletionQueue {
  public:
    CompletionQueue() = delete;
    CompletionQueue(std::shared_ptr<Context> ctx, int depth);
    ~CompletionQueue();
    CompletionQueue(const CompletionQueue &) = delete;
    CompletionQueue &operator=(const CompletionQueue &) = delete;
    ibv_cq *get() const;

  private:
    std::shared_ptr<Context> ctx_;
    ibv_cq *cq_ = nullptr;
};
} // namespace rdma::devices::common
