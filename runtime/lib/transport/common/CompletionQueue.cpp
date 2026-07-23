#include "CompletionQueue.hpp"

#include <utility>

#include "Error.hpp"

namespace rdma::devices::common {
CompletionQueue::CompletionQueue(std::shared_ptr<Context> ctx, int depth) : ctx_(std::move(ctx))
{
    cq_ = ibv_create_cq(ctx_->get(), depth, nullptr, nullptr, 0);
    RDMA_CHECK(cq_, "ibv_create_cq");
}
CompletionQueue::~CompletionQueue()
{
    if (cq_)
        ibv_destroy_cq(cq_);
}
ibv_cq *CompletionQueue::get() const { return cq_; }
} // namespace rdma::devices::common
