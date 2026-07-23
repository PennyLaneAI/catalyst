#include "ProtectionDomain.hpp"

#include <utility>

#include "Error.hpp"

namespace rdma::devices::common {
ProtectionDomain::ProtectionDomain(std::shared_ptr<Context> ctx) : ctx_(std::move(ctx))
{
    pd_ = ibv_alloc_pd(ctx_->get());
    RDMA_CHECK(pd_, "ibv_alloc_pd");
}
ProtectionDomain::~ProtectionDomain()
{
    if (pd_)
        ibv_dealloc_pd(pd_);
}
ibv_pd *ProtectionDomain::get() const { return pd_; }
} // namespace rdma::devices::common
