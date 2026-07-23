#include "Context.hpp"

#include <algorithm>

#include "Error.hpp"
namespace rdma::devices::common {
Context::Context(const std::string &dev_name)
{
    int n = 0;
    ibv_device **devs = ibv_get_device_list(&n);
    RDMA_CHECK(devs && n > 0, "ibv_get_device_list");
    auto it = std::find_if(devs, devs + n,
                           [&](ibv_device *d) { return dev_name == ibv_get_device_name(d); });
    ibv_device *dev = (it != devs + n) ? *it : nullptr;
    if (!dev) {
        ibv_free_device_list(devs);
        RDMA_FAIL("device %s not found", dev_name.c_str());
    }
    ctx_ = ibv_open_device(dev);
    ibv_free_device_list(devs);
    RDMA_CHECK(ctx_, "ibv_open_device(%s)", dev_name.c_str());
}
Context::~Context()
{
    if (ctx_)
        ibv_close_device(ctx_);
}
ibv_context *Context::get() const { return ctx_; }
ibv_port_attr Context::port_attr(std::uint8_t port) const
{
    ibv_port_attr attr{};
    RDMA_CHECK(ibv_query_port(ctx_, port, &attr) == 0, "ibv_query_port(%u)", port);
    return attr;
}
ibv_gid Context::gid(std::uint8_t port, int idx) const
{
    ibv_gid gid{};
    RDMA_CHECK(ibv_query_gid(ctx_, port, idx, &gid) == 0, "ibv_query_gid(%u,%d)", port, idx);
    return gid;
}
} // namespace rdma::devices::common
