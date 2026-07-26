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

#include "Context.hpp"

#include <algorithm>
#include <string>

#include "Error.hpp"
namespace catalyst::transport::common {
Context::Context(const std::string &dev_name)
{
    int n = 0;
    ibv_device **devs = ibv_get_device_list(&n);
    RDMA_CHECK(devs && n > 0, "ibv_get_device_list");
    auto it = std::find_if(devs, devs + n,
                           [&](ibv_device *d) { return dev_name == ibv_get_device_name(d); });
    ibv_device *dev = (it != devs + n) ? *it : nullptr;
    if (!dev) {
        std::string avail;
        for (int i = 0; i < n; ++i) {
            avail += (i ? ", " : "");
            avail += ibv_get_device_name(devs[i]);
        }
        ibv_free_device_list(devs);
        RDMA_FAIL("device %s not found (available: %s)", dev_name.c_str(),
                  avail.empty() ? "<none>" : avail.c_str());
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
} // namespace catalyst::transport::common
