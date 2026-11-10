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
#include <cstdio>
#include <cstring>
#include <string>

#include "Error.hpp"
namespace catalyst::transport::common {

namespace {
// One line of a sysfs file, without its newline; empty if it cannot be read.
std::string read_sysfs(const std::string &path) {
    std::FILE *f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
        return {};
    }
    char buf[128] = {};
    const bool ok = std::fgets(buf, sizeof(buf), f) != nullptr;
    std::fclose(f);
    if (!ok) {
        return {};
    }
    buf[std::strcspn(buf, "\n")] = 0;
    return buf;
}

constexpr std::uint8_t V4_PREFIX[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
} // namespace

ibv_gid ipv4_mapped_gid(std::uint32_t ip) {
    ibv_gid gid{};
    std::memcpy(gid.raw, V4_PREFIX, sizeof(V4_PREFIX));
    gid.raw[12] = static_cast<std::uint8_t>(ip >> 24);
    gid.raw[13] = static_cast<std::uint8_t>(ip >> 16);
    gid.raw[14] = static_cast<std::uint8_t>(ip >> 8);
    gid.raw[15] = static_cast<std::uint8_t>(ip);
    return gid;
}

Context::Context(const std::string &dev_name) : dev_name_(dev_name) {
    int n = 0;
    ibv_device **devs = ibv_get_device_list(&n);
    TP_CHECK_ERRNO(devs && n > 0, "ibv_get_device_list");
    auto it = std::find_if(devs, devs + n,
                           [&](ibv_device *d) { return dev_name == ibv_get_device_name(d); });
    ibv_device *dev = (it != devs + n) ? *it : nullptr;
    if (!dev) {
        std::string avail;
        for (int i = 0; i < n; i++) {
            avail += (i ? ", " : "");
            avail += ibv_get_device_name(devs[i]);
        }
        ibv_free_device_list(devs);
        TP_FAIL("device %s not found (available: %s)", dev_name.c_str(),
                avail.empty() ? "<none>" : avail.c_str());
    }
    ctx_ = ibv_open_device(dev);
    ibv_free_device_list(devs);
    TP_CHECK_ERRNO(ctx_, "ibv_open_device(%s)", dev_name.c_str());
}
Context::~Context() {
    if (ctx_) {
        ibv_close_device(ctx_);
    }
}
ibv_context *Context::get() const { return ctx_; }
ibv_port_attr Context::port_attr(std::uint8_t port) const {
    ibv_port_attr attr{};
    const int rc = ibv_query_port(ctx_, port, &attr);
    TP_CHECK(rc == 0, "ibv_query_port(%u) rc=%d (%s)", port, rc, std::strerror(rc));
    return attr;
}
ibv_gid Context::gid(std::uint8_t port, int idx) const {
    ibv_gid gid{};
    const int rc = ibv_query_gid(ctx_, port, idx, &gid);
    TP_CHECK(rc == 0, "ibv_query_gid(%u,%d) rc=%d (%s)", port, idx, rc, std::strerror(rc));
    return gid;
}
Ipv4Gid Context::ipv4_gid(std::uint8_t port, int idx) const {
    const ibv_gid g = gid(port, idx);
    TP_CHECK(std::memcmp(g.raw, V4_PREFIX, sizeof(V4_PREFIX)) == 0,
             "GID %d of %s is not an IPv4 GID; pick the RoCE v2 IPv4 one (gid=)", idx,
             dev_name_.c_str());
    Ipv4Gid out;
    out.ip = (static_cast<std::uint32_t>(g.raw[12]) << 24) |
             (static_cast<std::uint32_t>(g.raw[13]) << 16) |
             (static_cast<std::uint32_t>(g.raw[14]) << 8) | g.raw[15];
    const std::string attrs =
        "/sys/class/infiniband/" + dev_name_ + "/ports/" + std::to_string(port) + "/gid_attrs/";
    const std::string type = read_sysfs(attrs + "types/" + std::to_string(idx));
    if (!type.empty() && type.find("v2") == std::string::npos) {
        std::fprintf(stderr, "warning: GID %d of %s is \"%s\", not RoCE v2\n", idx,
                     dev_name_.c_str(), type.c_str());
    }
    const std::string netdev = read_sysfs(attrs + "ndevs/" + std::to_string(idx));
    TP_CHECK(!netdev.empty(), "no netdev behind GID %d of %s", idx, dev_name_.c_str());
    unsigned m[6] = {};
    const std::string mac = read_sysfs("/sys/class/net/" + netdev + "/address");
    TP_CHECK(std::sscanf(mac.c_str(), "%x:%x:%x:%x:%x:%x", &m[0], &m[1], &m[2], &m[3], &m[4],
                         &m[5]) == 6,
             "cannot read the MAC of %s", netdev.c_str());
    for (int i = 0; i < 6; ++i) {
        out.mac[i] = static_cast<std::uint8_t>(m[i]);
    }
    return out;
}
} // namespace catalyst::transport::common
