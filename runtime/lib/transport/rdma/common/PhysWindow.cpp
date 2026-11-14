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

#include "PhysWindow.hpp"

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/mman.h>

#include "Error.hpp"

#include <fcntl.h>
#include <unistd.h>

namespace catalyst::transport::common {

namespace {

// A hex number from a sysfs file, such as a UIO map's addr or size.
bool read_sysfs_hex(const std::string &path, std::uint64_t *v) {
    std::FILE *f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
        return false;
    }
    const bool ok = std::fscanf(f, "%" SCNx64, v) == 1;
    std::fclose(f);
    return ok;
}

} // namespace

void *map_phys_window(std::uint64_t phys, std::size_t len, const char *who) {
    for (int i = 0; i < 64; ++i) {
        const std::string dir = "/sys/class/uio/uio" + std::to_string(i) + "/maps/map0/";
        std::uint64_t addr = 0, size = 0;
        if (!read_sysfs_hex(dir + "addr", &addr) || addr != phys ||
            !read_sysfs_hex(dir + "size", &size) || size < len) {
            continue;
        }
        const std::string dev = "/dev/uio" + std::to_string(i);
        const int fd = ::open(dev.c_str(), O_RDWR | O_SYNC);
        TP_CHECK(fd >= 0, "%s: %s (0x%" PRIx64 "): %s (the controller needs root)", who,
                 dev.c_str(), phys, std::strerror(errno));
        void *p = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        TP_CHECK(p != MAP_FAILED, "%s: mmap %s: %s", who, dev.c_str(), std::strerror(errno));
        return p;
    }

    const int fd = ::open("/dev/mem", O_RDWR | O_SYNC);
    TP_CHECK(fd >= 0, "%s: no UIO device at 0x%" PRIx64 " and /dev/mem: %s", who, phys,
             std::strerror(errno));
    void *p =
        ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, static_cast<off_t>(phys));
    ::close(fd);
    TP_CHECK(p != MAP_FAILED, "%s: mmap /dev/mem at 0x%" PRIx64 ": %s", who, phys,
             std::strerror(errno));
    return p;
}

void unmap_phys_window(volatile void *p, std::size_t len) {
    if (p != nullptr) {
        ::munmap(const_cast<void *>(p), len);
    }
}

} // namespace catalyst::transport::common
