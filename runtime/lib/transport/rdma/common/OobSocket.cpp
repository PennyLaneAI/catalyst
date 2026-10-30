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

#include "OobSocket.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>

#include "Error.hpp"

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

namespace catalyst::transport::common {

namespace {
constexpr int CONNECT_ATTEMPTS = 200;
constexpr auto CONNECT_RETRY_DELAY = std::chrono::milliseconds(50);

void set_tcp_nodelay(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}
} // namespace

FdGuard tcp_listen_accept(std::uint16_t port) {
    FdGuard listener(socket(AF_INET, SOCK_STREAM, 0));
    TP_CHECK_ERRNO(listener.valid(), "socket");
    int one = 1;
    setsockopt(listener.get(), SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in sa{
        .sin_family = AF_INET,
        .sin_port = htons(port),
        .sin_addr = {.s_addr = INADDR_ANY},
    };
    TP_CHECK_ERRNO(bind(listener.get(), reinterpret_cast<sockaddr *>(&sa), sizeof(sa)) == 0,
                   "bind(%u)", port);
    TP_CHECK_ERRNO(listen(listener.get(), 1) == 0, "listen");
    FdGuard client(accept(listener.get(), nullptr, nullptr));
    TP_CHECK_ERRNO(client.valid(), "accept");
    set_tcp_nodelay(client.get());
    return client; // listener closed by its FdGuard on return
}

FdGuard tcp_connect(const char *host, std::uint16_t port) {
    // getaddrinfo resolves both numeric IPs ("127.0.0.1") and hostnames
    // ("localhost", "node01"). IPv4-only.
    addrinfo hints{
        .ai_family = AF_INET,
        .ai_socktype = SOCK_STREAM,
    };
    char port_str[6];
    std::snprintf(port_str, sizeof(port_str), "%u", port);
    addrinfo *res = nullptr;
    int rc = getaddrinfo(host, port_str, &hints, &res);
    TP_CHECK(rc == 0, "getaddrinfo(%s:%s): %s", host, port_str, gai_strerror(rc));
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> res_guard(res, freeaddrinfo);

    for (int attempt = 0; attempt < CONNECT_ATTEMPTS; attempt++) {
        FdGuard s(socket(res->ai_family, res->ai_socktype, res->ai_protocol));
        TP_CHECK_ERRNO(s.valid(), "socket");
        if (connect(s.get(), res->ai_addr, res->ai_addrlen) == 0) {
            set_tcp_nodelay(s.get());
            return s;
        }
        std::this_thread::sleep_for(CONNECT_RETRY_DELAY);
    }
    TP_FAIL("tcp_connect(%s:%u) failed after %d attempts", host, port, CONNECT_ATTEMPTS);
}

void send_exact(int fd, const void *buf, std::size_t n) {
    std::size_t done = 0;
    const char *p = static_cast<const char *>(buf);
    while (done < n) {
        ssize_t r = ::send(fd, p + done, n - done, 0);
        if (r < 0) {
            if (errno == EINTR) {
                continue; // interrupted by signal; retry
            }
            TP_FAIL("send: %s", std::strerror(errno));
        }
        done += static_cast<std::size_t>(r);
    }
}

void recv_exact(int fd, void *buf, std::size_t n) {
    std::size_t done = 0;
    char *p = static_cast<char *>(buf);
    while (done < n) {
        ssize_t r = ::recv(fd, p + done, n - done, 0);
        if (r < 0) {
            if (errno == EINTR) {
                continue; // interrupted by signal; retry
            }
            TP_FAIL("recv: %s", std::strerror(errno));
        }
        TP_CHECK(r > 0, "recv: peer closed connection (%zu/%zu bytes)", done, n);
        done += static_cast<std::size_t>(r);
    }
}

} // namespace catalyst::transport::common
