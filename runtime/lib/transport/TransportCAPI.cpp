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

#include "TransportCAPI.h"

#include <algorithm>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "DynamicLibraryLoader.hpp"
#include "Transport.hpp"
#include "TransportBackend.h"

using catalyst::transport::ChannelDesc;
using catalyst::transport::ConnectInfo;
using catalyst::transport::ControllerSession;
using catalyst::transport::CoprocessorFn;
using catalyst::transport::CoprocessorSession;
using catalyst::transport::MemKind;
using catalyst::transport::MemRegion;
using catalyst::transport::PeerRef;
using catalyst::transport::TransportSession;

// The opaque handle. `sess` is the base type; the concrete role is chosen by the factory at
// create. The peer region learned in exchange_keys is kept here so establish_channel needs no peer
// argument (matches the dialect, where the peer is session-held).
struct CatalystTransportSession {
    std::unique_ptr<DynamicLibraryLoader> backend;
    TransportSession *sess = nullptr; // heap-allocated by the backend factory
    MemRegion reply;                  // local region advertised in exchange_keys
    bool reply_ready = false;
    PeerRef peer; // peer region learned in exchange_keys
    bool peer_ready = false;
};

namespace {

// (role, key) -> live session registry. Populated at create, read by get_session so a session
// brought up in one function can be resolved in another.
std::unordered_map<std::string, CatalystTransportSession *> g_registry;

std::string registry_key(std::int32_t role, const char *key)
{
    return std::to_string(role) + "/" + (key ? key : "");
}

// Local reply region size, provisioned automatically at exchange_keys.
constexpr std::uint64_t kReplyBytes = 16 * 1024;

// Run fn, logging and swallowing any exception. Returns fn()'s result, or CATALYST_TRANSPORT_ERR
// if it threw; when fn() returns void there is nothing to return.
template <typename Fn> auto guard(Fn &&fn) -> decltype(fn())
{
    try {
        return fn();
    }
    catch (const std::exception &e) {
        std::cerr << "[transport] " << e.what() << "\n";
    }
    catch (...) {
    }
    if constexpr (!std::is_void_v<decltype(fn())>) {
        return CATALYST_TRANSPORT_ERR;
    }
}

// Provision the local reply region on first use (idempotent).
void ensure_reply(CatalystTransportSession *s)
{
    if (!s->reply_ready) {
        s->reply = s->sess->alloc_memory(kReplyBytes, MemKind::CpuRam);
        s->reply_ready = true;
    }
}

ControllerSession *as_controller(CatalystTransportSession *s)
{
    return s ? dynamic_cast<ControllerSession *>(s->sess) : nullptr;
}

// Bring-up bodies shared by the blocking and async (worker-thread) entry points.
int do_connect(CatalystTransportSession *s, std::string peer, std::uint16_t oob_port)
{
    ConnectInfo info;
    info.peer = std::move(peer);
    info.oob_port = oob_port;
    return s->sess->connect(info);
}

int do_exchange_keys(CatalystTransportSession *s)
{
    ensure_reply(s);
    s->peer = s->sess->exchange_keys(s->reply);
    s->peer_ready = true;
    return CATALYST_TRANSPORT_OK;
}

// Built-in fallback coprocessor function: echo the input back.
std::size_t echo_fn(const void *in, std::size_t in_len, void *out, std::size_t out_cap, void *)
{
    std::size_t n = std::min(in_len, out_cap);
    if (n && in && out) {
        std::memcpy(out, in, n);
    }
    return n;
}

// Async task registry: connect_async / exchange_keys_async run on a worker thread and return a
// token; barrier awaits it. Tokens start at 1 so a 0 return can signal a dispatch failure.
std::mutex g_async_mtx;
std::int64_t g_next_token = 1;
std::unordered_map<std::int64_t, std::future<int>> g_async_tasks;

std::int64_t dispatch_async(std::function<int()> fn)
{
    std::lock_guard<std::mutex> lk(g_async_mtx);
    std::int64_t token = g_next_token++;
    g_async_tasks.emplace(token, std::async(std::launch::async, std::move(fn)));
    return token;
}

int await_token(std::int64_t token)
{
    std::future<int> fut;
    {
        std::lock_guard<std::mutex> lk(g_async_mtx);
        auto it = g_async_tasks.find(token);
        if (it == g_async_tasks.end()) {
            return CATALYST_TRANSPORT_ERR;
        }
        fut = std::move(it->second);
        g_async_tasks.erase(it);
    }
    return guard([&] { return fut.get(); });
}

} // namespace

extern "C" {

CatalystTransportSession *__catalyst__transport__create(const char *backend_lib, const char *config,
                                                        std::int32_t role, const char *key)
{
    try {
        if (!backend_lib || !*backend_lib) {
            std::cerr << "[transport] no backend library given\n";
            return nullptr;
        }
        auto h = std::make_unique<CatalystTransportSession>();
        h->backend = std::make_unique<DynamicLibraryLoader>(backend_lib);
        const char *cfg = config ? config : "";
        if (role == CATALYST_TRANSPORT_ROLE_COPROCESSOR) {
            auto *factory = h->backend->getSymbol<CatalystTransportCoprocessorFactoryFn *>(
                CATALYST_TRANSPORT_COPROCESSOR_FACTORY_SYMBOL);
            h->sess = factory(cfg);
        }
        else {
            auto *factory = h->backend->getSymbol<CatalystTransportControllerFactoryFn *>(
                CATALYST_TRANSPORT_CONTROLLER_FACTORY_SYMBOL);
            h->sess = factory(cfg);
        }
        if (!h->sess) {
            std::cerr << "[transport] backend factory returned null for config: " << cfg << "\n";
            return nullptr;
        }
        auto *raw = h.release();
        if (key && *key) {
            g_registry[registry_key(role, key)] = raw; // resolved later via get_session
        }
        return raw;
    }
    catch (const std::exception &e) {
        std::cerr << "[transport] create: " << e.what() << "\n";
        return nullptr;
    }
    catch (...) {
        return nullptr;
    }
}

int __catalyst__transport__connect(CatalystTransportSession *s, const char *peer,
                                   std::uint16_t oob_port)
{
    if (!s || !s->sess) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] { return do_connect(s, peer ? peer : "", oob_port); });
}

std::int64_t __catalyst__transport__connect_async(CatalystTransportSession *s, const char *peer,
                                                  std::uint16_t oob_port)
{
    if (!s || !s->sess) {
        return 0;
    }
    return dispatch_async(
        [s, p = std::string(peer ? peer : ""), oob_port] { return do_connect(s, p, oob_port); });
}

int __catalyst__transport__exchange_keys(CatalystTransportSession *s)
{
    if (!s || !s->sess) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] { return do_exchange_keys(s); });
}

std::int64_t __catalyst__transport__exchange_keys_async(CatalystTransportSession *s)
{
    if (!s || !s->sess) {
        return 0;
    }
    return dispatch_async([s] { return do_exchange_keys(s); });
}

int __catalyst__transport__barrier(std::int64_t token) { return await_token(token); }

int __catalyst__transport__establish_channel(CatalystTransportSession *s, const char *data_path)
{
    if (!s || !s->sess) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] {
        ChannelDesc desc;
        desc.data_path = data_path ? data_path : ""; // opaque; the backend interprets it
        s->sess->establish_channel(desc, s->reply, s->peer);
        return CATALYST_TRANSPORT_OK;
    });
}

int __catalyst__transport__set_coprocessor_fn(CatalystTransportSession *s, const char *symbol)
{
    if (!s || !s->sess) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] {
        auto *co = dynamic_cast<CoprocessorSession *>(s->sess);
        if (!co) {
            std::cerr << "[transport] set_coprocessor_fn on a non-coprocessor session\n";
            return CATALYST_TRANSPORT_ERR;
        }
        // Empty symbol selects the built-in echo; a named-but-unresolved symbol is a hard error
        CoprocessorFn fn = &echo_fn;
        if (symbol && *symbol) {
            fn = reinterpret_cast<CoprocessorFn>(dlsym(RTLD_DEFAULT, symbol));
            if (!fn) {
                std::cerr << "[transport] set_coprocessor_fn: symbol not found: " << symbol << "\n";
                return CATALYST_TRANSPORT_ERR;
            }
        }
        co->set_coprocessor_fn(fn, nullptr);
        return CATALYST_TRANSPORT_OK;
    });
}

int __catalyst__transport__commit_work_item(CatalystTransportSession *s,
                                            std::uint32_t work_item_idx, std::uint64_t in_bytes,
                                            std::uint64_t out_bytes)
{
    auto *c = as_controller(s);
    if (!c) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] {
        if (s->reply_ready && out_bytes > s->reply.size) {
            std::cerr << "[transport] commit_work_item: out_bytes (" << out_bytes
                      << ") exceeds the reply region (" << s->reply.size << ")\n";
            return CATALYST_TRANSPORT_ERR;
        }
        c->commit_work_item(work_item_idx, in_bytes, out_bytes);
        return CATALYST_TRANSPORT_OK;
    });
}

void *__catalyst__transport__data_slot(CatalystTransportSession *s)
{
    auto *c = as_controller(s);
    void *slot = nullptr;
    if (c) {
        guard([&] { slot = c->data_slot(); });
    }
    return slot;
}

int __catalyst__transport__kick(CatalystTransportSession *s, std::uint32_t work_item_idx)
{
    auto *c = as_controller(s);
    if (!c) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] { return c->kick(work_item_idx); });
}

int __catalyst__transport__collect(CatalystTransportSession *s, void *reply,
                                   std::uint64_t reply_bytes)
{
    if (!s || !s->sess) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] {
        void *replies[1] = {reply};
        std::uint64_t replies_bytes[1] = {reply_bytes};
        return s->sess->collect(replies, replies_bytes, 1);
    });
}

std::uint64_t __catalyst__transport__last_rtt_ns(CatalystTransportSession *s)
{
    if (!s || !s->sess) {
        return 0;
    }
    return s->sess->last_rtt_ns();
}

void __catalyst__transport__start(CatalystTransportSession *s)
{
    if (s && s->sess) {
        guard([&] { s->sess->start(); });
    }
}

void __catalyst__transport__stop(CatalystTransportSession *s)
{
    if (s && s->sess) {
        guard([&] { s->sess->stop(); });
    }
}

CatalystTransportSession *__catalyst__transport__get_session(std::int32_t role, const char *key)
{
    auto it = g_registry.find(registry_key(role, key));
    if (it == g_registry.end()) {
        std::cerr << "[transport] get_session: no session registered for role " << role << " key '"
                  << (key ? key : "") << "'\n";
        return nullptr;
    }
    return it->second;
}

void __catalyst__transport__destroy(CatalystTransportSession *s)
{
    if (!s) {
        return;
    }
    for (auto it = g_registry.begin(); it != g_registry.end();)
        it = (it->second == s) ? g_registry.erase(it) : std::next(it);
    delete s->sess; // owned by the backend factory
    s->backend.reset();
    delete s;
}

} // extern "C"
