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
#include <ctime>
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
#include <vector>

#include "ConfigParser.hpp"
#include "DynamicLibraryLoader.hpp"
#include "Transport.hpp"
#include "TransportBackend.h"
#include "WireProtocol.hpp"

using catalyst::transport::ChannelDesc;
using catalyst::transport::ConnectInfo;
using catalyst::transport::ControllerSession;
using catalyst::transport::CoprocConvention;
using catalyst::transport::CoprocessorFn;
using catalyst::transport::CoprocessorLauncherFn;
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
    std::vector<std::int64_t> pending_tokens;

    std::uint32_t work_item = 0;
    std::uint64_t in_bytes = 0;
    std::uint64_t out_bytes = 0;
    bool work_item_ready = false;
};

namespace {

std::uint64_t now_ns() {
    timespec ts = {};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ull +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

// (role, key) -> live session registry. Populated at create, read by get_session so a session
// brought up in one function can be resolved in another.
std::unordered_map<std::string, CatalystTransportSession *> g_registry;

std::string registry_key(std::int32_t role, const char *key) {
    return std::to_string(role) + "/" + (key ? key : "");
}

// Local reply region size, provisioned automatically at exchange_keys.
// TODO: Make this configurable rather than hard-coded. Expose an API such as
// __catalyst__transport__set_reply_bytes(s, n) to override the region size per session, keeping
// 16KB as the default when it is not set.
constexpr std::uint64_t kReplyBytes = 16 * 1024;

// Run fn, logging and swallowing any exception. Returns fn()'s result, or CATALYST_TRANSPORT_ERR
// if it threw; when fn() returns void there is nothing to return.
template <typename Fn> auto guard(Fn &&fn) -> decltype(fn()) {
    try {
        return fn();
    } catch (const std::exception &e) {
        std::cerr << "[transport] " << e.what() << "\n";
    } catch (...) {
    }
    if constexpr (!std::is_void_v<decltype(fn())>) {
        return CATALYST_TRANSPORT_ERR;
    }
}

const char *collect_error_name(int rc) {
    switch (rc) {
    case CATALYST_TRANSPORT_ERR_MEMORY:
        return "memory";
    case CATALYST_TRANSPORT_ERR_TIMEOUT:
        return "timeout";
    case CATALYST_TRANSPORT_ERR_STUCK:
        return "stuck - no reply before the deadline";
    default:
        return "error";
    }
}

// Provision the local reply region on first use (idempotent).
void ensure_reply(CatalystTransportSession *s) {
    if (!s->reply_ready) {
        s->reply = s->sess->alloc_memory(kReplyBytes, s->sess->preferred_mem_kind());
        s->reply_ready = true;
    }
}

ControllerSession *cast_to_controller(CatalystTransportSession *s) {
    return s ? dynamic_cast<ControllerSession *>(s->sess) : nullptr;
}

// Bring-up bodies shared by the blocking and async (worker-thread) entry points.
int do_connect(CatalystTransportSession *s, std::string peer, std::uint16_t oob_port) {
    ConnectInfo info;
    info.peer = std::move(peer);
    info.oob_port = oob_port;
    return s->sess->connect(info);
}

int do_exchange_keys(CatalystTransportSession *s) {
    ensure_reply(s);
    s->peer = s->sess->exchange_keys(s->reply);
    s->peer_ready = true;
    return CATALYST_TRANSPORT_OK;
}

// Built-in fallback coprocessor function: echo the input back.
std::size_t echo_fn(const void *in, std::size_t in_len, void *out, std::size_t out_cap, void *) {
    std::size_t n = std::min(in_len, out_cap);
    if (n && in && out) {
        std::memcpy(out, in, n);
    }
    return n;
}

// Async task registry: connect_async / exchange_keys_async run on a worker thread and return a
// token; barrier awaits it. Tokens start at 1 so a 0 return can signal a dispatch failure.
// Each token is also recorded on the owning session so destroy can drain outstanding work.
std::mutex g_async_mtx;
std::int64_t g_next_token = 1;
std::unordered_map<std::int64_t, std::future<int>> g_async_tasks;
std::unordered_map<std::int64_t, CatalystTransportSession *> g_token_owner;

void forget_token_locked(std::int64_t token) {
    auto oit = g_token_owner.find(token);
    if (oit == g_token_owner.end()) {
        return;
    }
    auto &pending = oit->second->pending_tokens;
    pending.erase(std::remove(pending.begin(), pending.end(), token), pending.end());
    g_token_owner.erase(oit);
}

std::int64_t dispatch_async(CatalystTransportSession *s, std::function<int()> fn) {
    std::lock_guard<std::mutex> lk(g_async_mtx);
    std::int64_t token = g_next_token++;
    g_async_tasks.emplace(token, std::async(std::launch::async, std::move(fn)));
    g_token_owner.emplace(token, s);
    s->pending_tokens.push_back(token);
    return token;
}

int await_token(std::int64_t token) {
    std::future<int> fut;
    {
        std::lock_guard<std::mutex> lk(g_async_mtx);
        auto it = g_async_tasks.find(token);
        if (it == g_async_tasks.end()) {
            return CATALYST_TRANSPORT_ERR;
        }
        fut = std::move(it->second);
        g_async_tasks.erase(it);
        forget_token_locked(token);
    }
    return guard([&] { return fut.get(); });
}

// Await any connect_async / exchange_keys_async work still outstanding for this session.
void drain_pending(CatalystTransportSession *s) {
    std::vector<std::int64_t> tokens;
    {
        std::lock_guard<std::mutex> lk(g_async_mtx);
        tokens.swap(s->pending_tokens);
    }
    for (std::int64_t token : tokens) {
        (void)await_token(token);
    }
}

// Fold the compiler-emitted session key (unique per controller/coprocessor pair, as emitted by
// inject-transport-session in MLIR) into the config as `pair=<key>` so backends that need to
// pair endpoints in-process (e.g. memcpy) share one identifier for both sides. Transparent to
// backends that don't parse `pair`.
std::string fold_pair_key(const char *config, const char *key) {
    std::string cfg = config ? config : "";
    if (!key || !*key) {
        return cfg;
    }
    const std::string_view k(key);
    if (k.find(';') != std::string_view::npos) {
        throw std::runtime_error("session key must not contain ';' (got '" + std::string(k) +
                                 "'); it would split the backend config into two entries");
    }
    // A caller-supplied `pair=` and ours would coexist, and the backend would silently pick one.
    bool reserved_in_use = false;
    catalyst::transport::common::configparser::for_each_kv(
        cfg, [&](std::string_view entry_key, std::string_view) {
            if (entry_key == "pair") {
                reserved_in_use = true;
            }
        });
    if (reserved_in_use) {
        throw std::runtime_error("backend config must not set 'pair'; it is reserved for the "
                                 "compiler-emitted session key");
    }
    if (!cfg.empty()) {
        cfg += ";";
    }
    cfg += "pair=";
    cfg += k;
    return cfg;
}

// Try the plugin handle first, then the process-global namespace (main image).
void *resolve_coprocessor_fn_symbol(CatalystTransportSession *s, const char *symbol) {
    dlerror();
    if (s->backend && s->backend->handle) {
        if (void *sym = dlsym(s->backend->handle, symbol)) {
            return sym;
        }
        dlerror();
    }
    return dlsym(RTLD_DEFAULT, symbol);
}

} // namespace

extern "C" {

CatalystTransportSession *__catalyst__transport__create(const char *backend_lib, const char *config,
                                                        std::int32_t role, const char *key) {
    try {
        if (!backend_lib || !*backend_lib) {
            std::cerr << "[transport] no backend library given\n";
            return nullptr;
        }
        auto h = std::make_unique<CatalystTransportSession>();
        h->backend = std::make_unique<DynamicLibraryLoader>(backend_lib);
        const std::string cfg = fold_pair_key(config, key);
        if (role == CATALYST_TRANSPORT_ROLE_COPROCESSOR) {
            auto *factory = h->backend->getSymbol<CatalystTransportCoprocessorFactoryFn *>(
                CATALYST_TRANSPORT_COPROCESSOR_FACTORY_SYMBOL);
            h->sess = factory(cfg.c_str());
        } else {
            auto *factory = h->backend->getSymbol<CatalystTransportControllerFactoryFn *>(
                CATALYST_TRANSPORT_CONTROLLER_FACTORY_SYMBOL);
            h->sess = factory(cfg.c_str());
        }
        if (!h->sess) {
            std::cerr << "[transport] backend factory returned null for config: " << cfg << "\n";
            return nullptr;
        }
        auto *raw = h.release();
        if (key && *key) {
            const std::string rk = registry_key(role, key);
            auto it = g_registry.find(rk);
            if (it != g_registry.end() && it->second != raw) {
                // Registry is a soft lookup only; the prior session is still owned by its
                // create() return value and must be destroy()'d by the caller.
                std::cerr << "[transport] create: overwriting registry entry for role " << role
                          << " key '" << key
                          << "' without destroying the previous session; caller must still "
                             "destroy the old handle\n";
            }
            g_registry[rk] = raw; // resolved later via get_session
        }
        return raw;
    } catch (const std::exception &e) {
        std::cerr << "[transport] create: " << e.what() << "\n";
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

int __catalyst__transport__connect(CatalystTransportSession *s, const char *peer,
                                   std::uint16_t oob_port) {
    if (!s || !s->sess) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] { return do_connect(s, peer ? peer : "", oob_port); });
}

std::int64_t __catalyst__transport__connect_async(CatalystTransportSession *s, const char *peer,
                                                  std::uint16_t oob_port) {
    if (!s || !s->sess) {
        return 0;
    }
    return dispatch_async(
        s, [s, p = std::string(peer ? peer : ""), oob_port] { return do_connect(s, p, oob_port); });
}

int __catalyst__transport__exchange_keys(CatalystTransportSession *s) {
    if (!s || !s->sess) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] { return do_exchange_keys(s); });
}

std::int64_t __catalyst__transport__exchange_keys_async(CatalystTransportSession *s) {
    if (!s || !s->sess) {
        return 0;
    }
    return dispatch_async(s, [s] { return do_exchange_keys(s); });
}

int __catalyst__transport__await(std::int64_t token) { return await_token(token); }

int __catalyst__transport__establish_channel(CatalystTransportSession *s, const char *transport) {
    if (!s || !s->sess) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] {
        ChannelDesc desc;
        desc.transport = transport ? transport : ""; // opaque; the backend interprets it
        s->sess->establish_channel(desc, s->reply, s->peer);
        return CATALYST_TRANSPORT_OK;
    });
}

int __catalyst__transport__set_coprocessor_fn(CatalystTransportSession *s, const char *symbol) {
    if (!s || !s->sess) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] {
        auto *co = dynamic_cast<CoprocessorSession *>(s->sess);
        if (!co) {
            std::cerr << "[transport] set_coprocessor_fn on a non-coprocessor session\n";
            return CATALYST_TRANSPORT_ERR;
        }
        void *resolved_fn = nullptr;
        if (symbol && *symbol) {
            resolved_fn = resolve_coprocessor_fn_symbol(s, symbol);
            if (!resolved_fn) {
                std::cerr << "[transport] set_coprocessor_fn: symbol not found: " << symbol << "\n";
                return CATALYST_TRANSPORT_ERR;
            }
        }
        switch (co->coprocessor_fn_convention()) {
        case CoprocConvention::PerMessage:
            // No symbol -> the core's built-in echo.
            co->set_coprocessor_fn(
                resolved_fn ? reinterpret_cast<CoprocessorFn>(resolved_fn) : &echo_fn, nullptr);
            return CATALYST_TRANSPORT_OK;
        case CoprocConvention::LaunchOnce:
            // No symbol -> null, letting the backend pick its own default
            // launcher; the core holds no device launcher of its own.
            co->set_coprocessor_launcher(reinterpret_cast<CoprocessorLauncherFn>(resolved_fn),
                                         nullptr);
            return CATALYST_TRANSPORT_OK;
        }
        std::cerr << "[transport] set_coprocessor_fn: backend reported an unknown convention "
                  << static_cast<std::int32_t>(co->coprocessor_fn_convention()) << "\n";
        return CATALYST_TRANSPORT_ERR;
    });
}

int __catalyst__transport__set_message_sizes(CatalystTransportSession *s,
                                             std::uint32_t work_item_idx, std::uint64_t in_bytes,
                                             std::uint64_t out_bytes) {
    auto *c = cast_to_controller(s);
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
        s->work_item = work_item_idx;
        s->in_bytes = in_bytes;
        s->out_bytes = out_bytes;
        s->work_item_ready = true;
        return CATALYST_TRANSPORT_OK;
    });
}

void *__catalyst__transport__request_slot(CatalystTransportSession *s) {
    auto *c = cast_to_controller(s);
    void *slot = nullptr;
    if (c) {
        guard([&] { slot = c->data_slot(); });
    }
    return slot;
}

int __catalyst__transport__stage_payload(CatalystTransportSession *s, const void *src,
                                         std::uint64_t bytes, std::uint32_t decoder_id) {
    auto *c = cast_to_controller(s);
    if (!c) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] {
        c->write_data_slot(src, bytes, decoder_id);
        return 0;
    });
}

void *__catalyst__transport__reply_slot(CatalystTransportSession *s) {
    auto *c = cast_to_controller(s);
    void *slot = nullptr;
    if (c) {
        guard([&] { slot = c->reply_slot(); });
    }
    return slot;
}

int __catalyst__transport__post(CatalystTransportSession *s, std::uint32_t work_item_idx) {
    auto *c = cast_to_controller(s);
    if (!c) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] { return c->kick(work_item_idx); });
}

int __catalyst__transport__collect(CatalystTransportSession *s, void *reply,
                                   std::uint64_t reply_bytes) {
    if (!s || !s->sess) {
        return CATALYST_TRANSPORT_ERR;
    }
    if (!reply) {
        return CATALYST_TRANSPORT_ERR;
    }
    const int rc = guard([&] {
        void *replies[1] = {reply};
        std::uint64_t replies_bytes[1] = {reply_bytes};
        return s->sess->collect(replies, replies_bytes, 1);
    });
    if (rc != CATALYST_TRANSPORT_OK) {
        // The generated code discards this return value, so a failed round is otherwise silent:
        // `reply` keeps whatever it held, and the caller consumes that as a valid result.
        std::cerr << "[transport] collect failed (rc=" << rc << ": " << collect_error_name(rc)
                  << "); the reply buffer was not written\n";
    }
    return rc;
}

std::uint64_t __catalyst__transport__last_rtt_ns(CatalystTransportSession *s) {
    if (s && s->sess) {
        return guard([&] { return s->sess->last_rtt_ns(); });
    }
    return 0;
}

// The benchmark loop
// Flags:
//     CATALYST_BENCH_FORCE_SW_RTT: Force the use of the software RTT.
//     CATALYST_BENCH_PROGRESS: Print progress information.
int __catalyst__transport__start_benchmark(CatalystTransportSession *s, std::uint32_t iters,
                                           std::uint32_t decoder_id, std::uint32_t flags,
                                           std::uint64_t *samples, std::uint64_t samples_bytes,
                                           std::uint64_t *rounds) {
    if (rounds) {
        *rounds = 0;
    }
    auto *c = cast_to_controller(s);
    if (!c || (iters && !samples)) {
        return CATALYST_TRANSPORT_ERR;
    }

    const std::uint64_t samples_wanted = static_cast<std::uint64_t>(iters) * sizeof(std::uint64_t);
    if (samples_bytes < samples_wanted) {
        std::cerr << "[transport] start_benchmark: samples buffer is " << samples_bytes
                  << " B, short of the " << samples_wanted << " B that " << iters
                  << " rounds report\n";
        return CATALYST_TRANSPORT_ERR;
    }

    if (!s->work_item_ready) {
        std::cerr << "[transport] start_benchmark: no work item committed; call "
                     "commit_work_item first so the round's sizes are known\n";
        return CATALYST_TRANSPORT_ERR;
    }

    // The round's shape is whatever was committed, so it cannot disagree with it.
    const std::uint32_t work_item_idx = s->work_item;
    const auto outgoing_message_bytes = static_cast<std::uint32_t>(s->in_bytes);
    const auto reply_bytes = static_cast<std::uint32_t>(s->out_bytes);

    const bool force_sw_rtt = (flags & CATALYST_BENCH_FORCE_SW_RTT) != 0;
    const bool progress = (flags & CATALYST_BENCH_PROGRESS) != 0;

    return guard([&] {
        std::vector<std::uint8_t> outgoing_message(
            std::max<std::size_t>(outgoing_message_bytes, sizeof(std::uint64_t)), 0);
        std::uint64_t written = 0;
        int rc = CATALYST_TRANSPORT_OK;

        for (std::uint32_t i = 0; i < iters; ++i) {
            const std::uint64_t value = static_cast<std::uint64_t>(i) + 1;
            std::memcpy(outgoing_message.data(), &value, sizeof(value));

            c->write_data_slot(outgoing_message.data(), outgoing_message_bytes, decoder_id);

            void *rslot = c->reply_slot();
            std::uint64_t t0 = now_ns();

            rc = c->kick(work_item_idx);
            if (rc == CATALYST_TRANSPORT_OK) {
                void *replies[1] = {rslot};
                std::uint64_t replies_bytes[1] = {reply_bytes};
                rc = s->sess->collect(replies, replies_bytes, 1);
            }
            std::uint64_t sw_rtt = now_ns() - t0;

            if (rc != CATALYST_TRANSPORT_OK) {
                std::uint64_t reply_value = 0;
                std::uint32_t reply_seq = 0;
                std::memcpy(&reply_value, rslot, sizeof(reply_value));
                std::memcpy(&reply_seq,
                            static_cast<const std::uint8_t *>(rslot) +
                                offsetof(catalyst::transport::common::Payload, seq_num),
                            sizeof(reply_seq));
                std::cerr << "[transport] start_benchmark: round " << i << " failed rc=" << rc
                          << " [sent 0x" << std::hex << value << ", reply slot 0x" << reply_value
                          << std::dec << " seq=" << reply_seq << "; work_item=" << work_item_idx
                          << " in=" << outgoing_message_bytes << "B out=" << reply_bytes << "B]\n";
                break;
            }

            std::uint64_t hw_rtt = force_sw_rtt ? 0 : s->sess->last_rtt_ns();
            samples[written++] = (hw_rtt != 0) ? hw_rtt : sw_rtt;

            if (progress && ((i & 1023) == 0 || i == iters - 1)) {
                std::uint64_t cval = 0;
                std::memcpy(&cval, rslot, std::min<std::size_t>(reply_bytes, sizeof(cval)));
                std::cerr << "[transport] round " << i << " rtt=" << samples[written - 1] << " ns ["
                          << (hw_rtt ? "hw" : "sw") << "] reply[0:8]=0x" << std::hex << cval
                          << std::dec << "\n";
            }
        }

        if (rounds) {
            *rounds = written;
        }
        return rc;
    });
}

void __catalyst__transport__start(CatalystTransportSession *s) {
    if (s && s->sess) {
        guard([&] { s->sess->start(); });
    }
}

void __catalyst__transport__stop(CatalystTransportSession *s) {
    if (s && s->sess) {
        guard([&] { s->sess->stop(); });
    }
}

CatalystTransportSession *__catalyst__transport__get_session(std::int32_t role, const char *key) {
    auto it = g_registry.find(registry_key(role, key));
    if (it == g_registry.end()) {
        std::cerr << "[transport] get_session: no session registered for role " << role << " key '"
                  << (key ? key : "") << "'\n";
        return nullptr;
    }
    return it->second;
}

void __catalyst__transport__destroy(CatalystTransportSession *s) {
    if (!s) {
        return;
    }
    for (auto it = g_registry.begin(); it != g_registry.end();) {
        it = (it->second == s) ? g_registry.erase(it) : std::next(it);
    }
    // Wait for any in-flight async bring-up before tearing down
    drain_pending(s);
    if (s->sess) {
        guard([&] { s->sess->stop(); });
    }
    delete s->sess; // owned by the backend factory
    s->backend.reset();
    delete s;
}

} // extern "C"
