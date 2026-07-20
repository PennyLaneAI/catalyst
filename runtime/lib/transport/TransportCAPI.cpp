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

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "DynamicLibraryLoader.hpp"
#include "Transport.hpp"
#include "TransportBackend.h"

using catalyst::transport::ChannelDesc;
using catalyst::transport::ConnectInfo;
using catalyst::transport::ControllerSession;
using catalyst::transport::DataPath;
using catalyst::transport::MemKind;
using catalyst::transport::MemRegion;
using catalyst::transport::PeerRef;

// The opaque handle
struct CatalystTransportSession {
    std::unique_ptr<DynamicLibraryLoader> backend;
    ControllerSession *sess = nullptr; // heap-allocated by the backend factory
    MemRegion reply = {};
    bool have_reply = false;
};

namespace {

MemKind to_mem_kind(std::int32_t k)
{
    switch (k) {
    case CATALYST_TRANSPORT_MEM_CPU_RAM:
        return MemKind::CpuRam;
    case CATALYST_TRANSPORT_MEM_GPU_HBM:
        return MemKind::GpuHbm;
    case CATALYST_TRANSPORT_MEM_DDR:
        return MemKind::Ddr;
    case CATALYST_TRANSPORT_MEM_OTHER:
        return MemKind::Other;
    default:
        return MemKind::Ddr;
    }
}

DataPath to_data_path(std::int32_t p)
{
    switch (p) {
    case CATALYST_TRANSPORT_PATH_CPU_VERBS:
        return DataPath::CpuVerbs;
    case CATALYST_TRANSPORT_PATH_GPU_ENGINE:
        return DataPath::GpuEngine;
    case CATALYST_TRANSPORT_PATH_OTHER:
    default:
        return DataPath::Other;
    }
}

template <typename Fn> int guard(Fn &&fn)
{
    try {
        return fn();
    }
    catch (const std::exception &e) {
        std::cerr << "[transport] " << e.what() << "\n";
        return CATALYST_TRANSPORT_ERR;
    }
    catch (...) {
        return CATALYST_TRANSPORT_ERR;
    }
}

} // namespace

extern "C" {

CatalystTransportSession *__catalyst__transport__controller_create(const char *backend_lib,
                                                                   const char *config)
{
    try {
        if (!backend_lib || !*backend_lib) {
            std::cerr << "[transport] no backend library given\n";
            return nullptr;
        }
        auto h = std::make_unique<CatalystTransportSession>();
        h->backend = std::make_unique<DynamicLibraryLoader>(backend_lib);
        auto *factory = h->backend->getSymbol<CatalystTransportControllerFactoryFn *>(
            CATALYST_TRANSPORT_CONTROLLER_FACTORY_SYMBOL);
        h->sess = factory(config ? config : "");
        if (!h->sess) {
            std::cerr << "[transport] backend factory returned null for config: "
                      << (config ? config : "") << "\n";
            return nullptr;
        }
        return h.release();
    }
    catch (const std::exception &e) {
        std::cerr << "[transport] controller_create: " << e.what() << "\n";
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
    return guard([&] {
        ConnectInfo info;
        info.peer = peer ? peer : "";
        info.oob_port = oob_port;
        return s->sess->connect(info);
    });
}

int __catalyst__transport__alloc_reply(CatalystTransportSession *s, std::uint64_t size,
                                       std::int32_t mem_kind, std::uint32_t access,
                                       CatalystTransportMemRegion *out)
{
    if (!s || !s->sess) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] {
        MemRegion r = s->sess->alloc_memory(size, to_mem_kind(mem_kind), access);
        s->reply = r;
        s->have_reply = true;
        if (out) {
            out->addr = r.addr;
            out->size = r.size;
            out->lkey = r.lkey;
            out->rkey = r.rkey;
            out->kind = mem_kind;
        }
        return CATALYST_TRANSPORT_OK;
    });
}

int __catalyst__transport__exchange_keys(CatalystTransportSession *s, CatalystTransportPeerRef *out)
{
    if (!s || !s->sess) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] {
        PeerRef p = s->sess->exchange_keys(s->have_reply ? s->reply : MemRegion{});
        if (out) {
            out->rkey = p.rkey;
            out->remote_addr = p.remote_addr;
            out->size = p.size;
        }
        return CATALYST_TRANSPORT_OK;
    });
}

int __catalyst__transport__establish_channel(CatalystTransportSession *s, std::int32_t data_path,
                                             const CatalystTransportPeerRef *peer)
{
    if (!s || !s->sess || !peer) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] {
        ChannelDesc desc;
        desc.data_path = to_data_path(data_path);
        PeerRef p;
        p.rkey = peer->rkey;
        p.remote_addr = peer->remote_addr;
        p.size = peer->size;
        s->sess->establish_channel(desc, s->have_reply ? s->reply : MemRegion{}, p);
        return CATALYST_TRANSPORT_OK;
    });
}

int __catalyst__transport__commit_work_item(CatalystTransportSession *s,
                                            std::uint32_t work_item_idx, std::uint64_t in_bytes,
                                            std::uint64_t out_bytes)
{
    if (!s || !s->sess) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] {
        s->sess->commit_work_item(work_item_idx, in_bytes, out_bytes);
        return CATALYST_TRANSPORT_OK;
    });
}

void *__catalyst__transport__data_slot(CatalystTransportSession *s)
{
    if (!s || !s->sess) {
        return nullptr;
    }

    void *slot = nullptr;
    try {
        slot = s->sess->data_slot();
    }
    catch (const std::exception &e) {
        std::cerr << "[transport] data_slot: " << e.what() << "\n";
        return nullptr;
    }
    catch (...) {
        return nullptr;
    }
    return slot;
}

int __catalyst__transport__kick(CatalystTransportSession *s, std::uint32_t work_item_idx)
{
    if (!s || !s->sess) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] { return s->sess->kick(work_item_idx); });
}

int __catalyst__transport__collect(CatalystTransportSession *s, void *correction,
                                   std::uint64_t bytes)
{
    if (!s || !s->sess) {
        return CATALYST_TRANSPORT_ERR;
    }
    return guard([&] {
        void *outputs[1] = {correction};
        std::size_t caps[1] = {static_cast<std::size_t>(bytes)};
        return s->sess->collect(outputs, caps, 1);
    });
}

std::uint64_t __catalyst__transport__last_rtt_ns(CatalystTransportSession *s)
{
    if (!s || !s->sess) {
        return 0;
    }
    return s->sess->last_rtt_ns();
}

void __catalyst__transport__stop(CatalystTransportSession *s)
{
    if (s && s->sess) {
        try {
            s->sess->stop();
        }
        catch (...) {
        }
    }
}

void __catalyst__transport__destroy(CatalystTransportSession *s)
{
    if (!s) {
        return;
    }
    delete s->sess; // owned by the backend factory
    s->backend.reset();
    delete s;
}

void __catalyst__transport__close(CatalystTransportSession *s)
{
    __catalyst__transport__stop(s);
    __catalyst__transport__destroy(s);
}

} // extern "C"
