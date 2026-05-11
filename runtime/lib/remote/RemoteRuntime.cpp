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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>

#include "Exception.hpp"
#include "RemoteCAPI.h"
#include "RemoteSession.hpp"

namespace {

struct RemoteEntry {
    catalyst::remote::RemoteSession *session = nullptr; // The remote session handle.
    std::set<std::string> loaded_paths; // The paths of the binaries that are going to be loaded
                                        // into the remote session.
    std::mutex mu;                      // The mutex to protect the loaded paths.
};

std::mutex g_map_mu;

// Each address has its own RemoteEntry, so we can dispatch the object file to different remote
// sessions.
std::map<std::string, std::unique_ptr<RemoteEntry>> remote_sessions;

thread_local std::string g_remote_runtime_error;
void set_remote_runtime_error(const char *msg)
{
    if (msg) {
        g_remote_runtime_error = msg;
    }
    else {
        g_remote_runtime_error = "(unknown)";
    }
}

// DEBUG logs
bool remote_verbose()
{
    static const bool v = []() {
        const char *e = std::getenv("CATALYST_REMOTE_VERBOSE");
        return e && *e && *e != '0';
    }();
    return v;
}

// Look up or create the entry for `addr`.
RemoteEntry *find_or_create_entry(const char *addr, bool create_if_missing)
{
    if (!addr || !*addr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_map_mu);
    auto it = remote_sessions.find(addr);
    if (it != remote_sessions.end()) {
        return it->second.get();
    }
    if (!create_if_missing) {
        return nullptr;
    }
    auto inserted = remote_sessions.emplace(std::string(addr), std::make_unique<RemoteEntry>());
    return inserted.first->second.get();
}

} // namespace

extern "C" {

int __catalyst__remote__open(const char *addr)
{
    if (!addr || !*addr) {
        set_remote_runtime_error("Empty address");
        return -1;
    }
    RemoteEntry *entry = find_or_create_entry(addr, /*create_if_missing=*/true);
    std::lock_guard<std::mutex> lock(entry->mu);
    if (entry->session) {
        return 0; // idempotent per addr
    }
    if (remote_verbose()) {
        std::fprintf(stderr, "[remote] open(addr=%s)\n", addr);
    }
    entry->session = catalyst::remote::open(addr);
    if (!entry->session) {
        std::string msg = "Could not connect to catalyst-executor at ";
        msg += addr;
        msg += ": ";
        msg += catalyst::remote::last_error();
        set_remote_runtime_error(msg.c_str());
        std::lock_guard<std::mutex> mapLock(g_map_mu);
        remote_sessions.erase(addr);
        return -1;
    }
    if (remote_verbose()) {
        std::fprintf(stderr, "[remote] open(%s) OK\n", addr);
    }
    return 0;
}

int __catalyst__remote__send_binary(const char *addr, const char *path, uint32_t format)
{
    RemoteEntry *entry = find_or_create_entry(addr, /*create_if_missing=*/false);
    if (!entry) {
        set_remote_runtime_error("No session found, call __catalyst__remote__open first.");
        return -1;
    }
    std::lock_guard<std::mutex> lock(entry->mu);
    if (!entry->session) {
        std::string msg = "__catalyst__remote__send_binary(";
        msg += addr;
        msg += "): session is closed.";
        set_remote_runtime_error(msg.c_str());
        return -1;
    }
    if (!path || !*path) {
        return 0;
    }
    std::string key(path);
    if (!entry->loaded_paths.insert(key).second) {
        return 0;
    }
    if (remote_verbose()) {
        std::fprintf(stderr, "[remote] send_binary(addr=%s, path=%s, format=%u)\n", addr, path,
                     format);
    }

    int rc = 0;
    switch (format) {
    case 0:
        rc = catalyst::remote::load_object_path(entry->session, path);
        break;
    default:
        std::string msg = "unknown binary format tag ";
        msg += std::to_string(format);
        set_remote_runtime_error(msg.c_str());
        rc = -1;
    }

    if (rc != 0) {
        set_remote_runtime_error(catalyst::remote::last_error());
        entry->loaded_paths.erase(key);
        return -1;
    }
    return 0;
}

int __catalyst__remote__close()
{
    std::lock_guard<std::mutex> mapLock(g_map_mu);
    for (auto &[addr, entry] : remote_sessions) {
        std::lock_guard<std::mutex> lock(entry->mu);
        if (entry->session) {
            catalyst::remote::close(entry->session);
            entry->session = nullptr;
            entry->loaded_paths.clear();
        }
    }
    remote_sessions.clear();
    return 0;
}

const char *__catalyst__remote__last_error() { return g_remote_runtime_error.c_str(); }

void __catalyst__remote__launch(const char *addr, const char *entry_symbol, size_t num_inputs,
                                void *const *input_descs, const size_t *input_ranks,
                                const size_t *input_elem_sizes, size_t num_outputs,
                                void *const *output_descs, const size_t *output_ranks,
                                const size_t *output_elem_sizes)
{
    RemoteEntry *entry = find_or_create_entry(addr, /*create_if_missing=*/false);
    if (!entry) {
        RT_FAIL("Can't find opened session");
    }

    std::lock_guard<std::mutex> lock(entry->mu);
    if (remote_verbose()) {
        std::fprintf(stderr, "[remote] launch(addr=%s, symbol=%s, n_in=%zu, n_out=%zu)\n", addr,
                     entry_symbol, num_inputs, num_outputs);
    }
    if (!entry->session) {
        RT_FAIL("Session is closed");
    }
    uint64_t entry_addr = catalyst::remote::lookup(entry->session, entry_symbol);
    if (!entry_addr) {
        RT_FAIL(catalyst::remote::last_error());
    }
    if (catalyst::remote::invoke_kernel(entry->session, entry_addr, num_inputs, input_descs,
                                        input_ranks, input_elem_sizes, num_outputs, output_descs,
                                        output_ranks, output_elem_sizes) != 0) {
        RT_FAIL(catalyst::remote::last_error());
    }
}

} // extern "C"
