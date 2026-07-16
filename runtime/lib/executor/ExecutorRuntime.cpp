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
#include "ExecutorCAPI.h"
#include "ExecutorSession.hpp"

namespace {

struct RemoteEntry {
    catalyst::executor::ExecutorSession *session = nullptr; // The remote session handle.
    std::set<std::string> loaded_paths; // The paths of the binaries that are going to be loaded
                                        // into the remote session.
    std::mutex mu;                      // The mutex to protect the loaded paths.

    ~RemoteEntry()
    {
        if (session) {
            catalyst::executor::close(session);
            session = nullptr;
        }
    }
};

std::mutex g_map_mu;

// Each address has its own RemoteEntry, so we can dispatch the object file to different remote
// sessions.
std::map<std::string, std::unique_ptr<RemoteEntry>> remote_sessions;

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

int __catalyst__executor__open(const char *addr)
{
    if (!addr || !*addr) {
        RT_FAIL("Empty address");
    }
    RemoteEntry *entry = find_or_create_entry(addr, /*create_if_missing=*/true);
    std::string err_msg;
    {
        std::lock_guard<std::mutex> lock(entry->mu);
        if (entry->session) {
            return 0; // idempotent per addr
        }
        if (remote_verbose()) {
            std::fprintf(stderr, "[remote] open(addr=%s)\n", addr);
        }
        entry->session = catalyst::executor::open(addr);
        if (entry->session) {
            if (remote_verbose()) {
                std::fprintf(stderr, "[remote] open(%s) OK\n", addr);
            }
            return 0;
        }
        err_msg = "Could not connect to catalyst-executor at ";
        err_msg += addr;
        err_msg += ": ";
        err_msg += catalyst::executor::last_error();
    }
    {
        std::lock_guard<std::mutex> mapLock(g_map_mu);
        remote_sessions.erase(addr);
    }
    RT_FAIL(err_msg.c_str());
}

int __catalyst__executor__send_binary(const char *addr, const char *path, uint32_t format)
{
    RemoteEntry *entry = find_or_create_entry(addr, /*create_if_missing=*/false);
    if (!entry) {
        RT_FAIL("No session found, call __catalyst__executor__open first.");
    }
    std::lock_guard<std::mutex> lock(entry->mu);
    if (!entry->session) {
        std::string msg = "__catalyst__executor__send_binary(";
        msg += addr;
        msg += "): session is closed.";
        RT_FAIL(msg.c_str());
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
        rc = catalyst::executor::load_object_path(entry->session, path);
        break;
    case 1:
        rc = catalyst::executor::load_asset_path(entry->session, path);
        break;
    default: {
        std::string msg = "unknown binary format tag ";
        msg += std::to_string(format);
        entry->loaded_paths.erase(key);
        RT_FAIL(msg.c_str());
    }
    }

    if (rc != 0) {
        std::string msg = catalyst::executor::last_error();
        entry->loaded_paths.erase(key);
        RT_FAIL(msg.c_str());
    }
    return 0;
}

/**
 * @brief Generic ORC wrapper-function call by symbol name. Returns 0 on success, -1 on error.
 *
 * @param addr The address of the remote session.
 * @param symbol The symbol of the function to call.
 * @param args_buf The buffer of the arguments.
 * @param args_size The size of the arguments.
 * @param out_buf The buffer of the result.
 * @param out_size The size of the result.
 * @return int 0 on success, -1 on error.
 */
int __catalyst__executor__call_wrapper(const char *addr, const char *symbol, const char *args_buf,
                                       size_t args_size, void **out_buf, size_t *out_size)
{
    if (out_buf) {
        *out_buf = nullptr;
    }
    if (out_size) {
        *out_size = 0;
    }
    RemoteEntry *entry = find_or_create_entry(addr, /*create_if_missing=*/false);
    if (!entry) {
        RT_FAIL("No session found, call __catalyst__executor__open first.");
    }
    std::lock_guard<std::mutex> lock(entry->mu);
    if (!entry->session) {
        RT_FAIL("Session is closed");
    }
    if (!symbol || !*symbol) {
        RT_FAIL("Empty symbol passed to __catalyst__executor__call_wrapper");
    }
    if (remote_verbose()) {
        std::fprintf(stderr, "[remote] call_wrapper(addr=%s, sym=%s, in_size=%zu)\n", addr, symbol,
                     args_size);
    }
    char *buf = nullptr;
    size_t n = 0;
    int rc =
        catalyst::executor::call_wrapper_raw(entry->session, symbol, args_buf, args_size, &buf, &n);
    if (rc != 0) {
        RT_FAIL(catalyst::executor::last_error());
    }
    if (out_buf) {
        *out_buf = buf;
    }
    else {
        std::free(buf); // caller didn't want the bytes back
    }
    if (out_size) {
        *out_size = n;
    }
    return 0;
}

void __catalyst__executor__free_result(void *buf) { std::free(buf); }

int __catalyst__executor__close()
{
    std::lock_guard<std::mutex> mapLock(g_map_mu);
    for (auto &[addr, entry] : remote_sessions) {
        std::lock_guard<std::mutex> lock(entry->mu);
        if (entry->session) {
            catalyst::executor::close(entry->session);
            entry->session = nullptr;
            entry->loaded_paths.clear();
        }
    }
    remote_sessions.clear();
    return 0;
}

void __catalyst__executor__launch(const char *addr, const char *entry_symbol, const char *object,
                                  size_t num_inputs, void *const *input_descs,
                                  const size_t *input_ranks, const size_t *input_elem_sizes,
                                  size_t num_outputs, void *const *output_descs,
                                  const size_t *output_ranks, const size_t *output_elem_sizes)
{
    RemoteEntry *entry = find_or_create_entry(addr, /*create_if_missing=*/false);
    if (!entry) {
        RT_FAIL("Can't find opened session");
    }

    std::lock_guard<std::mutex> lock(entry->mu);
    if (remote_verbose()) {
        std::fprintf(stderr,
                     "[remote] launch(addr=%s, symbol=%s, object=%s, n_in=%zu, n_out=%zu)\n", addr,
                     entry_symbol, object ? object : "<none>", num_inputs, num_outputs);
    }
    if (!entry->session) {
        RT_FAIL("Session is closed");
    }
    // Resolve the entry within the JITDylib of its own object, so same-named entries shipped from
    // different kernel objects don't collide in one symbol namespace.
    uint64_t entry_addr = catalyst::executor::lookup(entry->session, entry_symbol, object);
    if (!entry_addr) {
        RT_FAIL(catalyst::executor::last_error());
    }
    if (catalyst::executor::invoke_kernel(entry->session, entry_addr, num_inputs, input_descs,
                                          input_ranks, input_elem_sizes, num_outputs, output_descs,
                                          output_ranks, output_elem_sizes) != 0) {
        RT_FAIL(catalyst::executor::last_error());
    }
}

} // extern "C"
