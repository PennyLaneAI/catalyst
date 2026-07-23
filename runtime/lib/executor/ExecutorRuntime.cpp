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

struct ExecutorEntry {
    catalyst::executor::ExecutorSession *session = nullptr; // The executor session handle.
    int64_t handle = 0;                 // i64 handle handed to the compiled program.
    std::string address;                // The endpoint this entry connects to.
    std::set<std::string> loaded_paths; // Paths of the binaries already loaded into the session.
    std::mutex mu;                      // Guards this entry's `session` and `loaded_paths`.

    ~ExecutorEntry()
    {
        if (session) {
            catalyst::executor::close(session);
            session = nullptr;
        }
    }
};

// Guards both session maps and `g_next_handle`.
std::mutex g_map_mu;

// Entries are held by shared_ptr so a handle returned by the lookup helpers stays valid across
// concurrent map mutation.
std::map<std::string, std::shared_ptr<ExecutorEntry>> g_sessions_by_addr;
std::map<int64_t, std::shared_ptr<ExecutorEntry>> g_sessions_by_handle;

// Monotonic handle allocator. 0 is reserved for the invalid handle.
int64_t g_next_handle = 1;

// DEBUG logs
bool remote_verbose()
{
    static const bool v = []() {
        const char *e = std::getenv("CATALYST_REMOTE_VERBOSE");
        return e && *e && *e != '0';
    }();
    return v;
}

// Resolve a session handle to its entry.
//
// Returns a `shared_ptr` so the entry outlives concurrent map mutation: if another thread erases
// this handle from the maps (e.g. __catalyst__executor__close) right after we return, the returned
// handle keeps the ExecutorEntry alive. The caller must lock `entry->mu` before touching `session`,
// and re-check `session` (it may have been closed in the meantime).
//
// Returns nullptr if `session` is 0 or unknown.
std::shared_ptr<ExecutorEntry> find_entry_by_handle(int64_t session)
{
    if (session == 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_map_mu);
    auto it = g_sessions_by_handle.find(session);
    if (it == g_sessions_by_handle.end()) {
        return nullptr;
    }
    return it->second;
}

} // namespace

extern "C" {

int64_t __catalyst__executor__open(const char *addr)
{
    if (!addr || !*addr) {
        RT_FAIL("Empty address");
    }
    std::shared_ptr<ExecutorEntry> entry;
    {
        std::lock_guard<std::mutex> mapLock(g_map_mu);
        auto it = g_sessions_by_addr.find(addr);
        if (it != g_sessions_by_addr.end()) {
            entry = it->second;
        }
        else {
            entry = std::make_shared<ExecutorEntry>();
            entry->address = addr;
            entry->handle = g_next_handle++;
            g_sessions_by_addr.emplace(std::string(addr), entry);
            g_sessions_by_handle.emplace(entry->handle, entry);
        }
    }
    std::string err_msg;
    {
        std::lock_guard<std::mutex> lock(entry->mu);
        if (entry->session) {
            return entry->handle; // idempotent per addr: reuse the live session
        }
        if (remote_verbose()) {
            std::fprintf(stderr, "[remote] open(addr=%s)\n", addr);
        }
        entry->session = catalyst::executor::open(addr);
        if (entry->session) {
            if (remote_verbose()) {
                std::fprintf(stderr, "[remote] open(%s) OK -> session %lld\n", addr,
                             static_cast<long long>(entry->handle));
            }
            return entry->handle;
        }
        err_msg = "Could not connect to catalyst-executor at ";
        err_msg += addr;
        err_msg += ": ";
        err_msg += catalyst::executor::last_error();
    }
    // Connect failed: drop the entry so a later open retries from scratch.
    {
        std::lock_guard<std::mutex> mapLock(g_map_mu);
        g_sessions_by_addr.erase(addr);
        g_sessions_by_handle.erase(entry->handle);
    }
    RT_FAIL(err_msg.c_str());
}

int64_t __catalyst__executor__send_binary(int64_t session, const char *path, uint32_t format)
{
    auto entry = find_entry_by_handle(session);
    if (!entry) {
        RT_FAIL("Invalid session handle, call __catalyst__executor__open first.");
    }
    std::lock_guard<std::mutex> lock(entry->mu);
    if (!entry->session) {
        std::string msg = "__catalyst__executor__send_binary(session=";
        msg += std::to_string(session);
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
        std::fprintf(stderr, "[remote] send_binary(session=%lld, addr=%s, path=%s, format=%u)\n",
                     static_cast<long long>(session), entry->address.c_str(), path, format);
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
 * @param session The i64 handle of the remote session.
 * @param symbol The symbol of the function to call.
 * @param args_buf The buffer of the arguments.
 * @param args_size The size of the arguments.
 * @param out_buf The buffer of the result.
 * @param out_size The size of the result.
 * @return int 0 on success, -1 on error.
 */
int32_t __catalyst__executor__call_wrapper(int64_t session, const char *symbol,
                                           const char *args_buf, size_t args_size, void **out_buf,
                                           size_t *out_size)
{
    if (out_buf) {
        *out_buf = nullptr;
    }
    if (out_size) {
        *out_size = 0;
    }
    auto entry = find_entry_by_handle(session);
    if (!entry) {
        RT_FAIL("Invalid session handle, call __catalyst__executor__open first.");
    }
    std::lock_guard<std::mutex> lock(entry->mu);
    if (!entry->session) {
        RT_FAIL("Session is closed");
    }
    if (!symbol || !*symbol) {
        RT_FAIL("Empty symbol passed to __catalyst__executor__call_wrapper");
    }
    if (remote_verbose()) {
        std::fprintf(stderr, "[remote] call_wrapper(session=%lld, sym=%s, in_size=%zu)\n",
                     static_cast<long long>(session), symbol, args_size);
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

int64_t __catalyst__executor__close(int64_t session)
{
    auto entry = find_entry_by_handle(session);
    if (!entry) {
        return 0;
    }
    std::string addr = entry->address;
    {
        // Close under the entry lock only; do not hold g_map_mu here (see lock ordering above).
        std::lock_guard<std::mutex> lock(entry->mu);
        if (entry->session) {
            if (remote_verbose()) {
                std::fprintf(stderr, "[remote] close(session=%lld, addr=%s)\n",
                             static_cast<long long>(session), addr.c_str());
            }
            catalyst::executor::close(entry->session);
            entry->session = nullptr;
            entry->loaded_paths.clear();
        }
    }

    {
        std::lock_guard<std::mutex> mapLock(g_map_mu);
        g_sessions_by_handle.erase(session);
        g_sessions_by_addr.erase(addr);
    }
    return 0;
}

void __catalyst__executor__launch(int64_t session, const char *entry_symbol, size_t num_inputs,
                                  void *const *input_descs, const size_t *input_ranks,
                                  const size_t *input_elem_sizes, size_t num_outputs,
                                  void *const *output_descs, const size_t *output_ranks,
                                  const size_t *output_elem_sizes)
{
    auto entry = find_entry_by_handle(session);
    if (!entry) {
        RT_FAIL("Can't find opened session");
    }

    std::lock_guard<std::mutex> lock(entry->mu);
    if (remote_verbose()) {
        std::fprintf(stderr, "[remote] launch(session=%lld, symbol=%s, n_in=%zu, n_out=%zu)\n",
                     static_cast<long long>(session), entry_symbol, num_inputs, num_outputs);
    }
    if (!entry->session) {
        RT_FAIL("Session is closed");
    }
    uint64_t entry_addr = catalyst::executor::lookup(entry->session, entry_symbol);
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
