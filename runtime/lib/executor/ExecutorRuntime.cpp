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

// Guards the `remote_sessions` map structure itself
std::mutex g_map_mu;

// Entries are held by shared_ptr so a handle returned by `find_or_create_entry` stays valid
std::map<std::string, std::shared_ptr<ExecutorEntry>> remote_sessions;

// DEBUG logs
bool remote_verbose()
{
    static const bool v = []() {
        const char *e = std::getenv("CATALYST_REMOTE_VERBOSE");
        return e && *e && *e != '0';
    }();
    return v;
}

// Look up (or optionally create) the entry for `addr`.
//
// Returns a `shared_ptr` so the entry outlives concurrent map mutation: if another thread
// erases this address from `remote_sessions` (e.g. __catalyst__executor__close) right after we
// return, the returned handle keeps the ExecutorEntry alive. The caller must lock `entry->mu`
// before touching `session`, and re-check `session` (it may have been closed in the meantime).
//
// Returns nullptr if `addr` is empty, or if the entry is absent and `create_if_missing` is false.
std::shared_ptr<ExecutorEntry> find_or_create_entry(const char *addr, bool create_if_missing)
{
    if (!addr || !*addr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_map_mu);
    auto it = remote_sessions.find(addr);
    if (it != remote_sessions.end()) {
        return it->second;
    }
    if (!create_if_missing) {
        return nullptr;
    }
    auto inserted = remote_sessions.emplace(std::string(addr), std::make_shared<ExecutorEntry>());
    return inserted.first->second;
}

} // namespace

extern "C" {

int64_t __catalyst__executor__open(const char *addr)
{
    if (!addr || !*addr) {
        RT_FAIL("Empty address");
    }
    auto entry = find_or_create_entry(addr, /*create_if_missing=*/true);
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

int64_t __catalyst__executor__send_binary(const char *addr, const char *path, uint32_t format)
{
    auto entry = find_or_create_entry(addr, /*create_if_missing=*/false);
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
    auto entry = find_or_create_entry(addr, /*create_if_missing=*/false);
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

int64_t __catalyst__executor__close(const char *addr)
{
    auto entry = find_or_create_entry(addr, /*create_if_missing=*/false);
    if (!entry) {
        return 0;
    }
    {
        // Close under the entry lock only; do not hold g_map_mu here (see lock ordering above).
        std::lock_guard<std::mutex> lock(entry->mu);
        if (entry->session) {
            if (remote_verbose()) {
                std::fprintf(stderr, "[remote] close(addr=%s)\n", addr);
            }
            catalyst::executor::close(entry->session);
            entry->session = nullptr;
            entry->loaded_paths.clear();
        }
    }

    {
        std::lock_guard<std::mutex> mapLock(g_map_mu);
        remote_sessions.erase(addr);
    }
    return 0;
}

void __catalyst__executor__launch(const char *addr, const char *entry_symbol, size_t num_inputs,
                                  void *const *input_descs, const size_t *input_ranks,
                                  const size_t *input_elem_sizes, size_t num_outputs,
                                  void *const *output_descs, const size_t *output_ranks,
                                  const size_t *output_elem_sizes)
{
    auto entry = find_or_create_entry(addr, /*create_if_missing=*/false);
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
