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

#include "Quantum/Transforms/DecompCallbacksLoader.h"

#include <atomic>
#include <dlfcn.h>
#include <string>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include "Quantum/Transforms/DecompCallbacks.h"

namespace catalyst::quantum {

namespace {

// Cache the resolution attempt so
// repeated calls don't keep paying
// the symbol lookup cost.
std::atomic<bool> resolutionAttempted{false};

using RegisterFn = void (*)();

static const std::vector<std::string> kAlternativePluginNames = {
    "libQuantumPythonCallbacks.so",
    "libQuantumPythonCallbacks.abi3.so",
    "libQuantumPythonCallbacks.cpython-311-darwin.so",
    "libQuantumPythonCallbacks.dylib",
    "libQuantumPythonCallbacks.abi3.dylib",
    "libQuantumPythonCallbacks.cpython-311-aarch64-linux-gnu.so",
    "libQuantumPythonCallbacks.cpython-311-x86_64-linux-gnu.so",
};

// Resolve the plugin path. Search order:
// 1. $CATALYST_PYTHON_CALLBACK_PLUGIN (explicit override).
// 2. function parameter
// 3. <exe_dir>/../lib/libQuantumPythonCallbacks.so (build & install)
// 4. <exe_dir>/libQuantumPythonCallbacks.so (alongside)
std::string resolvePluginPath(std::string callbackPluginPath)
{
    if (auto override_ = llvm::sys::Process::GetEnv("CATALYST_PYTHON_CALLBACK_PLUGIN")) {
        return *override_;
    }

    auto checkCandidate = [](llvm::SmallString<256> &path) -> std::string {
        if (!llvm::sys::fs::exists(path)) {
            return {};
        }

        if (llvm::sys::fs::is_directory(path)) {
            for (const auto &name : kAlternativePluginNames) {
                llvm::SmallString<256> candidate(path);
                llvm::sys::path::append(candidate, name);
                if (llvm::sys::fs::exists(candidate)) {
                    return std::string(candidate);
                }
            }
            return {};
        }
        return std::string(path);
    };

    llvm::SmallString<256> inputPath(callbackPluginPath);
    std::string result = checkCandidate(inputPath);
    if (!result.empty()) {
        LDBG(2) << "Found libQuantumPythonCallbacks path from function parameter:" << result
                << "\n";
        return result;
    }

    std::string exe = llvm::sys::fs::getMainExecutable(nullptr, nullptr);
    if (exe.empty()) {
        return {};
    }
    llvm::SmallString<256> exeDir(exe);
    llvm::sys::path::remove_filename(exeDir);

    for (llvm::StringRef rel : {"../lib", "."}) {
        llvm::SmallString<256> base(exeDir);
        llvm::sys::path::append(base, rel);
        result = checkCandidate(base);
        if (!result.empty()) {
            return result;
        }
    }
    return {};
}

// Ensure libpython is loaded into the process before the plugin .so is opened.
bool tryLoadLibpython(llvm::StringRef where)
{
    if (where.empty()) {
        return false;
    }

    auto h = ::dlopen(where.str().c_str(), RTLD_GLOBAL | RTLD_LAZY);
    if (!h) {
        llvm::errs() << "[decomp-callbacks-loader] libpython dlopen failed: " << ::dlerror()
                     << "\n";
        return false;
    }
    LDBG() << "[decomp-callbacks-loader] libpython loaded from: " << where << "\n";
    return true;
}

// Try, in order:
//   1. $CATALYST_LIBPYTHON: explicit user override (any deployment)
//   2. function parameter
//   3. CATALYST_LIBPYTHON_PATH: absolute path the configure-time Python uses
//   4. CATALYST_LIBPYTHON_SONAME: bare SONAME via the dynamic loader search
//   (manylinux wheels)
void ensureLibpythonLoaded(std::string libpythonPath)
{
    if (auto over = llvm::sys::Process::GetEnv("CATALYST_LIBPYTHON")) {
        if (tryLoadLibpython(*over))
            LDBG() << "Found python from CATALYST_LIBPYTHON environment variable:" << over;
        return;
    }

    if (tryLoadLibpython(libpythonPath)) {
        LDBG() << "Found python from input path: " << libpythonPath;
        return;
    }

#ifdef CATALYST_LIBPYTHON_PATH
    if (tryLoadLibpython(CATALYST_LIBPYTHON_PATH))
        return;
#endif
#ifdef CATALYST_LIBPYTHON_SONAME
    if (tryLoadLibpython(CATALYST_LIBPYTHON_SONAME))
        return;
#endif
    llvm::errs() << "[decomp-callbacks-loader] libpython could not be resolved, "
                    "the plugin will likely "
                    "fail with undefined symbols.\n";
}

RegisterFn loadAndResolve(std::string callbackPluginPath, std::string libpythonPath)
{
    std::string path = resolvePluginPath(callbackPluginPath);
    if (path.empty()) {
        llvm::errs() << "[decomp-callbacks-loader] The plugin path could not be resolved.\n";
        return nullptr;
    }
    LDBG() << "[decomp-callbacks-loader] plugin resolved at: " << path << "\n";

    ensureLibpythonLoaded(libpythonPath);

    // Open with RTLD_GLOBAL so nanobind's internals map to Python's runtime
    // memory space
    void *libHandle = ::dlopen(path.c_str(), RTLD_GLOBAL | RTLD_LAZY);
    if (!libHandle) {
        llvm::errs() << "[decomp-callbacks-loader] dlopen('" << path << "') failed: " << ::dlerror()
                     << "\n";
        return nullptr;
    }

    auto *sym = reinterpret_cast<RegisterFn>(::dlsym(libHandle, "registerPythonDecompCallback"));
    if (!sym) {
        llvm::errs() << "[decomp-callbacks-loader] dlopen succeeded but symbol "
                        "'registerPythonDecompCallback' not found in '"
                     << path << "'\n";
    }
    return sym;
}

} // namespace

bool loadPythonCallbackPlugin(std::string callbackPluginPath, std::string libpythonPath)
{
    if (getLowerPauliRot()) {
        return true;
    }

    if (resolutionAttempted.exchange(true)) {
        return getLowerPauliRot() != nullptr;
    }

    RegisterFn reg = loadAndResolve(callbackPluginPath, libpythonPath);
    if (!reg) {
        return false;
    }

    reg();
    return getLowerPauliRot() != nullptr;
}

} // namespace catalyst::quantum
