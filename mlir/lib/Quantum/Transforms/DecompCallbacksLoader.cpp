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
#include <string>

#include "llvm/ADT/SmallString.h"
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

#if defined(__APPLE__)
constexpr const char *kPluginFileName = "libQuantumPythonCallbacks.dylib";
#else
constexpr const char *kPluginFileName = "libQuantumPythonCallbacks.so";
#endif

// Resolve the plugin path. Search order:
// 1. $CATALYST_PYTHON_CALLBACK_PLUGIN (explicit override).
// 2. <exe_dir>/../lib/libQuantumPythonCallbacks.so (build & install)
// 3. <exe_dir>/libQuantumPythonCallbacks.so (alongside)
std::string resolvePluginPath()
{
    if (auto override_ = llvm::sys::Process::GetEnv("CATALYST_PYTHON_CALLBACK_PLUGIN")) {
        return *override_;
    }

    std::string exe = llvm::sys::fs::getMainExecutable(nullptr, nullptr);
    if (exe.empty()) {
        return {};
    }
    llvm::SmallString<256> exeDir(exe);
    llvm::sys::path::remove_filename(exeDir);

    for (llvm::StringRef rel : {"../lib", "."}) {
        llvm::SmallString<256> candidate(exeDir);
        llvm::sys::path::append(candidate, rel, kPluginFileName);
        if (llvm::sys::fs::exists(candidate)) {
            return std::string(candidate);
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
    std::string err;
    auto h = llvm::sys::DynamicLibrary::getPermanentLibrary(where.str().c_str(), &err);
    if (!h.isValid()) {
        llvm::errs() << "[decomp-callbacks-loader] libpython dlopen of '" << where
                     << "' failed: " << err << "\n";
        return false;
    }
    // FIXME(Ali): remove this after testing
    llvm::errs() << "[decomp-callbacks-loader] libpython loaded: " << where << "\n";
    return true;
}

// Try, in order:
//   1. $CATALYST_LIBPYTHON: explicit user override (any deployment)
//   2. CATALYST_LIBPYTHON_PATH: absolute path the configure-time Python uses
//   3. CATALYST_LIBPYTHON_SONAME: bare SONAME via the dynamic loader search (manylinux wheels)
void ensureLibpythonLoaded()
{
    if (auto over = llvm::sys::Process::GetEnv("CATALYST_LIBPYTHON")) {
        if (tryLoadLibpython(*over))
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
    llvm::errs() << "[decomp-callbacks-loader] no libpython candidate succeeded; "
                    "the plugin dlopen will likely fail with undefined symbols\n";
}

RegisterFn loadAndResolve()
{
    std::string path = resolvePluginPath();
    if (path.empty()) {
        llvm::errs() << "[decomp-callbacks-loader] plugin path could not be resolved\n";
        return nullptr;
    }
    // FIXME(Ali): remove this after testing
    llvm::errs() << "[decomp-callbacks-loader] plugin resolved at: " << path << "\n";

    ensureLibpythonLoaded();

    std::string err;
    auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(path.c_str(), &err);
    if (!lib.isValid()) {
        llvm::errs() << "[decomp-callbacks-loader] dlopen('" << path << "') failed: " << err
                     << "\n";
        return nullptr;
    }

    auto *sym =
        reinterpret_cast<RegisterFn>(lib.getAddressOfSymbol("registerPythonDecompCallback"));
    if (!sym) {
        llvm::errs() << "[decomp-callbacks-loader] dlopen succeeded but symbol "
                        "'registerPythonDecompCallback' not found in '"
                     << path << "'\n";
    }
    return sym;
}

} // namespace

bool loadPythonCallbackPlugin()
{
    if (getLowerPauliRot()) {
        return true;
    }

    if (resolutionAttempted.exchange(true)) {
        return getLowerPauliRot() != nullptr;
    }

    RegisterFn reg = loadAndResolve();
    if (!reg) {
        return false;
    }

    reg();
    return getLowerPauliRot() != nullptr;
}

} // namespace catalyst::quantum
