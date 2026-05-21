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

#include "llvm/Support/DynamicLibrary.h"

#include "Quantum/Transforms/DecompCallbacks.h"

namespace catalyst::quantum {

namespace {

// Cache the resolution attempt so
// repeated calls don't keep paying
// the symbol lookup cost.
std::atomic<bool> resolutionAttempted{false};

using RegisterFn = void (*)();

RegisterFn resolveRegisterSymbol()
{
    void *addr =
        llvm::sys::DynamicLibrary::SearchForAddressOfSymbol("registerPythonDecompCallback");
    return reinterpret_cast<RegisterFn>(addr);
}

} // namespace

bool loadPythonCallbackPlugin()
{
    if (getDecompCallback()) {
        return true;
    }

    if (resolutionAttempted.exchange(true)) {
        return getDecompCallback() != nullptr;
    }

    RegisterFn reg = resolveRegisterSymbol();
    if (!reg) {
        return false;
    }

    reg();
    return getDecompCallback() != nullptr;
}

} // namespace catalyst::quantum
