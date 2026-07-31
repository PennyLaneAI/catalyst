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

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/WithColor.h"

#include "Catalyst/Analysis/ResourceResultExtension.h"

namespace catalyst {

// Global registry of ResourceAnalysisExtension providers.
// Dialects / plugins self-register via the ResourceAnalysisRegistry::add method,
// or the REGISTER_RESOURCE_ANALYSIS_EXTENSION macro.
class ResourceAnalysisRegistry {
  public:
    using ExtensionProvider = std::function<std::unique_ptr<ResourceAnalysisExtension>()>;

    static ResourceAnalysisRegistry &get();

    void add(ExtensionProvider extensionProvider) {
        std::string name = extensionProvider()->name().str();
        if (llvm::is_contained(names, name)) {
            llvm::WithColor::warning() << "ResourceAnalysisExtension '" << name
                                       << "' is already registered; ignoring duplicate\n";
            return;
        }
        names.push_back(std::move(name));
        extensionProviders.push_back(std::move(extensionProvider));
    }

    llvm::ArrayRef<ExtensionProvider> all() const { return extensionProviders; }

  private:
    llvm::SmallVector<ExtensionProvider> extensionProviders;

    // parallel to `extensionProviders`, avoids re-invoking them
    llvm::SmallVector<std::string> names;
};

// Self-register a ResourceAnalysisExtension factory.
// Example:
// REGISTER_RESOURCE_ANALYSIS_EXTENSION(std::make_unique<ABCAnalysis>());
#define RES_EXT_CONCAT_(a, b) a##b
#define RES_EXT_CONCAT(a, b) RES_EXT_CONCAT_(a, b)
#define REGISTER_RESOURCE_ANALYSIS_EXTENSION(CTOR_EXPR)                                            \
    static const int LLVM_ATTRIBUTE_UNUSED _resExtReg_##__COUNTER__ = []() {       \
        ::catalyst::ResourceAnalysisRegistry::get().add([] { return CTOR_EXPR; });                 \
        return 0;                                                                                  \
    }()

} // namespace catalyst
