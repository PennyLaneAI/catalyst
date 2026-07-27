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

#include <cassert>
#include <functional>
#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"

#include "Catalyst/Analysis/ResourceExtension.h"

namespace catalyst {

/// Global registry of ResourceExtensionAnalysis providers.
/// Dialects / plugins self-register
// (e.g. pbc::registerPBCResourceExtensions, or REGISTER_RESOURCE_EXTENSION in a shared library).
class ResourceExtensionRegistry {
  public:
    using ExtensionProvider = std::function<std::unique_ptr<ResourceExtensionAnalysis>()>;

    static ResourceExtensionRegistry &get();

    void add(ExtensionProvider extensionProvider)
    {
        llvm::StringRef name = extensionProvider()->name();
        assert(!llvm::is_contained(names, name) && "ResourceExtension name already registered");
        names.push_back(name.str());
        extensionProviders.push_back(std::move(extensionProvider));
    }

    llvm::ArrayRef<ExtensionProvider> all() const { return extensionProviders; }

  private:
    llvm::SmallVector<ExtensionProvider> extensionProviders;

    // parallel to `extensionProviders`, avoids re-invoking them
    llvm::SmallVector<std::string> names;
};

/// Self-register a ResourceExtensionAnalysis factory.
// Example:
// REGISTER_RESOURCE_EXTENSION(std::make_unique<ABCAnalysis>());
#define RES_EXT_CONCAT_(a, b) a##b
#define RES_EXT_CONCAT(a, b) RES_EXT_CONCAT_(a, b)
#define REGISTER_RESOURCE_EXTENSION(CTOR_EXPR)                                                     \
    static const int LLVM_ATTRIBUTE_UNUSED RES_EXT_CONCAT(_resExtReg_, __COUNTER__) = []() {       \
        ::catalyst::ResourceExtensionRegistry::get().add([] { return CTOR_EXPR; });                \
        return 0;                                                                                  \
    }()

} // namespace catalyst
