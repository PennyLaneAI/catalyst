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

#include <cstdlib>
#include <map>
#include <set>
#include <string>

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "Catalyst/IR/CatalystOps.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_RESOLVERUNTIMESYMBOLSPASS
#include "Catalyst/Transforms/Passes.h.inc"
} // namespace catalyst

namespace {

// The C++ functions a library exports, indexed by qualified name.
using ExportIndex = llvm::StringMap<SmallVector<std::string, 1>>;

// Whether a callee names a C++ function.
bool isCppName(StringRef callee) { return callee.contains("::") || callee.contains('('); }

// `text` without whitespace
std::string withoutSpaces(StringRef text) {
    std::string squeezed;
    for (char c : text) {
        if (!llvm::isSpace(c)) {
            squeezed += c;
        }
    }
    return squeezed;
}

// Collects the C++ functions the library exports.
void collectExports(const llvm::object::ObjectFile &object, ExportIndex &index) {
    llvm::ItaniumPartialDemangler demangler;
    char *buffer = nullptr;
    size_t size = 0;

    auto add = [&](const llvm::object::SymbolRef &sym) {
        llvm::Expected<uint32_t> flags = sym.getFlags();
        llvm::Expected<StringRef> name = sym.getName();
        if (!flags || !name) {
            llvm::consumeError(flags.takeError());
            llvm::consumeError(name.takeError());
            return;
        }
        if ((*flags & llvm::object::SymbolRef::SF_Undefined) ||
            !(*flags & llvm::object::SymbolRef::SF_Global)) {
            return;
        }
        StringRef symbol = *name;
        if (object.isMachO()) {
            symbol.consume_front("_"); // Mach-O has a prefix `_` for C symbols
        }
        if (!symbol.starts_with("_Z")) {
            return; // a C symbol
        }

        std::string mangled = symbol.str();
        if (demangler.partialDemangle(mangled.c_str()) || !demangler.isFunction()) {
            return;
        }
        char *qualified = demangler.getFunctionName(buffer, &size);
        if (!qualified) {
            return;
        }
        buffer = qualified;
        SmallVector<std::string, 1> &overloads = index[qualified];
        if (!llvm::is_contained(overloads, mangled)) {
            overloads.push_back(std::move(mangled));
        }
    };

    for (const llvm::object::SymbolRef &sym : object.symbols()) {
        add(sym);
    }

    if (const auto *elf = dyn_cast<llvm::object::ELFObjectFileBase>(&object)) {
        for (const llvm::object::ELFSymbolRef &sym : elf->getDynamicSymbolIterators()) {
            add(sym);
        }
    }
    std::free(buffer);
}

// The C++ functions the library at `path` exports.
llvm::Expected<ExportIndex> exportsOf(StringRef path) {
    auto binary = llvm::object::createBinary(path);
    if (!binary) {
        return binary.takeError();
    }
    ExportIndex exports;
    llvm::object::Binary *contents = binary->getBinary();
    if (auto *object = dyn_cast<llvm::object::ObjectFile>(contents)) {
        collectExports(*object, exports);
    } else if (auto *universal = dyn_cast<llvm::object::MachOUniversalBinary>(contents)) {
        for (const auto &slice : universal->objects()) {
            auto object = slice.getAsObjectFile();
            if (!object) {
                llvm::consumeError(object.takeError());
                continue;
            }
            collectExports(**object, exports);
        }
    } else {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "not a shared library or object file");
    }
    return exports;
}

// The libraries the modules enclosing `op` record.
SmallVector<StringRef> librariesOf(Operation *op) {
    SmallVector<StringRef> libraries;
    for (auto mod = op->getParentOfType<ModuleOp>(); mod; mod = mod->getParentOfType<ModuleOp>()) {
        if (auto paths = mod->getAttrOfType<ArrayAttr>("catalyst.runtime_artifacts")) {
            for (auto path : paths.getAsRange<StringAttr>()) {
                libraries.push_back(path.getValue());
            }
        }
    }
    return libraries;
}

} // namespace

namespace catalyst {

struct ResolveRuntimeSymbolsPass : impl::ResolveRuntimeSymbolsPassBase<ResolveRuntimeSymbolsPass> {
    using ResolveRuntimeSymbolsPassBase::ResolveRuntimeSymbolsPassBase;

    void runOnOperation() final {
        WalkResult result = getOperation()->walk([&](RuntimeCallOp op) {
            if (op.isDispatched()) {
                return WalkResult::advance();
            }

            if (isCppName(op.getCallee()) && mlir::failed(resolve(op))) {
                return WalkResult::interrupt();
            }

            return WalkResult::advance();
        });

        if (result.wasInterrupted()) {
            signalPassFailure();
        }
    }

  private:
    // stores the exports of the libraries, indexed by library path.
    std::map<std::string, ExportIndex> exports;

    // Replaces the C++ function `op` names with the symbol from the library
    LogicalResult resolve(RuntimeCallOp op) {
        StringRef callee = op.getCallee();
        StringRef name = callee.split('(').first.trim();
        bool bySignature = callee.contains('(');
        std::string wanted = withoutSpaces(callee);

        SmallVector<StringRef> libraries = librariesOf(op);
        if (libraries.empty()) {
            return op.emitOpError() << "cannot resolve the C++ function '" << callee
                                    << "' because no library was specified";
        }

        // Find the symbols that match the function name.
        std::set<std::string> matches;
        std::set<std::string> candidates;
        for (StringRef library : libraries) {
            auto known = exports.find(library.str());
            if (known == exports.end()) {
                auto read = exportsOf(library);
                if (!read) {
                    return op.emitOpError() << "cannot read the symbols of '" << library
                                            << "': " << llvm::toString(read.takeError());
                }
                known = exports.emplace(library.str(), std::move(*read)).first;
            }
            auto overloads = known->second.find(name);
            if (overloads == known->second.end()) {
                continue;
            }
            for (const std::string &symbol : overloads->second) {
                std::string signature = llvm::demangle(symbol);
                if (!bySignature || withoutSpaces(signature) == wanted) {
                    matches.insert(symbol);
                }
                candidates.insert(std::move(signature));
            }
        }

        if (matches.size() == 1) {
            op.setCallee(*matches.begin());
            return success();
        }
        auto diag = op.emitOpError();
        if (matches.empty()) {
            diag << "cannot resolve the C++ function '" << callee
                 << "' because no library exports it";
        } else {
            diag << "cannot resolve the C++ function '" << callee
                 << "' because it is overloaded; give its full signature";
        }
        if (!candidates.empty()) {
            diag << ". Libraries exporting '" << name << "': " << llvm::join(candidates, "; ");
        }
        return diag;
    }
};

} // namespace catalyst
