// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "llvm/Support/MemoryBuffer.h"

#include "mlir/IR/BuiltinTypes.h"

#include "Catalyst/Utils/JSONUtils.h"

using namespace mlir;

namespace catalyst {

Attribute jsonToAttribute(MLIRContext *ctx, const llvm::json::Value &json)
{
    if (auto str = json.getAsString()) {
        return StringAttr::get(ctx, *str);
    }
    if (auto num = json.getAsInteger()) {
        return IntegerAttr::get(IntegerType::get(ctx, 64), *num);
    }
    if (auto num = json.getAsNumber()) {
        return FloatAttr::get(Float64Type::get(ctx), *num);
    }
    if (auto b = json.getAsBoolean()) {
        return BoolAttr::get(ctx, *b);
    }
    if (auto arr = json.getAsArray()) {
        SmallVector<Attribute> attrs;
        for (const auto &elem : *arr) {
            attrs.push_back(jsonToAttribute(ctx, elem));
        }
        return ArrayAttr::get(ctx, attrs);
    }
    if (auto *obj = json.getAsObject()) {
        SmallVector<NamedAttribute> entries;
        for (const auto &kv : *obj) {
            StringRef key = kv.first;
            entries.emplace_back(StringAttr::get(ctx, key), jsonToAttribute(ctx, kv.second));
        }
        // Sort entries by name for DictionaryAttr
        llvm::sort(entries, [](const NamedAttribute &lhs, const NamedAttribute &rhs) {
            return lhs.getName().getValue() < rhs.getName().getValue();
        });
        return DictionaryAttr::get(ctx, entries);
    }
    // null
    return UnitAttr::get(ctx);
}

FailureOr<DictionaryAttr> loadJsonFileAsDict(MLIRContext *ctx, StringRef filePath)
{
    auto fileOrErr = llvm::MemoryBuffer::getFile(filePath);
    if (!fileOrErr) {
        return failure();
    }

    auto json = llvm::json::parse((*fileOrErr)->getBuffer());
    if (!json) {
        llvm::errs() << "Failed to parse JSON: " << llvm::toString(json.takeError()) << "\n";
        return failure();
    }

    auto *obj = json->getAsObject();
    if (!obj) {
        llvm::errs() << "JSON file must contain a root object\n";
        return failure();
    }

    // Convert JSON object to DictionaryAttr
    SmallVector<NamedAttribute> entries;
    for (const auto &kv : *obj) {
        StringRef key = kv.first;
        entries.emplace_back(StringAttr::get(ctx, key), jsonToAttribute(ctx, kv.second));
    }
    llvm::sort(entries, [](const NamedAttribute &lhs, const NamedAttribute &rhs) {
        return lhs.getName().getValue() < rhs.getName().getValue();
    });

    return DictionaryAttr::get(ctx, entries);
}

} // namespace catalyst
