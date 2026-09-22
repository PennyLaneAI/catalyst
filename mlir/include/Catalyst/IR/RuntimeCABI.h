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

#include <optional>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

namespace catalyst {

enum class RuntimeCABIKind {
    Void,    ///< "void"
    Pointer, ///< "ptr", "str", "buf", "out"
    Float,   ///< "f32", "f64"
    Integer, ///< "i1", "i8"..."i64", "u8"..."u64"
};

struct RuntimeCABIType {
    RuntimeCABIKind kind;
    unsigned width;
};

/// Classify a C ABI type name, return std::nullopt when the runtime does not support it.
inline std::optional<RuntimeCABIType> classifyRuntimeCABIType(llvm::StringRef name) {
    using Kind = RuntimeCABIKind;
    return llvm::StringSwitch<std::optional<RuntimeCABIType>>(name)
        .Case("void", RuntimeCABIType{Kind::Void, 0})
        .Cases({"ptr", "str", "buf", "out"}, RuntimeCABIType{Kind::Pointer, 0})
        .Case("f32", RuntimeCABIType{Kind::Float, 32})
        .Case("f64", RuntimeCABIType{Kind::Float, 64})
        .Case("i1", RuntimeCABIType{Kind::Integer, 1})
        .Cases({"i8", "u8"}, RuntimeCABIType{Kind::Integer, 8})
        .Cases({"i16", "u16"}, RuntimeCABIType{Kind::Integer, 16})
        .Cases({"i32", "u32"}, RuntimeCABIType{Kind::Integer, 32})
        .Cases({"i64", "u64"}, RuntimeCABIType{Kind::Integer, 64})
        .Default(std::nullopt);
}

} // namespace catalyst
