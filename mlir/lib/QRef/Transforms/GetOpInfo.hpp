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

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Value.h"

// A struct to store the register and the index of rQubits from a qref.get operation.
// This struct is intended to be the keys in `llvm::DenseMap`s.
struct rQubitGetOpInfo {
    mlir::Value reg;
    int64_t idxAttr;
    mlir::Value idx;

    rQubitGetOpInfo(mlir::Value _reg, mlir::Value _idx) : reg(_reg), idxAttr(-1), idx(_idx) {}

    rQubitGetOpInfo(mlir::Value _reg, int64_t _idxAttr)
        : reg(_reg), idxAttr(_idxAttr), idx(nullptr) {}

    bool operator==(const rQubitGetOpInfo &other) const {
        return reg == other.reg && idxAttr == other.idxAttr && idx == other.idx;
    }
};

namespace llvm {

// Boilerplate to enable using `rQubitGetOpInfo` as DenseMap keys.
template <> struct DenseMapInfo<rQubitGetOpInfo> {
    static inline rQubitGetOpInfo getEmptyKey() {
        return rQubitGetOpInfo(DenseMapInfo<mlir::Value>::getEmptyKey(), -1);
    }

    static inline rQubitGetOpInfo getTombstoneKey() {
        return rQubitGetOpInfo(DenseMapInfo<mlir::Value>::getTombstoneKey(), -2);
    }

    static unsigned getHashValue(const rQubitGetOpInfo &val) {
        return hash_combine(hash_value(val.reg.getAsOpaquePointer()), val.idxAttr,
                            val.idx ? static_cast<size_t>(hash_value(val.idx.getAsOpaquePointer()))
                                    : 0);
    }

    static bool isEqual(const rQubitGetOpInfo &lhs, const rQubitGetOpInfo &rhs) {
        return lhs == rhs;
    }
};
} // namespace llvm
