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

#include <cstdint>
#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"

namespace catalyst::pbc {

// Resolve bounded classical tensors without rewriting the circuit. Integer wire
// arithmetic and structural tensor operations are handled locally; other pure
// classical operations can use the runtime evaluator. Copies retain captures;
// binding a callee argument invalidates previously derived constants.
class WireIndexResolver {
    using PendingValues = llvm::SmallVector<mlir::Value>;
    llvm::DenseMap<mlir::Value, mlir::Attribute> bindings;
    mutable llvm::DenseMap<mlir::Value, mlir::Attribute> cache;

    void invalidate(mlir::Value value);
    mlir::Attribute resolve(mlir::Value value, PendingValues &pending) const;
    mlir::Attribute evaluate(mlir::Value value, PendingValues &pending) const;

  public:
    void bind(mlir::Value target, mlir::Value source, const WireIndexResolver &from);

    // Restore a resolved value from a call summary; null records an unknown value.
    void bindConstant(mlir::Value target, mlir::Attribute value);

    // Reevaluate an operation result when its enclosing loop executes it again.
    void forget(mlir::Value value);

    // Return a scalar numeric attribute or bounded DenseElementsAttr, or null.
    // Attributes are uniqued and can be included in call-summary cache keys.
    mlir::Attribute resolveConstant(mlir::Value value) const;

    // Resolve a scalar integer/index value or a rank-zero integer tensor.
    std::optional<int64_t> resolveIndex(mlir::Value value) const;
};

} // namespace catalyst::pbc
