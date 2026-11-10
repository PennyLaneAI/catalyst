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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace catalyst {

// Make a buffer contiguous
inline mlir::FailureOr<mlir::Value>
makeContiguous(mlir::RewriterBase &rewriter, mlir::Value buffer,
               const mlir::bufferization::BufferizationOptions &options) {
    auto memrefType = mlir::cast<mlir::MemRefType>(buffer.getType());

    // The buffer is already contiguous.
    if (memrefType.getLayout().isIdentity()) {
        return buffer;
    }

    // Materialize a contiguous copy.
    mlir::MemRefType contiguous = mlir::MemRefType::get(
        memrefType.getShape(), memrefType.getElementType(),
        /*layout=*/mlir::MemRefLayoutAttrInterface{}, memrefType.getMemorySpace());
    return mlir::bufferization::castOrReallocMemRefValue(rewriter, buffer, contiguous, options);
}

} // namespace catalyst
